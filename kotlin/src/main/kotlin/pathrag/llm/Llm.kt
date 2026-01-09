package pathrag.llm

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.model.ollama.OllamaEmbeddingModel
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import dev.langchain4j.model.output.Response
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import pathrag.prompt.Prompts
import pathrag.utils.EmbeddingFunc
import java.util.concurrent.ConcurrentHashMap
import kotlin.random.Random

private val logger = KotlinLogging.logger("PathRAG-LLM")
private const val DEFAULT_CHAT_MODEL = "gpt-4o-mini"
private const val DEFAULT_OLLAMA_MODEL = "llama3"
private const val DEFAULT_EMBED_MODEL = "text-embedding-3-small"
private const val DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
private const val DEFAULT_EMBED_DIM = 1536
private const val DEFAULT_EMBED_CTX = 8192
private const val DEFAULT_OLLAMA_EMBED_DIM = 768

private val chatModels = ConcurrentHashMap<String, ChatModel>()
private val embeddingModels = ConcurrentHashMap<String, EmbeddingModel>()

/**
 * Send a chat completion request to OpenAI (with retries/backoff) and return the model's response.
 *
 * If the OPENAI_API_KEY environment variable is not set, returns a deterministic stub:
 * - When `keywordExtraction` is true: a JSON-like string with `high_level_keywords` and `low_level_keywords`.
 * - Otherwise: a concatenation of `systemPrompt`, `prompt`, and an indicator of history size truncated to `maxTokens` (or 4000).
 *
 * The call will retry on transient failures (configured by OPENAI_RETRY_ATTEMPTS and OPENAI_RETRY_BACKOFF_MS) and, if all attempts fail, returns Prompts.FAIL_RESPONSE.
 *
 * @param historyMessages A list of maps representing prior messages; each map may contain "role" and "content" keys used to build conversation history.
 * @param keywordExtraction When true, request/produce keyword-extraction-formatted output instead of a normal chat response.
 * @param maxTokens When no OPENAI_API_KEY is present, limits the length of the returned stubbed response; otherwise not used directly.
 * @param hashingKv Opaque value accepted for compatibility with callers (not used by this function).
 * @return The chat model's textual response, a structured stub when the API key is absent, or Prompts.FAIL_RESPONSE if remote calls fail after retries.
 */
@Suppress("TooGenericExceptionCaught")
suspend fun openAiComplete(
    model: String,
    prompt: String,
    systemPrompt: String? = null,
    historyMessages: List<Map<String, String>> = emptyList(),
    keywordExtraction: Boolean = false,
    stream: Boolean = false,
    maxTokens: Int? = null,
    hashingKv: Any? = null,
): String {
    val apiKey = System.getenv("OPENAI_API_KEY")
    if (apiKey.isNullOrBlank()) {
        logger.warn { "OPENAI_API_KEY not set. Falling back to stubbed response." }
        delay(50)
        return if (keywordExtraction) {
            """{"high_level_keywords": ["${prompt.take(10)}"], "low_level_keywords": ["${prompt.takeLast(10)}"]}"""
        } else {
            val history = if (historyMessages.isEmpty()) "" else " History size=${historyMessages.size}."
            "${systemPrompt.orEmpty()} $prompt$history".trim().take(maxTokens ?: 4000)
        }
    }

    val logRequests = System.getenv("OPENAI_LOG_REQUESTS")?.toBoolean() ?: false
    val logResponses = System.getenv("OPENAI_LOG_RESPONSES")?.toBoolean() ?: false
    val baseUrl = System.getenv("OPENAI_API_BASE")
    val modelName = model.ifBlank { DEFAULT_CHAT_MODEL }
    val chatModel: ChatModel =
        chatModels.computeIfAbsent("$modelName|$baseUrl") {
            val builder =
                OpenAiChatModel
                    .builder()
                    .apiKey(apiKey)
                    .logRequests(logRequests)
                    .logResponses(logResponses)
                    .modelName(modelName)
            baseUrl?.takeIf { it.isNotBlank() }?.let { builder.baseUrl(it) }
            builder.build()
        }

    val fullPrompt =
        buildString {
            if (!systemPrompt.isNullOrBlank()) {
                appendLine(systemPrompt)
            }
            if (historyMessages.isNotEmpty()) {
                historyMessages.forEach { msg ->
                    appendLine("${msg["role"] ?: "user"}: ${msg["content"] ?: ""}")
                }
            }
            append(prompt)
        }

    val result: String =
        withContext(Dispatchers.IO) {
            val maxAttempts = (System.getenv("OPENAI_RETRY_ATTEMPTS")?.toIntOrNull() ?: 3).coerceAtLeast(1)
            val backoffMs = (System.getenv("OPENAI_RETRY_BACKOFF_MS")?.toLongOrNull() ?: 500L).coerceAtLeast(0L)
            var lastError: IllegalStateException? = null
            repeat(maxAttempts) { attempt ->
                try {
                    return@withContext chatModel.chat(fullPrompt)
                } catch (e: IllegalStateException) {
                    lastError = e
                    logger.warn(e) { "OpenAI chat attempt ${attempt + 1} failed for model $modelName" }
                    if (attempt < maxAttempts - 1) {
                        delay(backoffMs)
                    }
                }
            }
            logger.error(lastError) { "OpenAI chat call failed after $maxAttempts attempts for model $modelName" }
            Prompts.FAIL_RESPONSE
        }
    return result
}

/**
 * Request a completion or keyword-extraction response from an Ollama chat model.
 *
 * @param model The Ollama model name to use; if blank the environment variable `OLLAMA_MODEL` or the default model is used.
 * @param prompt The primary user prompt to send to the model.
 * @param systemPrompt Optional system prompt prepended to the conversation.
 * @param historyMessages Conversation history as a list of maps with keys `"role"` and `"content"`; each entry is appended before `prompt`.
 * @param keywordExtraction When `true`, request the model produce keyword extraction instead of a standard completion.
 * @param stream Reserved for streaming usage (not used by this function implementation).
 * @param maxTokens Reserved for token-limit usage (not used by this function implementation).
 * @param hashingKv Reserved hook for optional hashing/storage keys (not used by this function implementation).
 * @return The model's text response, or a predefined failure string if the call fails after retries.
 */
@Suppress("TooGenericExceptionCaught")
suspend fun ollamaComplete(
    model: String,
    prompt: String,
    systemPrompt: String? = null,
    historyMessages: List<Map<String, String>> = emptyList(),
    keywordExtraction: Boolean = false,
    stream: Boolean = false,
    maxTokens: Int? = null,
    hashingKv: Any? = null,
): String {
    val baseUrl = System.getenv("OLLAMA_BASE_URL") ?: System.getenv("OLLAMA_HOST") ?: "http://localhost:11434"
    val modelName = model.ifBlank { System.getenv("OLLAMA_MODEL") ?: DEFAULT_OLLAMA_MODEL }
    val chatModel: ChatModel =
        chatModels.computeIfAbsent("ollama|$modelName|$baseUrl") {
            OllamaChatModel
                .builder()
                .baseUrl(baseUrl)
                .modelName(modelName)
                .build()
        }

    val fullPrompt =
        buildString {
            if (!systemPrompt.isNullOrBlank()) {
                appendLine(systemPrompt)
            }
            if (historyMessages.isNotEmpty()) {
                historyMessages.forEach { msg ->
                    appendLine("${msg["role"] ?: "user"}: ${msg["content"] ?: ""}")
                }
            }
            append(prompt)
        }

    return withContext(Dispatchers.IO) {
        val maxAttempts = (System.getenv("OLLAMA_RETRY_ATTEMPTS")?.toIntOrNull() ?: 3).coerceAtLeast(1)
        val backoffMs = (System.getenv("OLLAMA_RETRY_BACKOFF_MS")?.toLongOrNull() ?: 500L).coerceAtLeast(0L)
        var lastError: IllegalStateException? = null
        repeat(maxAttempts) { attempt ->
            try {
                return@withContext chatModel.chat(fullPrompt)
            } catch (e: IllegalStateException) {
                lastError = e
                logger.warn(e) { "Ollama chat attempt ${attempt + 1} failed for model $modelName" }
                if (attempt < maxAttempts - 1) {
                    delay(backoffMs)
                }
            }
        }
        logger.error(lastError) { "Ollama chat call failed after $maxAttempts attempts for model $modelName" }
        Prompts.FAIL_RESPONSE
    }
}

/**
 * Generate embeddings for the provided texts using the configured OpenAI embedding model.
 *
 * Blank strings are filtered out; if all inputs are blank an empty list is returned.
 * If the `OPENAI_API_KEY` environment variable is not set, deterministic stub embeddings
 * are generated for each non-blank input.
 *
 * @param inputs The list of input texts to embed. Order of returned embeddings corresponds to the order of non-blank inputs.
 * @return A list of embedding vectors (DoubleArray), one per non-blank input.
 * @throws IllegalStateException If the OpenAI embedding call fails after the configured retry attempts.
 */
@Suppress("TooGenericExceptionCaught")
suspend fun openAiEmbedding(inputs: List<String>): List<DoubleArray> {
    val apiKey = System.getenv("OPENAI_API_KEY")
    val sanitized = inputs.filter { it.isNotBlank() }
    if (sanitized.isEmpty()) return emptyList()
    if (apiKey.isNullOrBlank()) {
        logger.warn { "OPENAI_API_KEY not set. Falling back to stubbed embeddings." }
        return sanitized.map { text ->
            val seed = text.hashCode()
            val random = Random(seed)
            DoubleArray(1536) { random.nextDouble() }
        }
    }
    val baseUrl = System.getenv("OPENAI_API_BASE")
    val modelName = System.getenv("OPENAI_EMBEDDING_MODEL") ?: DEFAULT_EMBED_MODEL
    val embedModel: EmbeddingModel =
        embeddingModels.computeIfAbsent("$modelName|$baseUrl") {
            val builder = OpenAiEmbeddingModel.builder().apiKey(apiKey).modelName(modelName)
            baseUrl?.takeIf { it.isNotBlank() }?.let { builder.baseUrl(it) }
            builder.build()
        }

    return withContext(Dispatchers.IO) {
        val maxAttempts = (System.getenv("OPENAI_RETRY_ATTEMPTS")?.toIntOrNull() ?: 3).coerceAtLeast(1)
        val backoffMs = (System.getenv("OPENAI_RETRY_BACKOFF_MS")?.toLongOrNull() ?: 500L).coerceAtLeast(0L)
        var lastError: IllegalStateException? = null
        repeat(maxAttempts) { attempt ->
            try {
                val segments = sanitized.map { TextSegment.from(it) }
                val response: Response<List<Embedding>> = embedModel.embedAll(segments)
                return@withContext response.content().map { embedding ->
                    val vector = embedding.vector()
                    DoubleArray(vector.size) { idx -> vector[idx].toDouble() }
                }
            } catch (e: IllegalStateException) {
                lastError = e
                logger.warn(e) { "OpenAI embedding attempt ${attempt + 1} failed for model $modelName" }
                if (attempt < maxAttempts - 1) {
                    delay(backoffMs)
                }
            }
        }
        logger.error(lastError) { "OpenAI embedding call failed after $maxAttempts attempts for model $modelName" }
        throw IllegalStateException("OpenAI embedding call failed after $maxAttempts attempts for model $modelName", lastError)
    }
}

/**
 * Generate embedding vectors for the provided texts using an Ollama embedding model.
 *
 * Blank or whitespace-only inputs are ignored; if no valid inputs remain, an empty list is returned.
 *
 * The function will retry on transient failures according to environment-configured retry attempts and backoff, and will throw if all attempts fail.
 *
 * @param inputs List of input texts to embed.
 * @return A list of embedding vectors; each element is a DoubleArray representing the embedding for the corresponding non-blank input (order preserved).
 * @throws IllegalStateException if the embedding operation fails after the configured number of retry attempts.
 */
@Suppress("TooGenericExceptionCaught")
suspend fun ollamaEmbedding(inputs: List<String>): List<DoubleArray> {
    val sanitized = inputs.filter { it.isNotBlank() }
    if (sanitized.isEmpty()) return emptyList()
    val baseUrl = System.getenv("OLLAMA_BASE_URL") ?: System.getenv("OLLAMA_HOST") ?: "http://localhost:11434"
    val modelName = System.getenv("OLLAMA_EMBED_MODEL") ?: DEFAULT_OLLAMA_EMBED_MODEL
    val embedModel: EmbeddingModel =
        embeddingModels.computeIfAbsent("ollama|$modelName|$baseUrl") {
            OllamaEmbeddingModel
                .builder()
                .baseUrl(baseUrl)
                .modelName(modelName)
                .build()
        }

    return withContext(Dispatchers.IO) {
        val maxAttempts = (System.getenv("OLLAMA_RETRY_ATTEMPTS")?.toIntOrNull() ?: 3).coerceAtLeast(1)
        val backoffMs = (System.getenv("OLLAMA_RETRY_BACKOFF_MS")?.toLongOrNull() ?: 500L).coerceAtLeast(0L)
        var lastError: IllegalStateException? = null
        repeat(maxAttempts) { attempt ->
            try {
                val segments = sanitized.map { TextSegment.from(it) }
                val response: Response<List<Embedding>> = embedModel.embedAll(segments)
                return@withContext response.content().map { embedding ->
                    val vector = embedding.vector()
                    DoubleArray(vector.size) { idx -> vector[idx].toDouble() }
                }
            } catch (e: IllegalStateException) {
                lastError = e
                logger.warn(e) { "Ollama embedding attempt ${attempt + 1} failed for model $modelName" }
                if (attempt < maxAttempts - 1) {
                    delay(backoffMs)
                }
            }
        }
        logger.error(lastError) { "Ollama embedding call failed after $maxAttempts attempts for model $modelName" }
        throw IllegalStateException("Ollama embedding call failed after $maxAttempts attempts for model $modelName", lastError)
    }
}

/**
     * Create an EmbeddingFunc configured from environment variables.
     *
     * The returned EmbeddingFunc contains an embedding dimension, a maximum token size,
     * and an embedding function chosen according to the embedding provider configuration
     * (uses Ollama when the provider is "ollama", otherwise uses the OpenAI embedding function).
     *
     * @return An EmbeddingFunc configured with the resolved `embeddingDim`, `maxTokenSize`, and `func`.
     */
fun defaultEmbeddingFunc(): EmbeddingFunc =
    embeddingModelConfig().let { (provider, dim, ctx) ->
        val func =
            when (provider) {
                "ollama" -> ::ollamaEmbedding
                else -> ::openAiEmbedding
            }
        EmbeddingFunc(
            embeddingDim = dim,
            maxTokenSize = ctx,
            func = func,
        )
    }

private fun embeddingModelConfig(): Triple<String, Int, Int> {
    val provider = System.getenv("EMBED_PROVIDER")?.lowercase() ?: "openai"
    return when (provider) {
        "ollama" -> {
            val dim = System.getenv("OLLAMA_EMBED_DIM")?.toIntOrNull() ?: DEFAULT_OLLAMA_EMBED_DIM
            Triple("ollama", dim, DEFAULT_EMBED_CTX)
        }

        else -> {
            val modelName = System.getenv("OPENAI_EMBEDDING_MODEL") ?: DEFAULT_EMBED_MODEL
            val dim =
                when {
                    modelName.contains("3-large", ignoreCase = true) -> {
                        3072
                    }

                    modelName.contains("3-small", ignoreCase = true) -> {
                        1536
                    }

                    else -> {
                        logger.warn { "Unrecognized embedding model '$modelName'; using default dim $DEFAULT_EMBED_DIM" }
                        DEFAULT_EMBED_DIM
                    }
                }
            val ctx =
                when {
                    modelName.contains("3-large", ignoreCase = true) ||
                        modelName.contains("3-small", ignoreCase = true) -> 8192

                    else -> DEFAULT_EMBED_CTX
                }
            Triple("openai", dim, ctx)
        }
    }
}