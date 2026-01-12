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
import dev.langchain4j.service.AiServices
import dev.langchain4j.service.SystemMessage
import dev.langchain4j.service.UserMessage
import dev.langchain4j.service.V
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
private const val DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

private val chatModels = ConcurrentHashMap<String, ChatModel>()
private val embeddingModels = ConcurrentHashMap<String, EmbeddingModel>()
private val chatServices = ConcurrentHashMap<String, TemplateChatService>()

private interface TemplateChatService {
    @SystemMessage("{{systemPrompt}}")
    @UserMessage("{{history}}{{prompt}}")
    fun chat(
        @V("systemPrompt") systemPrompt: String,
        @V("history") history: String,
        @V("prompt") prompt: String,
    ): String
}

private fun templateClient(
    key: String,
    model: ChatModel,
): TemplateChatService = chatServices.computeIfAbsent(key) { AiServices.create(TemplateChatService::class.java, model) }

private fun buildHistoryBlock(historyMessages: List<Map<String, String>>): String {
    if (historyMessages.isEmpty()) return ""
    val joined =
        historyMessages.joinToString("\n") { msg ->
            val role = msg["role"] ?: "user"
            val content = msg["content"] ?: ""
            "$role: $content"
        }
    return "$joined\n"
}

/**
 * Obtain a completion from an OpenAI chat model for the given prompt.
 *
 * If `keywordExtraction` is true the returned string is a JSON object containing
 * `high_level_keywords` and `low_level_keywords`. If the environment variable
 * `OPENAI_API_KEY` is not set a stubbed response derived from the prompt is
 * returned instead of calling the API. The call is retried on transient failures;
 * if all retry attempts fail the function returns `Prompts.FAIL_RESPONSE`.
 *
 * @param prompt The user prompt to send to the model.
 * @param systemPrompt Optional system-level prompt to prepend to the conversation.
 * @param historyMessages Conversation history as a list of maps with keys like `"role"` and `"content"`.
 * @param keywordExtraction When true, request keyword-style output (JSON with keyword arrays) instead of a normal completion.
 * @param maxTokens Optional maximum length (number of characters) to trim the returned stubbed response when the API key is missing.
 * @return The model's text response, a keyword JSON string when `keywordExtraction` is true, a prompt-derived stub if the API key is missing, or `Prompts.FAIL_RESPONSE` after persistent failures.
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
            val history = buildHistoryBlock(historyMessages)
            "${systemPrompt.orEmpty()} $history$prompt".trim().take(maxTokens ?: 4000)
        }
    }

    val logRequests = System.getenv("OPENAI_LOG_REQUESTS")?.toBoolean() ?: false
    val logResponses = System.getenv("OPENAI_LOG_RESPONSES")?.toBoolean() ?: false
    val baseUrl = System.getenv("OPENAI_API_BASE")
    val modelName = model.ifBlank { DEFAULT_CHAT_MODEL }
    val chatKey = "$modelName|$baseUrl"
    val chatModel: ChatModel =
        chatModels.computeIfAbsent(chatKey) {
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

    val historyBlock = buildHistoryBlock(historyMessages)
    val systemBlock = systemPrompt?.takeIf { it.isNotBlank() } ?: DEFAULT_SYSTEM_PROMPT

    return withContext(Dispatchers.IO) {
        val maxAttempts = (System.getenv("OPENAI_RETRY_ATTEMPTS")?.toIntOrNull() ?: 3).coerceAtLeast(1)
        val backoffMs = (System.getenv("OPENAI_RETRY_BACKOFF_MS")?.toLongOrNull() ?: 500L).coerceAtLeast(0L)
        var lastError: RuntimeException? = null
        val result =
            retryWithBackoff(
                maxAttempts,
                backoffMs,
                operation = { templateClient(chatKey, chatModel).chat(systemBlock, historyBlock, prompt) },
                onError = { e: RuntimeException, attempt: Int ->
                    lastError = e
                    logger.warn(e) { "OpenAI chat attempt ${attempt + 1} failed for model $modelName" }
                },
            )
        result
            ?: run {
                logger.error(lastError) { "OpenAI chat call failed after $maxAttempts attempts for model $modelName" }
                Prompts.FAIL_RESPONSE
            }
    }
}

/**
 * Produces a completion from an Ollama chat model using the provided inputs.
 *
 * Builds a full prompt from an optional system prompt, an ordered list of history messages (each map expected to contain "role" and "content"), and the user prompt, then calls the configured Ollama model with retry/backoff. The function resolves the model and base URL from the provided `model` argument and environment variables when needed.
 *
 * @param model The Ollama model name to use; if blank the `OLLAMA_MODEL` environment variable or a default model is used.
 * @param prompt The user prompt to be appended to the constructed conversation.
 * @param systemPrompt Optional system instruction to prepend to the conversation.
 * @param historyMessages Ordered conversation history as a list of maps with keys "role" and "content".
 * @param keywordExtraction If true, indicates the caller is requesting keyword-style output (handled by the model or fallback).
 * @param stream Reserved for streaming behavior (ignored if the underlying model/client does not support it).
 * @param maxTokens Optional maximum token limit to request from the model (honored if supported by the client).
 * @param hashingKv Optional opaque value used for request hashing or deduplication by callers.
 * @return The model's text completion, or the sentinel Prompts.FAIL_RESPONSE if the call fails after retries.
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
    val chatKey = "ollama|$modelName|$baseUrl"
    val chatModel: ChatModel =
        chatModels.computeIfAbsent(chatKey) {
            OllamaChatModel
                .builder()
                .baseUrl(baseUrl)
                .modelName(modelName)
                .build()
        }

    val historyBlock = buildHistoryBlock(historyMessages)
    val systemBlock = systemPrompt?.takeIf { it.isNotBlank() } ?: DEFAULT_SYSTEM_PROMPT

    return withContext(Dispatchers.IO) {
        val maxAttempts = (System.getenv("OLLAMA_RETRY_ATTEMPTS")?.toIntOrNull() ?: 3).coerceAtLeast(1)
        val backoffMs = (System.getenv("OLLAMA_RETRY_BACKOFF_MS")?.toLongOrNull() ?: 500L).coerceAtLeast(0L)
        var lastError: RuntimeException? = null
        val result =
            retryWithBackoff(
                maxAttempts,
                backoffMs,
                operation = { templateClient(chatKey, chatModel).chat(systemBlock, historyBlock, prompt) },
                onError = { e: RuntimeException, attempt: Int ->
                    lastError = e
                    logger.warn(e) { "Ollama chat attempt ${attempt + 1} failed for model $modelName" }
                },
            )
        result
            ?: run {
                logger.error(lastError) { "Ollama chat call failed after $maxAttempts attempts for model $modelName" }
                Prompts.FAIL_RESPONSE
            }
    }
}

/**
 * Generate embeddings for the provided text inputs using the configured OpenAI embedding model.
 *
 * Blank inputs are ignored; for each non-blank input this returns a dense embedding vector as a DoubleArray.
 * If the OPENAI_API_KEY environment variable is not set, returns a deterministic pseudo-random embedding per input seeded by the input text.
 *
 * @param inputs List of text values to embed; blank entries are filtered out before embedding.
 * @return A list of embedding vectors (one DoubleArray per non-blank input) in the same order as the filtered inputs.
 * @throws IllegalStateException If the OpenAI embedding API fails after configured retry attempts.
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
        var lastError: RuntimeException? = null
        val result =
            retryWithBackoff(
                maxAttempts,
                backoffMs,
                operation = {
                    val segments = sanitized.map { TextSegment.from(it) }
                    val response: Response<List<Embedding>> = embedModel.embedAll(segments)
                    response.content().map { embedding ->
                        val vector = embedding.vector()
                        DoubleArray(vector.size) { idx -> vector[idx].toDouble() }
                    }
                },
                onError = { e: RuntimeException, attempt: Int ->
                    lastError = e
                    logger.warn(e) { "OpenAI embedding attempt ${attempt + 1} failed for model $modelName" }
                },
            )
        result
            ?: run {
                logger.error(lastError) { "OpenAI embedding call failed after $maxAttempts attempts for model $modelName" }
                throw IllegalStateException("OpenAI embedding call failed after $maxAttempts attempts for model $modelName", lastError)
            }
    }
}

/**
 * Generate embedding vectors for a list of input texts using an Ollama embedding model.
 *
 * Blank input strings are ignored before calling the model; the returned list corresponds
 * to the non-blank inputs in the same order.
 *
 * @param inputs The texts to embed; blank entries are filtered out.
 * @return A list of embedding vectors where each vector is a DoubleArray representing
 *         the embedding for the corresponding non-blank input.
 * @throws IllegalStateException If the embedding call fails after the configured retry attempts.
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
        var lastError: RuntimeException? = null
        val result =
            retryWithBackoff(
                maxAttempts,
                backoffMs,
                operation = {
                    val segments = sanitized.map { TextSegment.from(it) }
                    val response: Response<List<Embedding>> = embedModel.embedAll(segments)
                    response.content().map { embedding ->
                        val vector = embedding.vector()
                        DoubleArray(vector.size) { idx -> vector[idx].toDouble() }
                    }
                },
                onError = { e: RuntimeException, attempt: Int ->
                    lastError = e
                    logger.warn(e) { "Ollama embedding attempt ${attempt + 1} failed for model $modelName" }
                },
            )
        result
            ?: run {
                logger.error(lastError) { "Ollama embedding call failed after $maxAttempts attempts for model $modelName" }
                throw IllegalStateException("Ollama embedding call failed after $maxAttempts attempts for model $modelName", lastError)
            }
    }
}

/**
     * Create an embedding function wrapper configured from environment variables.
     *
     * Constructs an EmbeddingFunc populated with the provider-selected embedding implementation,
     * the provider's embedding dimension, and the provider's maximum token size.
     *
     * @return An EmbeddingFunc configured with the chosen provider's embedding dimension, maximum token size, and implementation function.
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

/**
 * Determine the embedding provider, embedding vector dimension, and context window size from environment settings.
 *
 * Reads EMBED_PROVIDER to choose the provider ("ollama" or default "openai"). When using Ollama, OLLAMA_EMBED_DIM
 * (if present) controls the dimension. When using OpenAI, OPENAI_EMBEDDING_MODEL is inspected to pick a likely
 * embedding dimension and context size; unrecognized models fall back to defaults.
 *
 * @return Triple where:
 *   - first: the provider identifier ("openai" or "ollama"),
 *   - second: the embedding vector dimension (number of floats),
 *   - third: the context window size in tokens for that embedding model.
 */
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

                    modelName.contains("ada-002", ignoreCase = true) ||
                        modelName.contains("text-embedding-ada", ignoreCase = true) -> {
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

/**
 * Retry a suspending operation up to a fixed number of attempts, applying a fixed delay between retries when a RuntimeException occurs.
 *
 * @param maxAttempts The maximum number of attempts to run `operation`.
 * @param backoffMs Milliseconds to wait between attempts when retrying; no delay if zero or negative.
 * @param operation The suspending operation to execute.
 * @param onError Callback invoked for each caught `RuntimeException` with the exception and the zero-based attempt index.
 * @return The successful result of `operation` if any attempt succeeds, or `null` if all attempts throw `RuntimeException`.
 */
private suspend fun <T> retryWithBackoff(
    maxAttempts: Int,
    backoffMs: Long,
    operation: suspend () -> T,
    onError: (RuntimeException, Int) -> Unit,
): T? {
    var lastError: RuntimeException? = null
    repeat(maxAttempts) { attempt ->
        try {
            return operation()
        } catch (e: RuntimeException) {
            lastError = e
            onError(e, attempt)
            if (attempt < maxAttempts - 1 && backoffMs > 0) {
                delay(backoffMs)
            }
        }
    }
    return null
}