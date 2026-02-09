package pathrag.utils

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import pathrag.prompt.Prompts
import java.io.File
import java.math.BigInteger
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.pow
import kotlin.math.sqrt

private val internalLogger = KotlinLogging.logger("PathRAG")

/**
 * Provides access to the internal logger used by the PathRAG utilities.
 *
 * @return The internal logger instance.
 */
fun log() = internalLogger

/**
 * Wrapper around an embedding function with dimension enforcement and optional concurrency limiting.
 *
 * @property embeddingDim expected dimension of embedding vectors.
 * @property maxTokenSize maximum token size supported by the embedding model.
 * @property func underlying embedding generator.
 * @property concurrentLimit maximum concurrent invocations.
 */
data class EmbeddingFunc(
    val embeddingDim: Int,
    val maxTokenSize: Int,
    val func: suspend (List<String>) -> List<DoubleArray>,
    val concurrentLimit: Int = 16,
) {
    private val semaphore = if (concurrentLimit > 0) Semaphore(concurrentLimit) else null

    /**
     * Invoke the embedding function to compute embedding vectors for the provided inputs.
     *
     * Computes an embedding for each input string, enforcing the configured embedding dimension
     * and respecting any concurrency limit configured for the embedding function.
     *
     * @param inputs The list of input strings to embed; result order corresponds to this list.
     * @return A list of `DoubleArray` where each array is the embedding vector for the corresponding input.
     * @throws IllegalStateException If any returned embedding does not have length equal to `embeddingDim`.
     */
    suspend operator fun invoke(inputs: List<String>): List<DoubleArray> {
        val exec: suspend () -> List<DoubleArray> = { func(inputs) }
        val lock = semaphore
        val vectors =
            if (lock == null) {
                exec()
            } else {
                lock.withPermit { exec() }
            }
        vectors.forEachIndexed { idx, vec ->
            if (vec.size != embeddingDim) {
                val message = "Embedding dimension mismatch at index $idx: expected $embeddingDim, got ${vec.size}"
                log().error { message }
                throw IllegalStateException(message)
            }
        }
        return vectors
    }
}

/**
 * Extracts the first JSON object found in the input string and parses it into a map.
 *
 * The returned map preserves any nested `JsonObject` values and converts JSON primitives to their string content.
 *
 * @param response String that contains a JSON object (possibly embedded in other text).
 * @return A map of the parsed JSON object's keys to their values (`JsonObject` for objects, `String` for primitives).
 * @throws IllegalStateException if no JSON object can be found or parsed from the input.
 */
fun convertResponseToJson(response: String): Map<String, Any?> {
    val regex = Regex("\\{.*\\}", RegexOption.DOT_MATCHES_ALL)
    val jsonString = regex.find(response)?.value ?: error("Unable to parse JSON from response: $response")
    val parsed = Json.parseToJsonElement(jsonString).jsonObject
    return parsed.mapValues { entry ->
        when (val value = entry.value) {
            is JsonObject -> value
            else -> value.jsonPrimitive.content
        }
    }
}

/**
 * Produce a 32-character lowercase MD5 hex digest of the given content, optionally prefixed.
 *
 * @param content Input string to hash.
 * @param prefix Optional string to prepend to the resulting hex hash.
 * @return The `prefix` concatenated with the 32-character lowercase hexadecimal MD5 hash of `content`.
 */
fun computeMdHashId(
    content: String,
    prefix: String = "",
): String {
    val md = MessageDigest.getInstance("MD5")
    val digest = md.digest(content.toByteArray())
    val bigInt = BigInteger(1, digest)
    val hashText = bigInt.toString(16).padStart(32, '0')
    return prefix + hashText
}

/**
 * Produce a higher-order wrapper that enforces a maximum number of concurrent invocations for a suspending no-argument function.
 *
 * @param maxSize The maximum number of concurrent executions allowed.
 * @param waitingTimeMillis Reserved for a wait/backoff duration in milliseconds (currently not used by the implementation).
 * @return A function that accepts a suspending no-argument function and returns a suspending function which enforces the specified concurrency limit when invoked.
 */
fun limitAsyncFuncCall(
    maxSize: Int,
    waitingTimeMillis: Long = 1,
): (suspend (() -> Unit) -> suspend () -> Unit) =
    { func ->
        val semaphore = Semaphore(maxSize)
        suspend {
            semaphore.withPermit { func() }
        }
    }

/**
 * Produces a deterministic identifier by hashing the string representation of the given arguments.
 *
 * @param args The values to include in the hash; the function uses their string representations.
 * @return A 32-character hexadecimal hash representing the provided arguments.
 */
fun computeArgsHash(vararg args: Any?): String = computeMdHashId(args.toList().toString())

/**
 * Computes the cosine similarity between two vectors.
 *
 * Returns 0.0 if either vector is empty, their lengths differ, or either has zero magnitude.
 *
 * @param a First vector.
 * @param b Second vector.
 * @return The cosine similarity value between -1.0 and 1.0, or 0.0 for invalid or degenerate inputs.
 */
fun cosineSimilarity(
    a: DoubleArray,
    b: DoubleArray,
): Double {
    if (a.isEmpty() || b.isEmpty() || a.size != b.size) return 0.0
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for (i in a.indices) {
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    }
    val denom = sqrt(na) * sqrt(nb)
    return if (denom == 0.0) 0.0 else dot / denom
}

/**
 * Cached payload for persisted responses.
 *
 * @property argsHash hash of query args.
 * @property content cached response content.
 * @property prompt prompt that produced the content.
 * @property embedding optional embedding for similarity reuse.
 * @property mode mode associated with the cache entry.
 */
data class CacheData(
    val argsHash: String,
    val content: String,
    val prompt: String,
    val embedding: DoubleArray? = null,
    val mode: String = "default",
)

@Serializable
private data class PersistEntry(
    val content: String,
    val prompt: String,
    val embedding: String? = null,
    val shape: List<Int>? = null,
    val min: Double? = null,
    val max: Double? = null,
)

/**
 * In-memory response cache with optional disk persistence and embedding similarity.
 *
 * @property globalConfig shared configuration containing cache and embedding options.
 */
class ResponseCache(
    val globalConfig: Map<String, Any?> = emptyMap(),
) {
    /**
     * In-memory cache entry.
     *
     * @property content cached content.
     * @property prompt originating prompt.
     * @property embedding optional embedding used for similarity.
     */
    data class Entry(
        val content: String,
        val prompt: String,
        val embedding: DoubleArray?,
    )

    private val store = ConcurrentHashMap<String, MutableMap<String, Entry>>()
    private val cachePath: String? =
        (globalConfig["cache_path"] as? String)
            ?: (globalConfig["working_dir"] as? String)?.let { "$it/llm_cache.json" }

    init {
        loadFromDisk()
    }

    /**
     * Retrieve all cached entries for a mode.
     */
    suspend fun getById(mode: String): Map<String, Entry>? = store[mode]

    /**
     * Insert or update a cache entry for the given mode and arguments, optionally computing
     * and storing an embedding for the prompt when embedding caching is enabled, then persist to disk.
     *
     * @param mode The cache namespace or mode.
     * @param argsHash Deterministic hash identifying the call arguments.
     * @param content The response content to cache.
     * @param prompt The prompt associated with the content (used for embedding computation when enabled).
     */
    suspend fun upsert(
        mode: String,
        argsHash: String,
        content: String,
        prompt: String,
    ) {
        val embedCfg =
            globalConfig["embedding_cache_config"]?.takeIf { it is Map<*, *> }?.let {
                @Suppress("UNCHECKED_CAST")
                it as? Map<String, Any?>
            }
                ?: mapOf("enabled" to false, "similarity_threshold" to 0.95, "use_llm_check" to false)
        val embedEnabled = embedCfg["enabled"] as? Boolean ?: false
        val embedding =
            if (embedEnabled) {
                val func = globalConfig["embedding_func"] as? EmbeddingFunc
                func?.invoke(listOf(prompt))?.firstOrNull()
            } else {
                null
            }
        store.computeIfAbsent(mode) { ConcurrentHashMap() }[argsHash] = Entry(content, prompt, embedding)
        persist()
    }

    /**
     * Attempt to retrieve a cached response either by exact args hash or by embedding similarity.
     *
     * When a direct cache hit for the given `mode` and `argsHash` exists, that content is returned.
     * If not, and embedding-based caching is enabled in `globalConfig` (via `embedding_cache_config`),
     * the function computes an embedding for `prompt`, finds the best-matching cached entry by cosine similarity,
     * and returns the cached content if the best similarity meets or exceeds the configured threshold.
     * If `use_llm_check` is enabled in the embedding cache config and an LLM checker is available in `globalConfig`,
     * the cached match is optionally validated by the LLM before returning.
     *
     * @param argsHash The deterministic hash of the request arguments used for exact cache lookup.
     * @param prompt The prompt text whose embedding may be used for similarity-based matching.
     * @param mode The cache namespace or mode to search within.
     * @param allowSimilar When false, disables embedding-similarity cache lookup.
     * @return The cached content when an exact or sufficiently similar entry is found, `null` otherwise.
     */
    suspend fun handleCache(
        argsHash: String,
        prompt: String,
        mode: String,
        allowSimilar: Boolean = true,
    ): String? {
        val modeCache = store[mode]
        if (modeCache != null) {
            val direct = modeCache[argsHash]
            if (direct != null) return direct.content
        }

        if (!allowSimilar) return null

        val embedCfg =
            globalConfig["embedding_cache_config"]?.takeIf { it is Map<*, *> }?.let {
                @Suppress("UNCHECKED_CAST")
                it as? Map<String, Any?>
            }
                ?: mapOf("enabled" to false, "similarity_threshold" to 0.95, "use_llm_check" to false)
        val embedEnabled = embedCfg["enabled"] as? Boolean ?: false
        if (!embedEnabled) return null

        val similarityThreshold = (embedCfg["similarity_threshold"] as? Number)?.toDouble() ?: 0.95
        val embeddingFunc = globalConfig["embedding_func"] as? EmbeddingFunc ?: return null
        val currentEmbedding = embeddingFunc(listOf(prompt)).firstOrNull() ?: return null
        val useLlmCheck = embedCfg["use_llm_check"] as? Boolean ?: false
        val llmFunc =
            globalConfig["llm_model_func"] as? suspend (
                String,
                String?,
                List<Map<String, String>>,
                Boolean,
                Boolean,
                Int?,
                Any?,
            ) -> String

        var best: Entry? = null
        var bestSim = -1.0
        modeCache?.values?.forEach { entry ->
            val cachedEmb = entry.embedding ?: return@forEach
            val sim = cosineSimilarity(currentEmbedding, cachedEmb)
            if (!sim.isNaN() && sim > bestSim) {
                bestSim = sim
                best = entry
            }
        }
        if (best != null && bestSim >= similarityThreshold) {
            if (useLlmCheck && llmFunc != null) {
                val promptTemplate =
                    (globalConfig["similarity_check_prompt"] as? String)
                        ?: Prompts.SIMILARITY_CHECK
                val promptCheck =
                    runCatching {
                        Prompts.render(
                            promptTemplate,
                            mapOf(
                                "original_prompt" to prompt,
                                "cached_prompt" to best.prompt,
                            ),
                        )
                    }.getOrElse {
                        promptTemplate
                            .replace("{original_prompt}", prompt)
                            .replace("{cached_prompt}", best.prompt)
                    }
                val llmScore =
                    runCatching { llmFunc(promptCheck, null, emptyList(), false, false, 32, null).trim() }
                        .getOrNull()
                        ?.toDoubleOrNull()
                if (llmScore != null && llmScore < similarityThreshold) {
                    return null
                }
            }
            return best.content
        }
        return null
    }

    private fun persist() {
        val path = cachePath ?: return
        val payload =
            store.mapValues { (_, v) ->
                v.mapValues { (_, entry) ->
                    val q = entry.embedding?.let { quantizeEmbedding(it) }
                    PersistEntry(
                        content = entry.content,
                        prompt = entry.prompt,
                        embedding = q?.first?.let { bytesToHex(it) },
                        shape = q?.fourth,
                        min = q?.second,
                        max = q?.third,
                    )
                }
            }
        runCatching {
            val f = File(path)
            f.parentFile?.mkdirs()
            f.writeText(Json.encodeToString(payload))
        }.onFailure { internalLogger.warn(it) { "Failed to persist cache to $path" } }
    }

    /**
     * Loads persisted cache entries from disk into the in-memory store.
     *
     * Reads the JSON file at the configured cache path (if any), reconstructs each persisted entry,
     * dequantizes stored embeddings when present, and populates the in-memory store by mode and args hash.
     * Failures are logged and do not throw.
     */
    private fun loadFromDisk() {
        val path = cachePath ?: return
        val f = File(path)
        if (!f.exists()) return
        runCatching {
            val text = f.readText()
            val decoded: Map<String, Map<String, PersistEntry>> = Json.decodeFromString(text)
            decoded.forEach { (mode, entries) ->
                entries.forEach { (hash, entry) ->
                    val embedding =
                        if (entry.embedding != null && entry.shape != null && entry.min != null && entry.max != null) {
                            val bytes = hexToBytes(entry.embedding)
                            dequantizeEmbedding(bytes, entry.min, entry.max, entry.shape)
                        } else {
                            null
                        }
                    store.computeIfAbsent(mode) { ConcurrentHashMap() }[hash] =
                        Entry(entry.content, entry.prompt, embedding)
                }
            }
        }.onFailure { internalLogger.warn(it) { "Failed to load cache from $path" } }
    }
}

/**
 * Quantizes a non-empty embedding vector into 8-bit bytes and returns the quantized bytes with metadata.
 *
 * @param embedding The embedding values to quantize; must not be empty.
 * @param bits Quantization bit width (only `8` is supported).
 * @return A Quadruple containing:
 *   - `ByteArray`: quantized values (length equals `embedding.size`),
 *   - `Double`: minimum value from the original embedding,
 *   - `Double`: maximum value from the original embedding,
 *   - `List<Int>`: shape metadata (single-element list with the embedding length).
 *   Returns `null` if `embedding` is empty.
 * @throws IllegalArgumentException If `bits` is not `8`.
 */
private fun quantizeEmbedding(
    embedding: DoubleArray,
    bits: Int = 8,
): Quadruple<ByteArray, Double, Double, List<Int>>? {
    if (embedding.isEmpty()) return null
    // Only 8-bit quantization is supported because we store values in a ByteArray and dequantize with 255-scale.
    require(bits == 8) { "Only 8-bit quantization is supported." }
    val min = embedding.min()
    val max = embedding.max()
    val maxVal = (1 shl bits) - 1
    if (max == min) {
        return Quadruple(ByteArray(embedding.size) { 0 }, min, max, listOf(embedding.size))
    }
    val scale = (max - min) / maxVal
    val bytes = ByteArray(embedding.size)
    for (i in embedding.indices) {
        val q = ((embedding[i] - min) / scale).toInt().coerceIn(0, maxVal)
        bytes[i] = q.toByte()
    }
    return Quadruple(bytes, min, max, listOf(embedding.size))
}

private fun dequantizeEmbedding(
    quantized: ByteArray,
    min: Double,
    max: Double,
    shape: List<Int>,
): DoubleArray {
    val size = shape.firstOrNull() ?: quantized.size
    val scale = (max - min) / 255.0
    val result = DoubleArray(size)
    for (i in 0 until size.coerceAtMost(quantized.size)) {
        val v = quantized[i].toInt() and 0xFF
        result[i] = v * scale + min
    }
    return result
}

private fun bytesToHex(bytes: ByteArray): String = bytes.joinToString("") { "%02x".format(it) }

private fun hexToBytes(hex: String): ByteArray {
    val clean = hex.trim()
    require(clean.length % 2 == 0) { "Hex string must have an even length." }
    val len = clean.length
    val data = ByteArray(len / 2)
    var i = 0
    while (i < len) {
        val high = Character.digit(clean[i], 16)
        val low = Character.digit(clean[i + 1], 16)
        require(high >= 0 && low >= 0) { "Invalid hex character at position $i" }
        data[i / 2] = ((high shl 4) + low).toByte()
        i += 2
    }
    return data
}

private data class Quadruple<A, B, C, D>(
    val first: A,
    val second: B,
    val third: C,
    val fourth: D,
)
