package pathrag

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import java.nio.file.Files
import java.nio.file.Paths

private val prettyJson = Json { prettyPrint = true }

/**
 * Demonstrates a minimal PathRAG run: loads environment settings, initializes a PathRAG instance,
 * seeds it with example Dickens passages, and performs local, global, and hybrid queries printing each result.
 *
 * The function loads configuration from "../.env", applies defaults for KV, vector, and graph storages when unset,
 * configures chunking, language, keyword and addon parameters, inserts three demo documents, then queries
 * the RAG with the question "What themes does Dickens explore?" in three modes and prints the answers.
 */
fun main() =
    runBlocking {
        val env = EnvironmentConfig.load(Paths.get("../.env"))
        val kvStorage = env["KV_STORAGE"] ?: "JsonKVStorage"
        val vectorStorage = env["VECTOR_STORAGE"] ?: "NanoVectorDBStorage"
        val graphStorage = env["GRAPH_STORAGE"] ?: "NetworkXStorage"
        val workingDir = env["WORKING_DIR"] ?: "./sample_cache"
        val rag =
            PathRAG(
                workingDir = workingDir,
                kvStorage = kvStorage,
                vectorStorage = vectorStorage,
                graphStorage = graphStorage,
                chunkTokenSize = 800,
                chunkOverlapTokenSize = 120,
                language = env["LANGUAGE"] ?: "English",
                keywordExamples = "",
                // Optional: pin keywords instead of calling the LLM extractor
                // highLevelKeywords = listOf("themes", "Dickens"),
                // lowLevelKeywords = listOf("poverty", "class struggle", "redemption"),
                similarityCheckPrompt = pathrag.prompt.Prompts.SIMILARITY_CHECK,
                embeddingCacheConfig =
                    mapOf(
                        "enabled" to true,
                        "similarity_threshold" to 0.9,
                        "use_llm_check" to false,
                    ),
                addonParams =
                    pathrag.base.AddonParams(
                        entityTypes = listOf("organization", "person", "geo", "event", "category"),
                        // language is set at top-level already
                        exampleNumber = 3,
                    ),
            )

        // Insert demo content (replace with real documents).
        rag.insert(
            listOf(
                """
                Charles Dickens was an English writer and social critic.
                He created some of the world's best-known fictional characters
                and is regarded as one of the greatest novelists of the Victorian era.
                """.trimIndent(),
                """
                Oliver Twist is a novel by Dickens that critiques workhouses and child poverty.
                It follows an orphan navigating criminal underworlds and harsh social systems.
                """.trimIndent(),
                """
                A Christmas Carol tells the redemption story of Ebenezer Scrooge, shifting from greed to generosity.
                It explores themes of morality, compassion, and social responsibility.
                """.trimIndent(),
            ),
        )

        val graphSnapshot = snapshotGraphToJson(rag.graph())
        val graphPath = Paths.get(workingDir, "knowledge-graph.json")
        Files.createDirectories(graphPath.parent)
        Files.writeString(
            graphPath,
            prettyJson.encodeToString(graphSnapshot),
        )
        println("Knowledge graph saved to: ${graphPath.toAbsolutePath()}")

        val question = "What themes does Dickens explore?"

        // Local mode: entity-centric context only.
        val localAnswer = rag.query(question, param = pathrag.base.QueryParam(mode = "local", onlyNeedContext = true))
        println("Q (local): $question")
        println("A: $localAnswer\n")

        // Global mode: relationship-centric context.
        val globalAnswer = rag.query(question, param = pathrag.base.QueryParam(mode = "global", onlyNeedContext = true))
        println("Q (global): $question")
        println("A: $globalAnswer\n")

        // Hybrid mode: intentionally fetches context and full answer separately (two queries).
        val context = rag.query(question, param = pathrag.base.QueryParam(mode = "hybrid", onlyNeedContext = true))
        val hybridAnswer = rag.query(question, param = pathrag.base.QueryParam(mode = "hybrid", onlyNeedContext = false))
        println("Context: $context")
        println("Q (hybrid): $question")
        println("A: $hybridAnswer\n")

        println("\nDone!")
    }

@Serializable
private data class GraphSnapshot(
    val nodes: List<GraphNode>,
    val edges: List<GraphEdge>,
)

@Serializable
private data class GraphNode(
    val id: String,
    val data: JsonObject = JsonObject(emptyMap()),
)

@Serializable
private data class GraphEdge(
    val source: String,
    val target: String,
    val data: JsonObject = JsonObject(emptyMap()),
)

private suspend fun snapshotGraphToJson(graph: pathrag.base.BaseGraphStorage): GraphSnapshot {
    val nodes =
        graph.nodes().map { id ->
            val data = graph.getNode(id)
            GraphNode(id, toJsonObject(data))
        }
    val edges =
        graph.edges().map { (source, target) ->
            val data = graph.getEdge(source, target) ?: graph.getEdge(target, source)
            GraphEdge(source, target, toJsonObject(data))
        }
    return GraphSnapshot(nodes, edges)
}

private fun toJsonObject(data: Map<String, Any?>?): JsonObject =
    if (data == null) {
        JsonObject(emptyMap())
    } else {
        buildJsonObject {
            data.forEach { (key, value) ->
                put(key, toJsonElement(value))
            }
        }
    }

private fun toJsonElement(value: Any?): JsonElement =
    when (value) {
        null -> {
            JsonNull
        }

        is JsonElement -> {
            value
        }

        is String -> {
            JsonPrimitive(value)
        }

        is Number -> {
            JsonPrimitive(value)
        }

        is Boolean -> {
            JsonPrimitive(value)
        }

        is Map<*, *> -> {
            buildJsonObject {
                value.forEach { (k, v) ->
                    if (k != null) {
                        put(k.toString(), toJsonElement(v))
                    }
                }
            }
        }

        is List<*> -> {
            JsonArray(value.map { toJsonElement(it) })
        }

        is Array<*> -> {
            JsonArray(value.map { toJsonElement(it) })
        }

        is IntArray -> {
            JsonArray(value.map { JsonPrimitive(it) })
        }

        is LongArray -> {
            JsonArray(value.map { JsonPrimitive(it) })
        }

        is DoubleArray -> {
            JsonArray(value.map { JsonPrimitive(it) })
        }

        is FloatArray -> {
            JsonArray(value.map { JsonPrimitive(it) })
        }

        is BooleanArray -> {
            JsonArray(value.map { JsonPrimitive(it) })
        }

        is ShortArray -> {
            JsonArray(value.map { JsonPrimitive(it) })
        }

        is ByteArray -> {
            JsonArray(value.map { JsonPrimitive(it) })
        }

        is CharArray -> {
            JsonArray(value.map { JsonPrimitive(it.toString()) })
        }

        else -> {
            JsonPrimitive(value.toString())
        }
    }
