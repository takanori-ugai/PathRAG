package pathrag

import kotlinx.coroutines.runBlocking
import java.nio.file.Paths

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
        val rag =
            PathRAG(
                workingDir = env["WORKING_DIR"] ?: "./sample_cache",
                kvStorage = kvStorage,
                vectorStorage = vectorStorage,
                graphStorage = graphStorage,
                chunkTokenSize = 800,
                chunkOverlapTokenSize = 120,
                language = env["LANGUAGE"] ?: "English",
                keywordExamples = "",
                // Optional: pin keywords instead of calling the LLM extractor
                highLevelKeywords = listOf("themes", "Dickens"),
                lowLevelKeywords = listOf("poverty", "class struggle", "redemption"),
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

        val question = "What themes does Dickens explore?"

        // Local mode: entity-centric context only.
        val localAnswer = rag.query(question, param = pathrag.base.QueryParam(mode = "local"))
        println("Q (local): $question")
        println("A: $localAnswer\n")

        // Global mode: relationship-centric context.
        val globalAnswer = rag.query(question, param = pathrag.base.QueryParam(mode = "global"))
        println("Q (global): $question")
        println("A: $globalAnswer\n")

        // Hybrid mode: uses existing hybrid flow.
        val hybridAnswer = rag.query(question, param = pathrag.base.QueryParam(mode = "hybrid"))
        println("Q (hybrid): $question")
        println("A: $hybridAnswer\n")

        println("\nDone!")
    }
