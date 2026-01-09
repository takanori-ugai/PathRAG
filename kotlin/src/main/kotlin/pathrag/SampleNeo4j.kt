package pathrag

import kotlinx.coroutines.runBlocking
import java.nio.file.Paths

/**
 * Sample showing how to run PathRAG with Neo4j-backed graph storage.
 *
 * Set Neo4j connection via env (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD),
 * then run: ./gradlew execute -PmainClass=pathrag.SampleNeo4jKt
 */
/**
     * Runs a sample PathRAG demonstration using a Neo4j-backed graph store.
     *
     * Loads environment variables from ../.env, constructs a PathRAG instance with sensible defaults
     * (including Neo4j connection via ExtraConfig and addon parameters), inserts three sample Dickens-related
     * documents, performs the same query in "local", "global", and "hybrid" modes, and prints each result.
     */
    fun main() =
    runBlocking {
        val env = EnvironmentConfig.load(Paths.get("../.env"))
        val kvStorage = env["KV_STORAGE"] ?: "JsonKVStorage"
        val vectorStorage = env["VECTOR_STORAGE"] ?: "NanoVectorDBStorage"
        val graphStorage = env["GRAPH_STORAGE"] ?: "Neo4jStorage"
        val neo4jConfig =
            pathrag.base.ExtraConfig(
                neo4jUri = env["NEO4J_URI"],
                neo4jUser = env["NEO4J_USER"],
                neo4jPassword = env["NEO4J_PASSWORD"],
            )
        val rag =
            PathRAG(
                workingDir = env["WORKING_DIR"] ?: "./sample_cache_neo4j",
                kvStorage = kvStorage,
                vectorStorage = vectorStorage,
                graphStorage = graphStorage,
                extraConfig = neo4jConfig,
                chunkTokenSize = 800,
                chunkOverlapTokenSize = 120,
                language = env["LANGUAGE"] ?: "English",
                // Optional: pin keywords to bypass LLM keyword extraction
                highLevelKeywords = listOf("themes", "Dickens"),
                lowLevelKeywords = listOf("poverty", "class struggle", "redemption"),
                embeddingCacheConfig =
                    mapOf(
                        "enabled" to true,
                        "similarity_threshold" to 0.9,
                        "use_llm_check" to false,
                    ),
                addonParams =
                    pathrag.base.AddonParams(
                        entityTypes = listOf("organization", "person", "geo", "event", "category"),
                        exampleNumber = 3,
                    ),
            )

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

        val localAnswer = rag.query(question, param = pathrag.base.QueryParam(mode = "local"))
        println("Q (local): $question")
        println("A: $localAnswer\n")

        val globalAnswer = rag.query(question, param = pathrag.base.QueryParam(mode = "global"))
        println("Q (global): $question")
        println("A: $globalAnswer\n")

        val hybridAnswer = rag.query(question, param = pathrag.base.QueryParam(mode = "hybrid"))
        println("Q (hybrid): $question")
        println("A: $hybridAnswer\n")

        println("\nDone with Neo4j-backed graph storage.")
    }