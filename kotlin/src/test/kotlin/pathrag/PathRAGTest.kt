package pathrag

import dev.langchain4j.model.chat.ChatModel
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockkConstructor
import io.mockk.mockkStatic
import io.mockk.unmockkAll
import io.mockk.verify
import kotlinx.coroutines.runBlocking
import org.neo4j.driver.GraphDatabase
import pathrag.base.QueryParam
import pathrag.utils.computeMdHashId
import java.io.File
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PathRAGTest {
    @AfterTest
    fun cleanup() {
        unmockkAll()
        File(".")
            .listFiles { file -> file.isDirectory && file.name.startsWith("PathRAG_cache_") }
            ?.forEach { dir ->
                if (!dir.deleteRecursively()) {
                    System.err.println("Warning: failed to delete ${dir.absolutePath}")
                }
            }
    }

    @Test
    fun upsertAndDeleteEntityAndEdge() =
        runBlocking {
            PathRAG(chunkTokenSize = 32, chunkOverlapTokenSize = 8).use { rag ->
                rag.aupsertEntity("Alpha", "desc", "TYPE", "src")
                rag.aupsertEntity("Beta", "desc", "TYPE", "src")
                rag.aupsertEdge("Alpha", "Beta", description = "link", keywords = "k")

                val graph = rag.graph()
                assertTrue(graph.hasNode("ALPHA"))
                assertTrue(graph.hasEdge("ALPHA", "BETA"))

                rag.adeleteEdge("Alpha", "Beta")
                assertFalse(graph.hasEdge("ALPHA", "BETA"))

                rag.adeleteByEntity("Alpha")
                assertFalse(graph.hasNode("ALPHA"))
            }
        }

    @Test
    fun cleanupGraphRemovesDanglingEdges() =
        runBlocking {
            PathRAG(chunkTokenSize = 32, chunkOverlapTokenSize = 8).use { rag ->
                // Add an edge without nodes to simulate a dangling relation.
                rag.aupsertEdge("ghost1", "ghost2", description = "dangling")
                val result = rag.acleanupGraph()
                assertTrue((result["removed_edges"] ?: 0) >= 1)
            }
        }

    @Test
    fun insertCustomKgPopulatesGraph() =
        runBlocking {
            PathRAG(chunkTokenSize = 32, chunkOverlapTokenSize = 8).use { rag ->
                val payload =
                    PathRAG.CustomKgPayload(
                        chunks = listOf(PathRAG.CustomKgChunk(content = "Entity text", sourceId = "s1")),
                        entities =
                            listOf(
                                PathRAG.CustomKgEntityInput(
                                    entityName = "NodeA",
                                    entityType = "THING",
                                    description = "desc",
                                    sourceId = "s1",
                                ),
                            ),
                        relationships =
                            listOf(
                                PathRAG.CustomKgRelationshipInput(
                                    srcId = "NodeA",
                                    tgtId = "NodeB",
                                    description = "related",
                                    keywords = "k",
                                    weight = 1.0,
                                    sourceId = "s1",
                                ),
                            ),
                    )
                rag.ainsertCustomKg(payload)
                val graph = rag.graph()
                assertTrue(graph.hasNode("NODEA"))
                assertTrue(graph.hasEdge("NODEA", "NODEB"))
                rag.dropGraph()
            }
        }

    @Test
    fun graphAndQueryWithInMemoryBackendsAndMockedLlm() =
        runBlocking {
            mockkStatic("pathrag.llm.LlmKt")
            coEvery {
                pathrag.llm.openAiComplete(
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                )
            } returns "LLM-ANSWER"

            PathRAG(chunkTokenSize = 16, chunkOverlapTokenSize = 4).use { rag ->
                rag.aupsertEntity("Alpha", "desc", "TYPE", "src")
                rag.aupsertEntity("Beta", "desc", "TYPE", "src")
                rag.aupsertEdge("Alpha", "Beta", description = "link", keywords = "k")

                val graph = rag.graph()
                assertTrue(graph.hasNode("ALPHA"))
                assertTrue(graph.hasEdge("ALPHA", "BETA"))

                val result = rag.query("Tell me about Alpha", QueryParam(mode = "hybrid", topK = 3))
                assertTrue(result.contains("LLM-ANSWER"), "Expected mocked LLM response to be returned")
            }
        }

    @Test
    fun queryReturnsContextWhenOnlyNeedContextIsSet() =
        runBlocking {
            mockkStatic("pathrag.llm.LlmKt")
            coEvery {
                pathrag.llm.openAiComplete(
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                )
            } returns """{"high_level_keywords":["alpha"],"low_level_keywords":["alpha"]}"""

            val chunkContent = "Alpha and Beta share a context chunk."
            val chunkId = computeMdHashId(chunkContent.trim(), prefix = "chunk-")

            PathRAG(chunkTokenSize = 16, chunkOverlapTokenSize = 4).use { rag ->
                rag.insertCustomKg(
                    PathRAG.CustomKgPayload(
                        chunks = listOf(PathRAG.CustomKgChunk(content = chunkContent, sourceId = chunkId)),
                    ),
                )

                rag.aupsertEntity("Alpha", "alpha description", "THING", chunkId)
                rag.aupsertEntity("Beta", "beta description", "THING", chunkId)
                rag.aupsertEdge("Alpha", "Beta", description = "Alpha Beta link", keywords = "k", sourceId = chunkId)

                val graph = rag.graph()
                assertTrue(graph.hasNode("ALPHA"))
                assertTrue(graph.hasEdge("ALPHA", "BETA"))

                val context =
                    rag.query(
                        "What is the relationship between Alpha and Beta?",
                        QueryParam(mode = "hybrid", onlyNeedContext = true, topK = 3),
                    )

                assertTrue(context.contains("local-information"), "Context should include local section")
                assertTrue(context.contains("global-information"), "Context should include global section")
                assertTrue(context.contains("ALPHA"))
                assertTrue(context.contains("BETA"))
                assertTrue(context.contains(chunkContent), "Context should include text chunk content")
                assertTrue(context.contains("Alpha Beta link"), "Context should include relationship description")
                coVerify(exactly = 1) {
                    pathrag.llm.openAiComplete(
                        any(),
                        any(),
                        any(),
                        any(),
                        any(),
                        any(),
                        any(),
                        any(),
                    )
                }
            }
        }

    @Test
    fun graphAndQueryWithInMemoryBackendsAndMockedChatModelContextOnly() =
        runBlocking {
            val chatModel = io.mockk.mockk<ChatModel>()
            every { chatModel.chat(any<String>()) } returns """{"high_level_keywords":["alpha"],"low_level_keywords":["beta"]}"""

            mockkStatic("pathrag.llm.LlmKt")
            coEvery {
                pathrag.llm.openAiComplete(
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                )
            } answers { chatModel.chat(it.invocation.args[1] as String) }
            coEvery { pathrag.llm.openAiEmbedding(any()) } returns listOf(DoubleArray(1536) { 0.01 })

            PathRAG(chunkTokenSize = 16, chunkOverlapTokenSize = 4).use { rag ->
                val chunkContent = "Alpha and Beta share history."
                val chunkId = computeMdHashId(chunkContent.trim(), prefix = "chunk-")
                rag.aupsertEntity("Alpha", "alpha description", "THING", chunkId)
                rag.aupsertEntity("Beta", "beta description", "THING", chunkId)
                rag.aupsertEdge("Alpha", "Beta", description = "Alpha Beta link", keywords = "k", sourceId = chunkId)
                rag.insertCustomKg(
                    PathRAG.CustomKgPayload(
                        chunks = listOf(PathRAG.CustomKgChunk(content = chunkContent, sourceId = chunkId)),
                    ),
                )

                val graph = rag.graph()
                assertTrue(graph.hasNode("ALPHA"), "Graph nodes: ${graph.nodes()}")
                assertTrue(graph.hasEdge("ALPHA", "BETA"), "Graph edges: ${graph.edges()}")

                val context =
                    rag.query(
                        "Tell me about Alpha and Beta",
                        QueryParam(mode = "local", onlyNeedContext = true, topK = 5),
                    )

                assertTrue(context.contains("local-information"), "Context was: $context")
                assertTrue(context.contains("ALPHA"), "Context was: $context")
                assertTrue(context.contains("BETA"), "Context was: $context")
                assertTrue(context.contains("Alpha Beta link"), "Context was: $context")
                assertTrue(context.contains(chunkContent), "Context was: $context")

                coVerify(atLeast = 1) {
                    pathrag.llm.openAiComplete(
                        any(),
                        any(),
                        any(),
                        any(),
                        any(),
                        any(),
                        any(),
                        any(),
                    )
                }
                verify(atLeast = 1) { chatModel.chat(any<String>()) }
            }
        }

    @Test
    fun queryReturnsContextWithNeo4jBackendsAndOnlyNeedContext() =
        runBlocking {
            mockkStatic("pathrag.llm.LlmKt")
            mockkStatic(GraphDatabase::class)

            coEvery {
                pathrag.llm.openAiEmbedding(any())
            } returns listOf(DoubleArray(1536) { 0.1 })

            coEvery {
                pathrag.llm.openAiComplete(
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                )
            } returns """{"high_level_keywords":["alpha"],"low_level_keywords":["alpha"]}"""

            mockkConstructor(pathrag.storage.Neo4jStorage::class)
            mockkConstructor(pathrag.storage.Neo4jVectorStorage::class)
            mockkConstructor(pathrag.storage.Neo4jKVStorage::class)

            val driver = io.mockk.mockk<org.neo4j.driver.Driver>()
            val session = io.mockk.mockk<org.neo4j.driver.Session>()
            val tx = io.mockk.mockk<org.neo4j.driver.TransactionContext>()

            every { GraphDatabase.driver(any<String>(), any<org.neo4j.driver.AuthToken>()) } returns driver
            every { driver.session() } returns session
            io.mockk.justRun { driver.close() }
            io.mockk.justRun { session.close() }

            fun emptyResult(): org.neo4j.driver.Result =
                io.mockk.mockk<org.neo4j.driver.Result>().apply {
                    every { hasNext() } returns false
                    every { list() } returns emptyList()
                    every { list<Any?>(any()) } returns emptyList()
                }

            every { tx.run(any<String>(), any<org.neo4j.driver.Value>()) } returns emptyResult()
            every { tx.run(any<String>()) } returns emptyResult()
            every { session.executeWrite<Any>(any()) } answers {
                val cb = invocation.args[0] as org.neo4j.driver.TransactionCallback<Any>
                cb.execute(tx)
            }
            every { session.executeRead<Any>(any()) } answers {
                val cb = invocation.args[0] as org.neo4j.driver.TransactionCallback<Any>
                cb.execute(tx)
            }

            PathRAG(
                kvStorage = "Neo4jKVStorage",
                vectorStorage = "Neo4jVectorStorage",
                graphStorage = "Neo4jStorage",
                chunkTokenSize = 16,
                chunkOverlapTokenSize = 4,
                extraConfig =
                    pathrag.base.ExtraConfig(
                        neo4jUri = "bolt://mock",
                        neo4jUser = "user",
                        neo4jPassword = "pass",
                    ),
            ).use { rag ->
                rag.aupsertEntity("Alpha", "alpha description", "THING", null)
                rag.aupsertEntity("Beta", "beta description", "THING", null)
                rag.aupsertEdge("Alpha", "Beta", description = "Alpha Beta link", keywords = "k", sourceId = null)

                val context =
                    rag.query(
                        "How are Alpha and Beta related?",
                        QueryParam(mode = "hybrid", onlyNeedContext = true, topK = 5),
                    )

                assertTrue(context.contains("local-information"))
                assertTrue(context.contains("global-information"))

                coVerify(exactly = 1) {
                    pathrag.llm.openAiComplete(
                        any(),
                        any(),
                        any(),
                        any(),
                        any(),
                        any(),
                        any(),
                        any(),
                    )
                }
            }
        }

    @Test
    fun insertSingleAndListWithMockedChatAndEmbeddingModels() =
        runBlocking {
            val chatModel = io.mockk.mockk<ChatModel>()
            val embeddingModel = io.mockk.mockk<dev.langchain4j.model.embedding.EmbeddingModel>()
            every { chatModel.chat(any<String>()) } answers {
                val prompt = it.invocation.args[0] as String
                if (prompt.contains("Extract entities")) {
                    """
                    {
                      "entities": [
                        { "entity_name": "ALPHA", "entity_type": "THING", "description": "alpha desc" },
                        { "entity_name": "BETA", "entity_type": "THING", "description": "beta desc" }
                      ],
                      "relationships": [
                        { "src_id": "ALPHA", "tgt_id": "BETA", "description": "connects", "keywords": "k", "weight": 1.0 }
                      ]
                    }
                    """.trimIndent()
                } else {
                    """{"high_level_keywords":["alpha"],"low_level_keywords":["beta"]}"""
                }
            }

            mockkStatic("pathrag.llm.LlmKt")
            coEvery {
                pathrag.llm.openAiComplete(
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                )
            } answers { chatModel.chat(it.invocation.args[1] as String) }
            coEvery { pathrag.llm.openAiEmbedding(any()) } answers { call ->
                val inputs = call.invocation.args[0] as List<String>
                inputs.map { DoubleArray(1536) { 0.02 } }
            }

            PathRAG(chunkTokenSize = 32, chunkOverlapTokenSize = 8).use { rag ->
                rag.insert(listOf("Alpha and Beta are here.", "Another Alpha mention."))
                rag.insert("Solo insertion for Beta.")

                val graph = rag.graph()
                assertTrue(graph.hasNode("ALPHA"), "Graph nodes: ${graph.nodes()}")
                assertTrue(graph.hasNode("BETA"), "Graph nodes: ${graph.nodes()}")
                assertTrue(graph.hasEdge("ALPHA", "BETA"), "Graph edges: ${graph.edges()}")

                val context =
                    rag.query(
                        "Tell me about Alpha",
                        QueryParam(mode = "local", onlyNeedContext = true, topK = 3),
                    )
                assertTrue(context.contains("local-information"), "Context was: $context")
                assertTrue(context.contains("ALPHA"), "Context was: $context")
                assertTrue(context.contains("BETA"), "Context was: $context")
            }
        }

    @Test
    fun insertSingleAndListWithNeo4jBackendsAndMockedChatEmbeddingAndDriver() =
        runBlocking {
            val chatModel = io.mockk.mockk<ChatModel>()
            every { chatModel.chat(any<String>()) } answers {
                val p = it.invocation.args[0] as String
                if (p.contains("Extract entities")) {
                    """
                    {
                      "entities": [
                        { "entity_name": "ALPHA", "entity_type": "THING", "description": "alpha desc" },
                        { "entity_name": "BETA", "entity_type": "THING", "description": "beta desc" }
                      ],
                      "relationships": [
                        { "src_id": "ALPHA", "tgt_id": "BETA", "description": "connects", "keywords": "k", "weight": 1.0 }
                      ]
                    }
                    """.trimIndent()
                } else {
                    """{"high_level_keywords":["alpha"],"low_level_keywords":["beta"]}"""
                }
            }

            mockkStatic("pathrag.llm.LlmKt")
            coEvery {
                pathrag.llm.openAiComplete(
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                )
            } answers { chatModel.chat(it.invocation.args[1] as String) }
            coEvery { pathrag.llm.openAiEmbedding(any()) } answers { call ->
                val inputs = call.invocation.args[0] as List<String>
                inputs.map { DoubleArray(1536) { 0.03 } }
            }

            mockkStatic(GraphDatabase::class)
            val driver = io.mockk.mockk<org.neo4j.driver.Driver>()
            val session = io.mockk.mockk<org.neo4j.driver.Session>()
            val tx = io.mockk.mockk<org.neo4j.driver.TransactionContext>()
            every { GraphDatabase.driver(any<String>(), any<org.neo4j.driver.AuthToken>()) } returns driver
            every { driver.session() } returns session
            io.mockk.justRun { driver.close() }
            io.mockk.justRun { session.close() }

            fun emptyResult(): org.neo4j.driver.Result =
                io.mockk.mockk<org.neo4j.driver.Result>().apply {
                    every { hasNext() } returns false
                    every { list() } returns emptyList()
                    every { list<Any?>(any()) } returns emptyList()
                }
            every { tx.run(any<String>(), any<org.neo4j.driver.Value>()) } returns emptyResult()
            every { tx.run(any<String>()) } returns emptyResult()
            every { session.executeWrite<Any>(any()) } answers {
                val cb = it.invocation.args[0] as org.neo4j.driver.TransactionCallback<Any>
                cb.execute(tx)
            }
            every { session.executeRead<Any>(any()) } answers {
                val cb = it.invocation.args[0] as org.neo4j.driver.TransactionCallback<Any>
                cb.execute(tx)
            }

            PathRAG(
                kvStorage = "Neo4jKVStorage",
                vectorStorage = "Neo4jVectorStorage",
                graphStorage = "Neo4jStorage",
                chunkTokenSize = 32,
                chunkOverlapTokenSize = 8,
                extraConfig =
                    pathrag.base.ExtraConfig(
                        neo4jUri = "bolt://mock",
                        neo4jUser = "user",
                        neo4jPassword = "pass",
                    ),
            ).use { rag ->
                rag.insert(listOf("Alpha and Beta are here.", "Another Alpha mention."))
                rag.insert("Solo insertion for Beta.")

                val graph = rag.graph()
                val context =
                    rag.query(
                        "Tell me about Alpha",
                        QueryParam(mode = "local", onlyNeedContext = true, topK = 3),
                    )
                assertTrue(context.contains("local-information"), "Context was: $context")
            }
        }

    @Test
    fun queryResultsMatchAcrossInMemoryAndNeo4jBackendsWithMockedChat() =
        runBlocking {
            mockkStatic("pathrag.llm.LlmKt")
            mockkStatic(GraphDatabase::class)

            coEvery {
                pathrag.llm.openAiComplete(
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                )
            } answers { call ->
                val keyword = call.invocation.args[4] as Boolean
                if (keyword) {
                    """{"high_level_keywords":["alpha"],"low_level_keywords":["beta"]}"""
                } else {
                    "ANSWER"
                }
            }
            coEvery { pathrag.llm.openAiEmbedding(any()) } answers { call ->
                val inputs = call.invocation.args[0] as List<String>
                inputs.map { DoubleArray(1536) { 0.05 } }
            }

            val driver = io.mockk.mockk<org.neo4j.driver.Driver>()
            val session = io.mockk.mockk<org.neo4j.driver.Session>()
            val tx = io.mockk.mockk<org.neo4j.driver.TransactionContext>()
            every { GraphDatabase.driver(any<String>(), any<org.neo4j.driver.AuthToken>()) } returns driver
            every { driver.session() } returns session
            io.mockk.justRun { driver.close() }
            io.mockk.justRun { session.close() }

            fun emptyResult(): org.neo4j.driver.Result =
                io.mockk.mockk<org.neo4j.driver.Result>().apply {
                    every { hasNext() } returns false
                    every { list() } returns emptyList()
                    every { list<Any?>(any()) } returns emptyList()
                }
            every { tx.run(any<String>(), any<org.neo4j.driver.Value>()) } returns emptyResult()
            every { tx.run(any<String>()) } returns emptyResult()
            every { session.executeWrite<Any>(any()) } answers {
                val cb = it.invocation.args[0] as org.neo4j.driver.TransactionCallback<Any>
                cb.execute(tx)
            }
            every { session.executeRead<Any>(any()) } answers {
                val cb = it.invocation.args[0] as org.neo4j.driver.TransactionCallback<Any>
                cb.execute(tx)
            }

            val inMemory =
                PathRAG(chunkTokenSize = 16, chunkOverlapTokenSize = 4)
            val neo =
                PathRAG(
                    kvStorage = "Neo4jKVStorage",
                    vectorStorage = "Neo4jVectorStorage",
                    graphStorage = "Neo4jStorage",
                    chunkTokenSize = 16,
                    chunkOverlapTokenSize = 4,
                    extraConfig =
                        pathrag.base.ExtraConfig(
                            neo4jUri = "bolt://mock",
                            neo4jUser = "user",
                            neo4jPassword = "pass",
                        ),
                )

            inMemory.use { mem ->
                neo.use { neoRag ->
                    val param = QueryParam(mode = "hybrid", topK = 3)
                    val memResult = mem.query("Tell me about Alpha", param)
                    val neoResult = neoRag.query("Tell me about Alpha", param)
                    assertEquals(memResult, neoResult)
                    assertEquals("ANSWER", memResult)
                }
            }
        }
}
