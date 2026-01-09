package pathrag

import kotlinx.coroutines.runBlocking
import java.io.File
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PathRAGTest {
    @AfterTest
    fun cleanup() {
        File(".")
            .listFiles { file -> file.isDirectory && file.name.startsWith("PathRAG_cache_") }
            ?.forEach { it.deleteRecursively() }
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
}
