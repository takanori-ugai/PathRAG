package pathrag.storage

import kotlinx.coroutines.runBlocking
import pathrag.utils.EmbeddingFunc
import pathrag.utils.computeMdHashId
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

private fun stubEmbeddingFunc(dim: Int = 3): EmbeddingFunc =
    EmbeddingFunc(
        embeddingDim = dim,
        maxTokenSize = 16,
        func = { inputs ->
            inputs.map { text ->
                DoubleArray(dim) { idx -> (text.length + idx).toDouble() }
            }
        },
    )

class InMemoryStorageTest {
    @Test
    fun jsonKvStorage_persistsAndListsKeys() =
        runBlocking {
            val kv = JsonKVStorage<Map<String, Any>>("kv", emptyMap(), null)
            val payload =
                mapOf(
                    "id1" to mapOf("value" to 1),
                    "id2" to mapOf("value" to 2),
                )

            kv.upsert(payload)
            assertEquals(setOf("id1", "id2"), kv.allKeys().toSet())
            assertEquals(mapOf("value" to 1), kv.getById("id1"))

            val batch = kv.getByIds(listOf("id1", "missing", "id2"))
            assertEquals(listOf(mapOf("value" to 1), null, mapOf("value" to 2)), batch)

            val missing = kv.filterKeys(listOf("id1", "id3"))
            assertEquals(setOf("id3"), missing)

            kv.drop()
            assertTrue(kv.allKeys().isEmpty())
        }

    @Test
    fun nanoVectorDbStorage_upsertQueryAndDelete() =
        runBlocking {
            val embed = stubEmbeddingFunc()
            val store = NanoVectorDBStorage("vec", emptyMap(), embed, metaFields = setOf("entity_name", "src_id", "tgt_id"))
            val entKey = computeMdHashId("ALPHA", prefix = "ent-")
            store.upsert(mapOf(entKey to mapOf("content" to "alpha", "entity_name" to "ALPHA")))

            val results = store.query("alpha", topK = 5)
            assertEquals(1, results.size)
            assertEquals("ALPHA", results.first()["entity_name"])

            store.deleteEntity("ALPHA")
            assertTrue(store.query("alpha", topK = 5).isEmpty())

            val relKey = computeMdHashId("SRC" + "TGT", prefix = "rel-")
            store.upsert(mapOf(relKey to mapOf("content" to "relation", "src_id" to "SRC", "tgt_id" to "TGT")))
            assertTrue(store.query("relation", topK = 5).isNotEmpty())
            store.deleteRelationBetween("SRC", "TGT")
            assertTrue(store.query("relation", topK = 5).isEmpty())
        }

    @Test
    fun networkXStorage_handlesGraphOperationsAndPagerank() =
        runBlocking {
            val graph = NetworkXStorage("g", emptyMap(), stubEmbeddingFunc(dim = 2))
            graph.upsertNode("A", mapOf("description" to "first"))
            graph.upsertNode("B", mapOf("description" to "second"))
            graph.upsertEdge("A", "B", mapOf("weight" to 1.0))

            assertTrue(graph.hasNode("A"))
            assertTrue(graph.hasEdge("A", "B"))
            assertEquals(1, graph.nodeDegree("A"))
            assertEquals(1, graph.edgeDegree("A", "B"))
            assertNotNull(graph.getNode("A"))
            assertEquals(listOf("A", "B").toSet(), graph.nodes().toSet())

            val rankA = graph.getPagerank("A")
            val rankB = graph.getPagerank("B")
            assertTrue(rankA > 0.0 && rankB > 0.0)

            val (emb, labels) = graph.embedNodes("metadata")
            assertEquals(2, labels.size)
            assertEquals(4, emb.size) // two nodes * dim 2

            graph.deleteEdge("A", "B")
            assertFalse(graph.hasEdge("A", "B"))
            graph.deleteNode("A")
            assertFalse(graph.hasNode("A"))
            graph.drop()
            assertTrue(graph.nodes().isEmpty())
        }
}
