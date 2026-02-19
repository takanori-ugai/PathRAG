package pathrag

import kotlinx.coroutines.runBlocking
import pathrag.base.BaseGraphStorage
import pathrag.base.BaseKVStorage
import pathrag.base.BaseVectorStorage
import pathrag.base.QueryParam
import pathrag.operate.chunkingByTokenSize
import pathrag.operate.kgQuery
import pathrag.utils.ResponseCache
import pathrag.utils.computeArgsHash
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

private class FakeGraphStorage : BaseGraphStorage("test", emptyMap()) {
    private val nodes = mutableMapOf<String, MutableMap<String, Any?>>()
    private val edges = mutableMapOf<Pair<String, String>, MutableMap<String, Any?>>()

    override suspend fun hasNode(nodeId: String): Boolean = nodes.containsKey(nodeId)

    override suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean = edges.containsKey(sourceNodeId to targetNodeId)

    override suspend fun nodeDegree(nodeId: String): Int = edges.keys.count { it.first == nodeId || it.second == nodeId }

    override suspend fun getNode(nodeId: String): Map<String, Any?>? = nodes[nodeId]

    override suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, Any?>? = edges[sourceNodeId to targetNodeId]

    override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>> =
        edges.keys.filter { it.first == sourceNodeId || it.second == sourceNodeId }

    override suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, Any?>,
    ) {
        nodes[nodeId] = (nodes[nodeId] ?: mutableMapOf()).apply { putAll(nodeData) }
    }

    override suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, Any?>,
    ) {
        edges[sourceNodeId to targetNodeId] = (edges[sourceNodeId to targetNodeId] ?: mutableMapOf()).apply { putAll(edgeData) }
    }

    override suspend fun deleteEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ) {
        edges.remove(sourceNodeId to targetNodeId)
    }

    override suspend fun deleteNode(nodeId: String) {
        nodes.remove(nodeId)
        edges.keys.filter { it.first == nodeId || it.second == nodeId }.forEach { edges.remove(it) }
    }

    override suspend fun nodes(): List<String> = nodes.keys.toList()

    override suspend fun edges(): List<Pair<String, String>> = edges.keys.toList()
}

private class FakeVectorStorage(
    override val namespace: String,
    private val results: List<Map<String, Any?>>,
) : BaseVectorStorage(namespace, emptyMap()) {
    override suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, Any?>> = results.take(topK)

    override suspend fun upsert(data: Map<String, Map<String, Any?>>) {}
}

private class FakeKVStorage(
    override val namespace: String,
    private val records: Map<String, Map<String, Any>?>,
) : BaseKVStorage<Map<String, Any>>(namespace, emptyMap()) {
    override suspend fun allKeys(): List<String> = records.keys.toList()

    override suspend fun getById(id: String): Map<String, Any>? = records[id]

    override suspend fun getByIds(
        ids: List<String>,
        fields: Set<String>?,
    ): List<Map<String, Any>?> = ids.map { records[it] }

    override suspend fun filterKeys(data: List<String>): Set<String> = data.filterNot { records.containsKey(it) }.toSet()

    override suspend fun upsert(data: Map<String, Map<String, Any>>) {}

    override suspend fun drop() {}
}

class OperateTest {
    @Test
    fun chunkingByTokenSize_splitsContent() {
        val text = "Kotlin ".repeat(300)
        val chunks = chunkingByTokenSize(text, overlapTokenSize = 10, maxTokenSize = 50, tiktokenModel = "gpt-4o-mini")
        assertTrue(chunks.size > 1)
        val indices = chunks.map { it["chunk_order_index"] as Int }
        assertEquals(indices.sorted(), indices)
        assertTrue(chunks.all { (it["tokens"] as Int) <= 50 })
    }

    @Test
    fun kgQuery_returnsLocalContext() =
        runBlocking {
            val graph = FakeGraphStorage()
            graph.upsertNode("NODE", mapOf("entity_type" to "THING", "description" to "desc", "source_id" to "doc1"))
            val entitiesVdb =
                FakeVectorStorage(
                    "entities",
                    listOf(mapOf("entity_name" to "NODE", "content" to "desc", "full_doc_id" to "doc1")),
                )
            val relationshipsVdb = FakeVectorStorage("rels", emptyList())
            val kv = FakeKVStorage("chunks", mapOf("doc1" to mapOf("content" to "Document content")))

            val context =
                kgQuery(
                    query = "question",
                    knowledgeGraphInst = graph,
                    entitiesVdb = entitiesVdb,
                    relationshipsVdb = relationshipsVdb,
                    textChunksDb = kv,
                    queryParam = QueryParam(mode = "local", onlyNeedContext = true, topK = 5),
                    globalConfig = emptyMap(),
                    llmModel = { prompt, system, _, _, _, _, _ -> "llm-$prompt-$system" },
                )
            assertTrue(context.contains("local-information"))
            assertTrue(context.contains("NODE"))
            assertTrue(context.contains("Document content"))
        }

    @Test
    fun kgQuery_hybridUsesCacheOnSecondCall() =
        runBlocking {
            val graph = FakeGraphStorage()
            graph.upsertNode("NODE", mapOf("entity_type" to "THING", "description" to "desc", "source_id" to "doc1"))
            graph.upsertEdge("NODE", "NODE2", mapOf("description" to "linked", "keywords" to "k"))
            val entitiesVdb =
                FakeVectorStorage(
                    "entities",
                    listOf(mapOf("entity_name" to "NODE", "content" to "desc", "full_doc_id" to "doc1")),
                )
            val relationshipsVdb =
                FakeVectorStorage(
                    "rels",
                    listOf(mapOf("src_id" to "NODE", "tgt_id" to "NODE2", "content" to "linked", "keywords" to "k", "weight" to 1.0)),
                )
            val kv = FakeKVStorage("chunks", mapOf("doc1" to mapOf("content" to "Document content")))
            var calls = 0
            val llm: suspend (
                String,
                String?,
                List<Map<String, String>>,
                Boolean,
                Boolean,
                Int?,
                Any?,
            ) -> String = { prompt, _, _, _, _, _, _ ->
                calls += 1
                "answer-$prompt"
            }

            val param = QueryParam(mode = "hybrid", topK = 5)
            val cache = ResponseCache()
            val first =
                kgQuery(
                    query = "question",
                    knowledgeGraphInst = graph,
                    entitiesVdb = entitiesVdb,
                    relationshipsVdb = relationshipsVdb,
                    textChunksDb = kv,
                    queryParam = param,
                    globalConfig = emptyMap(),
                    llmModel = llm,
                    hashingKv = cache,
                )
            val callsAfterFirst = calls
            val second =
                kgQuery(
                    query = "question",
                    knowledgeGraphInst = graph,
                    entitiesVdb = entitiesVdb,
                    relationshipsVdb = relationshipsVdb,
                    textChunksDb = kv,
                    queryParam = param,
                    globalConfig = emptyMap(),
                    llmModel = llm,
                    hashingKv = cache,
                )
            assertEquals(callsAfterFirst, calls, "LLM should not be called again on cache hit")
            val cached = cache.handleCache(computeArgsHash(param.mode, "question"), "question", param.mode)
            assertEquals(first, second)
            assertEquals(first, cached)
        }
}
