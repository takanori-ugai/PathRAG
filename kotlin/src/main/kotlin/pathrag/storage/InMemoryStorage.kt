package pathrag.storage

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import pathrag.base.BaseGraphStorage
import pathrag.base.BaseKVStorage
import pathrag.base.BaseVectorStorage
import pathrag.utils.EmbeddingFunc
import pathrag.utils.computeMdHashId
import pathrag.utils.computePagerankLocal
import pathrag.utils.cosineSimilarity
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.abs

/**
 * Simple in-memory key-value storage used as the default KV backend.
 */
class JsonKVStorage<T : Any>(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
    private val embeddingFunc: EmbeddingFunc?,
) : BaseKVStorage<T>(namespace, globalConfig) {
    private val mutex = Mutex()
    private val data = ConcurrentHashMap<String, T>()

    /**
     * Return all stored keys.
     */
    override suspend fun allKeys(): List<String> = data.keys().toList()

    /**
     * Fetch a value by id.
     */
    override suspend fun getById(id: String): T? = data[id]

    /**
     * Fetch multiple values by ids.
     */
    override suspend fun getByIds(
        ids: List<String>,
        fields: Set<String>?,
    ): List<T?> = ids.map { data[it] }

    /**
     * Identify which ids are not already stored.
     */
    override suspend fun filterKeys(data: List<String>): Set<String> {
        val existing = allKeys().toSet()
        return data.filterNot { existing.contains(it) }.toSet()
    }

    /**
     * Insert or update values.
     */
    override suspend fun upsert(data: Map<String, T>) {
        mutex.withLock {
            this.data.putAll(data)
        }
    }

    /**
     * Remove all stored values.
     */
    override suspend fun drop() {
        mutex.withLock { data.clear() }
    }
}

/**
 * Lightweight in-memory vector store with cosine similarity search.
 */
class NanoVectorDBStorage(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
    private val embeddingFunc: EmbeddingFunc,
    private val metaFields: Set<String> = setOf("entity_name", "full_doc_id", "source_id"),
) : BaseVectorStorage(namespace, globalConfig) {
    private val logger = KotlinLogging.logger("PathRAG-NanoVectorDBStorage")
    private val mutex = Mutex()
    private val entries = ConcurrentHashMap<String, StoredVector>()

    /**
     * Stored vector entry with optional metadata.
     *
     * @property embedding vector values.
     * @property content raw content tied to the vector.
     * @property meta metadata persisted alongside the vector.
     */
    data class StoredVector(
        val embedding: DoubleArray,
        val content: String,
        val meta: Map<String, Any?> = emptyMap(),
    )

    /**
     * Query vectors by similarity using embeddings generated for the query text.
     */
    @Suppress("TooGenericExceptionCaught")
    override suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, Any?>> {
        if (entries.isEmpty()) return emptyList()
        if (query.isBlank()) return emptyList()
        val queryEmbeddings =
            try {
                embeddingFunc(listOf(query))
            } catch (e: IllegalStateException) {
                logger.error(e) { "Embedding generation failed during query in namespace '$namespace'." }
                throw e
            }
        if (queryEmbeddings.isEmpty()) return emptyList()
        val queryEmbedding = queryEmbeddings.first()
        return entries.values
            .map { stored ->
                val similarity = cosineSimilarity(queryEmbedding, stored.embedding)
                mapOf("content" to stored.content, "score" to similarity) + stored.meta
            }.sortedByDescending { it["score"] as Double }
            .take(topK)
    }

    /**
     * Insert or update vectors with metadata.
     */
    @Suppress("TooGenericExceptionCaught")
    override suspend fun upsert(data: Map<String, Map<String, Any?>>) {
        val items = data.entries.toList()
        val contents = items.map { it.value["content"]?.toString().orEmpty() }
        val validPairs = items.zip(contents).filter { it.second.isNotBlank() }
        if (validPairs.isEmpty()) return
        val embeddings =
            try {
                embeddingFunc(validPairs.map { it.second })
            } catch (e: IllegalStateException) {
                logger.error(e) { "Embedding generation failed during upsert in namespace '$namespace'." }
                throw e
            }
        mutex.withLock {
            embeddings.forEachIndexed { index, vector ->
                val (entry, content) = validPairs[index]
                val key = entry.key
                val value = entry.value
                val meta = value.filterKeys { metaFields.contains(it) }
                entries[key] = StoredVector(vector, content, meta)
            }
        }
    }

    /**
     * Delete all vectors related to an entity.
     */
    override suspend fun deleteEntity(entityName: String) {
        val entityId = computeMdHashId(entityName, prefix = "ent-")
        mutex.withLock {
            entries.remove(entityId)
        }
    }

    /**
     * Delete relation vectors that reference the entity.
     */
    override suspend fun deleteRelation(entityName: String) {
        // Remove any relation vectors involving this entity (matches src_id or tgt_id in metadata)
        mutex.withLock {
            val toRemove =
                entries.keys.filter { key ->
                    val stored = entries[key]
                    stored?.meta?.get("src_id") == entityName || stored?.meta?.get("tgt_id") == entityName
                }
            toRemove.forEach { entries.remove(it) }
        }
    }

    /**
     * Delete a specific relationship vector.
     */
    override suspend fun deleteRelationBetween(
        srcId: String,
        tgtId: String,
    ) {
        val relId = computeMdHashId(srcId + tgtId, prefix = "rel-")
        mutex.withLock {
            entries.remove(relId)
        }
    }

    /**
     * Drop the entire vector namespace.
     */
    override suspend fun drop() {
        mutex.withLock { entries.clear() }
    }
}

/**
 * In-memory graph storage that mirrors NetworkX behavior for small graphs.
 */
class NetworkXStorage(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
    private val embeddingFunc: EmbeddingFunc?,
) : BaseGraphStorage(namespace, globalConfig) {
    private val mutex = Mutex()
    private val nodes = ConcurrentHashMap<String, MutableMap<String, Any?>>()
    private val edges = ConcurrentHashMap<Pair<String, String>, MutableMap<String, Any?>>()
    private var cachedPagerank: Map<String, Double>? = null

    /**
     * Check whether a node exists.
     */
    override suspend fun hasNode(nodeId: String): Boolean = nodes.containsKey(nodeId)

    /**
     * Check whether an edge exists.
     */
    override suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean = edges.containsKey(sourceNodeId to targetNodeId)

    /**
     * Compute degree for a node.
     */
    override suspend fun nodeDegree(nodeId: String): Int = edges.keys.count { it.first == nodeId || it.second == nodeId }

    /**
     * Compute degree for a specific edge.
     */
    override suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int = if (edges.containsKey(srcId to tgtId)) 1 else 0

    /**
     * Fetch node properties.
     */
    override suspend fun getNode(nodeId: String): Map<String, Any?>? = nodes[nodeId]?.toMap()

    /**
     * Fetch edge properties.
     */
    override suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, Any?>? = edges[sourceNodeId to targetNodeId]?.toMap()

    /**
     * List all edges touching a node.
     */
    override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>> =
        edges.keys.filter { it.first == sourceNodeId || it.second == sourceNodeId }

    /**
     * Insert or update a node.
     */
    override suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, Any?>,
    ) {
        mutex.withLock {
            cachedPagerank = null
            val existing = nodes[nodeId] ?: mutableMapOf()
            existing.putAll(nodeData)
            nodes[nodeId] = existing
        }
    }

    /**
     * Insert or update an edge.
     */
    override suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, Any?>,
    ) {
        mutex.withLock {
            cachedPagerank = null
            val existing = edges[sourceNodeId to targetNodeId] ?: mutableMapOf()
            existing.putAll(edgeData)
            edges[sourceNodeId to targetNodeId] = existing
        }
    }

    /**
     * Delete a node and attached edges.
     */
    override suspend fun deleteNode(nodeId: String) {
        mutex.withLock {
            nodes.remove(nodeId)
            edges.keys.filter { it.first == nodeId || it.second == nodeId }.forEach { edges.remove(it) }
            cachedPagerank = null
        }
    }

    /**
     * Delete an edge.
     */
    override suspend fun deleteEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ) {
        mutex.withLock {
            edges.remove(sourceNodeId to targetNodeId)
            cachedPagerank = null
        }
    }

    /**
     * List node identifiers.
     */
    override suspend fun nodes(): List<String> = nodes.keys().toList()

    /**
     * List edges as pairs.
     */
    override suspend fun edges(): List<Pair<String, String>> = edges.keys.toList()

    /**
     * Retrieve cached or computed PageRank.
     */
    override suspend fun getPagerank(nodeId: String): Double {
        val ranks =
            cachedPagerank ?: mutex.withLock {
                cachedPagerank ?: computePagerank().also { cachedPagerank = it }
            }
        return ranks[nodeId] ?: 0.0
    }

    /**
     * Embed nodes using metadata or node2vec.
     */
    override suspend fun embedNodes(algorithm: String): Pair<DoubleArray, List<String>> {
        val labels = nodes.keys().toList()
        if (labels.isEmpty()) return DoubleArray(0) to emptyList()
        return when (algorithm.lowercase()) {
            "node2vec" -> runNode2Vec(labels)
            else -> runMetadataEmbedding(labels)
        }
    }

    private suspend fun runMetadataEmbedding(labels: List<String>): Pair<DoubleArray, List<String>> {
        val func = embeddingFunc
        return if (func != null) {
            val texts =
                labels.map { id ->
                    val n = nodes[id] ?: emptyMap()
                    val desc = n["description"]?.toString().orEmpty()
                    "$id ${n["entity_type"] ?: ""} $desc"
                }
            val vectors = func(texts)
            val flat = vectors.flatMap { it.asIterable() }.toDoubleArray()
            flat to labels
        } else {
            val ranks = computePagerank()
            val degs = labels.map { nodeDegree(it).toDouble() }
            val vecs =
                labels.mapIndexed { idx, id ->
                    doubleArrayOf(ranks[id] ?: 0.0, degs[idx])
                }
            val flat = vecs.flatMap { it.asIterable() }.toDoubleArray()
            flat to labels
        }
    }

    private suspend fun runNode2Vec(labels: List<String>): Pair<DoubleArray, List<String>> {
        val dim = globalConfig["node2vec_dim"] as? Int ?: 64
        if (labels.isEmpty()) return DoubleArray(0) to emptyList()
        val ranks = computePagerank()
        val degs = labels.map { nodeDegree(it).toDouble() }
        val vectors =
            labels.mapIndexed { idx, id ->
                DoubleArray(dim) { i ->
                    val r = ranks[id] ?: 0.0
                    val d = degs[idx]
                    if (i % 2 == 0) r else d
                }
            }
        val flat = vectors.flatMap { it.asIterable() }.toDoubleArray()
        return flat to labels
    }

    private fun computePagerank(
        damping: Double = 0.85,
        maxIter: Int = 100,
        tol: Double = 1e-6,
    ): Map<String, Double> {
        val nodeList = nodes.keys().toList()
        val edgeList = edges.keys.toList()
        return computePagerankLocal(nodeList, edgeList, damping, maxIter, tol)
    }

    override suspend fun drop() {
        mutex.withLock {
            nodes.clear()
            edges.clear()
            cachedPagerank = null
        }
    }
}
