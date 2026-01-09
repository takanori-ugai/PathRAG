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
 * List all keys currently stored in this in-memory key-value storage.
 *
 * @return A list of all stored keys.
 */
    override suspend fun allKeys(): List<String> = data.keys().toList()

    /**
 * Retrieve the value associated with the given id from storage.
 *
 * @return The stored value for the id, or `null` if no entry exists.
 */
    override suspend fun getById(id: String): T? = data[id]

    /**
     * Retrieve values for the given ids preserving input order.
     *
     * Returns a list aligned with the input `ids` where each element is the stored value for the corresponding id or `null` if that id is absent. The `fields` parameter is ignored by this implementation.
     *
     * @param ids The list of ids to fetch; result list preserves this order.
     * @param fields Ignored by this storage implementation.
     * @return A list of values or `null` for each requested id, in the same order as `ids`.
     */
    override suspend fun getByIds(
        ids: List<String>,
        fields: Set<String>?,
    ): List<T?> = ids.map { data[it] }

    /**
     * Identify which input ids are not present in the storage.
     *
     * @param data The list of ids to check.
     * @return A set containing the ids from `data` that are not currently stored.
     */
    override suspend fun filterKeys(data: List<String>): Set<String> {
        val existing = allKeys().toSet()
        return data.filterNot { existing.contains(it) }.toSet()
    }

    /**
     * Insert or update multiple key-value pairs in the storage under mutual exclusion.
     *
     * @param data Map of keys to values to upsert.
     */
    override suspend fun upsert(data: Map<String, T>) {
        mutex.withLock {
            this.data.putAll(data)
        }
    }

    /**
     * Clears all entries from the storage.
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
     * Find stored vectors most similar to the provided text query using the configured embedding function.
     *
     * @param query Text to embed and search with.
     * @param topK Maximum number of results to return.
     * @return A list of result maps sorted by descending similarity. Each map contains the keys:
     *  - "content": the stored content string,
     *  - "score": cosine similarity (Double) between the query and the stored vector,
     *  plus any stored metadata fields.
     * @throws IllegalStateException If embedding generation fails.
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
     * Inserts or updates stored vectors and their metadata for the provided entries.
     *
     * For each map entry whose "content" field is non-blank, generates an embedding using the configured embedding function,
     * then stores a StoredVector containing the embedding, the content string, and the metadata filtered to the configured metaFields.
     * The storage update is performed under a coroutine mutex to ensure consistency.
     *
     * @throws IllegalStateException if embedding generation fails; the exception is logged and rethrown.
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
     * Removes the stored vector entry for the specified entity.
     *
     * @param entityName The entity name used to compute the storage key (prefixed with "ent-").
     */
    override suspend fun deleteEntity(entityName: String) {
        val entityId = computeMdHashId(entityName, prefix = "ent-")
        mutex.withLock {
            entries.remove(entityId)
        }
    }

    /**
     * Removes all stored relation vectors whose metadata references the given entity.
     *
     * Deletes any entry whose metadata `src_id` or `tgt_id` equals `entityName`.
     *
     * @param entityName The entity identifier to match against stored vectors' `src_id` or `tgt_id` metadata.
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
     * Remove the stored vector representing the relation from the source entity to the target entity.
     *
     * @param srcId The source entity identifier.
     * @param tgtId The target entity identifier.
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
     * Removes all stored vectors and associated metadata from this vector namespace.
     *
     * This is a destructive operation that clears the in-memory index for this storage instance.
     */
    override suspend fun drop() {
        mutex.withLock { entries.clear() }
    }

    /**
     * Compute the cosine similarity between two vectors.
     *
     * @param a The first vector.
     * @param b The second vector.
     * @return Cosine similarity score (between -1.0 and 1.0). Returns `0.0` if the vectors have different lengths, are empty, or either vector has zero magnitude.
     */
    private fun cosineSimilarity(
        a: DoubleArray,
        b: DoubleArray,
    ): Double {
        if (a.size != b.size) {
            logger.warn { "Cannot compute cosine similarity for vectors of different dimensions: ${a.size} vs ${b.size}" }
            return 0.0
        }
        if (a.isEmpty()) return 0.0
        val dot = a.zip(b).sumOf { it.first * it.second }
        val normA = kotlin.math.sqrt(a.sumOf { it * it })
        val normB = kotlin.math.sqrt(b.sumOf { it * it })
        return if (normA == 0.0 || normB == 0.0) 0.0 else dot / (normA * normB)
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
 * Determines whether a node with the specified identifier exists in the graph.
 *
 * @return `true` if a node with the given id exists, `false` otherwise.
 */
    override suspend fun hasNode(nodeId: String): Boolean = nodes.containsKey(nodeId)

    /**
     * Determine whether a directed edge from the source node to the target node exists.
     *
     * @param sourceNodeId ID of the source node.
     * @param targetNodeId ID of the target node.
     * @return `true` if the edge exists, `false` otherwise.
     */
    override suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean = edges.containsKey(sourceNodeId to targetNodeId)

    /**
 * Get the degree (number of edges incident to the specified node).
 *
 * @param nodeId The node identifier.
 * @return The number of edges where the node is either the source or the target.
 */
    override suspend fun nodeDegree(nodeId: String): Int = edges.keys.count { it.first == nodeId || it.second == nodeId }

    /**
     * Determine whether a directed edge exists between two nodes.
     *
     * @param srcId The source node identifier.
     * @param tgtId The target node identifier.
     * @return `1` if an edge from `srcId` to `tgtId` exists, `0` otherwise.
     */
    override suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int = if (edges.containsKey(srcId to tgtId)) 1 else 0

    /**
 * Retrieve properties for a node by its identifier.
 *
 * @param nodeId The node identifier.
 * @return The properties map for the node, or `null` if the node does not exist.
 */
    override suspend fun getNode(nodeId: String): Map<String, Any?>? = nodes[nodeId]

    /**
     * Retrieve the property map for the edge from the given source to target node.
     *
     * @param sourceNodeId ID of the source node.
     * @param targetNodeId ID of the target node.
     * @return The properties map for the specified edge, or `null` if the edge does not exist.
     */
    override suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, Any?>? = edges[sourceNodeId to targetNodeId]

    /**
         * Return all edges incident to the specified node.
         *
         * @param sourceNodeId The node identifier to find incident edges for.
         * @return A list of pairs `(sourceId, targetId)` for each edge where `sourceNodeId` is either the source or the target.
         */
    override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>> =
        edges.keys.filter { it.first == sourceNodeId || it.second == sourceNodeId }

    /**
     * Insert a new node or merge properties into an existing node.
     *
     * Merges the provided `nodeData` into the stored properties for `nodeId`, creating the node if it does not exist.
     * This operation invalidates the cached PageRank values.
     *
     * @param nodeId The identifier of the node to insert or update.
     * @param nodeData Map of node properties to merge into the existing node entry.
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
     * Upserts an edge between two nodes and merges the provided properties into the stored edge.
     *
     * The stored PageRank cache is cleared as a result of this mutation.
     *
     * @param sourceNodeId The ID of the source node for the edge.
     * @param targetNodeId The ID of the target node for the edge.
     * @param edgeData Properties to merge into the existing edge entry (added or overwritten).
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
     * Remove the node identified by [nodeId] and all edges incident to it.
     *
     * Also clears the cached PageRank data so ranks will be recomputed on next request.
     *
     * @param nodeId The identifier of the node to delete.
     */
    override suspend fun deleteNode(nodeId: String) {
        mutex.withLock {
            nodes.remove(nodeId)
            edges.keys.filter { it.first == nodeId || it.second == nodeId }.forEach { edges.remove(it) }
            cachedPagerank = null
        }
    }

    /**
     * Remove the edge between the given source and target nodes.
     *
     * Also clears the cached PageRank so ranks will be recomputed on next request.
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
 * List all node identifiers currently stored in the graph.
 *
 * @return A list of node IDs present in the storage.
 */
    override suspend fun nodes(): List<String> = nodes.keys().toList()

    /**
 * List all edges as pairs of source and target node IDs.
 *
 * @return A list of pairs where each pair is (sourceNodeId, targetNodeId) representing an edge.
 */
    override suspend fun edges(): List<Pair<String, String>> = edges.keys.toList()

    /**
     * Get the PageRank score for the specified node, computing and caching PageRank if not already available.
     *
     * @param nodeId The identifier of the node whose PageRank score is requested.
     * @return The PageRank score for `nodeId`, or `0.0` if the node is not present in the computed ranks.
     */
    override suspend fun getPagerank(nodeId: String): Double {
        val ranks =
            cachedPagerank ?: mutex.withLock {
                cachedPagerank ?: computePagerank().also { cachedPagerank = it }
            }
        return ranks[nodeId] ?: 0.0
    }

    /**
     * Produce embeddings for all stored nodes using the specified algorithm.
     *
     * Uses "node2vec" to generate structural embeddings; any other value selects
     * a metadata-based embedding strategy. If there are no nodes, returns an
     * empty embedding array and an empty label list.
     *
     * @param algorithm The embedding algorithm name; "node2vec" selects node2vec,
     *                  any other value selects metadata-based embedding.
     * @return A pair where the first element is a flattened `DoubleArray` containing
     *         embeddings (concatenated per node) and the second element is the list
     *         of node IDs in the same order as the embeddings.
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