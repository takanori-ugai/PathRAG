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
 * Lists all stored keys.
 *
 * @return A list of all keys currently stored.
 */
    override suspend fun allKeys(): List<String> = data.keys().toList()

    /**
 * Retrieve the value associated with the given id.
 *
 * @return The value for `id` if present, or `null` if no entry exists.
 */
    override suspend fun getById(id: String): T? = data[id]

    /**
     * Retrieve stored values for the given ids in the same order as provided.
     *
     * @param ids The identifiers to fetch.
     * @param fields Optional set of field names to project; currently ignored and full objects are returned.
     * @return A list where each element is the value corresponding to the id at the same index, or `null` if that id is not present.
     */
    override suspend fun getByIds(
        ids: List<String>,
        fields: Set<String>?,
    ): List<T?> = ids.map { data[it] }

    /**
     * Determine which identifiers from the provided list are not present in storage.
     *
     * @param data The list of identifiers to check for presence in the store.
     * @return The set of identifiers from `data` that are not currently stored.
     */
    override suspend fun filterKeys(data: List<String>): Set<String> {
        val existing = allKeys().toSet()
        return data.filterNot { existing.contains(it) }.toSet()
    }

    /**
     * Inserts or updates the provided key-value pairs in the storage atomically.
     *
     * @param data Map of keys to values to insert or update.
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
     * Finds stored vectors most similar to the provided text query.
     *
     * Returns an ordered list of up to `topK` result maps. Each result contains:
     * - `"content"`: the stored content string,
     * - `"score"`: the cosine similarity score as a `Double`,
     * - plus all key/value pairs from the stored metadata.
     *
     * Returns an empty list if there are no stored entries, if `query` is blank,
     * or if embedding generation produces no vectors.
     *
     * @param query Text to embed and use for similarity search.
     * @param topK Maximum number of results to return.
     * @return A list of result maps sorted by descending `"score"`.
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
     * Insert or update vectors and their metadata into the vector store.
     *
     * Accepts a map of entry id to a property map that must include a "content" value; entries with missing or blank "content" are ignored.
     * For each kept entry this generates an embedding (via the storage's embedding function) and stores the embedding, the content string, and the entry's metadata filtered by `metaFields`.
     *
     * @param data Map from entry id to a map of properties; the property "content" is used as the text to embed and other keys are considered metadata.
     * @throws IllegalStateException If embedding generation fails.
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
     * Remove the stored vector entry associated with the given entity.
     *
     * @param entityName The entity identifier used to compute the stored entry key (prefixed with "ent-").
     */
    override suspend fun deleteEntity(entityName: String) {
        val entityId = computeMdHashId(entityName, prefix = "ent-")
        mutex.withLock {
            entries.remove(entityId)
        }
    }

    /**
     * Remove stored relation vectors whose metadata references the given entity.
     *
     * Matches entries where the metadata key "src_id" or "tgt_id" equals the provided `entityName`
     * and removes those entries while holding the storage mutex.
     *
     * @param entityName The entity identifier to match against relation metadata (`src_id` or `tgt_id`).
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
     * Remove the stored vector representing a relation between two entities.
     *
     * @param srcId Identifier of the source entity participating in the relation.
     * @param tgtId Identifier of the target entity participating in the relation.
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
     * Remove all stored vectors and associated metadata from this namespace.
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
 * Determine whether a node with the given identifier exists in the graph.
 *
 * @param nodeId The node identifier to check.
 * @return `true` if a node with the given identifier exists, `false` otherwise.
 */
    override suspend fun hasNode(nodeId: String): Boolean = nodes.containsKey(nodeId)

    /**
     * Determine whether an edge from one node to another exists.
     *
     * @param sourceNodeId Identifier of the source node.
     * @param targetNodeId Identifier of the target node.
     * @return `true` if an edge from `sourceNodeId` to `targetNodeId` exists, `false` otherwise.
     */
    override suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean = edges.containsKey(sourceNodeId to targetNodeId)

    /**
 * Get the number of edges connected to the given node.
 *
 * @return The count of edges where the node is either the source or the target.
 */
    override suspend fun nodeDegree(nodeId: String): Int = edges.keys.count { it.first == nodeId || it.second == nodeId }

    /**
     * Determine whether a specific directed edge exists.
     *
     * @return `1` if the edge exists, `0` otherwise.
     */
    override suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int = if (edges.containsKey(srcId to tgtId)) 1 else 0

    /**
 * Retrieve a copy of the properties for the node with the given id.
 *
 * @param nodeId Identifier of the node to fetch.
 * @return A map containing the node's properties if the node exists, or `null` if it does not.
 */
    override suspend fun getNode(nodeId: String): Map<String, Any?>? = nodes[nodeId]?.toMap()

    /**
     * Retrieve a copy of properties for the edge from sourceNodeId to targetNodeId.
     *
     * @param sourceNodeId Source node identifier.
     * @param targetNodeId Target node identifier.
     * @return A map of edge properties, or `null` if the edge does not exist.
     */
    override suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, Any?>? = edges[sourceNodeId to targetNodeId]?.toMap()

    /**
         * List all edges that include the given node.
         *
         * @param sourceNodeId The node identifier to search for.
         * @return A list of pairs `(sourceId, targetId)` for each edge where the node is either the source or the target.
         */
    override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>> =
        edges.keys.filter { it.first == sourceNodeId || it.second == sourceNodeId }

    /**
     * Upserts a node's properties into the graph and invalidates the cached PageRank.
     *
     * Merges the provided properties into any existing properties for the node; keys in `nodeData`
     * overwrite existing keys. Also clears the stored PageRank cache so it will be recomputed when needed.
     *
     * @param nodeId The identifier of the node to insert or update.
     * @param nodeData A map of properties to merge into the node; values may be `null` to represent absent properties.
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
     * Insert or update properties for the edge from sourceNodeId to targetNodeId.
     *
     * Merges the provided `edgeData` into existing edge properties (overwriting any matching keys) or creates the edge if it does not exist. Resets the cached PageRank so ranks will be recomputed on next request.
     *
     * @param sourceNodeId Identifier of the source node.
     * @param targetNodeId Identifier of the target node.
     * @param edgeData Map of edge property names to values; existing properties with the same keys are overwritten.
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
     * Remove the node identified by [nodeId] and all edges connected to it, and invalidate cached PageRank.
     *
     * @param nodeId Identifier of the node to remove.
     */
    override suspend fun deleteNode(nodeId: String) {
        mutex.withLock {
            nodes.remove(nodeId)
            edges.keys.filter { it.first == nodeId || it.second == nodeId }.forEach { edges.remove(it) }
            cachedPagerank = null
        }
    }

    /**
     * Removes the directed edge from sourceNodeId to targetNodeId and invalidates the cached PageRank.
     *
     * @param sourceNodeId Identifier of the source node.
     * @param targetNodeId Identifier of the target node.
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
 * Retrieves all node identifiers present in the graph.
 *
 * @return A list of node identifiers.
 */
    override suspend fun nodes(): List<String> = nodes.keys().toList()

    /**
 * Return all stored edges as a list of (sourceNodeId, targetNodeId) pairs.
 *
 * @return A list containing a pair for each edge where the first element is the source node id and the second is the target node id.
 */
    override suspend fun edges(): List<Pair<String, String>> = edges.keys.toList()

    /**
     * Get the PageRank score for a node.
     *
     * Uses a cached PageRank mapping when available; computes and caches PageRank otherwise.
     *
     * @return The PageRank score for the given node identifier, or `0.0` if the node is not present.
     */
    override suspend fun getPagerank(nodeId: String): Double {
        val ranks =
            cachedPagerank ?: mutex.withLock {
                cachedPagerank ?: computePagerank().also { cachedPagerank = it }
            }
        return ranks[nodeId] ?: 0.0
    }

    /**
     * Produce embeddings for all nodes using the specified algorithm.
     *
     * If `algorithm` equals `"node2vec"` (case-insensitive) Node2Vec-based embeddings are produced; any other value falls back to metadata-based embeddings.
     *
     * @param algorithm The embedding algorithm name (`"node2vec"` or other to select metadata embedding).
     * @return A pair where the first element is a flattened DoubleArray of embeddings (concatenated per-node vectors) and the second element is the list of node IDs in the same order as the embeddings.
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