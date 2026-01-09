package pathrag.storage

import com.mongodb.client.model.Filters
import com.mongodb.client.model.ReplaceOneModel
import com.mongodb.client.model.ReplaceOptions
import com.mongodb.kotlin.client.coroutine.MongoClient
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.toList
import pathrag.base.BaseGraphStorage
import pathrag.base.BaseKVStorage
import pathrag.base.BaseVectorStorage
import pathrag.utils.EmbeddingFunc
import pathrag.utils.computeMdHashId
import pathrag.utils.computePagerankLocal
import pathrag.utils.log

/**
 * MongoDB-backed key-value storage that persists documents per namespace.
 */
class MongoKVStorage<T : Any>(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
) : BaseKVStorage<T>(namespace, globalConfig),
    AutoCloseable {
    private val client: MongoClient = MongoClient.create(globalConfig["mongo_uri"] as? String ?: "mongodb://localhost:27017")
    private val database = client.getDatabase(globalConfig["mongo_database"] as? String ?: "pathrag")
    private val collection = database.getCollection<org.bson.Document>("${namespace}_kv")

    /**
     * Close the MongoDB client.
     */
    override fun close() {
        client.close()
    }

    /**
             * Retrieve all document IDs stored in the collection for this namespace.
             *
             * @return A list of `_id` values for every document in the collection.
             */
            override suspend fun allKeys(): List<String> =
        collection
            .find()
            .projection(org.bson.Document("_id", 1))
            .map { it.getString("_id") }
            .toList()

    /**
             * Retrieve the document with the given id and return its stored value with the internal `_id` field removed.
             *
             * @return The document converted to `T` with the `_id` field removed, or `null` if no document with the id exists.
             */
            override suspend fun getById(id: String): T? =
        collection
            .find(Filters.eq("_id", id))
            .firstOrNull()
            ?.let { doc ->
                doc.remove("_id")
                @Suppress("UNCHECKED_CAST")
                doc.toMap() as T
            }

    /**
     * Retrieves values for the given IDs from the collection, preserving the input order.
     *
     * If `fields` is provided and non-empty, only those fields (plus `_id`) are included in each returned document.
     *
     * @param ids The list of document IDs to fetch; order of the returned list matches this list.
     * @param fields Optional set of field names to include in the result documents; when null or empty, all fields are returned.
     * @return A list of values cast to `T?` aligned with `ids`: each element is the corresponding document data if found, or `null` if no document exists for that ID.
     */
    override suspend fun getByIds(
        ids: List<String>,
        fields: Set<String>?,
    ): List<T?> {
        if (ids.isEmpty()) return emptyList()
        val projection =
            if (fields != null && fields.isNotEmpty()) {
                org.bson.Document(fields.associateWith { 1 }).append("_id", 1)
            } else {
                null
            }
        val results =
            collection
                .find(Filters.`in`("_id", ids))
                .let { cursor -> if (projection != null) cursor.projection(projection) else cursor }
                .map { doc ->
                    val id = doc.getString("_id")
                    doc.remove("_id")
                    id to doc.toMap()
                }.toList()
                .toMap()
        return ids.map { key ->
            results[key]?.let {
                @Suppress("UNCHECKED_CAST")
                it as T
            }
        }
    }

    /**
     * Compute which input keys do not exist in the MongoDB collection.
     *
     * @param data List of candidate document IDs to check for existence.
     * @return A set containing the IDs from `data` that are not present in the collection.
     */
    override suspend fun filterKeys(data: List<String>): Set<String> {
        if (data.isEmpty()) return emptySet()
        val existing =
            collection
                .find(Filters.`in`("_id", data))
                .projection(org.bson.Document("_id", 1))
                .map { it.getString("_id") }
                .toList()
                .toSet()
        return data.toSet() - existing
    }

    /**
     * Upserts multiple key-value entries into the collection, storing each value as a MongoDB document.
     *
     * Each map entry is written as a document whose `_id` is the map key; any `_id` field present in the value map is ignored.
     *
     * @param data Mapping from document id to the document fields (must be a Map of field names to values).
     * @throws IllegalArgumentException if any value in `data` is not a Map.
     */
    override suspend fun upsert(data: Map<String, T>) {
        if (data.isEmpty()) return
        val opts = ReplaceOptions().upsert(true)
        data.forEach { (id, value) ->
            val doc =
                when (value) {
                    is Map<*, *> -> org.bson.Document(value.filterKeys { it != "_id" } as Map<String, Any?>)

                    else -> throw IllegalArgumentException(
                        "MongoKVStorage only supports storing Map values. Got: ${value::class.simpleName}",
                    )
                }.append("_id", id)
            collection.replaceOne(Filters.eq("_id", id), doc, opts)
        }
    }

    /**
     * Removes the entire MongoDB collection backing this storage namespace.
     *
     * This permanently deletes all documents and indexes in the collection.
     */
    override suspend fun drop() {
        collection.drop()
    }
}

/**
 * MongoDB-backed vector storage that stores embeddings and metadata.
 */
class MongoVectorStorage(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
    private val embeddingFunc: EmbeddingFunc,
    private val metaFields: Set<String> = setOf("entity_name", "full_doc_id", "source_id", "src_id", "tgt_id"),
) : BaseVectorStorage(namespace, globalConfig),
    AutoCloseable {
    private val client: MongoClient = MongoClient.create(globalConfig["mongo_uri"] as? String ?: "mongodb://localhost:27017")
    private val database = client.getDatabase(globalConfig["mongo_database"] as? String ?: "pathrag")
    private val collection = database.getCollection<org.bson.Document>("${namespace}_vector")

    /**
     * Close the MongoDB client.
     */
    override fun close() {
        client.close()
    }

    /**
     * Finds and ranks stored documents by cosine similarity to the embedding of the given query.
     *
     * Queries the collection, computes the query embedding, scores each document by cosine similarity against its stored embedding,
     * and returns the top K results sorted by score descending.
     *
     * @param query Text to embed and match against stored documents. Blank queries produce an empty result.
     * @param topK Maximum number of results to return.
     * @return A list of maps for the top matches; each map contains a "content" string, a "score" double, and any metadata fields present in the document.
     */
    @Suppress("TooGenericExceptionCaught")
    override suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, Any?>> {
        if (query.isBlank()) return emptyList()
        val embeddings =
            try {
                embeddingFunc(listOf(query))
            } catch (e: IllegalStateException) {
                log().warn(e) { "Embedding generation failed for query in MongoVectorStorage ($namespace)" }
                return emptyList()
            }
        val queryEmbedding = embeddings.firstOrNull() ?: return emptyList()
        val docs = collection.find().toList()
        return docs
            .mapNotNull { doc ->
                val embList = (doc["embedding"] as? List<*>)?.mapNotNull { (it as? Number)?.toDouble() } ?: return@mapNotNull null
                val embedding = embList.toDoubleArray()
                val content = doc.getString("content") ?: ""
                val meta = doc.filterKeys { metaFields.contains(it) }
                val score = cosineSimilarity(queryEmbedding, embedding)
                mapOf("content" to content, "score" to score) + meta
            }.sortedByDescending { (it["score"] as? Double) ?: 0.0 }
            .take(topK)
    }

    /**
     * Upserts vector documents for entries that contain non-blank `content`.
     *
     * For each input entry whose `"content"` is non-blank, computes an embedding using the configured
     * embedding function and upserts a document with `_id` = entry key, `content`, `embedding` (as a list),
     * and any metadata fields present in `metaFields`. If `data` is empty or contains no entries with
     * non-blank content, the function performs no writes. Entries for which an embedding is not produced
     * are skipped.
     *
     * @param data Map from document id to a map of document fields; each value is expected to contain a
     * `"content"` entry whose string value will be embedded and stored.
     */
    override suspend fun upsert(data: Map<String, Map<String, Any?>>) {
        if (data.isEmpty()) return
        val items = data.entries.toList()
        val contents = items.map { it.value["content"]?.toString().orEmpty() }
        val validPairs = items.zip(contents).filter { it.second.isNotBlank() }
        if (validPairs.isEmpty()) return
        val embeddings = embeddingFunc(validPairs.map { it.second })
        val opts = ReplaceOptions().upsert(true)
        val writes =
            validPairs.mapIndexedNotNull { idx, pair ->
                val (entry, content) = pair
                val vector = embeddings.getOrNull(idx) ?: return@mapIndexedNotNull null
                val meta = entry.value.filterKeys { metaFields.contains(it) }
                val doc =
                    org.bson.Document(
                        mapOf(
                            "_id" to entry.key,
                            "content" to content,
                            "embedding" to vector.toList(),
                        ) + meta,
                    )
                ReplaceOneModel(Filters.eq("_id", entry.key), doc, opts)
            }
        if (writes.isNotEmpty()) {
            collection.bulkWrite(writes)
        }
    }

    /**
     * Deletes the stored entity document corresponding to the given entity name.
     *
     * The document's `_id` is derived from the entity name using an MD-based hash with the prefix `"ent-"`; the matching document is removed if present.
     *
     * @param entityName The name of the entity to delete.
     */
    override suspend fun deleteEntity(entityName: String) {
        val entityId = computeMdHashId(entityName, prefix = "ent-")
        collection.deleteOne(Filters.eq("_id", entityId))
    }

    /**
     * Deletes all relation documents that reference the given entity as either source or target.
     *
     * @param entityName The entity identifier whose relations (where `src_id` or `tgt_id` equals this value) should be removed.
     */
    override suspend fun deleteRelation(entityName: String) {
        collection.deleteMany(Filters.or(Filters.eq("src_id", entityName), Filters.eq("tgt_id", entityName)))
    }

    /**
     * Deletes the relation document that represents the edge from the source node to the target node.
     *
     * The relation document removed is the one whose identifier is deterministically derived from the
     * concatenation of `srcId` and `tgtId` with the prefix "rel-".
     *
     * @param srcId The source node's identifier.
     * @param tgtId The target node's identifier.
     */
    override suspend fun deleteRelationBetween(
        srcId: String,
        tgtId: String,
    ) {
        val relId = computeMdHashId(srcId + tgtId, prefix = "rel-")
        collection.deleteOne(Filters.eq("_id", relId))
    }

    /**
     * Removes the entire MongoDB collection backing this storage namespace.
     *
     * This permanently deletes all documents and indexes in the collection.
     */
    override suspend fun drop() {
        collection.drop()
    }

    /**
     * Compute the cosine similarity between two numeric vectors.
     *
     * @param a The first vector.
     * @param b The second vector.
     * @return A value between -1.0 and 1.0 representing the cosine similarity of `a` and `b`, or `0.0` if either array is empty, their sizes differ, or one of the vectors has zero norm.
     */
    private fun cosineSimilarity(
        a: DoubleArray,
        b: DoubleArray,
    ): Double {
        if (a.isEmpty() || b.isEmpty() || a.size != b.size) return 0.0
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        val denom = kotlin.math.sqrt(na) * kotlin.math.sqrt(nb)
        return if (denom == 0.0) 0.0 else dot / denom
    }
}

/**
 * MongoDB-backed graph storage using separate node and edge collections.
 */
class MongoGraphStorage(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
    private val embeddingFunc: EmbeddingFunc?,
) : BaseGraphStorage(namespace, globalConfig),
    AutoCloseable {
    private val client: MongoClient = MongoClient.create(globalConfig["mongo_uri"] as? String ?: "mongodb://localhost:27017")
    private val database = client.getDatabase(globalConfig["mongo_database"] as? String ?: "pathrag")
    private val nodeCollection = database.getCollection<org.bson.Document>("${namespace}_nodes")
    private val edgeCollection = database.getCollection<org.bson.Document>("${namespace}_edges")

    /**
     * Close the MongoDB client.
     */
    override fun close() {
        client.close()
    }

    /**
 * Determines whether a node with the given ID exists in the node collection.
 *
 * @param nodeId The node identifier to check for existence.
 * @return `true` if a node with the specified ID exists, `false` otherwise.
 */
override suspend fun hasNode(nodeId: String): Boolean = nodeCollection.countDocuments(Filters.eq("_id", nodeId)) > 0

    /**
     * Determines whether an edge exists from the given source node to the given target node.
     *
     * @param sourceNodeId ID of the source node.
     * @param targetNodeId ID of the target node.
     * @return `true` if an edge with `src == sourceNodeId` and `tgt == targetNodeId` exists, `false` otherwise.
     */
    override suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean = edgeCollection.countDocuments(Filters.and(Filters.eq("src", sourceNodeId), Filters.eq("tgt", targetNodeId))) > 0

    /**
         * Count edges connected to the specified node.
         *
         * @param nodeId The node identifier to count incident edges for.
         * @return The number of edges where `nodeId` appears as either `src` or `tgt`.
         */
        override suspend fun nodeDegree(nodeId: String): Int =
        edgeCollection.countDocuments(Filters.or(Filters.eq("src", nodeId), Filters.eq("tgt", nodeId))).toInt()

    /**
     * Determine whether an edge exists between two nodes.
     *
     * @param srcId The source node identifier.
     * @param tgtId The target node identifier.
     * @return `1` if an edge from `srcId` to `tgtId` exists, `0` otherwise.
     */
    override suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int = if (hasEdge(srcId, tgtId)) 1 else 0

    /**
             * Retrieves the node document for the given node ID from the node collection.
             *
             * If a document is found, its `_id` field is removed and the remaining document is returned as a map.
             *
             * @param nodeId The identifier of the node to retrieve.
             * @return The node document as a Map of field names to values, or `null` if no node with the given ID exists.
             */
            override suspend fun getNode(nodeId: String): Map<String, Any?>? =
        nodeCollection
            .find(Filters.eq("_id", nodeId))
            .firstOrNull()
            ?.let { doc ->
                doc.remove("_id")
                doc.toMap()
            }

    /**
             * Retrieve the edge document connecting the given source and target node IDs.
             *
             * Searches the edge collection for a document where `src` equals `sourceNodeId` and `tgt` equals `targetNodeId`.
             *
             * @param sourceNodeId The ID of the source node.
             * @param targetNodeId The ID of the target node.
             * @return A map of the edge document's fields with the `_id` field removed, or `null` if no matching edge exists.
             */
            override suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, Any?>? =
        edgeCollection
            .find(Filters.and(Filters.eq("src", sourceNodeId), Filters.eq("tgt", targetNodeId)))
            .firstOrNull()
            ?.let { doc ->
                doc.remove("_id")
                doc.toMap()
            }

    /**
             * Fetches all edges where the specified node appears as the source or target.
             *
             * @return A list of pairs `(sourceId, targetId)` for each edge connected to the node.
             */
            override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>> =
        edgeCollection
            .find(Filters.or(Filters.eq("src", sourceNodeId), Filters.eq("tgt", sourceNodeId)))
            .map { doc -> doc.getString("src") to doc.getString("tgt") }
            .toList()

    /**
     * Insert or replace a node document using the provided identifier and data.
     *
     * Stores `nodeData` as a MongoDB document with its `_id` set to `nodeId`; if a document with
     * the same `_id` exists it will be replaced, otherwise a new document will be inserted.
     *
     * @param nodeId The identifier to use as the document `_id`.
     * @param nodeData A map of fields and values to store for the node (values may be `null`).
     */
    override suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, Any?>,
    ) {
        val doc = org.bson.Document(nodeData).append("_id", nodeId)
        nodeCollection.replaceOne(Filters.eq("_id", nodeId), doc, ReplaceOptions().upsert(true))
    }

    /**
     * Inserts or updates an edge between two nodes using the provided edge data.
     *
     * Creates or replaces the edge document in the edge collection, ensuring the document has
     * a deterministic `_id` derived from the source and target, and includes `src` and `tgt`
     * fields along with the provided `edgeData` as document fields.
     *
     * @param sourceNodeId The source node identifier.
     * @param targetNodeId The target node identifier.
     * @param edgeData Arbitrary metadata for the edge; its entries are stored as fields on the edge document.
     */
    override suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, Any?>,
    ) {
        val doc =
            org.bson
                .Document(edgeData)
                .append("_id", computeMdHashId(sourceNodeId + targetNodeId, prefix = "edge-"))
                .append("src", sourceNodeId)
                .append("tgt", targetNodeId)
        edgeCollection.replaceOne(
            Filters.and(Filters.eq("src", sourceNodeId), Filters.eq("tgt", targetNodeId)),
            doc,
            ReplaceOptions().upsert(true),
        )
    }

    /**
     * Deletes the edge document that connects the given source and target node IDs.
     *
     * If no matching edge exists, the call has no effect.
     *
     * @param sourceNodeId The ID of the source node.
     * @param targetNodeId The ID of the target node.
     */
    override suspend fun deleteEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ) {
        edgeCollection.deleteOne(Filters.and(Filters.eq("src", sourceNodeId), Filters.eq("tgt", targetNodeId)))
    }

    /**
     * Removes the node with the given ID and all edges connected to it from the storage.
     *
     * @param nodeId The ID of the node to delete.
     */
    override suspend fun deleteNode(nodeId: String) {
        nodeCollection.deleteOne(Filters.eq("_id", nodeId))
        edgeCollection.deleteMany(Filters.or(Filters.eq("src", nodeId), Filters.eq("tgt", nodeId)))
    }

    /**
     * Produce embeddings for all graph nodes using the named algorithm.
     *
     * @param algorithm The embedding algorithm name (case-insensitive). Currently accepts "node2vec"; other values are handled via the same metadata-based embedding fallback.
     * @return A pair whose first element is a flattened DoubleArray containing node embeddings concatenated in order, and whose second element is the list of node IDs corresponding to the embeddings' ordering.
     */
    override suspend fun embedNodes(algorithm: String): Pair<DoubleArray, List<String>> {
        val labels = nodes()
        if (labels.isEmpty()) return DoubleArray(0) to emptyList()
        return when (algorithm.lowercase()) {
            "node2vec" -> runMetadataEmbedding(labels)

            // placeholder; no random-walk embedding
            else -> runMetadataEmbedding(labels)
        }
    }

    /**
             * Retrieves all node identifiers stored in the node collection.
             *
             * @return A list of node `_id` values as strings, in iteration order from the collection.
             */
            override suspend fun nodes(): List<String> =
        nodeCollection
            .find()
            .projection(org.bson.Document("_id", 1))
            .map { it.getString("_id") }
            .toList()

    /**
             * Retrieves all edges from the edge collection as (source, target) ID pairs.
             *
             * @return A list of pairs where the first element is the source node ID (`src`) and the second is the target node ID (`tgt`).
             */
            override suspend fun edges(): List<Pair<String, String>> =
        edgeCollection
            .find()
            .map { doc -> doc.getString("src") to doc.getString("tgt") }
            .toList()

    /**
     * Removes the node and edge collections for this namespace from the database.
     *
     * This permanently deletes all stored nodes and edges for the graph namespace.
     */
    override suspend fun drop() {
        nodeCollection.drop()
        edgeCollection.drop()
    }

    /**
     * Produces embeddings for the given node IDs and returns them flattened alongside the input labels.
     *
     * If an embedding function is configured, returns the concatenated embedding vectors computed from
     * textual representations of each node ("<id> <entity_type> <description>"). If no embedding
     * function is available, returns for each node a two-dimensional fallback vector [pagerank, degree]
     * and concatenates those.
     *
     * @param labels The list of node IDs to embed (order is preserved).
     * @return A pair whose first element is a flattened DoubleArray of all node vectors (concatenated in the same order as `labels`),
     *         and whose second element is the `labels` list.
    private suspend fun runMetadataEmbedding(labels: List<String>): Pair<DoubleArray, List<String>> {
        val func = embeddingFunc
        return if (func != null) {
            val texts =
                labels.map { id ->
                    val n = getNode(id) ?: emptyMap()
                    val desc = n["description"]?.toString().orEmpty()
                    "$id ${n["entity_type"] ?: ""} $desc"
                }
            val vectors = func(texts)
            val flat = vectors.flatMap { it.asIterable() }.toDoubleArray()
            flat to labels
        } else {
            val ranks = computePagerankFallback()
            val degs = labels.map { nodeDegree(it).toDouble() }
            val vecs =
                labels.mapIndexed { idx, id ->
                    doubleArrayOf(ranks[id] ?: 0.0, degs[idx])
                }
            val flat = vecs.flatMap { it.asIterable() }.toDoubleArray()
            flat to labels
        }
    }

    /**
     * Computes PageRank scores for the storage's current nodes and edges using a local PageRank implementation.
     *
     * @param damping The damping factor for PageRank (commonly 0.85).
     * @param maxIter Maximum number of iterations to run the PageRank algorithm.
     * @param tol Convergence tolerance; iteration stops when change is below this value.
     * @return A map from node ID to its PageRank score. */
    private suspend fun computePagerankFallback(
        damping: Double = 0.85,
        maxIter: Int = 100,
        tol: Double = 1e-6,
    ): Map<String, Double> {
        val nodeList = nodes()
        val edgeList = edges()
        return computePagerankLocal(nodeList, edgeList, damping, maxIter, tol)
    }
}