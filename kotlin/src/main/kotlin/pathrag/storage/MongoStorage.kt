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
import pathrag.utils.cosineSimilarity
import pathrag.utils.log

/**
 * MongoDB-backed key-value storage that persists documents per namespace.
 *
 * Note: This implementation currently expects values to be `Map<String, Any?>`
 * and will throw if a different type is provided during `upsert`. It is
 * tailored for the existing PathRAG use cases rather than arbitrary payloads.
 */
class MongoKVStorage<T : Any>(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
) : BaseKVStorage<T>(namespace, globalConfig),
    AutoCloseable {
    private val mongoUri: String =
        globalConfig["mongo_uri"] as? String
            ?: error("mongo_uri is required when using MongoDB storage (namespace=$namespace)")
    private val mongoDatabase: String =
        globalConfig["mongo_database"] as? String
            ?: error("mongo_database is required when using MongoDB storage (namespace=$namespace)")
    private val client: MongoClient = MongoClient.create(mongoUri)
    private val database = client.getDatabase(mongoDatabase)
    private val collection = database.getCollection<org.bson.Document>("${namespace}_kv")

    /**
     * Close the MongoDB client.
     */
    override fun close() {
        client.close()
    }

    /**
     * Retrieve all document IDs (keys) from the storage collection.
     *
     * @return A list of document `_id` values as strings representing the stored keys.
     */
    override suspend fun allKeys(): List<String> =
        collection
            .find()
            .projection(org.bson.Document("_id", 1))
            .map { it.getString("_id") }
            .toList()

    /**
     * Retrieve the value stored under the given id from the collection.
     *
     * The returned value will not include the internal `_id` field stored in the database.
     *
     * @param id The document id/key to look up.
     * @return The document cast to `T` if a matching document exists, `null` otherwise.
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
     * Retrieves documents for the given list of IDs and returns results aligned to the input order.
     *
     * @param ids The list of document IDs to fetch.
     * @param fields Optional set of field names to project; if provided only these fields (plus `_id`) are returned.
     * @return A list whose elements correspond to the input `ids`: the mapped value cast to `T` when a document exists,
     *         or `null` when it does not.
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
     * Determine which keys from the provided list are missing in the storage.
     *
     * @param data List of keys to check for existence.
     * @return `Set<String>` containing the keys from `data` that do not exist in the collection.
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
     * Inserts or updates multiple key-value entries into the storage collection.
     *
     * Accepts a map of id -> value and upserts each entry as a MongoDB document with `_id` set to the id.
     * If `data` is empty the method returns immediately. Each `value` must be a Map with String keys;
     * any entry named `_id` in the value is ignored and the provided map key is used as the document `_id`.
     *
     * @param data Mapping of document id to value to upsert.
     * @throws IllegalArgumentException If a value is not a Map or if a map key is not a String.
     */
    override suspend fun upsert(data: Map<String, T>) {
        if (data.isEmpty()) return
        val opts = ReplaceOptions().upsert(true)
        val writes =
            data.map { (id, value) ->
                val doc =
                    when (value) {
                        is Map<*, *> -> {
                            val m =
                                value.entries
                                    .filter { (k, _) -> k != "_id" }
                                    .associate { (k, v) ->
                                        require(k is String) {
                                            "MongoKVStorage only supports String keys. Got key type=${k?.let { it::class.qualifiedName }}"
                                        }
                                        k to v
                                    }
                            org.bson.Document(m)
                        }

                        else -> {
                            throw IllegalArgumentException(
                                "MongoKVStorage only supports storing Map values. Got: ${value::class.simpleName}",
                            )
                        }
                    }.append("_id", id)
                ReplaceOneModel(Filters.eq("_id", id), doc, opts)
            }
        collection.bulkWrite(writes)
    }

    /**
     * Drops the MongoDB collection backing this storage namespace.
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
    private val mongoUri: String =
        globalConfig["mongo_uri"] as? String
            ?: error("mongo_uri is required when using MongoDB storage (namespace=$namespace)")
    private val mongoDatabase: String =
        globalConfig["mongo_database"] as? String
            ?: error("mongo_database is required when using MongoDB storage (namespace=$namespace)")
    private val client: MongoClient = MongoClient.create(mongoUri)
    private val database = client.getDatabase(mongoDatabase)
    private val collection = database.getCollection<org.bson.Document>("${namespace}_vector")

    /**
     * Close the MongoDB client.
     */
    override fun close() {
        client.close()
    }

    /**
     * Searches the vector collection for documents most similar to the provided query text.
     *
     * If the query is blank, if embedding generation fails, or if no embedding is produced, an empty list is returned.
     *
     * Results are ordered by cosine similarity (highest first) and limited to `topK`.
     *
     * @param query The text query to embed and use for similarity search.
     * @param topK Maximum number of results to return.
     * @return A list of maps for the top matching documents; each map contains the keys `content`, `score`, and any stored metadata fields.
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
     * Persists vector documents for entries that contain non-blank `content`.
     *
     * For each input entry with a non-blank `content` field, generates an embedding via the configured
     * embedding function and upserts a document containing `_id`, `content`, `embedding` (as a list),
     * and any configured metadata fields. If the input is empty, no entries contain `content`, or the
     * number of returned embeddings does not match the number of valid entries, no writes are performed.
     *
     * @param data Map of document id to document fields; each value should include a `content` field
     *             whose text will be embedded and persisted.
     */
    override suspend fun upsert(data: Map<String, Map<String, Any?>>) {
        if (data.isEmpty()) return
        val items = data.entries.toList()
        val contents = items.map { it.value["content"]?.toString().orEmpty() }
        val validPairs = items.zip(contents).filter { it.second.isNotBlank() }
        if (validPairs.isEmpty()) return
        val embeddings = embeddingFunc(validPairs.map { it.second })
        if (embeddings.size != validPairs.size) {
            log().warn {
                "MongoVectorStorage upsert skipped: embeddings (${embeddings.size}) != items (${validPairs.size}) for namespace=$namespace"
            }
            return
        }
        val opts = ReplaceOptions().upsert(true)
        val writes =
            validPairs.mapIndexedNotNull { idx, pair ->
                val (entry, content) = pair
                val vector = embeddings[idx]
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
     * Removes all vector documents for the given entity from the collection.
     *
     * Deletes documents whose `_id` matches the MD-hashed id of `entityName` (prefixed with `ent-`) and also deletes
     * documents with `entity_name` equal to `entityName`.
     *
     * @param entityName The entity's name used to compute the hashed id and to match the `entity_name` metadata field.
     */
    override suspend fun deleteEntity(entityName: String) {
        val entityId = computeMdHashId(entityName, prefix = "ent-")
        // Delete by both the hashed id (legacy) and the stored entity_name metadata.
        collection.deleteOne(Filters.eq("_id", entityId))
        collection.deleteMany(Filters.eq("entity_name", entityName))
    }

    /**
     * Deletes all relations involving the given entity.
     *
     * @param entityName The entity identifier used to match against the `src_id` and `tgt_id` fields; all matching
     *                   relation documents will be removed.
     */
    override suspend fun deleteRelation(entityName: String) {
        collection.deleteMany(
            Filters.or(Filters.eq("src_id", entityName), Filters.eq("tgt_id", entityName)),
        )
    }

    /**
     * Delete the relation between two entities identified by their IDs.
     *
     * Deletes a relation document using the legacy hashed relation ID and also removes any documents
     * whose `src_id` and `tgt_id` fields match the provided IDs.
     *
     * @param srcId ID of the source entity.
     * @param tgtId ID of the target entity.
     */
    override suspend fun deleteRelationBetween(
        srcId: String,
        tgtId: String,
    ) {
        val relId = computeMdHashId(srcId + tgtId, prefix = "rel-")
        // Delete by both the hashed id (legacy) and the src/tgt metadata.
        collection.deleteOne(Filters.eq("_id", relId))
        collection.deleteMany(Filters.and(Filters.eq("src_id", srcId), Filters.eq("tgt_id", tgtId)))
    }

    /**
     * Drops the MongoDB collection backing this storage namespace.
     */
    override suspend fun drop() {
        collection.drop()
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
    private val mongoUri: String =
        globalConfig["mongo_uri"] as? String
            ?: error("mongo_uri is required when using MongoDB storage (namespace=$namespace)")
    private val mongoDatabase: String =
        globalConfig["mongo_database"] as? String
            ?: error("mongo_database is required when using MongoDB storage (namespace=$namespace)")
    private val client: MongoClient = MongoClient.create(mongoUri)
    private val database = client.getDatabase(mongoDatabase)
    private val nodeCollection = database.getCollection<org.bson.Document>("${namespace}_nodes")
    private val edgeCollection = database.getCollection<org.bson.Document>("${namespace}_edges")

    /**
     * Close the MongoDB client.
     */
    override fun close() {
        client.close()
    }

    /**
     * Checks whether a node with the given ID exists in the node collection.
     *
     * @param nodeId The node identifier to check for existence.
     * @return `true` if a node with the given ID exists, `false` otherwise.
     */
    override suspend fun hasNode(nodeId: String): Boolean = nodeCollection.countDocuments(Filters.eq("_id", nodeId)) > 0

    /**
     * Determines whether an edge exists from the given source node to the given target node.
     *
     * @param sourceNodeId The ID of the source node.
     * @param targetNodeId The ID of the target node.
     * @return `true` if an edge with `src` equal to `sourceNodeId` and `tgt` equal to `targetNodeId` exists, `false` otherwise.
     */
    override suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean = edgeCollection.countDocuments(Filters.and(Filters.eq("src", sourceNodeId), Filters.eq("tgt", targetNodeId))) > 0

    /**
     * Compute the number of edges connected to a node.
     *
     * @param nodeId Identifier of the node whose incident edges are counted.
     * @return The number of documents where `src` or `tgt` equals the given `nodeId`.
     */
    override suspend fun nodeDegree(nodeId: String): Int =
        edgeCollection.countDocuments(Filters.or(Filters.eq("src", nodeId), Filters.eq("tgt", nodeId))).toInt()

    /**
     * Computes an integer indicator of whether an edge exists between two nodes.
     *
     * @return `1` if an edge exists from `srcId` to `tgtId`, `0` otherwise.
     */
    override suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int = if (hasEdge(srcId, tgtId)) 1 else 0

    /**
     * Retrieve a node document by its identifier.
     *
     * @param nodeId The node's identifier.
     * @return The node document as a map with the `_id` field removed, or `null` if no node exists for the given id.
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
     * Retrieve the edge document connecting two nodes.
     *
     * @param sourceNodeId The ID of the source node.
     * @param targetNodeId The ID of the target node.
     * @return The edge document as a map with the `_id` field removed, or `null` if no edge exists.
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
     * Retrieves all edges connected to the given node ID.
     *
     * @param sourceNodeId The node ID whose incident edges to fetch.
     * @return A list of pairs `(src, tgt)` for each edge where `src` or `tgt` equals `sourceNodeId`.
     */
    override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>> =
        edgeCollection
            .find(Filters.or(Filters.eq("src", sourceNodeId), Filters.eq("tgt", sourceNodeId)))
            .map { doc -> doc.getString("src") to doc.getString("tgt") }
            .toList()

    /**
     * Upserts a node document into the node collection for the given node id.
     *
     * The node's `_id` field is set to `nodeId`; any existing document with the same `_id` is replaced.
     *
     * @param nodeId The identifier to use as the document's `_id`.
     * @param nodeData Map of fields to store for the node; an existing `_id` entry, if present, will be overwritten.
     */
    override suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, Any?>,
    ) {
        val doc = org.bson.Document(nodeData).append("_id", nodeId)
        nodeCollection.replaceOne(Filters.eq("_id", nodeId), doc, ReplaceOptions().upsert(true))
    }

    /**
     * Insert or replace an edge document linking two nodes using the provided edge data.
     *
     * The stored document will include a computed `_id` (MD hash of `sourceNodeId + targetNodeId` with prefix `"edge-"`),
     * and will always contain `src` and `tgt` fields set to the provided node IDs.
     *
     * @param sourceNodeId The source node identifier for the edge.
     * @param targetNodeId The target node identifier for the edge.
     * @param edgeData Additional fields to persist on the edge document; these fields are stored alongside `src`, `tgt`,
     *                 and the computed `_id`.
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
     * Deletes the edge document that connects the given source node to the given target node.
     *
     * Removes a single document where the `src` field equals `sourceNodeId` and the `tgt` field equals `targetNodeId`,
     * if present.
     *
     * @param sourceNodeId The ID of the source node (matches the `src` field).
     * @param targetNodeId The ID of the target node (matches the `tgt` field).
     */
    override suspend fun deleteEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ) {
        edgeCollection.deleteOne(
            Filters.and(Filters.eq("src", sourceNodeId), Filters.eq("tgt", targetNodeId)),
        )
    }

    /**
     * Deletes the node identified by [nodeId] and removes all edges connected to it.
     *
     * @param nodeId The node identifier; deletes the node document with `_id` equal to this value and all edges where
     *               `src` or `tgt` equals this value.
     */
    override suspend fun deleteNode(nodeId: String) {
        nodeCollection.deleteOne(Filters.eq("_id", nodeId))
        edgeCollection.deleteMany(
            Filters.or(Filters.eq("src", nodeId), Filters.eq("tgt", nodeId)),
        )
    }

    /**
     * Generate embeddings for all nodes using the given algorithm.
     *
     * Currently supports "node2vec" (case-insensitive); other values fall back to the same metadata-based embedding path.
     *
     * @param algorithm The embedding algorithm name to use.
     * @return A pair where the first element is a flattened `DoubleArray` containing embedding vectors and the second
     *         element is the list of node IDs (labels) corresponding to those embeddings.
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
     * Retrieve all node IDs stored in the node collection.
     *
     * @return A list of node IDs as strings (order not guaranteed).
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
     * @return A list of pairs where the first element is the source node ID and the second element is the target node ID.
     */
    override suspend fun edges(): List<Pair<String, String>> =
        edgeCollection
            .find()
            .map { doc -> doc.getString("src") to doc.getString("tgt") }
            .toList()

    /**
     * Removes all graph data by dropping the node and edge collections.
     *
     * This permanently deletes every node and edge stored in the graph namespace.
     */
    override suspend fun drop() {
        nodeCollection.drop()
        edgeCollection.drop()
    }

    /**
     * Compute embeddings for the given node labels using stored metadata or a local fallback.
     *
     * If an `EmbeddingFunc` is available, generates a text string for each label in the form
     * "`<id> <entity_type> <description>`" and obtains an embedding vector for each text.
     * If no embedding function is provided, produces a 2-dimensional feature vector per label
     * containing `[pagerank, degree]` computed locally.
     *
     * @param labels The ordered list of node IDs to embed.
     * @return A pair where the first element is a flattened `DoubleArray` containing the concatenated
     *         embedding vectors in the same order as `labels` (length = labels.size * embeddingDim),
     *         and the second element is the input `labels` list aligned to those embeddings.
     */
    private suspend fun runMetadataEmbedding(labels: List<String>): Pair<DoubleArray, List<String>> {
        val func = embeddingFunc
        return if (func != null) {
            val nodesById =
                nodeCollection
                    .find(Filters.`in`("_id", labels))
                    .projection(org.bson.Document(mapOf("_id" to 1, "description" to 1, "entity_type" to 1)))
                    .map { doc -> doc.getString("_id") to doc }
                    .toList()
                    .toMap()
            val texts =
                labels.map { id ->
                    val doc = nodesById[id]
                    val desc = doc?.get("description")?.toString().orEmpty()
                    val entityType = doc?.get("entity_type")?.toString().orEmpty()
                    "$id $entityType $desc"
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
     * Computes PageRank scores for all nodes using the provided damping and convergence parameters.
     *
     * @param damping The damping factor (probability of following an outgoing edge) between 0.0 and 1.0.
     * @param maxIter Maximum number of iterations to perform.
     * @param tol Convergence tolerance; iteration stops when score changes are below this value.
     * @return A map from node id to its PageRank score.
     */
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
