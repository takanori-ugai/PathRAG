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
 */
class MongoKVStorage<T : Any>(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
) : BaseKVStorage<T>(namespace, globalConfig),
    AutoCloseable {
    private val mongoUri: String =
        globalConfig["mongo_uri"] as? String
            ?: error("MONGO_URI is required when using MongoDB storage (namespace=$namespace)")
    private val mongoDatabase: String =
        globalConfig["mongo_database"] as? String
            ?: error("MONGO_DATABASE is required when using MongoDB storage (namespace=$namespace)")
    private val client: MongoClient = MongoClient.create(mongoUri)
    private val database = client.getDatabase(mongoDatabase)
    private val collection = database.getCollection<org.bson.Document>("${namespace}_kv")

    /**
     * Close the MongoDB client.
     */
    override fun close() {
        client.close()
    }

    override suspend fun allKeys(): List<String> =
        collection
            .find()
            .projection(org.bson.Document("_id", 1))
            .map { it.getString("_id") }
            .toList()

    override suspend fun getById(id: String): T? =
        collection
            .find(Filters.eq("_id", id))
            .firstOrNull()
            ?.let { doc ->
                doc.remove("_id")
                @Suppress("UNCHECKED_CAST")
                doc.toMap() as T
            }

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

    override suspend fun upsert(data: Map<String, T>) {
        if (data.isEmpty()) return
        val opts = ReplaceOptions().upsert(true)
        val writes =
            data.map { (id, value) ->
                val doc =
                    when (value) {
                        is Map<*, *> -> org.bson.Document(value.filterKeys { it != "_id" } as Map<String, Any?>)

                        else -> throw IllegalArgumentException(
                            "MongoKVStorage only supports storing Map values. Got: ${value::class.simpleName}",
                        )
                    }.append("_id", id)
                ReplaceOneModel(Filters.eq("_id", id), doc, opts)
            }
        collection.bulkWrite(writes)
    }

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
            ?: error("MONGO_URI is required when using MongoDB storage (namespace=$namespace)")
    private val mongoDatabase: String =
        globalConfig["mongo_database"] as? String
            ?: error("MONGO_DATABASE is required when using MongoDB storage (namespace=$namespace)")
    private val client: MongoClient = MongoClient.create(mongoUri)
    private val database = client.getDatabase(mongoDatabase)
    private val collection = database.getCollection<org.bson.Document>("${namespace}_vector")

    /**
     * Close the MongoDB client.
     */
    override fun close() {
        client.close()
    }

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

    override suspend fun deleteEntity(entityName: String) {
        val entityId = computeMdHashId(entityName, prefix = "ent-")
        collection.deleteOne(Filters.eq("_id", entityId))
    }

    override suspend fun deleteRelation(entityName: String) {
        collection.deleteMany(Filters.or(Filters.eq("src_id", entityName), Filters.eq("tgt_id", entityName)))
    }

    override suspend fun deleteRelationBetween(
        srcId: String,
        tgtId: String,
    ) {
        val relId = computeMdHashId(srcId + tgtId, prefix = "rel-")
        collection.deleteOne(Filters.eq("_id", relId))
    }

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
            ?: error("MONGO_URI is required when using MongoDB storage (namespace=$namespace)")
    private val mongoDatabase: String =
        globalConfig["mongo_database"] as? String
            ?: error("MONGO_DATABASE is required when using MongoDB storage (namespace=$namespace)")
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

    override suspend fun hasNode(nodeId: String): Boolean = nodeCollection.countDocuments(Filters.eq("_id", nodeId)) > 0

    override suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean = edgeCollection.countDocuments(Filters.and(Filters.eq("src", sourceNodeId), Filters.eq("tgt", targetNodeId))) > 0

    override suspend fun nodeDegree(nodeId: String): Int =
        edgeCollection.countDocuments(Filters.or(Filters.eq("src", nodeId), Filters.eq("tgt", nodeId))).toInt()

    override suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int = if (hasEdge(srcId, tgtId)) 1 else 0

    override suspend fun getNode(nodeId: String): Map<String, Any?>? =
        nodeCollection
            .find(Filters.eq("_id", nodeId))
            .firstOrNull()
            ?.let { doc ->
                doc.remove("_id")
                doc.toMap()
            }

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

    override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>> =
        edgeCollection
            .find(Filters.or(Filters.eq("src", sourceNodeId), Filters.eq("tgt", sourceNodeId)))
            .map { doc -> doc.getString("src") to doc.getString("tgt") }
            .toList()

    override suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, Any?>,
    ) {
        val doc = org.bson.Document(nodeData).append("_id", nodeId)
        nodeCollection.replaceOne(Filters.eq("_id", nodeId), doc, ReplaceOptions().upsert(true))
    }

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

    override suspend fun deleteEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ) {
        edgeCollection.deleteOne(Filters.and(Filters.eq("src", sourceNodeId), Filters.eq("tgt", targetNodeId)))
    }

    override suspend fun deleteNode(nodeId: String) {
        nodeCollection.deleteOne(Filters.eq("_id", nodeId))
        edgeCollection.deleteMany(Filters.or(Filters.eq("src", nodeId), Filters.eq("tgt", nodeId)))
    }

    override suspend fun embedNodes(algorithm: String): Pair<DoubleArray, List<String>> {
        val labels = nodes()
        if (labels.isEmpty()) return DoubleArray(0) to emptyList()
        return when (algorithm.lowercase()) {
            "node2vec" -> runMetadataEmbedding(labels)

            // placeholder; no random-walk embedding
            else -> runMetadataEmbedding(labels)
        }
    }

    override suspend fun nodes(): List<String> =
        nodeCollection
            .find()
            .projection(org.bson.Document("_id", 1))
            .map { it.getString("_id") }
            .toList()

    override suspend fun edges(): List<Pair<String, String>> =
        edgeCollection
            .find()
            .map { doc -> doc.getString("src") to doc.getString("tgt") }
            .toList()

    override suspend fun drop() {
        nodeCollection.drop()
        edgeCollection.drop()
    }

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
