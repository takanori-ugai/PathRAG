package pathrag.storage

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.neo4j.driver.AuthTokens
import org.neo4j.driver.Driver
import org.neo4j.driver.GraphDatabase
import org.neo4j.driver.Record
import org.neo4j.driver.TransactionContext
import org.neo4j.driver.Values
import pathrag.base.BaseVectorStorage
import pathrag.utils.EmbeddingFunc
import pathrag.utils.computeMdHashId

/**
 * Neo4j-backed vector storage using native vector indexes when available.
 */
class Neo4jVectorStorage(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
    private val embeddingFunc: EmbeddingFunc,
    private val metaFields: Set<String> = setOf("entity_name", "full_doc_id", "source_id", "src_id", "tgt_id"),
) : BaseVectorStorage(namespace, globalConfig),
    AutoCloseable {
    private val logger = KotlinLogging.logger("PathRAG-Neo4jVectorStorage")

    private val uri: String =
        (globalConfig["neo4j_uri"] as? String)
            ?: System.getenv("NEO4J_URI")
            ?: "bolt://localhost:7687"
    private val user: String =
        (globalConfig["neo4j_user"] as? String)
            ?: System.getenv("NEO4J_USER")
            ?: "neo4j"
    private val password: String =
        (globalConfig["neo4j_password"] as? String)
            ?: System.getenv("NEO4J_PASSWORD")
            ?: run {
                logger.warn { "Using default Neo4j password; set NEO4J_PASSWORD or neo4j_password in config for production." }
                "password"
            }
    private val driver: Driver = GraphDatabase.driver(uri, AuthTokens.basic(user, password))

    private val nodeLabel = "${namespace.uppercase()}_VECTOR"
    private val vectorIndexName = "${nodeLabel}_EMBED_IDX"

    /**
     * Release Neo4j driver and associated resources.
     */
    override fun close() {
        driver.close()
    }

    private suspend fun <T> read(block: (TransactionContext) -> T): T =
        withContext(Dispatchers.IO) { driver.session().use { session -> session.executeRead { tx -> block(tx) } } }

    /**
     * Execute a write transaction against the Neo4j driver and return the block's result.
     *
     * The provided `block` is invoked with a `TransactionContext` inside a write transaction and its
     * return value is propagated to the caller.
     *
     * @param block Lambda that performs operations within the write transaction using the `TransactionContext`.
     * @return The value returned by `block`.
     */
    private suspend fun <T> write(block: (TransactionContext) -> T): T =
        withContext(Dispatchers.IO) { driver.session().use { session -> session.executeWrite { tx -> block(tx) } } }

    /**
     * Find the most relevant stored vectors for the given text, preferring a Neo4j vector index and falling back to
     * client-side cosine scoring.
     *
     * @param query The text query to embed and search.
     * @param topK Maximum number of results to return.
     * @return A list of result maps, each containing `content`, `score`, and any persisted metadata fields; results are
     *         ordered by descending `score` and limited to `topK`.
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
                logger.error(e) { "Failed to embed query for Neo4jVectorStorage ($namespace)" }
                throw e
            }
        if (embeddings.isEmpty()) return emptyList()
        val queryEmbedding = embeddings.first()
        queryWithIndex(queryEmbedding, topK)?.let { return it }

        val maxFallbackVectors = (globalConfig["max_fallback_vectors"] as? Int) ?: 10000
        val vectors =
            read { tx ->
                tx
                    .run(
                        "MATCH (v:$nodeLabel) RETURN v.id AS id, v.content AS content, v.embedding AS embedding, v AS props LIMIT \$limit",
                        Values.parameters("limit", maxFallbackVectors),
                    ).list { rec -> rec.toVectorEntry() }
            }
        if (vectors.size >= maxFallbackVectors) {
            logger.warn { "Fallback vector query reached limit $maxFallbackVectors for namespace $namespace; results may be truncated." }
        }
        return vectors
            .mapNotNull { entry ->
                val emb = entry["embedding"] as? DoubleArray ?: return@mapNotNull null
                val content = entry["content"] as? String ?: ""
                val meta =
                    (entry["meta"] as? Map<*, *>)
                        ?.mapNotNull { (key, value) ->
                            (key as? String)?.let { it to value }
                        }?.toMap()
                        ?: emptyMap()
                val score = cosineSimilarity(queryEmbedding, emb)
                mapOf("content" to content, "score" to score) + meta
            }.sortedByDescending { (it["score"] as? Double) ?: 0.0 }
            .take(topK)
    }

    /**
     * Insert or update vector nodes and their persisted metadata into the Neo4j store for this namespace.
     *
     * Processes the provided entries by embedding each non-blank `content`, ensuring (if possible) a vector index
     * for the embedding dimension, and merging nodes by id with properties `content`, `embedding`, and the subset
     * of metadata keys configured in `metaFields`.
     *
     * @param data Map from node id to a property map; each property map is expected to contain a `"content"` entry
     *             (the text to embed) and may contain additional metadata fields which will be persisted only if
     *             their keys are present in `metaFields`.
     * @throws IllegalStateException if embedding the provided contents fails.
     */
    @Suppress("TooGenericExceptionCaught")
    override suspend fun upsert(data: Map<String, Map<String, Any?>>) {
        if (data.isEmpty()) return
        val items = data.entries.toList()
        val contents = items.map { it.value["content"]?.toString().orEmpty() }
        val valid = items.zip(contents).filter { it.second.isNotBlank() }
        if (valid.isEmpty()) return
        val embeddings =
            try {
                embeddingFunc(valid.map { it.second })
            } catch (e: IllegalStateException) {
                logger.error(e) { "Failed to embed content for Neo4jVectorStorage ($namespace)" }
                throw e
            }
        runCatching { ensureVectorIndex(embeddings.firstOrNull()?.size ?: 0) }
            .onFailure { ex ->
                logger.warn(ex) { "Unable to ensure vector index for Neo4jVectorStorage ($namespace); continuing without index." }
            }
        write { tx ->
            embeddings.forEachIndexed { idx, vector ->
                val (entry, content) = valid[idx]
                val meta = entry.value.filterKeys { metaFields.contains(it) }
                tx.run(
                    "MERGE (v:$nodeLabel {id:\$id}) " +
                        "SET v.content = \$content, v.embedding = \$embedding " +
                        "SET v += \$meta",
                    Values.parameters("id", entry.key, "content", content, "embedding", vector.toList(), "meta", meta),
                )
            }
        }
    }

    /**
     * Delete all vector nodes that belong to the given entity.
     *
     * Deletes nodes whose stored `id` equals the hash derived from `entityName` with the `ent-` prefix.
     *
     * @param entityName The entity's canonical name used to compute the hashed entity id to delete.
     */
    override suspend fun deleteEntity(entityName: String) {
        val entityId = computeMdHashId(entityName, prefix = "ent-")
        write { tx -> tx.run("MATCH (v:$nodeLabel {id:\$id}) DETACH DELETE v", Values.parameters("id", entityId)) }
    }

    /**
     * Removes all relationship vector nodes that reference the given entity by matching `src_id` or `tgt_id`.
     *
     * @param entityName The entity identifier to match against nodes' `src_id` or `tgt_id`.
     */
    override suspend fun deleteRelation(entityName: String) {
        write { tx ->
            tx.run(
                "MATCH (v:$nodeLabel) WHERE v.src_id = \$ent OR v.tgt_id = \$ent DETACH DELETE v",
                Values.parameters("ent", entityName),
            )
        }
    }

    /**
     * Deletes the vector node representing the relationship between two entities.
     *
     * Removes any vector whose id equals the computed relation id for the pair or whose `src_id` and `tgt_id` properties
     * match the provided ids.
     *
     * @param srcId The source entity id.
     * @param tgtId The target entity id.
     */
    override suspend fun deleteRelationBetween(
        srcId: String,
        tgtId: String,
    ) {
        val relId = computeMdHashId(srcId + tgtId, prefix = "rel-")
        write { tx ->
            tx.run(
                "MATCH (v:$nodeLabel) WHERE v.id = \$id OR (v.src_id = \$src AND v.tgt_id = \$tgt) DETACH DELETE v",
                Values.parameters("id", relId, "src", srcId, "tgt", tgtId),
            )
        }
    }

    /**
     * Deletes all vector nodes and their relationships for this storage namespace from Neo4j.
     */
    override suspend fun drop() {
        write { tx -> tx.run("MATCH (v:$nodeLabel) DETACH DELETE v") }
    }

    private suspend fun vectorIndexState(): String? =
        runCatching {
            read { tx ->
                tx
                    .run("SHOW VECTOR INDEXES WHERE name = \$name RETURN state", Values.parameters("name", vectorIndexName))
                    .list()
                    .firstOrNull()
                    ?.get("state")
                    ?.asString()
            }
        }.getOrNull()

    private suspend fun ensureVectorIndex(dimension: Int) {
        if (dimension <= 0) return
        val currentState = vectorIndexState()
        if (currentState?.equals("ONLINE", ignoreCase = true) == true) return
        if (currentState == null) {
            write { tx ->
                tx.run(
                    "CREATE VECTOR INDEX $vectorIndexName IF NOT EXISTS FOR (v:$nodeLabel) ON (v.embedding) " +
                        "OPTIONS {indexConfig: {`vector.dimensions`: \$dim, `vector.similarity_function`: 'cosine'}}",
                    Values.parameters("dim", dimension),
                )
            }
            logger.info { "Created Neo4j vector index $vectorIndexName for label $nodeLabel with dimension $dimension" }
        }
        repeat(10) {
            val state = vectorIndexState()
            if (state?.equals("ONLINE", ignoreCase = true) == true) return
            delay(200)
        }
        val finalState = vectorIndexState()
        logger.warn { "Neo4j vector index $vectorIndexName not ONLINE after wait; last observed state=$finalState" }
    }

    private suspend fun queryWithIndex(
        queryEmbedding: DoubleArray,
        topK: Int,
    ): List<Map<String, Any?>>? =
        runCatching {
            ensureVectorIndex(queryEmbedding.size)
            read { tx ->
                tx
                    .run(
                        "CALL db.index.vector.queryNodes(\$indexName, \$k, \$embedding) " +
                            "YIELD node, score " +
                            "RETURN node, score",
                        Values.parameters("indexName", vectorIndexName, "k", topK, "embedding", queryEmbedding.toList()),
                    ).list { rec -> rec.toIndexedEntry() }
            }
        }.onFailure { ex ->
            logger.warn(
                ex,
            ) { "Vector index query unavailable for Neo4jVectorStorage ($namespace); falling back to client-side similarity." }
        }.getOrNull()

    private fun Record.toVectorEntry(): Map<String, Any?> {
        val props = this["props"].asNode().asMap { it.asObject() }
        val embeddingList = props["embedding"] as? List<*> ?: emptyList<Any?>()
        val embedding = embeddingList.filterIsInstance<Number>().map { it.toDouble() }.toDoubleArray()
        val meta = props.filterKeys { metaFields.contains(it) }
        return mapOf(
            "id" to (this["id"].takeIf { !it.isNull }?.asString() ?: ""),
            "content" to props["content"]?.toString().orEmpty(),
            "embedding" to embedding,
            "meta" to meta,
        )
    }

    private fun Record.toIndexedEntry(): Map<String, Any?> {
        val node = this["node"].asNode()
        val props = node.asMap { it.asObject() }
        val meta = props.filterKeys { metaFields.contains(it) }
        val content = props["content"]?.toString().orEmpty()
        val score = this["score"].asDouble()
        return mapOf("content" to content, "score" to score) + meta
    }

    private fun cosineSimilarity(
        a: DoubleArray,
        b: DoubleArray,
    ): Double {
        if (a.size != b.size) {
            logger.warn { "Cannot compute cosine similarity for vectors of different dimensions: ${a.size} vs ${b.size}" }
            return 0.0
        }
        if (a.isEmpty()) return 0.0
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        return if (normA == 0.0 || normB == 0.0) 0.0 else dot / (kotlin.math.sqrt(normA) * kotlin.math.sqrt(normB))
    }
}
