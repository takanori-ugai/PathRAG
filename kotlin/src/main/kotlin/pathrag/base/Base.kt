package pathrag.base

import kotlinx.coroutines.runBlocking

/**
 * Parameters controlling how PathRAG builds and returns context for a query.
 *
 * @property mode retrieval mode (local/global/hybrid).
 * @property onlyNeedContext return just the constructed context.
 * @property onlyNeedPrompt return only the generated prompt.
 * @property responseType target length/format hint for the LLM.
 * @property stream whether to stream responses when supported.
 * @property topK number of results to retrieve from vector stores.
 * @property maxTokenForTextUnit token cap for text units in context.
 * @property maxTokenForGlobalContext token cap for global context.
 * @property maxTokenForLocalContext token cap for local context.
 */
data class QueryParam(
    val mode: String = "hybrid",
    val onlyNeedContext: Boolean = false,
    val onlyNeedPrompt: Boolean = false,
    val responseType: String = "Multiple Paragraphs",
    val stream: Boolean = false,
    val topK: Int = 40,
    val maxTokenForTextUnit: Int = 4000,
    val maxTokenForGlobalContext: Int = 3000,
    val maxTokenForLocalContext: Int = 5000,
)

/**
 * Optional addon parameters for LLM prompts/extraction.
 */
data class AddonParams(
    val entityTypes: List<String> = emptyList(),
    val language: String? = null,
    val exampleNumber: Int? = null,
) {
    /**
     * Convert addon parameters to a configuration map suitable for downstream APIs.
     *
     * The resulting map contains the keys `entity_types`, `language`, and `example_number`
     * mapped to the corresponding fields of this instance.
     *
     * @return A map with keys `entity_types`, `language`, and `example_number` and their values.
     */
    fun asConfig(): Map<String, Any?> =
        mapOf(
            "entity_types" to entityTypes,
            "language" to language,
            "example_number" to exampleNumber,
        )
}

/**
 * Optional backend configuration such as Neo4j or Mongo connection details.
 */
data class ExtraConfig(
    val neo4jUri: String? = null,
    val neo4jUser: String? = null,
    val neo4jPassword: String? = null,
    val mongoUri: String? = null,
    val mongoDatabase: String? = null,
    val additional: Map<String, Any?> = emptyMap(),
) {
    /**
     * Builds a configuration map from this ExtraConfig, excluding any null fields and merging additional entries.
     *
     * The resulting map uses snake_case keys for Neo4j and Mongo fields and does not include keys whose values are null.
     *
     * @return A map of configuration entries containing non-null `neo4j_uri`, `neo4j_user`, `neo4j_password`,
     *         `mongo_uri`, and `mongo_database` values, merged with `additional`.
     */
    fun toMap(): Map<String, Any?> =
        mapOf(
            "neo4j_uri" to neo4jUri,
            "neo4j_user" to neo4jUser,
            "neo4j_password" to neo4jPassword,
            "mongo_uri" to mongoUri,
            "mongo_database" to mongoDatabase,
        ).filterValues { it != null } + additional
}

/**
 * Typed container for configuration passed to storage/LLM layers.
 */
data class GlobalConfig(
    val workingDir: String,
    val embeddingFunc: Any?,
    val llmModelFunc: Any?,
    val chunkTokenSize: Int,
    val chunkOverlapTokenSize: Int,
    val language: String,
    val keywordsExamples: String,
    val embeddingCacheConfig: Map<String, Any?>,
    val addonParams: AddonParams,
    val llmModelName: String,
    val similarityCheckPrompt: String,
    val fixedHighLevelKeywords: List<String>,
    val fixedLowLevelKeywords: List<String>,
) {
    /**
     * Convert the GlobalConfig into a plain Map representation for consumption by storage/LLM layers.
     *
     * The returned map contains the configuration fields keyed as expected by downstream components
     * (for example: "working_dir", "embedding_func", "llm_model_name", etc.).
     *
     * @param extra Additional entries to merge into the resulting map; entries in `extra` override existing keys on collision.
     * @return A map of configuration keys to their values, built from this GlobalConfig
     *         (including `addon_params` via `addonParams.asConfig()`), merged with `extra`.
     */
    fun toMap(extra: Map<String, Any?> = emptyMap()): Map<String, Any?> =
        mapOf(
            "working_dir" to workingDir,
            "embedding_func" to embeddingFunc,
            "llm_model_func" to llmModelFunc,
            "chunk_token_size" to chunkTokenSize,
            "chunk_overlap_token_size" to chunkOverlapTokenSize,
            "language" to language,
            "keywords_examples" to keywordsExamples,
            "embedding_cache_config" to embeddingCacheConfig,
            "addon_params" to addonParams.asConfig(),
            "llm_model_name" to llmModelName,
            "similarity_check_prompt" to similarityCheckPrompt,
            "fixed_high_level_keywords" to fixedHighLevelKeywords,
            "fixed_low_level_keywords" to fixedLowLevelKeywords,
        ).plus(extra)
}

/**
 * Shared namespace/config contract for storage implementations.
 *
 * @property namespace logical namespace for the storage instance.
 * @property globalConfig shared configuration map.
 */
open class StorageNameSpace(
    open val namespace: String,
    open val globalConfig: Map<String, Any?>,
) {
    /**
     * Hook invoked after indexing completes.
     */
    open suspend fun indexDoneCallback() = Unit

    /**
     * Called when a query operation finishes; override to perform cleanup or post-query actions.
     *
     * Default implementation performs no action.
     */
    open suspend fun queryDoneCallback() = Unit
}

/**
 * Base contract for vector storage backends.
 */
abstract class BaseVectorStorage(
    namespace: String,
    globalConfig: Map<String, Any?>,
) : StorageNameSpace(namespace, globalConfig) {
    /**
     * Perform a similarity search over stored vectors for the given query.
     *
     * Results are ordered by decreasing similarity and contain at most `topK` entries.
     *
     * @return A list of result records; each record is a map of field names to their values (nullable).
     */
    abstract suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, Any?>>

    /**
     * Bulk insert or update vector records in the storage backend.
     *
     * @param data Map where each key is a record identifier and each value is a map of fields for that record
     *             (for example vector and metadata fields).
     */
    abstract suspend fun upsert(data: Map<String, Map<String, Any?>>)

    /**
     * Remove all vectors associated with the given entity.
     *
     * The default implementation is a no-op; storage backends may override to delete related vectors.
     *
     * @param entityName The identifier or name of the entity whose vectors should be removed.
     */
    open suspend fun deleteEntity(entityName: String) {}

    /**
     * Delete all relationship vectors associated with the given entity within this namespace.
     *
     * Default implementation does nothing; storage backends should override to perform the actual deletion.
     *
     * @param entityName Identifier or name of the entity whose relationship vectors will be removed.
     */
    open suspend fun deleteRelation(entityName: String) {}

    /**
     * Delete the stored vector representing the relationship from a source node to a target node.
     *
     * @param srcId Identifier of the source node.
     * @param tgtId Identifier of the target node.
     */
    open suspend fun deleteRelationBetween(
        srcId: String,
        tgtId: String,
    ) {}

    /**
     * Remove all data and metadata associated with this storage namespace.
     *
     * Default implementation is a no-op; storage implementations should override to delete persisted data for the namespace.
     */
    open suspend fun drop() {}
}

/**
 * Base contract for key-value storage backends.
 */
abstract class BaseKVStorage<T>(
    namespace: String,
    globalConfig: Map<String, Any?>,
) : StorageNameSpace(namespace, globalConfig) {
    /**
     * Retrieve all keys stored in this namespace.
     *
     * @return A list containing every stored key.
     */
    abstract suspend fun allKeys(): List<String>

    /**
     * Retrieve a stored record by its identifier.
     *
     * @return The record matching `id`, or `null` if no record exists.
     */
    abstract suspend fun getById(id: String): T?

    /**
     * Retrieve multiple records for the given ids, optionally limiting returned fields.
     *
     * The returned list preserves the order of `ids`; each element is the record for the corresponding id or `null`
     * if no record exists for that id. When `fields` is provided, implementations should return only the specified fields
     * when supported.
     *
     * @param ids The list of record identifiers to fetch.
     * @param fields Optional set of field names to include in each returned record; if `null`, all available fields may be returned.
     * @return A list of records or `null` entries corresponding to each id in `ids`.
     */
    abstract suspend fun getByIds(
        ids: List<String>,
        fields: Set<String>? = null,
    ): List<T?>

    /**
     * Return the subset of input keys that are not present in the storage namespace.
     *
     * @param data The keys to check.
     * @return A set containing keys from `data` that do not already exist in storage.
     */
    abstract suspend fun filterKeys(data: List<String>): Set<String>

    /**
     * Bulk insert or update records in the storage namespace.
     *
     * Each entry's key is the record id and the value is the record to store; implementations should create missing records
     * and update existing ones.
     *
     * @param data Map from record id to record value to upsert.
     */
    abstract suspend fun upsert(data: Map<String, T>)

    /**
     * Drop the namespace and clear stored data.
     */
    abstract suspend fun drop()
}

/**
 * Base contract for graph storage backends.
 */
abstract class BaseGraphStorage(
    namespace: String,
    globalConfig: Map<String, Any?>,
) : StorageNameSpace(namespace, globalConfig) {
    /**
     * Check whether a node with the given identifier exists in the graph.
     *
     * @param nodeId The identifier of the node to check.
     * @return `true` if a node with the given id exists, `false` otherwise.
     */
    open suspend fun hasNode(nodeId: String): Boolean = false

    /**
     * Check whether an edge exists from the source node to the target node.
     *
     * @param sourceNodeId Identifier of the source node.
     * @param targetNodeId Identifier of the target node.
     * @return `true` if an edge from `sourceNodeId` to `targetNodeId` exists, `false` otherwise.
     */
    open suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean = false

    /**
     * Compute the degree of a node as the sum of its incoming and outgoing edges.
     *
     * @param nodeId Identifier of the node.
     * @return The number of incoming plus outgoing edges for the node.
     */
    open suspend fun nodeDegree(nodeId: String): Int = 0

    /**
     * Get the degree (number of connections) of the edge between two nodes.
     *
     * @param srcId Identifier of the source node.
     * @param tgtId Identifier of the target node.
     * @return The degree of the edge between the given source and target node.
     */
    open suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int = 0

    /**
     * Retrieve the PageRank score for the given node.
     *
     * @param nodeId Identifier of the node.
     * @return The node's PageRank score; `0.0` if the score is not available.
     */
    open suspend fun getPagerank(nodeId: String): Double = 0.0

    /**
     * Retrieve the stored properties for a node identified by the given id.
     *
     * @param nodeId The unique identifier of the node to fetch.
     * @return A map of the node's properties, or `null` if the node does not exist.
     */
    abstract suspend fun getNode(nodeId: String): Map<String, Any?>?

    /**
     * Retrieves properties for the edge between the given source and target node IDs.
     *
     * @param sourceNodeId The identifier of the source node.
     * @param targetNodeId The identifier of the target node.
     * @return A map of edge properties, or `null` if no edge exists between the specified nodes.
     */
    abstract suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, Any?>?

    /**
     * Retrieve edges incident to the specified node.
     *
     * @param sourceNodeId Identifier of the node whose edges should be fetched.
     * @return A list of pairs where the first element is the adjacent node's id and the second element is the edge id,
     *         or `null` if no edges are present.
     */
    abstract suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>>?

    /**
     * Retrieve incoming edges for the specified node.
     *
     * @param nodeId The target node identifier.
     * @return A list of pairs `(sourceNodeId, targetNodeId)` whose `targetNodeId` equals `nodeId`, or `null` if the storage
     *         does not expose edges.
     */
    open suspend fun getNodeInEdges(nodeId: String): List<Pair<String, String>>? = edges().filter { it.second == nodeId }

    /**
     * Return the outgoing edges that originate from the given node.
     *
     * @param nodeId The id of the source node whose outgoing edges are requested.
     * @return A list of `(sourceId, targetId)` pairs for edges originating from `nodeId`, or `null` if no edge list is available.
     */
    open suspend fun getNodeOutEdges(nodeId: String): List<Pair<String, String>>? = edges().filter { it.first == nodeId }

    /**
     * Inserts or updates a node in the graph storage.
     *
     * @param nodeId The unique identifier of the node.
     * @param nodeData A map of node properties; values may be null.
     */
    abstract suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, Any?>,
    )

    /**
     * Create or update an edge between two nodes in the namespace.
     *
     * Implementations should persist the provided edge properties for the directed edge
     * from `sourceNodeId` to `targetNodeId`, replacing or merging existing stored data
     * according to the backend's semantics.
     *
     * @param sourceNodeId Identifier of the source node.
     * @param targetNodeId Identifier of the target node.
     * @param edgeData Map of properties to associate with the edge.
     */
    abstract suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, Any?>,
    )

    /**
     * Remove the edge connecting two nodes identified by their IDs.
     *
     * @param sourceNodeId The identifier of the source node.
     * @param targetNodeId The identifier of the target node.
     */
    abstract suspend fun deleteEdge(
        sourceNodeId: String,
        targetNodeId: String,
    )

    /**
     * Delete a node and its edges.
     */
    abstract suspend fun deleteNode(nodeId: String)

    /**
     * Compute vector embeddings for nodes using the specified algorithm.
     *
     * @param algorithm The name of the embedding algorithm to apply.
     * @return A pair where the first element is a flat array of embedding values (embeddings for all nodes)
     *         and the second element is the ordered list of node IDs corresponding to those embeddings.
     * @throws NotImplementedError If node embedding is not implemented by the storage backend.
     */
    open suspend fun embedNodes(algorithm: String): Pair<DoubleArray, List<String>> =
        throw NotImplementedError("Node embedding is not implemented")

    /**
     * Retrieve the IDs of all nodes in this namespace.
     *
     * @return A list of node IDs; empty list if no nodes are present.
     */
    open suspend fun nodes(): List<String> = emptyList()

    /**
     * Return the stored edges as pairs of source and target node IDs.
     *
     * @return A list of pairs where each pair is `(sourceNodeId, targetNodeId)`. Returns an empty list by default.
     */
    open suspend fun edges(): List<Pair<String, String>> = emptyList()

    /**
     * Remove all data and metadata associated with this storage namespace.
     *
     * Default implementation is a no-op; storage implementations should override to delete persisted data for the namespace.
     */
    open suspend fun drop() {}
}

/**
 * Convenience helper to run suspend functions from blocking callers, mirroring the Python sync wrappers.
 */
fun <T> runBlockingMaybe(block: suspend () -> T): T = runBlocking { block() }
