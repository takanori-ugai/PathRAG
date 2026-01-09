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
         * Convert this AddonParams instance into a configuration map suitable for downstream use.
         *
         * The map contains the keys "entity_types", "language", and "example_number" mapped to the corresponding properties.
         *
         * @return A map with keys `"entity_types"` -> list of entity types, `"language"` -> language string or `null`, and `"example_number"` -> example count or `null`.
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
         * Builds a map of configured backend connection settings, omitting null values and merging with any additional entries.
         *
         * @return A map containing non-null Neo4j and Mongo configuration keys (e.g. `neo4j_uri`, `neo4j_user`, `neo4j_password`, `mongo_uri`, `mongo_database`) merged with the `additional` map.
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
         * Builds a configuration map representing this GlobalConfig, optionally merged with extra entries.
         *
         * The returned map contains all exposed configuration fields (including `addon_params` produced by `addonParams.asConfig()`)
         * and then overlays any key/value pairs from `extra`, with `extra` entries taking precedence.
         *
         * @param extra Additional configuration entries to merge into the resulting map; keys in `extra` override existing keys.
         * @return A map of configuration keys to values representing this GlobalConfig merged with `extra`.
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
 * Called after a query completes to allow implementations to perform post-query work (cleanup, metrics, etc.).
 *
 * Default implementation is a no-op; override to run custom suspendable tasks after queries.
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
     * Run a similarity search using the provided text to retrieve the most relevant stored vectors.
     *
     * @param query The query text used to compute similarity against stored vectors.
     * @param topK The maximum number of most-similar results to return.
     * @return A list of result records ordered by descending similarity. Each map represents a stored record's fields and may include implementation-specific metadata such as a similarity score.
     */
    abstract suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, Any?>>

    /**
 * Insert or update multiple vector records in the storage.
 *
 * @param data A map where each key is a record identifier and each value is a map of that record's fields (for example vector values and associated metadata). Implementations should upsert each entry. 
 */
    abstract suspend fun upsert(data: Map<String, Map<String, Any?>>)

    /**
 * Deletes all stored vectors associated with the specified entity.
 *
 * @param entityName The identifier of the entity whose vectors will be removed.
 */
    open suspend fun deleteEntity(entityName: String) {}

    /**
 * Remove relationship vectors touching the specified entity.
 *
 * Default implementation does nothing; storage implementations should override to delete any stored
 * relation vectors associated with the given entity.
 *
 * @param entityName The identifier of the entity whose relationship vectors should be removed.
 */
    open suspend fun deleteRelation(entityName: String) {}

    /**
     * Deletes the stored relation (edge) vector between two nodes in this namespace.
     *
     * @param srcId The source node identifier.
     * @param tgtId The target node identifier.
     */
    open suspend fun deleteRelationBetween(
        srcId: String,
        tgtId: String,
    ) {}

    /**
 * Removes all data associated with this namespace.
 *
 * The default implementation does nothing; storage backends should override this to drop the namespace and clear stored data.
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
 * List all keys stored in this namespace.
 *
 * @return A list containing every stored key. 
 */
    abstract suspend fun allKeys(): List<String>

    /**
 * Retrieve a stored record by its identifier.
 *
 * @return The record corresponding to the given id, or `null` if no record exists.
 */
    abstract suspend fun getById(id: String): T?

    /**
     * Retrieve multiple records by their ids with optional field filtering.
     *
     * @param ids The list of record ids to fetch, in the order to be preserved in the result.
     * @param fields If non-null, restrict each returned record to the specified field names; if null, return full records.
     * @return A list of records corresponding to the provided ids in the same order; an element is `null` when the record does not exist.
     */
    abstract suspend fun getByIds(
        ids: List<String>,
        fields: Set<String>? = null,
    ): List<T?>

    /**
 * Filters the provided keys to those not present in the storage namespace.
 *
 * @param data Candidate keys to check for existence.
 * @return A set containing the keys from `data` that do not currently exist in storage.
 */
    abstract suspend fun filterKeys(data: List<String>): Set<String>

    /**
 * Insert or update multiple records in a single bulk operation.
 *
 * @param data Map from record ID to record value; each entry will be created if missing or updated if already present.
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
 * Check whether a node with the given identifier exists in the namespace.
 *
 * @param nodeId The node identifier to check.
 * @return `true` if the node exists, `false` otherwise.
 */
    open suspend fun hasNode(nodeId: String): Boolean = false

    /**
     * Checks whether an edge exists from the source node to the target node in this namespace.
     *
     * @param sourceNodeId The identifier of the source node.
     * @param targetNodeId The identifier of the target node.
     * @return `true` if an edge from `sourceNodeId` to `targetNodeId` exists, `false` otherwise.
     */
    open suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean = false

    /**
 * Number of edges connected to the given node (incoming plus outgoing).
 *
 * @param nodeId Identifier of the node.
 * @return The node's degree (count of incoming and outgoing edges).
 */
    open suspend fun nodeDegree(nodeId: String): Int = 0

    /**
     * Get the degree of the edge between two nodes.
     *
     * @param srcId The source node identifier.
     * @param tgtId The target node identifier.
     * @return The number of edges connecting the source node to the target node.
     */
    open suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int = 0

    /**
 * Retrieve the PageRank score for the specified node.
 *
 * @param nodeId Identifier of the node whose PageRank to retrieve.
 * @return The PageRank score for the node. Returns 0.0 by default when PageRank is not implemented by the storage.
 */
    open suspend fun getPagerank(nodeId: String): Double = 0.0

    /**
 * Retrieve the properties of a graph node by its identifier.
 *
 * @param nodeId The identifier of the node to fetch.
 * @return A map of node properties keyed by property name, or `null` if the node does not exist.
 */
    abstract suspend fun getNode(nodeId: String): Map<String, Any?>?

    /**
     * Retrieves properties for the edge between the given source and target node IDs.
     *
     * @param sourceNodeId ID of the source node.
     * @param targetNodeId ID of the target node.
     * @return A map of edge properties keyed by property name, or `null` if the edge does not exist.
     */
    abstract suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, Any?>?

    /**
 * Retrieve all edges for the given node.
 *
 * @param sourceNodeId The node id whose edges are requested.
 * @return A list of pairs `(sourceNodeId, targetNodeId)` representing edges for the node, or `null` if no edges are found.
 */
    abstract suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>>?

    /**
 * Get the incoming edges for the given node.
 *
 * @param nodeId The node identifier to find incoming edges for.
 * @return A list of pairs `(sourceId, targetId)` whose `targetId` equals `nodeId`, or `null` if incoming edges are not available.
 */
    open suspend fun getNodeInEdges(nodeId: String): List<Pair<String, String>>? = edges().filter { it.second == nodeId }

    /**
 * Retrieve the outgoing edges for the given node.
 *
 * @param nodeId The identifier of the source node.
 * @return A list of pairs `(sourceNodeId, targetNodeId)` representing edges whose source equals `nodeId`, or `null` if the backing `edges()` implementation returns `null`.
 */
    open suspend fun getNodeOutEdges(nodeId: String): List<Pair<String, String>>? = edges().filter { it.first == nodeId }

    /**
     * Inserts or updates a node in the storage namespace.
     *
     * @param nodeId The node's identifier.
     * @param nodeData A map of node properties; values may be null.
     */
    abstract suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, Any?>,
    )

    /**
     * Insert or update an edge between two nodes using the provided edge properties.
     *
     * @param sourceNodeId ID of the source node.
     * @param targetNodeId ID of the target node.
     * @param edgeData Map of edge properties and metadata to store for the edge.
     */
    abstract suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, Any?>,
    )

    /**
     * Delete the edge that connects the specified source and target nodes.
     *
     * @param sourceNodeId The identifier of the source node.
     * @param targetNodeId The identifier of the target node.
     */
    abstract suspend fun deleteEdge(
        sourceNodeId: String,
        targetNodeId: String,
    )

    /**
 * Removes a node and all incident edges from the storage namespace.
 *
 * @param nodeId The identifier of the node to remove.
 */
    abstract suspend fun deleteNode(nodeId: String)

    /**
         * Compute vector embeddings for graph nodes using the specified algorithm.
         *
         * @param algorithm The name or identifier of the embedding algorithm to apply.
         * @return A pair where the first element is a flat `DoubleArray` containing embedding values
         *         (embeddings concatenated in the same order as the nodes), and the second element is
         *         a `List<String>` of node IDs corresponding to each embedding vector in order.
         * @throws NotImplementedError Thrown by the default implementation when node embedding is not supported.
         */
    open suspend fun embedNodes(algorithm: String): Pair<DoubleArray, List<String>> =
        throw NotImplementedError("Node embedding is not implemented")

    /**
 * List all node ids in the namespace.
 *
 * @return A list of node ids present in the namespace; an empty list if no nodes exist.
 */
    open suspend fun nodes(): List<String> = emptyList()

    /**
 * Return the graph's edges as (sourceId, targetId) pairs.
 *
 * @return A list of pairs where the first element is the source node id and the second is the target node id; returns an empty list if there are no edges.
 */
    open suspend fun edges(): List<Pair<String, String>> = emptyList()

    /**
 * Removes all data associated with this namespace.
 *
 * The default implementation does nothing; storage backends should override this to drop the namespace and clear stored data.
 */
    open suspend fun drop() {}
}

/**
 * Convenience helper to run suspend functions from blocking callers, mirroring the Python sync wrappers.
 */
fun <T> runBlockingMaybe(block: suspend () -> T): T = runBlocking { block() }