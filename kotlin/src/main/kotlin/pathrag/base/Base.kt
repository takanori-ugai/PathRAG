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
     * Hook invoked after a query completes.
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
     * Execute a similarity query against stored vectors.
     */
    abstract suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, Any?>>

    /**
     * Insert or update vector records in bulk.
     */
    abstract suspend fun upsert(data: Map<String, Map<String, Any?>>)

    /**
     * Remove all vectors related to an entity.
     */
    open suspend fun deleteEntity(entityName: String) {}

    /**
     * Remove relationship vectors touching an entity.
     */
    open suspend fun deleteRelation(entityName: String) {}

    /**
     * Remove a single relationship vector.
     */
    open suspend fun deleteRelationBetween(
        srcId: String,
        tgtId: String,
    ) {}

    /**
     * Drop the entire namespace.
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
     * List all stored keys.
     */
    abstract suspend fun allKeys(): List<String>

    /**
     * Fetch a single record by id.
     */
    abstract suspend fun getById(id: String): T?

    /**
     * Fetch multiple records by ids with optional field filtering.
     */
    abstract suspend fun getByIds(
        ids: List<String>,
        fields: Set<String>? = null,
    ): List<T?>

    /**
     * Filter out keys that already exist.
     */
    abstract suspend fun filterKeys(data: List<String>): Set<String>

    /**
     * Insert or update records in bulk.
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
     * Determine if a node exists.
     */
    open suspend fun hasNode(nodeId: String): Boolean = false

    /**
     * Determine if an edge exists.
     */
    open suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean = false

    /**
     * Degree of a node (incoming + outgoing).
     */
    open suspend fun nodeDegree(nodeId: String): Int = 0

    /**
     * Degree of a specific edge.
     */
    open suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int = 0

    /**
     * PageRank score for a node.
     */
    open suspend fun getPagerank(nodeId: String): Double = 0.0

    /**
     * Fetch node properties by id.
     */
    abstract suspend fun getNode(nodeId: String): Map<String, Any?>?

    /**
     * Fetch edge properties by source/target ids.
     */
    abstract suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, Any?>?

    /**
     * Fetch all edges for a node.
     */
    abstract suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>>?

    /**
     * Fetch incoming edges for a node.
     */
    open suspend fun getNodeInEdges(nodeId: String): List<Pair<String, String>>? = edges().filter { it.second == nodeId }

    /**
     * Fetch outgoing edges for a node.
     */
    open suspend fun getNodeOutEdges(nodeId: String): List<Pair<String, String>>? = edges().filter { it.first == nodeId }

    /**
     * Insert or update a node.
     */
    abstract suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, Any?>,
    )

    /**
     * Insert or update an edge.
     */
    abstract suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, Any?>,
    )

    /**
     * Delete an edge by endpoints.
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
     * Embed nodes with the requested algorithm.
     */
    open suspend fun embedNodes(algorithm: String): Pair<DoubleArray, List<String>> =
        throw NotImplementedError("Node embedding is not implemented")

    /**
     * List node ids.
     */
    open suspend fun nodes(): List<String> = emptyList()

    /**
     * List edges as source/target pairs.
     */
    open suspend fun edges(): List<Pair<String, String>> = emptyList()

    /**
     * Drop the namespace and clear stored data.
     */
    open suspend fun drop() {}
}

/**
 * Convenience helper to run suspend functions from blocking callers, mirroring the Python sync wrappers.
 */
fun <T> runBlockingMaybe(block: suspend () -> T): T = runBlocking { block() }
