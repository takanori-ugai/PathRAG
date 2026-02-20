package pathrag.storage

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.neo4j.driver.AuthTokens
import org.neo4j.driver.Driver
import org.neo4j.driver.GraphDatabase
import org.neo4j.driver.Record
import org.neo4j.driver.TransactionContext
import org.neo4j.driver.Values
import pathrag.base.BaseGraphStorage
import pathrag.utils.computePagerankLocal
import java.io.Closeable
import kotlin.math.abs

/**
 * Neo4j-backed graph storage that optionally uses GDS for analytics.
 */
class Neo4jStorage(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
) : BaseGraphStorage(namespace, globalConfig),
    Closeable {
    private val logger = KotlinLogging.logger("PathRAG-Neo4j")

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

    private val nodeLabel = namespace
    private val relType = "${namespace.uppercase()}_REL"

    /**
     * Close the underlying driver.
     */
    override fun close() {
        driver.close()
    }

    private suspend fun <T> read(block: (TransactionContext) -> T): T =
        withContext(Dispatchers.IO) { driver.session().use { session -> session.executeRead { tx -> block(tx) } } }

    /**
     * Execute a block within a Neo4j write transaction and return its result.
     *
     * @param block Lambda that receives a `TransactionContext` and produces a result of type `T`. It is executed inside
     *              a write transaction.
     * @return The value produced by the provided `block`.
     */
    private suspend fun <T> write(block: (TransactionContext) -> T): T =
        withContext(Dispatchers.IO) { driver.session().use { session -> session.executeWrite { tx -> block(tx) } } }

    /**
     * Determine whether a node with the given id exists in the storage.
     *
     * @param nodeId The node identifier to look up.
     * @return `true` if a node with the given id exists, `false` otherwise.
     */
    override suspend fun hasNode(nodeId: String): Boolean =
        read { tx ->
            tx
                .run(
                    "MATCH (n:$nodeLabel {id:\$id}) RETURN 1 LIMIT 1",
                    Values.parameters("id", nodeId),
                ).hasNext()
        }

    /**
     * Determine whether an edge exists from the source node to the target node.
     *
     * @param sourceNodeId The id of the source node.
     * @param targetNodeId The id of the target node.
     * @return `true` if an edge exists from `sourceNodeId` to `targetNodeId`, `false` otherwise.
     */
    override suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean =
        read { tx ->
            tx
                .run(
                    "MATCH (:$nodeLabel {id:\$src})-[r:$relType]->(:$nodeLabel {id:\$tgt}) RETURN 1 LIMIT 1",
                    Values.parameters("src", sourceNodeId, "tgt", targetNodeId),
                ).hasNext()
        }

    /**
     * Get the degree (number of incident relationships) of the node with the given id.
     *
     * @param nodeId The node's identifier.
     * @return The node's degree as an Int, or 0 if the node does not exist.
     */
    override suspend fun nodeDegree(nodeId: String): Int =
        read { tx ->
            tx
                .run(
                    "MATCH (n:$nodeLabel {id:\$id})-[r]-() RETURN count(r) AS deg",
                    Values.parameters("id", nodeId),
                ).list()
                .firstOrNull()
                ?.get("deg")
                ?.asInt() ?: 0
        }

    /**
     * Determine whether an edge exists from the source node to the target node.
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
     * Retrieve properties for a node by its id.
     *
     * @return A map of the node's properties keyed by property name, or `null` if no node with the given id exists.
     */
    override suspend fun getNode(nodeId: String): Map<String, Any?>? =
        read { tx ->
            tx
                .run(
                    "MATCH (n:$nodeLabel {id:\$id}) RETURN properties(n) AS props",
                    Values.parameters("id", nodeId),
                ).list()
                .firstOrNull()
                ?.get("props")
                ?.asMap { v -> v.asObject() }
        }

    /**
     * Retrieve the properties of the relationship from the source node to the target node.
     *
     * @param sourceNodeId The id of the source node.
     * @param targetNodeId The id of the target node.
     * @return A map of relationship properties keyed by property name, or `null` if no such relationship exists.
     */
    override suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, Any?>? =
        read { tx ->
            tx
                .run(
                    "MATCH (s:$nodeLabel {id:\$src})-[r:$relType]->(t:$nodeLabel {id:\$tgt}) RETURN properties(r) AS props",
                    Values.parameters("src", sourceNodeId, "tgt", targetNodeId),
                ).list()
                .firstOrNull()
                ?.get("props")
                ?.asMap { v -> v.asObject() }
        }

    /**
     * Retrieve all edges incident to the specified node.
     *
     * @param sourceNodeId The node identifier to find incident edges for.
     * @return A list of pairs (sourceId, targetId) for each edge where the specified node is either the source or the target.
     */
    override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>> =
        read { tx ->
            tx
                .run(
                    "MATCH (s:$nodeLabel {id:\$id})-[r:$relType]->(t:$nodeLabel) RETURN s.id AS src, t.id AS tgt " +
                        "UNION " +
                        "MATCH (s:$nodeLabel)-[r:$relType]->(t:$nodeLabel {id:\$id}) RETURN s.id AS src, t.id AS tgt",
                    Values.parameters("id", sourceNodeId),
                ).list { rec -> rec.get("src").asString() to rec.get("tgt").asString() }
        }

    /**
     * Retrieve all incoming edges to the specified node.
     *
     * @param nodeId The identifier of the target node whose incoming edges are fetched.
     * @return A list of pairs `(src, tgt)` where `src` is the source node id and `tgt` is the target node id (equal to
     *         `nodeId`) for each incoming edge.
     */
    override suspend fun getNodeInEdges(nodeId: String): List<Pair<String, String>> =
        read { tx ->
            tx
                .run(
                    "MATCH (s:$nodeLabel)-[r:$relType]->(t:$nodeLabel {id:\$id}) RETURN s.id AS src, t.id AS tgt",
                    Values.parameters("id", nodeId),
                ).list { rec -> rec.get("src").asString() to rec.get("tgt").asString() }
        }

    /**
     * Fetches outgoing edges from the node identified by the given id as a list of (sourceId, targetId) pairs.
     *
     * @param nodeId The node identifier to fetch outgoing edges for.
     * @return A list of pairs where each pair is the source node id and the target node id for an outgoing edge.
     */
    override suspend fun getNodeOutEdges(nodeId: String): List<Pair<String, String>> =
        read { tx ->
            tx
                .run(
                    "MATCH (s:$nodeLabel {id:\$id})-[r:$relType]->(t:$nodeLabel) RETURN s.id AS src, t.id AS tgt",
                    Values.parameters("id", nodeId),
                ).list { rec -> rec.get("src").asString() to rec.get("tgt").asString() }
        }

    /**
     * Create or update a node with the given identifier, merging the provided properties into it.
     *
     * The node's `id` property is set to `nodeId`. Provided entries in `nodeData` are added or overwrite
     * existing properties on the node; keys with `null` values will set the corresponding property to `null`.
     *
     * @param nodeId The unique identifier for the node (stored as the node's `id` property).
     * @param nodeData A map of property names to values to be merged into the node.
     */
    override suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, Any?>,
    ) {
        write { tx ->
            tx.run(
                "MERGE (n:$nodeLabel {id:\$id}) SET n += \$props",
                Values.parameters("id", nodeId, "props", nodeData),
            )
        }
    }

    /**
     * Ensure an edge from the given source to target exists, creating the endpoint nodes if missing, and update the
     * relationship's properties.
     *
     * @param sourceNodeId ID of the source node.
     * @param targetNodeId ID of the target node.
     * @param edgeData Properties to merge onto the relationship; `src_id` and `tgt_id` are set to the corresponding
     *                 node IDs.
     */
    override suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, Any?>,
    ) {
        write { tx ->
            tx.run(
                "MERGE (s:$nodeLabel {id:\$src}) " +
                    "MERGE (t:$nodeLabel {id:\$tgt}) " +
                    "MERGE (s)-[r:$relType]->(t) " +
                    "SET r += \$props, r.src_id = \$src, r.tgt_id = \$tgt",
                Values.parameters("src", sourceNodeId, "tgt", targetNodeId, "props", edgeData),
            )
        }
    }

    /**
     * Remove the node with the given id and all relationships attached to it.
     *
     * @param nodeId Identifier of the node to remove.
     */
    override suspend fun deleteNode(nodeId: String) {
        write { tx ->
            tx.run(
                "MATCH (n:$nodeLabel {id:\$id}) DETACH DELETE n",
                Values.parameters("id", nodeId),
            )
        }
    }

    /**
     * Delete the relationship between two nodes identified by their ids.
     *
     * @param sourceNodeId The id of the source node.
     * @param targetNodeId The id of the target node.
     */
    override suspend fun deleteEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ) {
        write { tx ->
            tx.run(
                "MATCH (s:$nodeLabel {id:\$src})-[r:$relType]->(t:$nodeLabel {id:\$tgt}) DELETE r",
                Values.parameters("src", sourceNodeId, "tgt", targetNodeId),
            )
        }
    }

    /**
     * Retrieve all node ids for this storage namespace.
     *
     * @return A list of node id strings; an empty list if no nodes exist.
     */
    override suspend fun nodes(): List<String> =
        read { tx -> tx.run("MATCH (n:$nodeLabel) RETURN n.id AS id").list { it.get("id").asString() } }

    /**
     * Retrieve all directed edges as (sourceId, targetId) pairs.
     *
     * @return A list of pairs where the first element is the source node id and the second element is the target node id.
     */
    override suspend fun edges(): List<Pair<String, String>> =
        read { tx ->
            tx
                .run("MATCH (s:$nodeLabel)-[r:$relType]->(t:$nodeLabel) RETURN s.id AS src, t.id AS tgt")
                .list { rec -> rec.get("src").asString() to rec.get("tgt").asString() }
        }

    /**
     * Return the PageRank score for the given node.
     *
     * @return The PageRank score for `nodeId`, or `0.0` if the node has no computed rank.
     */
    override suspend fun getPagerank(nodeId: String): Double {
        val ranks = computePagerank()
        return ranks[nodeId] ?: 0.0
    }

    /**
     * Removes all nodes with this storage's node label and detaches any related relationships.
     */
    override suspend fun drop() {
        write { tx -> tx.run("MATCH (n:$nodeLabel) DETACH DELETE n") }
    }

    /**
     * Produce numeric embeddings for all nodes using the specified algorithm.
     *
     * If `algorithm` equals "node2vec" (case-insensitive) attempts to compute node2vec embeddings via Neo4j GDS and falls
     * back to a simple PageRank/degree embedding if node2vec is unavailable; for any other algorithm uses a fallback
     * embedding composed of PageRank and node degree.
     *
     * @param algorithm Name of the embedding algorithm to use (e.g., "node2vec"). Case is ignored.
     * @return A pair where the first element is a flattened DoubleArray of embeddings (concatenated per-node vectors)
     *         and the second element is the list of node ids in the same order as the embeddings. If there are no nodes,
     *         returns an empty array and an empty list.
     */
    override suspend fun embedNodes(algorithm: String): Pair<DoubleArray, List<String>> {
        return if (algorithm.lowercase() == "node2vec") {
            runNode2VecGds(globalConfig["node2vec_dim"] as? Int ?: 64)
        } else {
            val labels = nodes()
            if (labels.isEmpty()) return DoubleArray(0) to emptyList()
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

    /**
     * Compute node embeddings using Neo4j GDS node2vec, falling back to a PageRank/degree-based embedding if node2vec is
     * not available or fails.
     *
     * @param dim The embedding dimensionality to request from node2vec.
     * @return A pair where the first element is a flattened DoubleArray containing the concatenated embedding vectors
     *         (ordered by node) and the second element is a List of node ids (as strings) in the same order as the
     *         embeddings. The flattened array length will be `labels.size * dim` when node2vec succeeds; fallback
     *         embeddings use a 2-dimensional scheme per node.
     */
    @Suppress("TooGenericExceptionCaught")
    private suspend fun runNode2VecGds(dim: Int): Pair<DoubleArray, List<String>> =
        withContext(Dispatchers.IO) {
            val labels = mutableListOf<String>()
            val vectors = mutableListOf<DoubleArray>()
            val graphName = "pathrag_${namespace}_gds"
            try {
                driver.session().use { session ->
                    runCatching {
                        session
                            .run(
                                "CALL gds.graph.drop(\$g, false) YIELD graphName",
                                Values.parameters("g", graphName),
                            ).consume()
                    }
                    session
                        .run(
                            "CALL gds.graph.project(\$g, \$labels, {`$relType`:{orientation:'UNDIRECTED'}})",
                            Values.parameters("g", graphName, "labels", listOf(nodeLabel)),
                        ).consume()
                    val result =
                        session.run(
                            "CALL gds.beta.node2vec.stream(\$g, {embeddingDimension:\$dim}) " +
                                "YIELD nodeId, embedding " +
                                "RETURN gds.util.asNode(nodeId).id AS key, embedding",
                            Values.parameters("g", graphName, "dim", dim),
                        )
                    result.list { rec: Record ->
                        val key = rec.get("key").asString()
                        val emb = rec.get("embedding").asList { (it as Number).toDouble() }.toDoubleArray()
                        labels.add(key)
                        vectors.add(emb)
                        Unit
                    }
                }
            } catch (ex: IllegalStateException) {
                logger.warn(ex) { "Neo4j node2vec failed; falling back to pagerank/degree." }
                return@withContext computeFallbackEmbeddings()
            } finally {
                runCatching {
                    driver.session().use {
                        it.run("CALL gds.graph.drop(\$g, false) YIELD graphName", Values.parameters("g", graphName)).consume()
                    }
                }
            }

            if (labels.isEmpty() || vectors.isEmpty()) {
                return@withContext computeFallbackEmbeddings()
            }
            val flat = vectors.flatMap { it.asIterable() }.toDoubleArray()
            flat to labels
        }

    private suspend fun computeFallbackEmbeddings(): Pair<DoubleArray, List<String>> {
        val fallbackLabels = nodes()
        if (fallbackLabels.isEmpty()) return DoubleArray(0) to emptyList()
        val ranks = computePagerank()
        val degs = fallbackLabels.map { nodeDegree(it).toDouble() }
        val vecs =
            fallbackLabels.mapIndexed { idx, id ->
                doubleArrayOf(ranks[id] ?: 0.0, degs[idx])
            }
        val flat = vecs.flatMap { it.asIterable() }.toDoubleArray()
        return flat to fallbackLabels
    }

    private suspend fun fetchGraph(): Pair<List<String>, List<Pair<String, String>>> {
        val allNodes = nodes()
        val allEdges = edges()
        return allNodes to allEdges
    }

    private suspend fun computePagerank(
        damping: Double = 0.85,
        maxIter: Int = 100,
        tol: Double = 1e-6,
    ): Map<String, Double> {
        val gdsRanks = computePagerankGds()
        if (gdsRanks.isNotEmpty()) return gdsRanks
        val (nodeList, edgeList) = fetchGraph()
        return computePagerankLocal(nodeList, edgeList, damping, maxIter, tol)
    }

    private suspend fun computePagerankGds(): Map<String, Double> {
        val ranks = mutableMapOf<String, Double>()
        val graphName = "pathrag_${namespace}_gds_pagerank"
        withContext(Dispatchers.IO) {
            runCatching {
                driver.session().use { session ->
                    runCatching { session.run("CALL gds.graph.drop(\$g, false)", Values.parameters("g", graphName)).consume() }
                    session
                        .run(
                            "CALL gds.graph.project(\$g, \$labels, {`$relType`:{orientation:'UNDIRECTED'}})",
                            Values.parameters("g", graphName, "labels", listOf(nodeLabel)),
                        ).consume()
                    val result =
                        session.run(
                            "CALL gds.pageRank.stream(\$g) YIELD nodeId, score " +
                                "RETURN gds.util.asNode(nodeId).id AS id, score",
                            Values.parameters("g", graphName),
                        )
                    result.list { rec: Record ->
                        ranks[rec.get("id").asString()] = rec.get("score").asDouble()
                        Unit
                    }
                    runCatching { session.run("CALL gds.graph.drop(\$g, false)", Values.parameters("g", graphName)).consume() }
                }
            }.onFailure { ex -> logger.warn(ex) { "Neo4j GDS PageRank failed; falling back to in-memory computation." } }
        }
        return ranks
    }
}
