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
     * Closes the underlying Neo4j driver and releases its resources.
     */
    override fun close() {
        driver.close()
    }

    private suspend fun <T> read(block: (TransactionContext) -> T): T =
        withContext(Dispatchers.IO) { driver.session().use { session -> session.executeRead { tx -> block(tx) } } }

    /**
         * Execute the given block inside a Neo4j write transaction on the IO dispatcher.
         *
         * @param block A function invoked with a `TransactionContext` to perform write operations; its return value is propagated.
         * @return The value returned by `block`.
         */
        private suspend fun <T> write(block: (TransactionContext) -> T): T =
        withContext(Dispatchers.IO) { driver.session().use { session -> session.executeWrite { tx -> block(tx) } } }

    /**
         * Determine whether a node with the given id exists.
         *
         * @param nodeId The node identifier to check.
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
         * Determines whether a directed relationship from the source node to the target node exists.
         *
         * Matches nodes by their `id` property and the storage's node label, and checks for a relationship
         * of the storage's relationship type from the source to the target.
         *
         * @param sourceNodeId The `id` of the source node.
         * @param targetNodeId The `id` of the target node.
         * @return `true` if such a relationship exists, `false` otherwise.
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
         * Computes the number of relationships incident to the node identified by [nodeId].
         *
         * @param nodeId The node's identifier.
         * @return The number of incident relationships (incoming and outgoing); `0` if the node does not exist.
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
     * Determine whether an edge exists from the source node to the target node and return its degree.
     *
     * @param srcId Identifier of the source node.
     * @param tgtId Identifier of the target node.
     * @return `1` if an edge exists from `srcId` to `tgtId`, `0` otherwise.
     */
    override suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int = if (hasEdge(srcId, tgtId)) 1 else 0

    /**
         * Retrieve the properties of the node with the given id.
         *
         * @return `Map<String, Any?>` of the node's properties if the node exists, `null` otherwise.
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
         * Retrieve the properties of the relationship from the node with id `sourceNodeId` to the node with id `targetNodeId`.
         *
         * @param sourceNodeId The identifier of the source node.
         * @param targetNodeId The identifier of the target node.
         * @return A map of the relationship's properties keyed by property name, or `null` if the edge does not exist.
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
         * @param sourceNodeId The id of the node whose incident edges to fetch.
         * @return A list of pairs `(srcId, tgtId)` for each incident relationship, including both outgoing and incoming edges.
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
         * Fetches incoming edges for the given node.
         *
         * @param nodeId The identifier of the target node.
         * @return A list of pairs where each pair is `(sourceNodeId, targetNodeId)` for an incoming edge; empty list if the node has no incoming edges.
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
         * Retrieve all outgoing edges from the given node.
         *
         * @return A list of pairs where each pair is (sourceNodeId, targetNodeId) for each outgoing edge.
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
     * Create or update a node with the storage's label using the provided identifier and properties.
     *
     * @param nodeId The node identifier (stored as the `id` property).
     * @param nodeData A map of property names to values to set on the node; values may be `null`.
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
     * Creates or updates a relationship from the source node to the target node, ensuring both endpoint nodes exist.
     *
     * Sets relationship properties from `edgeData` and stores `src_id` and `tgt_id` on the relationship.
     *
     * @param sourceNodeId The id of the source node.
     * @param targetNodeId The id of the target node.
     * @param edgeData Properties to merge onto the relationship; existing properties are overwritten by keys in this map.
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
     * Deletes the node with the specified id and removes all of its relationships.
     *
     * @param nodeId The value of the node's `id` property identifying which node to delete.
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
     * Delete the relationship of this storage's relationship type between the node identified by `sourceNodeId` and the node identified by `targetNodeId`.
     *
     * @param sourceNodeId The `id` property of the source node.
     * @param targetNodeId The `id` property of the target node.
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
         * Retrieve all node ids stored in this storage namespace.
         *
         * @return A list of node id strings.
         */
    override suspend fun nodes(): List<String> =
        read { tx -> tx.run("MATCH (n:$nodeLabel) RETURN n.id AS id").list { it.get("id").asString() } }

    /**
         * Return all edges as pairs of source and target node IDs.
         *
         * @return A list of pairs where the first element is the source node ID and the second is the target node ID.
         */
    override suspend fun edges(): List<Pair<String, String>> =
        read { tx ->
            tx
                .run("MATCH (s:$nodeLabel)-[r:$relType]->(t:$nodeLabel) RETURN s.id AS src, t.id AS tgt")
                .list { rec -> rec.get("src").asString() to rec.get("tgt").asString() }
        }

    /**
     * Retrieves the PageRank score for the node with the given id, computing PageRank if necessary.
     *
     * @param nodeId The identifier of the node whose PageRank is requested.
     * @return The PageRank score for the node, or 0.0 if the node has no computed score.
     */
    override suspend fun getPagerank(nodeId: String): Double {
        val ranks = computePagerank()
        return ranks[nodeId] ?: 0.0
    }

    /**
     * Deletes all nodes and relationships belonging to this storage's graph label.
     *
     * All nodes with the configured node label are detached from their relationships and removed.
     */
    override suspend fun drop() {
        write { tx -> tx.run("MATCH (n:$nodeLabel) DETACH DELETE n") }
    }

    /**
     * Produce embedding vectors for all nodes using the specified algorithm.
     *
     * If `algorithm` equals "node2vec" (case-insensitive), attempts to compute node2vec embeddings; otherwise constructs a 2-dimensional fallback embedding for each node where the vector is [pagerank, degree]. If the graph has no nodes, returns an empty array and list.
     *
     * @param algorithm The embedding algorithm name ("node2vec" to request node2vec; any other value triggers the pagerank/degree fallback).
     * @return A pair where the first element is a flattened array of embedding values (row-major, concatenated per-node vectors) and the second element is the list of node ids in the same order as the vectors.
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
         * Computes node embeddings using Neo4j Graph Data Science node2vec and returns them with node order.
         *
         * If GDS node2vec fails or returns no embeddings, falls back to `computeFallbackEmbeddings()`.
         *
         * @param dim The embedding dimensionality to request from node2vec.
         * @return A pair where the first element is a flattened array of embeddings (concatenated row-major:
         * each node's `dim` values in order) and the second element is the list of node IDs corresponding to
         * the embedding rows.
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