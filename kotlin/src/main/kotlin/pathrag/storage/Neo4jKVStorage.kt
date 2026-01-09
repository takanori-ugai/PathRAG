package pathrag.storage

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.neo4j.driver.AuthTokens
import org.neo4j.driver.Driver
import org.neo4j.driver.GraphDatabase
import org.neo4j.driver.TransactionContext
import org.neo4j.driver.Values
import pathrag.base.BaseKVStorage

/**
 * Neo4j-backed key-value store using a label per namespace.
 */
class Neo4jKVStorage<T : Any>(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
) : BaseKVStorage<T>(namespace, globalConfig),
    AutoCloseable {
    private val logger = KotlinLogging.logger("PathRAG-Neo4jKVStorage")

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

    private val nodeLabel = "${namespace.uppercase()}_KV"

    /**
     * Shuts down the Neo4j driver and releases associated resources.
     */
    override fun close() {
        driver.close()
    }

    private suspend fun <R> read(block: (TransactionContext) -> R): R =
        withContext(Dispatchers.IO) { driver.session().use { session -> session.executeRead { tx -> block(tx) } } }

    /**
         * Executes the provided block inside a Neo4j write transaction.
         *
         * The `block` is invoked with a `TransactionContext` scoped to the transaction and its return value is forwarded.
         *
         * @param block Lambda to execute within the write transaction; receives the transaction context.
         * @return The value produced by `block`.
         */
        private suspend fun <R> write(block: (TransactionContext) -> R): R =
        withContext(Dispatchers.IO) { driver.session().use { session -> session.executeWrite { tx -> block(tx) } } }

    /**
         * List all keys stored in this storage's namespace.
         *
         * @return A list of key strings present in the namespace.
         */
    override suspend fun allKeys(): List<String> =
        read { tx ->
            tx
                .run("MATCH (n:$nodeLabel) RETURN n.id AS id")
                .list { it.get("id").asString() }
        }

    /**
         * Retrieve the stored value for the given id.
         *
         * @param id The record identifier.
         * @return The node's properties as `T` with the `id` property removed, or `null` if no node exists.
         */
    override suspend fun getById(id: String): T? =
        read { tx ->
            tx
                .run("MATCH (n:$nodeLabel {id:\$id}) RETURN properties(n) AS props", Values.parameters("id", id))
                .list()
                .firstOrNull()
                ?.get("props")
                ?.asMap { v -> v.asObject() }
                ?.filterKeys { it != "id" }
                ?.let {
                    @Suppress("UNCHECKED_CAST")
                    (it as T)
                }
        }

    /**
     * Retrieve multiple records by their IDs while preserving input order.
     *
     * For each ID in `ids` returns the corresponding node's properties (the `id` property is excluded).
     * If `fields` is provided, only those property names are retained for each result. Missing IDs yield `null` entries,
     * and an empty `ids` list returns an empty result list.
     *
     * @param ids The ordered list of record IDs to fetch.
     * @param fields Optional set of property names to include; if `null`, all properties (except `id`) are returned.
     * @return A list whose elements correspond positionally to `ids`: the record properties cast to `T`, or `null` if not found.
     */
    override suspend fun getByIds(
        ids: List<String>,
        fields: Set<String>?,
    ): List<T?> {
        if (ids.isEmpty()) return emptyList()
        val results =
            read { tx ->
                tx
                    .run(
                        "MATCH (n:$nodeLabel) WHERE n.id IN \$ids RETURN n.id AS id, properties(n) AS props",
                        Values.parameters("ids", ids),
                    ).list { rec ->
                        rec.get("id").asString() to rec.get("props").asMap { v -> v.asObject() }
                    }.toMap()
            }
        return ids.map { id ->
            results[id]
                ?.filterKeys { it != "id" && (fields == null || fields.contains(it)) }
                ?.let {
                    @Suppress("UNCHECKED_CAST")
                    (it as T)
                }
        }
    }

    /**
     * Return the subset of provided IDs that are not present in this storage namespace.
     *
     * @param data The list of IDs to check for existence.
     * @return A set containing IDs from `data` that do not exist in storage.
     */
    override suspend fun filterKeys(data: List<String>): Set<String> {
        val existing = allKeys().toSet()
        return data.filterNot { existing.contains(it) }.toSet()
    }

    /**
     * Creates or updates nodes for the given id->value pairs in the namespace label.
     *
     * For each entry, a node with the storage label and the given `id` is created if missing or updated if present.
     * If a value is a Map, its entries (excluding an `"id"` key) are stored as node properties; otherwise the value is stored
     * under the `"value"` property. Calling with an empty map does nothing.
     *
     * @param data Mapping of record `id` to its value or property map to upsert.
     */
    override suspend fun upsert(data: Map<String, T>) {
        if (data.isEmpty()) return
        write { tx ->
            data.forEach { (id, value) ->
                val props =
                    when (value) {
                        is Map<*, *> -> value.filterKeys { it != "id" }
                        else -> mapOf("value" to value)
                    }
                tx.run(
                    "MERGE (n:$nodeLabel {id:\$id}) SET n += \$props, n.id = \$id",
                    Values.parameters("id", id, "props", props),
                )
            }
        }
    }

    /**
     * Drop all data in this namespace.
     */
    override suspend fun drop() {
        write { tx -> tx.run("MATCH (n:$nodeLabel) DETACH DELETE n") }
    }
}