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
     * Close the underlying Neo4j driver and release associated resources.
     */
    override fun close() {
        driver.close()
    }

    private suspend fun <R> read(block: (TransactionContext) -> R): R =
        withContext(Dispatchers.IO) { driver.session().use { session -> session.executeRead { tx -> block(tx) } } }

    /**
         * Execute the given block inside a Neo4j write transaction on the IO dispatcher.
         *
         * @param block Lambda invoked with a `TransactionContext` representing the write transaction; its return value is propagated.
         * @return The result produced by `block`.
         */
        private suspend fun <R> write(block: (TransactionContext) -> R): R =
        withContext(Dispatchers.IO) { driver.session().use { session -> session.executeWrite { tx -> block(tx) } } }

    /**
         * List all keys stored in this namespace.
         *
         * @return A list of keys present in this namespace.
         */
    override suspend fun allKeys(): List<String> =
        read { tx ->
            tx
                .run("MATCH (n:$nodeLabel) RETURN n.id AS id")
                .list { it.get("id").asString() }
        }

    /**
         * Retrieve the stored value for the given id from this namespace.
         *
         * @param id The key of the record to fetch.
         * @return The record's properties as `T` with the `"id"` property removed, or `null` if no node exists for the given id.
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
     * Fetches multiple records by their ids, preserving the order of the input list.
     *
     * @param ids The list of record ids to fetch.
     * @param fields Optional set of property keys to retain on each returned record; when `null`, all properties except `"id"` are retained.
     * @return A list whose elements correspond to `ids`: each element is the record (with the `"id"` property removed) cast to `T` if found, or `null` if no record exists for that id.
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
     * Compute the subset of the given keys that do not exist in storage.
     *
     * @param data List of keys to check for existence.
     * @return A set containing the keys from `data` that are not present in the storage.
     */
    override suspend fun filterKeys(data: List<String>): Set<String> {
        val existing = allKeys().toSet()
        return data.filterNot { existing.contains(it) }.toSet()
    }

    /**
     * Insert or update the provided entries into the namespace's Neo4j label.
     *
     * Each map entry is upserted by `id`: if `value` is a `Map`, its entries (except `"id"`) are written as node properties;
     * otherwise the value is stored under the `value` property. If `data` is empty the call is a no-op.
     *
     * @param data Map from record id to the value to store or update.
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
     * Removes all nodes belonging to this storage's namespace and deletes their relationships.
     *
     * This permanently deletes every node with the namespace-specific label using a Cypher
     * DETACH DELETE operation.
     */
    override suspend fun drop() {
        write { tx -> tx.run("MATCH (n:$nodeLabel) DETACH DELETE n") }
    }
}