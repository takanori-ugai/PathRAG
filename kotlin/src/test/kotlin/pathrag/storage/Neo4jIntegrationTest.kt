package pathrag.storage

import io.mockk.every
import io.mockk.justRun
import io.mockk.mockk
import io.mockk.mockkStatic
import io.mockk.unmockkAll
import kotlinx.coroutines.runBlocking
import org.neo4j.driver.AuthToken
import org.neo4j.driver.Driver
import org.neo4j.driver.GraphDatabase
import org.neo4j.driver.Record
import org.neo4j.driver.Result
import org.neo4j.driver.Session
import org.neo4j.driver.TransactionCallback
import org.neo4j.driver.TransactionContext
import org.neo4j.driver.Value
import org.neo4j.driver.Values
import pathrag.PathRAG
import java.util.function.Function
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertTrue

class Neo4jIntegrationTest {
    private val queries = mutableListOf<String>()

    @AfterTest
    fun tearDown() {
        unmockkAll()
        queries.clear()
    }

    @Test
    fun pathRag_usesNeo4jBackendsWithMockDriver() =
        runBlocking {
            mockkStatic(GraphDatabase::class)

            val driver = mockk<Driver>()
            val session = mockk<Session>()
            val tx = mockk<TransactionContext>()

            every { GraphDatabase.driver(any<String>(), any<AuthToken>()) } returns driver
            every { driver.session() } returns session
            justRun { driver.close() }
            justRun { session.close() }

            fun stubResult(): Result =
                mockk<Result>().apply {
                    every { hasNext() } returns false
                    every { list() } returns emptyList()
                    every { list<Any?>(any<Function<Record, Any?>>()) } returns emptyList()
                }

            every { tx.run(any<String>(), any<Value>()) } answers {
                val q = firstArg<String>()
                queries.add(q)
                stubResult()
            }
            every { tx.run(any<String>()) } answers {
                val q = firstArg<String>()
                queries.add(q)
                stubResult()
            }

            every { session.executeWrite<Any>(any<TransactionCallback<Any>>()) } answers {
                val callback = firstArg<TransactionCallback<Any>>()
                callback.execute(tx)
            }
            every { session.executeRead<Any>(any<TransactionCallback<Any>>()) } answers {
                val callback = firstArg<TransactionCallback<Any>>()
                callback.execute(tx)
            }

            val rag =
                PathRAG(
                    kvStorage = "Neo4jKVStorage",
                    vectorStorage = "Neo4jVectorStorage",
                    graphStorage = "Neo4jStorage",
                    chunkTokenSize = 16,
                    chunkOverlapTokenSize = 4,
                    extraConfig =
                        pathrag.base.ExtraConfig(
                            neo4jUri = "bolt://mock",
                            neo4jUser = "user",
                            neo4jPassword = "pass",
                        ),
                )

            rag.aupsertEntity("Alpha", "desc", "TYPE", "src")
            rag.aupsertEdge("Alpha", "Beta", "link", "k", 1.0, "src")
            rag.dropGraph()
            rag.close()

            val hasNamespaceQuery = queries.any { it.contains("chunk_entity_relation") }
            val hasMerge = queries.any { it.startsWith("MERGE") || it.contains("MERGE (") }
            assertTrue(hasNamespaceQuery, "Expected queries against the chunk_entity_relation namespace")
            assertTrue(hasMerge, "Expected at least one MERGE query to be issued")
        }
}
