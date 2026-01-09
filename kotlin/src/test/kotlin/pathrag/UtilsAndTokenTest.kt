package pathrag

import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import pathrag.utils.EmbeddingFunc
import pathrag.utils.ResponseCache
import pathrag.utils.Tokenizer
import pathrag.utils.computeArgsHash
import pathrag.utils.computeMdHashId
import pathrag.utils.computePagerankLocal
import pathrag.utils.convertResponseToJson
import pathrag.utils.limitAsyncFuncCall
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class UtilsAndTokenTest {
    @Test
    fun convertResponseToJson_extractsEmbeddedObject() {
        val response =
            """
            Noise before
            {"name": "alice", "details": {"age": "30"}}
            and after
            """.trimIndent()
        val parsed = convertResponseToJson(response)
        assertEquals("alice", parsed["name"])
        assertTrue(parsed["details"].toString().contains("age"), "Nested object should be preserved")
    }

    @Test
    fun computeMdHashId_isDeterministicAndPrefixed() {
        val h1 = computeMdHashId("content", prefix = "doc-")
        val h2 = computeMdHashId("content", prefix = "doc-")
        assertEquals(h1, h2)
        assertTrue(h1.startsWith("doc-"))
    }

    @Test
    fun computeArgsHash_changesWhenArgsChange() {
        val first = computeArgsHash("a", 1)
        val second = computeArgsHash("a", 2)
        assertNotEquals(first, second)
    }

    @Test
    fun limitAsyncFuncCall_respectsConcurrencyLimit() =
        runBlocking {
            val limiter = limitAsyncFuncCall(maxSize = 2)
            var running = 0
            var peak = 0
            val worker: () -> Unit = {
                runBlocking {
                    if (++running > peak) peak = running
                    delay(20)
                    running--
                }
            }
            val limited = limiter(worker)
            val jobs =
                (1..6).map {
                    async { limited() }
                }
            jobs.forEach { it.await() }
            assertTrue(peak <= 2, "Peak concurrency should not exceed limiter")
        }

    @Test
    fun embeddingFunc_throwsOnDimensionMismatch() {
        val func =
            EmbeddingFunc(
                embeddingDim = 2,
                maxTokenSize = 8,
                func = { _ -> listOf(doubleArrayOf(1.0)) },
            )
        assertFailsWith<IllegalStateException> {
            runBlocking { func(listOf("x")) }
        }
    }

    @Test
    fun responseCache_returnsCachedAndSimilarResults() =
        runBlocking {
            val embed =
                EmbeddingFunc(
                    embeddingDim = 2,
                    maxTokenSize = 8,
                    func = { _ -> listOf(doubleArrayOf(1.0, 0.0)) },
                )
            val cache =
                ResponseCache(
                    mapOf(
                        "embedding_func" to embed,
                        "embedding_cache_config" to mapOf("enabled" to true, "similarity_threshold" to 0.5, "use_llm_check" to false),
                    ),
                )
            cache.upsert("default", "hash-1", "cached-content", "prompt A")
            val direct = cache.handleCache("hash-1", "prompt A", "default")
            val similar = cache.handleCache("hash-2", "prompt B", "default")
            assertEquals("cached-content", direct)
            assertEquals("cached-content", similar)
        }

    @Test
    fun tokenizer_roundTripKeepsContent() {
        val text = "Hello Kotlin!"
        val tokens = Tokenizer.encode(text)
        val decoded = Tokenizer.decode(tokens)
        assertEquals(text, decoded)
    }

    @Test
    fun pagerank_computesScores() {
        val nodes = listOf("A", "B", "C")
        val edges =
            listOf(
                "A" to "B",
                "B" to "C",
                "C" to "A",
            )
        val ranks = computePagerankLocal(nodes, edges)
        assertEquals(3, ranks.size)
        val total = ranks.values.sum()
        assertTrue(total > 0.9 && total < 1.1, "Ranks should sum close to 1, got $total")
    }

    @Test
    fun tokenService_issuesAndVerifiesTokens() {
        val env = EnvironmentConfig.empty()
        TokenService.configure(env)
        val token = TokenService.issueToken("tester")
        val username = TokenService.usernameFromToken(token)
        assertEquals("tester", username)
    }
}
