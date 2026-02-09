package pathrag

import kotlinx.coroutines.runBlocking
import pathrag.base.AddonParams
import pathrag.eval.RagasEvaluator
import pathrag.eval.RagasInput
import pathrag.eval.RagasMetrics
import java.nio.file.Paths

/**
 * Demonstrates generating a RAGAS-compatible JSONL evaluation set from sample content.
 */
fun main() =
    runBlocking {
        val env = EnvironmentConfig.load(Paths.get("../.env"))
        val kvStorage = env["KV_STORAGE"] ?: "JsonKVStorage"
        val vectorStorage = env["VECTOR_STORAGE"] ?: "NanoVectorDBStorage"
        val graphStorage = env["GRAPH_STORAGE"] ?: "NetworkXStorage"

        val rag =
            PathRAG(
                workingDir = env["WORKING_DIR"] ?: "./sample_cache",
                kvStorage = kvStorage,
                vectorStorage = vectorStorage,
                graphStorage = graphStorage,
                chunkTokenSize = 800,
                chunkOverlapTokenSize = 120,
                language = env["LANGUAGE"] ?: "English",
                keywordExamples = "",
                highLevelKeywords = listOf("themes", "Dickens"),
                lowLevelKeywords = listOf("poverty", "class struggle", "redemption"),
                similarityCheckPrompt = pathrag.prompt.Prompts.SIMILARITY_CHECK,
                embeddingCacheConfig =
                    mapOf(
                        "enabled" to true,
                        "similarity_threshold" to 0.9,
                        "use_llm_check" to false,
                    ),
                addonParams =
                    AddonParams(
                        entityTypes = listOf("organization", "person", "geo", "event", "category"),
                        exampleNumber = 3,
                    ),
            )

        rag.insert(
            listOf(
                """
                Charles Dickens was an English writer and social critic.
                He created some of the world's best-known fictional characters
                and is regarded as one of the greatest novelists of the Victorian era.
                """.trimIndent(),
                """
                Oliver Twist is a novel by Dickens that critiques workhouses and child poverty.
                It follows an orphan navigating criminal underworlds and harsh social systems.
                """.trimIndent(),
                """
                A Christmas Carol tells the redemption story of Ebenezer Scrooge, shifting from greed to generosity.
                It explores themes of morality, compassion, and social responsibility.
                """.trimIndent(),
            ),
        )

        val inputs =
            listOf(
                RagasInput(
                    question = "What themes does Dickens explore?",
                    groundTruths = listOf("poverty", "class struggle", "redemption", "morality", "compassion"),
                    mode = "hybrid",
                    id = "sample-1",
                ),
            )

        val evaluator = RagasEvaluator(rag)
        val samples = evaluator.evaluateToJsonl(inputs)
        val metrics = RagasMetrics()
        val scores = metrics.scoreAll(samples)

        println("Wrote ${samples.size} samples to ${RagasEvaluator.DEFAULT_OUTPUT_PATH}")
        scores.forEachIndexed { idx, score ->
            println(
                "Sample ${idx + 1}: answerRelevancy=${score.answerRelevancy} " +
                    "contextRecall=${score.contextRecall} contextPrecision=${score.contextPrecision} " +
                    "faithfulness=${score.faithfulness} answerCorrectness=${score.answerCorrectness} " +
                    "answerPrecision=${score.answerPrecision} answerRecall=${score.answerRecall} " +
                    "answerF1=${score.answerF1}",
            )
        }
    }
