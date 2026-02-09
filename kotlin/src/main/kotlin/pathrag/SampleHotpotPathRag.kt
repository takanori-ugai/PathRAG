package pathrag

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import pathrag.base.QueryParam
import pathrag.eval.HotpotSample
import pathrag.eval.RagasContextExtractor
import pathrag.eval.RagasMetrics
import pathrag.eval.RagasSample
import pathrag.eval.RagasScores
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

data class SampleResult(
    val index: Int,
    val question: String,
    val expectedAnswer: String,
    val queryAnswer: String,
    val context: String,
    val matches: Boolean,
    val scores: RagasScores,
)

fun main(args: Array<String>) =
    runBlocking {
        val inputPath: Path = Path.of("data/data/musique_ans_v1.0_train-200.jsonl")
        val json = Json { ignoreUnknownKeys = true }

        val env = EnvironmentConfig.load(Paths.get("../.env"))
        val kvStorage = env["KV_STORAGE"] ?: "JsonKVStorage"
        val vectorStorage = env["VECTOR_STORAGE"] ?: "NanoVectorDBStorage"
        val graphStorage = env["GRAPH_STORAGE"] ?: "NetworkXStorage"
        val workingDir = env["WORKING_DIR"] ?: "./sample_cache"
        val configuredParallelism =
            args.firstOrNull()?.toIntOrNull()
                ?: env["PATHRAG_PARALLELISM"]?.toIntOrNull()
                ?: env["PARALLELISM"]?.toIntOrNull()
                ?: 1
        val parallelism = configuredParallelism.coerceAtLeast(1)

        var total = 0
        var correct = 0
        var sumAnswerRelevancy = 0.0
        var sumContextRecall = 0.0
        var sumContextPrecision = 0.0
        var sumFaithfulness = 0.0
        var sumAnswerCorrectness = 0.0
        var answerCorrectnessCount = 0
        var sumAnswerPrecision = 0.0
        var sumAnswerRecall = 0.0
        var sumAnswerF1 = 0.0
        var answerMetricCount = 0
        val results =
            Files.newBufferedReader(inputPath).useLines { lines ->
                val semaphore = Semaphore(parallelism)
                val deferred =
                    lines
                        .filter { it.isNotBlank() }
                        .mapIndexed { index, line ->
                            async(Dispatchers.IO) {
                                semaphore.withPermit {
                                    processSample(
                                        index = index,
                                        line = line,
                                        json = json,
                                        kvStorage = kvStorage,
                                        vectorStorage = vectorStorage,
                                        graphStorage = graphStorage,
                                        workingDir = workingDir,
                                    )
                                }
                            }
                        }.toList()
                deferred.awaitAll()
            }
        results.sortedBy { it.index }.forEach { result ->
            total += 1
            if (result.matches) {
                correct += 1
            }

            sumAnswerRelevancy += result.scores.answerRelevancy
            sumContextRecall += result.scores.contextRecall
            sumContextPrecision += result.scores.contextPrecision
            sumFaithfulness += result.scores.faithfulness
            if (result.scores.answerCorrectness != null) {
                sumAnswerCorrectness += result.scores.answerCorrectness
                answerCorrectnessCount += 1
            }
            if (result.scores.answerPrecision != null &&
                result.scores.answerRecall != null &&
                result.scores.answerF1 != null
            ) {
                sumAnswerPrecision += result.scores.answerPrecision
                sumAnswerRecall += result.scores.answerRecall
                sumAnswerF1 += result.scores.answerF1
                answerMetricCount += 1
            }

            println("Sample: ${result.index + 1}")
            println("Context: ${result.context}")
            println("Question: ${result.question}")
            println("Expected: ${result.expectedAnswer}")
            println("PathRAG: ${result.queryAnswer}")
            println("Match: ${result.matches}")
            println("Accuracy: $correct/$total")
            println(
                "RAGAS: relevancy=${String.format("%.3f", result.scores.answerRelevancy)}, " +
                    "ctx_recall=${String.format("%.3f", result.scores.contextRecall)}, " +
                    "ctx_precision=${String.format("%.3f", result.scores.contextPrecision)}, " +
                    "faithfulness=${String.format("%.3f", result.scores.faithfulness)}, " +
                    "answer_correctness=${result.scores.answerCorrectness?.let { String.format("%.3f", it) } ?: "n/a"}, " +
                    "answer_precision=${result.scores.answerPrecision?.let { String.format("%.3f", it) } ?: "n/a"}, " +
                    "answer_recall=${result.scores.answerRecall?.let { String.format("%.3f", it) } ?: "n/a"}, " +
                    "answer_f1=${result.scores.answerF1?.let { String.format("%.3f", it) } ?: "n/a"}",
            )
        }
        if (total > 0) {
            val accuracy = correct.toDouble() / total.toDouble()
            println("Summary: $correct/$total (${String.format("%.2f", accuracy * 100)}%)")
            println(
                "RAGAS Summary: " +
                    "relevancy=${String.format("%.3f", sumAnswerRelevancy / total)}, " +
                    "ctx_recall=${String.format("%.3f", sumContextRecall / total)}, " +
                    "ctx_precision=${String.format("%.3f", sumContextPrecision / total)}, " +
                    "faithfulness=${String.format("%.3f", sumFaithfulness / total)}, " +
                    "answer_correctness=${
                        if (answerCorrectnessCount > 0) {
                            String.format("%.3f", sumAnswerCorrectness / answerCorrectnessCount)
                        } else {
                            "n/a"
                        }
                    }, " +
                    "answer_precision=${
                        if (answerMetricCount > 0) {
                            String.format("%.3f", sumAnswerPrecision / answerMetricCount)
                        } else {
                            "n/a"
                        }
                    }, " +
                    "answer_recall=${
                        if (answerMetricCount > 0) {
                            String.format("%.3f", sumAnswerRecall / answerMetricCount)
                        } else {
                            "n/a"
                        }
                    }, " +
                    "answer_f1=${
                        if (answerMetricCount > 0) {
                            String.format("%.3f", sumAnswerF1 / answerMetricCount)
                        } else {
                            "n/a"
                        }
                    }",
            )
        } else {
            println("Summary: 0/0 (no samples processed)")
        }
    }

private suspend fun processSample(
    index: Int,
    line: String,
    json: Json,
    kvStorage: String,
    vectorStorage: String,
    graphStorage: String,
    workingDir: String,
): SampleResult {
    println("Processing sample: ${index + 1}")
    val sample: HotpotSample = json.decodeFromString(line)
    val metrics = RagasMetrics()
    val sampleWorkingDir = Path.of(workingDir, "sample_${index + 1}").toString()
    val rag =
        PathRAG(
            workingDir = sampleWorkingDir,
            kvStorage = kvStorage,
            vectorStorage = vectorStorage,
            graphStorage = graphStorage,
        )

    try {
        rag.clear()
        val paragraphs = sample.paragraphs.map { it.paragraphText }
        rag.insert(paragraphs)

        val queryAnswer =
            rag.aquery(
                "Answer in one or few words, no extra information: ${sample.question}",
                param = QueryParam(mode = "hybrid"),
            )
        val context =
            rag.aquery(
                "Answer in one or few words, no extra information: ${sample.question}",
                param = QueryParam(mode = "hybrid", onlyNeedContext = true),
            )
        val expectedAnswer = sample.answer
        val matches = queryAnswer.trim().equals(expectedAnswer.trim(), ignoreCase = true)

        val ragasSample =
            RagasSample(
                question = sample.question,
                answer = queryAnswer,
                contexts = RagasContextExtractor.extractContexts(context),
                ground_truths = listOf(expectedAnswer),
            )
        val scores = metrics.score(ragasSample)

        return SampleResult(
            index = index,
            question = sample.question,
            expectedAnswer = expectedAnswer,
            queryAnswer = queryAnswer,
            context = context,
            matches = matches,
            scores = scores,
        )
    } finally {
        rag.close()
    }
}
