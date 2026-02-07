package pathrag

import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import pathrag.base.QueryParam
import pathrag.eval.HotpotSample
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

fun main() {
    val inputPath: Path = Path.of("data/data/q1.json")
    val payload = Files.readString(inputPath)
    val json = Json { ignoreUnknownKeys = true }
    val sample: HotpotSample = json.decodeFromString(payload)

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
        )

    val paragraphs = sample.paragraphs.map { it.paragraphText }
    rag.insert(paragraphs)

    val queryAnswer = rag.query(sample.question, param = QueryParam(mode = "hybrid"))
    val expectedAnswer = sample.answer
    val matches = queryAnswer.trim().equals(expectedAnswer.trim(), ignoreCase = true)

    println("Question: ${sample.question}")
    println("Expected: $expectedAnswer")
    println("PathRAG: $queryAnswer")
    println("Match: $matches")
}
