package pathrag.eval

import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import pathrag.PathRAG
import pathrag.base.QueryParam
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardOpenOption

@Serializable
data class RagasInput(
    val question: String,
    val groundTruths: List<String> = emptyList(),
    val mode: String? = null,
    val id: String? = null,
)

@Serializable
data class RagasSample(
    val question: String,
    val answer: String,
    val contexts: List<String>,
    val ground_truths: List<String> = emptyList(),
    val metadata: Map<String, String> = emptyMap(),
)

class RagasEvaluator(
    private val rag: PathRAG,
    private val json: Json = Json { encodeDefaults = false },
) {
    companion object {
        val DEFAULT_OUTPUT_PATH: Path = Paths.get("sample_cache/ragas_eval.jsonl")
    }

    fun evaluateToJsonl(
        inputs: List<RagasInput>,
        outputPath: Path = DEFAULT_OUTPUT_PATH,
        defaultMode: String = "hybrid",
        baseParam: QueryParam = QueryParam(),
    ): List<RagasSample> {
        val samples =
            inputs.map { input ->
                val mode = input.mode ?: defaultMode
                val answer = rag.query(input.question, baseParam.copy(mode = mode))
                val context =
                    rag.query(
                        input.question,
                        baseParam.copy(mode = mode, onlyNeedContext = true),
                    )
                val contexts = RagasContextExtractor.extractContexts(context)
                val metadata =
                    buildMap {
                        put("mode", mode)
                        input.id?.let { put("id", it) }
                    }
                RagasSample(
                    question = input.question,
                    answer = answer,
                    contexts = contexts,
                    ground_truths = input.groundTruths,
                    metadata = metadata,
                )
            }

        writeJsonl(samples, outputPath)
        return samples
    }

    private fun writeJsonl(
        samples: List<RagasSample>,
        outputPath: Path,
    ) {
        outputPath.parent?.let { Files.createDirectories(it) }
        val lines = samples.joinToString("\n") { json.encodeToString(it) }
        Files.writeString(
            outputPath,
            lines,
            StandardOpenOption.CREATE,
            StandardOpenOption.TRUNCATE_EXISTING,
        )
    }
}
