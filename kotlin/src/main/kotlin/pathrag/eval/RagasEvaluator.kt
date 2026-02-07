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
                val contexts = extractContexts(context)
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

    private fun extractContexts(context: String): List<String> {
        val csvBlocks = extractCsvBlocks(context)
        if (csvBlocks.isEmpty()) return emptyList()
        val collected = mutableListOf<String>()
        for (csv in csvBlocks) {
            val rows = parseCsv(csv)
            if (rows.isEmpty()) continue
            val header = rows.first()
            val contentIdx = header.indexOf("content")
            val contextIdx = header.indexOf("context")
            when {
                contentIdx != -1 -> {
                    rows
                        .drop(1)
                        .mapNotNull { row -> row.getOrNull(contentIdx) }
                        .map { it.trim() }
                        .filter { it.isNotBlank() }
                        .forEach { collected.add(it) }
                }

                contextIdx != -1 -> {
                    rows
                        .drop(1)
                        .mapNotNull { row -> row.getOrNull(contextIdx) }
                        .map { it.trim() }
                        .filter { it.isNotBlank() }
                        .forEach { collected.add(it) }
                }

                else -> {
                    rows
                        .drop(1)
                        .map { row -> row.joinToString(" | ") { it.trim() } }
                        .map { it.trim() }
                        .filter { it.isNotBlank() }
                        .forEach { collected.add(it) }
                }
            }
        }
        return collected.distinct()
    }

    private fun extractCsvBlocks(context: String): List<String> {
        val regex =
            Regex(
                "```csv\\s*([\\s\\S]*?)\\s*```",
                RegexOption.MULTILINE,
            )
        return regex.findAll(context).mapNotNull { it.groupValues.getOrNull(1)?.trim() }.toList()
    }

    private fun parseCsv(csv: String): List<List<String>> =
        csv
            .split('\n')
            .map { it.trimEnd() }
            .filter { it.isNotBlank() }
            .map { parseCsvLine(it) }

    private fun parseCsvLine(line: String): List<String> {
        val result = mutableListOf<String>()
        val current = StringBuilder()
        var inQuotes = false
        var i = 0
        while (i < line.length) {
            val ch = line[i]
            if (inQuotes) {
                if (ch == '"') {
                    val next = i + 1
                    if (next < line.length && line[next] == '"') {
                        current.append('"')
                        i++
                    } else {
                        inQuotes = false
                    }
                } else {
                    current.append(ch)
                }
            } else {
                when (ch) {
                    '"' -> {
                        inQuotes = true
                    }

                    ',' -> {
                        result.add(current.toString())
                        current.setLength(0)
                    }

                    else -> {
                        current.append(ch)
                    }
                }
            }
            i++
        }
        result.add(current.toString())
        return result
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
