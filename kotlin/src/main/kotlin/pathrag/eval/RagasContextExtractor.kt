package pathrag.eval

import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVParser
import java.io.StringReader

object RagasContextExtractor {
    fun extractContexts(context: String): List<String> {
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
        StringReader(csv).use { reader ->
            val format =
                CSVFormat.DEFAULT
                    .builder()
                    .setIgnoreEmptyLines(true)
                    .build()
            CSVParser.parse(reader, format).use { parser ->
                return parser.records.map { record -> record.toList() }
            }
        }
}
