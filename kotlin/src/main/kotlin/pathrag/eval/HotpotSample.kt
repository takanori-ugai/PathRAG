package pathrag.eval

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class HotpotSample(
    val id: String,
    val paragraphs: List<HotpotParagraph>,
    val question: String,
    @SerialName("question_decomposition")
    val questionDecomposition: List<HotpotDecomposition>,
    val answer: String,
    @SerialName("answer_aliases")
    val answerAliases: List<String>,
    val answerable: Boolean,
)

@Serializable
data class HotpotParagraph(
    val idx: Int,
    val title: String,
    @SerialName("paragraph_text")
    val paragraphText: String,
    @SerialName("is_supporting")
    val isSupporting: Boolean,
)

@Serializable
data class HotpotDecomposition(
    val id: Int,
    val question: String,
    val answer: String,
    @SerialName("paragraph_support_idx")
    val paragraphSupportIdx: Int,
)

fun main() {
    val inputPath = java.nio.file.Path.of("data/data/q1.json")
    val payload = java.nio.file.Files.readString(inputPath)
    val json = kotlinx.serialization.json.Json { ignoreUnknownKeys = true }
    val sample: HotpotSample = json.decodeFromString(payload)
    println(sample)
}
