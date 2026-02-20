package pathrag

import pathrag.eval.RagasScores

data class SampleResult(
    val index: Int,
    val question: String,
    val expectedAnswer: String,
    val queryAnswer: String,
    val context: String,
    val matches: Boolean,
    val subEmCorrect: Int,
    val subEmTotal: Int,
    val scores: RagasScores,
)
