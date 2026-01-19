package pathrag.eval

data class RagasScores(
    val answerRelevancy: Double,
    val contextRecall: Double,
    val contextPrecision: Double,
    val faithfulness: Double,
    val answerCorrectness: Double?,
)

class RagasMetrics(
    private val stopwords: Set<String> = DEFAULT_STOPWORDS,
) {
    fun score(sample: RagasSample): RagasScores {
        val questionTokens = tokenize(sample.question)
        val answerTokens = tokenize(sample.answer)
        val contextTokens = sample.contexts.map { tokenize(it) }
        val gtTokens = sample.ground_truths.map { tokenize(it) }

        val answerRelevancy = jaccard(questionTokens, answerTokens)
        val contextRecall = meanMaxSimilarity(gtTokens, contextTokens)
        val contextPrecision = meanMaxSimilarity(contextTokens, gtTokens)
        val faithfulness = maxSimilarity(answerTokens, contextTokens)
        val answerCorrectness =
            if (gtTokens.isEmpty()) {
                null
            } else {
                maxSimilarity(answerTokens, gtTokens)
            }

        return RagasScores(
            answerRelevancy = answerRelevancy,
            contextRecall = contextRecall,
            contextPrecision = contextPrecision,
            faithfulness = faithfulness,
            answerCorrectness = answerCorrectness,
        )
    }

    fun scoreAll(samples: List<RagasSample>): List<RagasScores> = samples.map { score(it) }

    private fun meanMaxSimilarity(
        sources: List<Set<String>>,
        targets: List<Set<String>>,
    ): Double {
        if (sources.isEmpty() || targets.isEmpty()) return 0.0
        val total = sources.sumOf { src -> maxSimilarity(src, targets) }
        return total / sources.size.toDouble()
    }

    private fun maxSimilarity(
        source: Set<String>,
        targets: List<Set<String>>,
    ): Double {
        if (source.isEmpty() || targets.isEmpty()) return 0.0
        var best = 0.0
        for (target in targets) {
            val score = jaccard(source, target)
            if (score > best) best = score
        }
        return best
    }

    private fun jaccard(
        a: Set<String>,
        b: Set<String>,
    ): Double {
        if (a.isEmpty() || b.isEmpty()) return 0.0
        val intersection = a.intersect(b).size.toDouble()
        val union = (a.size + b.size - intersection).toDouble()
        return if (union == 0.0) 0.0 else intersection / union
    }

    private fun tokenize(text: String): Set<String> =
        text
            .lowercase()
            .split(Regex("[^a-z0-9]+"))
            .filter { it.length > 1 }
            .filterNot { stopwords.contains(it) }
            .toSet()

    companion object {
        private val DEFAULT_STOPWORDS =
            setOf(
                "a",
                "an",
                "and",
                "are",
                "as",
                "at",
                "be",
                "but",
                "by",
                "for",
                "from",
                "has",
                "have",
                "he",
                "in",
                "is",
                "it",
                "its",
                "of",
                "on",
                "or",
                "she",
                "that",
                "the",
                "their",
                "they",
                "this",
                "to",
                "was",
                "were",
                "will",
                "with",
            )
    }
}
