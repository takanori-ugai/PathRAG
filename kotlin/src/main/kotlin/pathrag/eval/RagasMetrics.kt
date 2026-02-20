package pathrag.eval

data class RagasScores(
    val answerRelevancy: Double,
    val contextRecall: Double,
    val contextPrecision: Double,
    val faithfulness: Double,
    val answerCorrectness: Double?,
    val answerPrecision: Double?,
    val answerRecall: Double?,
    val answerF1: Double?,
)

class RagasMetrics(
    private val stopwords: Set<String> = DEFAULT_STOPWORDS,
) {
    fun score(sample: RagasSample): RagasScores {
        val questionTokens = tokenize(sample.question)
        val answerTokens = tokenize(sample.answer)
        val contextTokens = sample.contexts.map { tokenize(it) }
        val gtTokens = sample.groundTruths.map { tokenize(it) }

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
        val answerTokenSet = answerTokens
        val gtTokenSet =
            gtTokens.fold(emptySet<String>()) { acc, tokens ->
                acc + tokens
            }
        val answerPrecision =
            if (gtTokenSet.isEmpty()) {
                null
            } else {
                precision(answerTokenSet, gtTokenSet)
            }
        val answerRecall =
            if (gtTokenSet.isEmpty()) {
                null
            } else {
                recall(answerTokenSet, gtTokenSet)
            }
        val answerF1 =
            if (answerPrecision == null || answerRecall == null) {
                null
            } else {
                f1(answerPrecision, answerRecall)
            }

        return RagasScores(
            answerRelevancy = answerRelevancy,
            contextRecall = contextRecall,
            contextPrecision = contextPrecision,
            faithfulness = faithfulness,
            answerCorrectness = answerCorrectness,
            answerPrecision = answerPrecision,
            answerRecall = answerRecall,
            answerF1 = answerF1,
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
        val union = a.size + b.size - intersection
        return if (union == 0.0) 0.0 else intersection / union
    }

    private fun precision(
        answer: Set<String>,
        groundTruth: Set<String>,
    ): Double {
        if (answer.isEmpty()) return 0.0
        val intersection = answer.intersect(groundTruth).size.toDouble()
        return intersection / answer.size.toDouble()
    }

    private fun recall(
        answer: Set<String>,
        groundTruth: Set<String>,
    ): Double {
        if (groundTruth.isEmpty()) return 0.0
        val intersection = answer.intersect(groundTruth).size.toDouble()
        return intersection / groundTruth.size.toDouble()
    }

    private fun f1(
        precision: Double,
        recall: Double,
    ): Double {
        if (precision == 0.0 && recall == 0.0) return 0.0
        return 2.0 * precision * recall / (precision + recall)
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
