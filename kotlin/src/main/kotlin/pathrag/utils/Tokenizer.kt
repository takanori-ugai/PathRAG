package pathrag.utils

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.IntArrayList

/**
 * Minimal tokenizer wrapper backed by JTokkit encodings.
 */
object Tokenizer {
    private val registry = Encodings.newDefaultEncodingRegistry()

    /**
     * Get the token encoding for a model, falling back to "cl100k_base" when necessary.
     *
     * @param model Model identifier to resolve the encoding for; defaults to "gpt-4o-mini".
     * @return The Encoding for the specified model, or the "cl100k_base" encoding if no model-specific encoding is found.
     */
    fun encoding(model: String = "gpt-4o-mini"): Encoding {
        val byModel = registry.getEncodingForModel(model)
        if (byModel.isPresent) return byModel.get()
        return registry.getEncoding("cl100k_base").get()
    }

    /**
     * Converts the given text into a list of token identifier integers for the specified model.
     *
     * Encodes `content` using the model-specific encoding resolved from the internal registry; if the registry
     * does not have an encoding for `model`, the registry's "cl100k_base" encoding is used as a fallback.
     *
     * @param content The text to be tokenized.
     * @param model The model name whose encoding to use (defaults to "gpt-4o-mini"). If no encoding for this model
     *     exists in the registry, the "cl100k_base" encoding will be used.
     * @return A list of token IDs representing the encoded `content`.
     */
    fun encode(
        content: String,
        model: String = "gpt-4o-mini",
    ): List<Int> {
        val encoded: IntArrayList = encoding(model).encode(content)
        val arr = IntArray(encoded.size())
        for (i in 0 until encoded.size()) {
            arr[i] = encoded.get(i)
        }
        return arr.toList()
    }

    /**
     * Convert a sequence of token IDs into the decoded text using the specified model's encoding.
     *
     * @param tokens The token ID sequence to decode.
     * @param model The model name whose encoding should be used to decode the tokens; defaults to "gpt-4o-mini".
     * @return The decoded text produced from the provided token IDs.
     */
    fun decode(
        tokens: List<Int>,
        model: String = "gpt-4o-mini",
    ): String {
        val arr = tokens.toIntArray()
        val intArrayList = IntArrayList(arr.size)
        arr.forEach { intArrayList.add(it) }
        return encoding(model).decode(intArrayList)
    }
}