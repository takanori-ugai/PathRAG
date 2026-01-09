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
     * Selects an encoding for the specified model, falling back to "cl100k_base" when no model-specific encoding exists.
     *
     * @param model The model name to resolve an encoding for; defaults to "gpt-4o-mini".
     * @return The resolved `Encoding` instance for the provided model or the `cl100k_base` encoding if none is registered for the model.
     */
    fun encoding(model: String = "gpt-4o-mini"): Encoding {
        val byModel = registry.getEncodingForModel(model)
        if (byModel.isPresent) return byModel.get()
        return registry.getEncoding("cl100k_base").get()
    }

    /**
     * Convert the given text into a list of token IDs using the encoding associated with the specified model.
     *
     * The function resolves an encoding for `model` and falls back to the "cl100k_base" encoding when a model-specific
     * encoding is not available.
     *
     * @param content The text to encode into tokens.
     * @param model The model name whose encoding should be used; defaults to "gpt-4o-mini".
     * @return A list of token IDs representing the encoded `content`.
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
     * Decode a sequence of token IDs into text using the encoding for the specified model.
     *
     * @param tokens List of token IDs to decode.
     * @param model Model name used to select the encoding; defaults to "gpt-4o-mini".
     * @return The decoded text.
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