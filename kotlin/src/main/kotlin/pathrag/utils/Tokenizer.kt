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
     * Resolve an encoding for a given model, falling back to cl100k_base.
     */
    fun encoding(model: String = "gpt-4o-mini"): Encoding {
        val byModel = registry.getEncodingForModel(model)
        if (byModel.isPresent) return byModel.get()
        return registry.getEncoding("cl100k_base").get()
    }

    /**
     * Encode content into token ids for the chosen model.
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
     * Decode token ids back into text for the chosen model.
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
