package pathrag

import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import pathrag.eval.HotpotSample
import java.nio.file.Files
import java.nio.file.Path

fun main() {
    val inputPath: Path = Path.of("data/data/q1.json")
    val payload = Files.readString(inputPath)
    val json = Json { ignoreUnknownKeys = true }
    val sample: HotpotSample = json.decodeFromString(payload)
    println(sample)
}
