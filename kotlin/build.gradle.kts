import org.jlleitschuh.gradle.ktlint.reporter.ReporterType
import java.io.File

plugins {
    kotlin("jvm") version "2.3.20"
    application
    id("com.gradleup.shadow") version "9.4.1"
    kotlin("plugin.serialization") version "2.3.20"
    id("org.jlleitschuh.gradle.ktlint") version "14.2.0"
    id("io.gitlab.arturbosch.detekt") version "1.23.8"
}

detekt {
    buildUponDefaultConfig = true
    config.setFrom(files("config/detekt.yml"))
}

group = "com.pathrag"
version = "0.0.1"

val ktorVersion = "3.4.3"
val koinVersion = "4.2.1"
val javafxVersion = "21.0.5"
val osName = System.getProperty("os.name").lowercase()
val archName = System.getProperty("os.arch").lowercase()
val javafxPlatform =
    when {
        osName.contains("mac") && archName.contains("aarch64") -> "mac-aarch64"
        osName.contains("mac") -> "mac"
        osName.contains("win") && archName.contains("aarch64") -> "win-aarch64"
        osName.contains("win") -> "win"
        archName.contains("aarch64") || archName.contains("arm64") -> "linux-aarch64"
        else -> "linux"
    }

application {
    val requestedMain =
        if (project.hasProperty("mainClass")) {
            project.property("mainClass") as String
        } else {
            null
        }
    mainClass.set(requestedMain ?: "pathrag.ApplicationKt")
}

repositories {
    mavenCentral()
    gradlePluginPortal()
}

dependencies {
    implementation(platform("io.ktor:ktor-bom:$ktorVersion"))
    implementation("io.ktor:ktor-server-core")
    implementation("io.ktor:ktor-server-netty")
    implementation("io.ktor:ktor-server-content-negotiation")
    implementation("io.ktor:ktor-serialization-kotlinx-json")
    implementation("io.ktor:ktor-server-swagger")
    implementation("io.ktor:ktor-server-cors")
    implementation("ch.qos.logback:logback-classic:1.5.32")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.10.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
    implementation("io.github.oshai:kotlin-logging-jvm:8.0.01")
    implementation("com.auth0:java-jwt:4.5.1")
    implementation("org.apache.commons:commons-csv:1.14.1")

    // Koin for Ktor
    implementation("io.insert-koin:koin-ktor:$koinVersion")
    implementation("io.insert-koin:koin-core:$koinVersion")
    implementation("io.insert-koin:koin-logger-slf4j:$koinVersion")

    // LangChain4j dependencies
    implementation("dev.langchain4j:langchain4j:1.13.1")
    implementation("dev.langchain4j:langchain4j-open-ai:1.13.1")
    implementation("dev.langchain4j:langchain4j-ollama:1.13.1")
    implementation("dev.langchain4j:langchain4j-community-neo4j:1.13.1-beta23")

    implementation("org.openjfx:javafx-base:$javafxVersion:$javafxPlatform")
    implementation("org.openjfx:javafx-graphics:$javafxVersion:$javafxPlatform")
    implementation("org.openjfx:javafx-controls:$javafxVersion:$javafxPlatform")

    // JTokkit
    implementation("com.knuddels:jtokkit:1.1.0")

    // MongoDB
    implementation("org.mongodb:mongodb-driver-kotlin-coroutine:5.6.5")
    implementation("org.mongodb:bson-kotlinx:5.6.5")
    implementation("org.neo4j.driver:neo4j-java-driver:6.0.3")

    testImplementation("org.jetbrains.kotlin:kotlin-test-junit:2.3.20")
    testImplementation("io.mockk:mockk:1.14.9")
}

tasks {
    withType<Test> {
        jvmArgs("-XX:+EnableDynamicAgentLoading")
        testLogging {
            events("passed", "skipped", "failed", "standardOut", "standardError")
            showStandardStreams = true
        }
    }

    val execute by registering(JavaExec::class) {
        group = "application"
        mainClass.set(
            if (project.hasProperty("mainClass")) {
                project.property("mainClass") as String
            } else {
                application.mainClass.get()
            },
        )
        classpath = sourceSets.main.get().runtimeClasspath
    }
}

// Configure JavaFX runtime options for the run/execute tasks to avoid module warnings and missing modules.
tasks.withType<JavaExec>().configureEach {
    doFirst {
        val javafxJars =
            configurations.runtimeClasspath
                .get()
                .filter { it.name.startsWith("javafx-") }
                .files
        if (javafxJars.isNotEmpty()) {
            jvmArgs(
                "--module-path",
                javafxJars.joinToString(File.pathSeparator),
                "--add-modules",
                "javafx.controls,javafx.graphics",
            )
        }
    }
}

ktlint {
    version.set("1.8.0")
    verbose.set(true)
    outputToConsole.set(true)
    coloredOutput.set(true)
    reporters {
        reporter(ReporterType.CHECKSTYLE)
        reporter(ReporterType.JSON)
        reporter(ReporterType.HTML)
    }
    filter {
        exclude("**/style-violations.kt")
    }
}
