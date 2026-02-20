package pathrag

import com.auth0.jwt.JWT
import com.auth0.jwt.algorithms.Algorithm
import com.auth0.jwt.exceptions.JWTVerificationException
import com.auth0.jwt.interfaces.JWTVerifier
import io.github.oshai.kotlinlogging.KotlinLogging
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpMethod
import io.ktor.http.HttpStatusCode
import io.ktor.http.charset
import io.ktor.http.content.PartData
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.ApplicationCall
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.http.content.staticResources
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.cors.routing.CORS
import io.ktor.server.plugins.swagger.swaggerUI
import io.ktor.server.request.receive
import io.ktor.server.request.receiveMultipart
import io.ktor.server.response.respond
import io.ktor.server.response.respondRedirect
import io.ktor.server.routing.Route
import io.ktor.server.routing.delete
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.put
import io.ktor.server.routing.route
import io.ktor.server.routing.routing
import io.ktor.utils.io.core.readBytes
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import pathrag.PathRAG
import pathrag.base.QueryParam
import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.security.MessageDigest
import java.time.Instant
import java.util.Date
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

private val logger = KotlinLogging.logger("pathrag")

/**
 * Starts the PathRAG application server using configuration from ../.env.
 *
 * Loads environment configuration, determines host and port (with defaults),
 * logs startup, and launches an embedded Netty server that mounts the application module.
 */
fun main() {
    val env = EnvironmentConfig.load(Paths.get("../.env"))
    val host = env["HOST"] ?: "0.0.0.0"
    val port = env["PORT"]?.toIntOrNull() ?: 8001

    logger.info { "Starting PathRAG API on $host:$port" }
    embeddedServer(Netty, port = port, host = host) {
        module(env)
    }.start(wait = true)
}

/**
 * Configures the Ktor Application: installs features, prepares storage and repositories, configures token handling,
 * preloads default data, and registers all HTTP routes used by the application.
 *
 * @param env Environment configuration providing runtime settings (loaded from .env and process environment); defaults
 *            to an empty config when not provided.
 */
fun Application.module(env: EnvironmentConfig = EnvironmentConfig.empty()) {
    warnOnMissingRequiredVars(env)
    install(ContentNegotiation) {
        json()
    }
    install(CORS) {
        val origins = env.corsOrigins()
        if (origins == "*") {
            allowOrigins { true }
            logger.warn { "CORS is configured to allow all origins. Avoid this configuration in production." }
        } else {
            val allowed =
                origins
                    .split(",")
                    .map { it.trim() }
                    .filter { it.isNotBlank() }
            allowOrigins { origin -> allowed.any { origin.contains(it, ignoreCase = true) } }
        }
        allowCredentials = true
        allowMethod(HttpMethod.Get)
        allowMethod(HttpMethod.Post)
        allowMethod(HttpMethod.Put)
        allowMethod(HttpMethod.Delete)
        allowHeader(HttpHeaders.Authorization)
        allowHeader(HttpHeaders.ContentType)
    }

    val workingDir = env["WORKING_DIR"] ?: "./data"
    val uploadDir = env["UPLOAD_DIR"] ?: "./uploads"
    createDirectories(listOf(workingDir, uploadDir))

    val kvStorage = env["KV_STORAGE"] ?: "JsonKVStorage"
    val vectorStorage = env["VECTOR_STORAGE"] ?: "NanoVectorDBStorage"
    val graphStorage = env["GRAPH_STORAGE"] ?: "NetworkXStorage"
    val chunkTokenSize = env.chunkTokenSize()
    val chunkOverlapTokenSize = env.chunkOverlapTokenSize()
    val extraConfig =
        pathrag.base.ExtraConfig(
            neo4jUri = env["NEO4J_URI"],
            neo4jUser = env["NEO4J_USER"],
            neo4jPassword = env["NEO4J_PASSWORD"],
            mongoUri = env["MONGO_URI"],
            mongoDatabase = env["MONGO_DATABASE"],
        )
    if (kvStorage.contains("Mongo") || vectorStorage.contains("Mongo") || graphStorage.contains("Mongo")) {
        val mongoUri = extraConfig.mongoUri
        val mongoDb = extraConfig.mongoDatabase
        if (mongoUri.isNullOrBlank() || mongoDb.isNullOrBlank()) {
            logger.warn { "Mongo storage selected but MONGO_URI or MONGO_DATABASE is not configured; falling back to defaults." }
        }
    }

    val rag =
        PathRAG(
            workingDir = workingDir,
            kvStorage = kvStorage,
            vectorStorage = vectorStorage,
            graphStorage = graphStorage,
            chunkTokenSize = chunkTokenSize,
            chunkOverlapTokenSize = chunkOverlapTokenSize,
            extraConfig = extraConfig,
        )
    val userRepository = UserRepository(Paths.get(workingDir, "users.json"))
    val chatRepository = ChatRepository(Paths.get(workingDir, "chats.json"))
    val documentRepository = DocumentRepository(uploadDir, Paths.get(workingDir, "documents.json"))
    TokenService.configure(env)
    runBlocking { createDefaultUsers(userRepository) }
    runBlocking { preloadKnowledgeGraphIfEmpty(rag, documentRepository) }

    routing {
        get("/") { call.respondRedirect("/swagger") }
        swaggerUI(path = "swagger", swaggerFile = "openapi/documentation.yaml")
        staticResources("/ui", "static")
        get("/app") { call.respondRedirect("/ui/index.html") }
        authRoutes(userRepository)
        userRoutes(userRepository)
        chatRoutes(userRepository, chatRepository)
        documentRoutes(documentRepository, rag, userRepository)
        knowledgeGraphRoutes(rag, userRepository)
        // query endpoint relocated under /documents/query
    }
}

private fun warnOnMissingRequiredVars(env: EnvironmentConfig) {
    val missing =
        buildList {
            if (!SecretKeyLoader.hasSecret(env)) add("SECRET_KEY_FILE (preferred) or SECRET_KEY")
        }
    if (missing.isNotEmpty()) {
        logger.warn { "Missing required environment variables: ${missing.joinToString(", ")}" }
        logger.warn { "Please set these variables in your .env file or environment. See sample.env for an example configuration." }
    }
}

private fun createDirectories(paths: List<String>) {
    paths.forEach { path ->
        runCatching {
            Files.createDirectories(Paths.get(path))
        }.onSuccess {
            logger.info { "Ensured directory exists: $path" }
        }.onFailure { ex ->
            logger.error(ex) { "Failed to create directory: $path" }
        }
    }
}

/**
 * In-memory repository used to mirror the Python default user creation.
 * Replace with a real database implementation when available.
 */
class UserRepository(
    private val filePath: Path? = null,
) {
    private val users = mutableListOf<User>()
    private var nextId = 1
    private val mutex = Mutex()
    private val json =
        Json {
            prettyPrint = true
            ignoreUnknownKeys = true
        }

    @Volatile
    private var initialized = false

    /**
     * Get the number of stored users, loading persisted users if necessary.
     *
     * @return The number of stored users.
     */
    suspend fun count(): Int {
        ensureLoaded()
        return mutex.withLock { users.size }
    }

    /**
     * Adds a new user or replaces an existing user in the repository.
     *
     * If the supplied user's `id` is null, a new unique id is assigned. The repository state is persisted to disk.
     *
     * @param user The user to add or update; may have a null `id` to request creation.
     * @return The stored `User` instance including the assigned `id`.
     */
    suspend fun add(user: User): User {
        ensureLoaded()
        return mutex.withLock {
            val stored =
                if (user.id == null) user.copy(id = nextId++) else user.copy(id = user.id)
            users.add(stored)
            persist()
            stored
        }
    }

    /**
     * Retrieve the user with the given username.
     *
     * @return The matching `User` if one exists, `null` otherwise.
     */
    suspend fun find(username: String): User? {
        ensureLoaded()
        return mutex.withLock { users.find { it.username == username } }
    }

    /**
     * Update the stored theme preference for the specified user.
     *
     * Updates the user's `theme`, sets `updatedAt` to the current time, persists the change,
     * and returns the updated user record.
     *
     * @param username The username whose theme will be updated.
     * @param theme The new theme value to set for the user.
     * @return The updated `User` if a matching user was found, `null` if no user with the given username exists.
     */
    suspend fun updateTheme(
        username: String,
        theme: String,
    ): User? {
        ensureLoaded()
        return mutex.withLock {
            val idx = users.indexOfFirst { it.username == username }
            if (idx == -1) return@withLock null
            val updated = users[idx].copy(theme = theme, updatedAt = Instant.now().toString())
            users[idx] = updated
            persist()
            updated
        }
    }

    /**
     * Retrieve all users currently stored in the repository as a snapshot list.
     *
     * @return A list containing all `User` objects present in the repository.
     */
    suspend fun list(): List<User> {
        ensureLoaded()
        return mutex.withLock { users.toList() }
    }

    /**
     * Ensures the repository's users are loaded from persistent storage into memory exactly once.
     *
     * If a persistence file is configured and readable, parses its contents and replaces the in-memory
     * user list and updates the next available id. Safe to call concurrently; subsequent calls are
     * no-ops after the first successful or attempted load.
     *
     * If the persistence file is missing, the in-memory store remains empty. If parsing fails, a
     * warning is logged and the in-memory store is not modified.
     */
    suspend fun ensureLoaded() {
        if (initialized) return
        mutex.withLock {
            if (initialized) return
            val file = filePath?.toFile()
            val parsed =
                if (file != null && file.exists()) {
                    withContext(Dispatchers.IO) {
                        runCatching { json.decodeFromString<List<User>>(file.readText()) }
                            .onFailure { ex -> logger.warn(ex) { "Failed to load users from ${file.absolutePath}" } }
                            .getOrNull()
                    }
                } else {
                    emptyList()
                }
            if (!parsed.isNullOrEmpty()) {
                users.clear()
                users.addAll(parsed)
                nextId = (users.maxOfOrNull { it.id ?: 0 } ?: 0) + 1
            }
            initialized = true
        }
    }

    /**
     * Persists the in-memory user list to the configured file path.
     *
     * If no file path is configured this is a no-op. Creates parent directories as needed,
     * writes the users as JSON, and logs a warning if writing fails.
     */
    private suspend fun persist() {
        val file = filePath?.toFile() ?: return
        withContext(Dispatchers.IO) {
            runCatching {
                file.parentFile?.mkdirs()
                file.writeText(json.encodeToString(users))
            }.onFailure { ex ->
                logger.warn(ex) { "Failed to persist users to ${file.absolutePath}" }
            }
        }
    }
}

/**
 * Basic user record persisted for demo authentication.
 *
 * @property id numeric user id (auto-assigned when null).
 * @property username login identifier.
 * @property email contact email.
 * @property hashedPassword SHA-256 hashed password string.
 * @property createdAt creation timestamp.
 * @property updatedAt last updated timestamp.
 * @property theme UI theme preference.
 */
@Serializable
data class User(
    val id: Int? = null,
    val username: String,
    val email: String,
    val hashedPassword: String,
    val createdAt: String = Instant.now().toString(),
    val updatedAt: String = Instant.now().toString(),
    val theme: String = "blue",
)

private val defaultUsersCreated = AtomicBoolean(false)

private suspend fun createDefaultUsers(repository: UserRepository) {
    if (defaultUsersCreated.get()) return

    if (repository.count() == 0) {
        val defaults =
            listOf(
                DefaultUser("user1", "user1@example.com", "Pass@123"),
                DefaultUser("user2", "user2@example.com", "Pass@123"),
                DefaultUser("user3", "user3@example.com", "Pass@123"),
            )

        defaults.forEach { user ->
            val hashed = PasswordHasher.hash(user.password)
            repository.add(
                User(
                    id = null,
                    username = user.username,
                    email = user.email,
                    hashedPassword = hashed,
                ),
            )
        }
        defaultUsersCreated.set(true)
        logger.info { "Default users created successfully" }
    }
}

private data class DefaultUser(
    val username: String,
    val email: String,
    val password: String,
)

/**
 * Simple password hashing that mirrors the intent of get_password_hash in the Python code.
 * Swap this out for a stronger hashing strategy (e.g., bcrypt) when integrating authentication.
 */
object PasswordHasher {
    /**
     * Compute the SHA-256 digest of the input and return it as a lowercase hexadecimal string.
     *
     * @return Lowercase hexadecimal SHA-256 digest of `input`.
     */
    fun hash(input: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val bytes = digest.digest(input.toByteArray())
        return bytes.joinToString(separator = "") { "%02x".format(it) }
    }
}

private object SecretKeyLoader {
    fun load(env: EnvironmentConfig): ByteArray {
        val secretFromFile =
            env["SECRET_KEY_FILE"]?.takeIf { it.isNotBlank() }?.let { pathString ->
                val path = runCatching { Paths.get(pathString) }.getOrNull()
                if (path == null) {
                    logger.error { "SECRET_KEY_FILE path is invalid: $pathString" }
                    null
                } else {
                    runCatching { Files.readString(path).trim() }
                        .onFailure { ex -> logger.error(ex) { "Failed to read SECRET_KEY_FILE at $pathString" } }
                        .getOrNull()
                        ?.takeIf { it.isNotBlank() }
                }
            }
        if (secretFromFile != null) {
            return secretFromFile.toByteArray(StandardCharsets.UTF_8)
        }

        val envSecret = env["SECRET_KEY"]?.takeIf { it.isNotBlank() }
        if (envSecret != null) {
            return envSecret.toByteArray(StandardCharsets.UTF_8)
        }

        logger.warn {
            "No SECRET_KEY_FILE configured and SECRET_KEY missing. Generated ephemeral secret; tokens will be invalidated on restart."
        }
        return UUID.randomUUID().toString().toByteArray(StandardCharsets.UTF_8)
    }

    /**
     * Determines whether a signing secret is available in the provided environment configuration.
     *
     * Checks first for `SECRET_KEY_FILE` pointing to an existing file whose trimmed contents are not blank;
     * if not present, checks that `SECRET_KEY` is not blank.
     *
     * @param env EnvironmentConfig to inspect for secret configuration variables.
     * @return `true` if a non-blank secret is available via `SECRET_KEY_FILE` (file exists and contains non-blank text)
     *         or via `SECRET_KEY`, `false` otherwise.
     */
    fun hasSecret(env: EnvironmentConfig): Boolean {
        val fileSecretPresent =
            env["SECRET_KEY_FILE"]?.takeIf { it.isNotBlank() }?.let { pathString ->
                val path = runCatching { Paths.get(pathString) }.getOrNull() ?: return@let false
                if (!Files.exists(path)) return@let false
                val content = runCatching { Files.readString(path).trim() }.getOrNull()
                !content.isNullOrBlank()
            } ?: false

        return fileSecretPresent || !env["SECRET_KEY"].isNullOrBlank()
    }
}

/**
 * JWT token helper for issuing and validating access tokens.
 */
object TokenService {
    private const val DEFAULT_TOKEN_TTL_MINUTES = 30L

    @Volatile
    private var secret: ByteArray? = null

    @Volatile
    private var tokenTtlSeconds: Long = DEFAULT_TOKEN_TTL_MINUTES * 60

    @Volatile
    private var issuer: String = "pathrag"

    @Volatile
    private var algorithm: Algorithm? = null

    @Volatile
    private var verifier: JWTVerifier? = null

    /**
     * Configure TokenService secret, issuer, and token TTL from the provided environment.
     *
     * Reads `ACCESS_TOKEN_EXPIRE_MINUTES` to set the token TTL (in seconds) and `TOKEN_ISSUER` to set the issuer.
     * If a secret is not already set, it is loaded via SecretKeyLoader.
     *
     * @param env EnvironmentConfig containing the relevant environment variables.
     */
    fun configure(env: EnvironmentConfig) {
        if (secret == null) {
            secret = SecretKeyLoader.load(env)
        }
        tokenTtlSeconds =
            env["ACCESS_TOKEN_EXPIRE_MINUTES"]?.toLongOrNull()?.takeIf { it > 0 }?.times(60)
                ?: DEFAULT_TOKEN_TTL_MINUTES * 60
        issuer = env["TOKEN_ISSUER"] ?: "pathrag"
        rebuildCrypto()
    }

    private fun secret(): ByteArray =
        secret ?: synchronized(this) {
            secret ?: SecretKeyLoader.load(EnvironmentConfig.empty()).also { secret = it }
        }

    private fun rebuildCrypto() =
        synchronized(this) {
            val alg = Algorithm.HMAC256(secret())
            algorithm = alg
            verifier = JWT.require(alg).withIssuer(issuer).build()
        }

    private fun ensureAlgorithm(): Algorithm =
        algorithm ?: run {
            rebuildCrypto()
            algorithm!!
        }

    /**
     * Ensure a `JWTVerifier` instance is available, creating and returning one if none exists.
     *
     * @return The initialized `JWTVerifier` instance.
     */
    private fun ensureVerifier(): JWTVerifier =
        verifier ?: run {
            rebuildCrypto()
            verifier!!
        }

    /**
     * Issue a signed JWT for a username.
     *
     * The token contains the configured issuer, the provided username as the subject,
     * issuance and expiration times based on the configured TTL, and a unique JWT ID.
     *
     * @return The signed JWT as a String.
     */
    fun issueToken(username: String): String {
        val issuedAt = Instant.now()
        val expiresAt = issuedAt.plusSeconds(tokenTtlSeconds)
        val algorithm = ensureAlgorithm()
        return JWT
            .create()
            .withIssuer(issuer)
            .withSubject(username)
            .withIssuedAt(Date.from(issuedAt))
            .withExpiresAt(Date.from(expiresAt))
            .withJWTId(UUID.randomUUID().toString())
            .sign(algorithm)
    }

    /**
     * Extracts the username contained in the JWT subject.
     *
     * @param token The JWT as a string, or `null`.
     * @return The username from the token's `subject`, or `null` if the token is missing, invalid, or expired.
     */
    fun usernameFromToken(token: String?): String? =
        try {
            ensureVerifier().verify(token).subject
        } catch (ex: JWTVerificationException) {
            logger.warn(ex) { "Invalid or expired token" }
            null
        }
}

@Serializable
private data class LoginRequest(
    val username: String,
    val password: String,
)

@Serializable
private data class RegisterRequest(
    val username: String,
    val email: String,
    val password: String,
)

@Serializable
private data class ThemeRequest(
    val username: String,
    val theme: String,
)

/**
 * Chat thread containing messages for a user.
 *
 * @property id numeric thread id.
 * @property uuid external thread identifier.
 * @property userId owner id.
 * @property title thread title.
 * @property createdAt creation timestamp.
 * @property updatedAt last updated timestamp.
 * @property isDeleted soft-delete flag.
 * @property chats messages in the thread.
 */
@Serializable
data class ChatThread(
    val id: Int,
    val uuid: String,
    val userId: Int,
    val title: String,
    val createdAt: String = Instant.now().toString(),
    val updatedAt: String = Instant.now().toString(),
    val isDeleted: Boolean = false,
    val chats: List<ChatMessage> = emptyList(),
)

/**
 * Individual chat message within a thread.
 *
 * @property id message id.
 * @property threadId owning thread id.
 * @property userId author id.
 * @property role sender role.
 * @property message message content.
 * @property createdAt creation timestamp.
 */
@Serializable
data class ChatMessage(
    val id: Int,
    val threadId: Int,
    val userId: Int,
    val role: String = "user",
    val message: String,
    val createdAt: String = Instant.now().toString(),
)

/**
 * Metadata for an uploaded document.
 *
 * @property id document id.
 * @property userId uploader id.
 * @property filename original name.
 * @property contentType mime type.
 * @property filePath path on disk.
 * @property fileSize file size in bytes.
 * @property uploadedAt upload timestamp.
 * @property status processing status.
 * @property processedAt processing timestamp.
 * @property errorMessage processing error message if any.
 */
@Serializable
data class DocumentInfo(
    val id: Int,
    val userId: Int,
    val filename: String,
    val contentType: String,
    val filePath: String,
    val fileSize: Long,
    val uploadedAt: String = Instant.now().toString(),
    val status: String = "uploaded",
    val processedAt: String? = null,
    val errorMessage: String? = null,
)

/**
 * In-memory chat/thread repository with optional persistence.
 */
class ChatRepository(
    private val filePath: Path? = null,
) {
    private val threads = mutableMapOf<String, ChatThread>()
    private var nextThreadId = 1
    private var nextMessageId = 1
    private val mutex = Mutex()
    private val json =
        Json {
            prettyPrint = true
            ignoreUnknownKeys = true
        }

    @Volatile
    private var initialized = false

    /**
     * Retrieves all chat threads.
     *
     * @return A list of all stored ChatThread objects.
     */
    suspend fun allThreads(): List<ChatThread> {
        ensureLoaded()
        return mutex.withLock { threads.values.toList() }
    }

    /**
     * Get the most recent chat threads ordered by update time (newest first).
     *
     * @param limit Maximum number of threads to return.
     * @return A list of ChatThread objects ordered newest first, containing at most `limit` entries.
     */
    suspend fun recentThreads(limit: Int = 5): List<ChatThread> {
        ensureLoaded()
        return mutex.withLock { threads.values.sortedByDescending { it.updatedAt }.take(limit) }
    }

    /**
     * Retrieve a chat thread by its UUID.
     *
     * @param id The UUID of the thread to fetch.
     * @return The `ChatThread` with the given UUID, or `null` if no matching thread exists.
     */
    suspend fun thread(id: String): ChatThread? {
        ensureLoaded()
        return mutex.withLock { threads[id] }
    }

    /**
     * Create a new chat thread for the specified user.
     *
     * @param title The thread title.
     * @param userId The id of the user who owns the thread.
     * @return The created `ChatThread` with assigned `id` and `uuid`.
     */
    suspend fun addThread(
        title: String,
        userId: Int,
    ): ChatThread {
        ensureLoaded()
        return mutex.withLock {
            val uuid =
                java.util.UUID
                    .randomUUID()
                    .toString()
            val thread =
                ChatThread(
                    id = nextThreadId++,
                    uuid = uuid,
                    userId = userId,
                    title = title,
                )
            threads[uuid] = thread
            persist()
            thread
        }
    }

    /**
     * Update the title of an existing chat thread.
     *
     * If a thread with the given `id` exists, updates its `title` and `updatedAt` timestamp,
     * persists the change, and returns the updated thread.
     *
     * @param id The thread's UUID.
     * @param title The new title to set.
     * @return The updated `ChatThread` if found, `null` otherwise.
     */
    suspend fun updateThreadTitle(
        id: String,
        title: String,
    ): ChatThread? {
        ensureLoaded()
        return mutex.withLock {
            val current = threads[id] ?: return@withLock null
            val updated = current.copy(title = title, updatedAt = Instant.now().toString())
            threads[id] = updated
            persist()
            updated
        }
    }

    /**
     * Soft-deletes the chat thread identified by the given UUID.
     *
     * Marks the thread as deleted, persists the change, and returns the updated thread.
     *
     * @param id The thread UUID.
     * @return The updated `ChatThread` with `isDeleted` set to `true`, or `null` if no thread with the given id exists.
     */
    suspend fun markDeleted(id: String): ChatThread? {
        ensureLoaded()
        return mutex.withLock {
            val current = threads[id] ?: return@withLock null
            val updated = current.copy(isDeleted = true, updatedAt = Instant.now().toString())
            threads[id] = updated
            persist()
            updated
        }
    }

    /**
     * Appends a new message to the specified chat thread and persists the updated thread.
     *
     * @param threadId The thread UUID identifying the target thread.
     * @param content The message text to append.
     * @param sender The role or sender label for the message (e.g., `"user"` or `"assistant"`).
     * @param userId The numeric ID of the user adding the message.
     * @return The created `ChatMessage`, or `null` if no thread with `threadId` exists.
     */
    suspend fun addChat(
        threadId: String,
        content: String,
        sender: String = "user",
        userId: Int,
    ): ChatMessage? {
        ensureLoaded()
        return mutex.withLock {
            val current = threads[threadId] ?: return@withLock null
            val message =
                ChatMessage(
                    id = nextMessageId++,
                    threadId = current.id,
                    userId = userId,
                    role = sender,
                    message = content,
                )
            val updated =
                current.copy(
                    chats = current.chats + message,
                    updatedAt = Instant.now().toString(),
                )
            threads[threadId] = updated
            persist()
            message
        }
    }

    private suspend fun ensureLoaded() {
        if (initialized) return
        mutex.withLock {
            if (initialized) return
            val file = filePath?.toFile()
            val parsed =
                if (file != null && file.exists()) {
                    withContext(Dispatchers.IO) {
                        runCatching { json.decodeFromString<List<ChatThread>>(file.readText()) }
                            .onFailure { ex -> logger.warn(ex) { "Failed to load chats from ${file.absolutePath}" } }
                            .getOrNull()
                    }
                } else {
                    emptyList()
                }
            if (!parsed.isNullOrEmpty()) {
                threads.clear()
                parsed.forEach { threads[it.uuid] = it }
                nextThreadId = (parsed.maxOfOrNull { it.id } ?: 0) + 1
                val maxMsgId =
                    parsed.flatMap { it.chats }.maxOfOrNull { it.id } ?: 0
                nextMessageId = maxMsgId + 1
            }
            initialized = true
        }
    }

    /**
     * Persists the in-memory chat threads to disk when a file path is configured.
     *
     * If `filePath` is null this function returns immediately. Otherwise it writes a JSON
     * array of the current threads to the target file on the IO dispatcher. Any I/O
     * errors are caught and reported via a warning log; they do not propagate.
     */
    private suspend fun persist() {
        val file = filePath?.toFile() ?: return
        withContext(Dispatchers.IO) {
            runCatching {
                file.parentFile?.mkdirs()
                file.writeText(json.encodeToString(threads.values.toList()))
            }.onFailure { ex ->
                logger.warn(ex) { "Failed to persist chats to ${file.absolutePath}" }
            }
        }
    }
}

/**
 * In-memory document repository that persists uploads to disk.
 */
class DocumentRepository(
    private val uploadDir: String,
    private val filePath: Path? = null,
) {
    private val documents = mutableMapOf<Int, DocumentInfo>()
    private var nextId = 1
    private val mutex = Mutex()
    private val json =
        Json {
            prettyPrint = true
            ignoreUnknownKeys = true
        }

    @Volatile
    private var initialized = false

    /**
     * Return a list of all stored documents.
     *
     * Ensures persisted documents are loaded (if persistence is configured) before returning.
     *
     * @return A list of all stored DocumentInfo objects.
     */
    suspend fun all(): List<DocumentInfo> {
        ensureLoaded()
        return mutex.withLock { documents.values.toList() }
    }

    /**
     * Retrieve the document with the given identifier.
     *
     * @return The matching DocumentInfo, or `null` if no document exists for the provided id.
     */
    suspend fun get(id: Int): DocumentInfo? {
        ensureLoaded()
        return mutex.withLock { documents[id] }
    }

    /**
     * Creates a new document record from provided text content, stores the content as a file in the upload directory,
     * and persists the document metadata.
     *
     * The new document is assigned a unique id, its status is set to "processing", and the file is saved using a filename
     * prefixed with the assigned id.
     *
     * @param name The original filename to record.
     * @param content The text content to save for the document.
     * @param contentType The MIME type to record for the document (default "text/plain").
     * @param userId The id of the user who owns the document (default 1).
     * @return The created DocumentInfo containing assigned id, file path, file size, status, and other metadata.
     */
    suspend fun add(
        name: String,
        content: String,
        contentType: String = "text/plain",
        userId: Int = 1,
    ): DocumentInfo {
        ensureLoaded()
        return mutex.withLock {
            val id = nextId++
            val filePath = File(uploadDir, "${id}_$name").absolutePath
            val info =
                DocumentInfo(
                    id = id,
                    userId = userId,
                    filename = name,
                    contentType = contentType,
                    filePath = filePath,
                    fileSize = content.toByteArray().size.toLong(),
                    status = "processing",
                )
            documents[id] = info
            saveToDisk(filePath, content)
            persist()
            info
        }
    }

    /**
     * Create a new document record from raw uploaded bytes and store the file on disk.
     *
     * @param name The filename to record and use when saving the file.
     * @param data File bytes to write to disk.
     * @param contentType The MIME type to record; defaults to "application/octet-stream" when null.
     * @param userId The owner user id for the document; defaults to 1.
     * @return The created DocumentInfo containing metadata including id, filename, filePath, fileSize, and status ("processing").
     */
    suspend fun addFile(
        name: String,
        data: ByteArray,
        contentType: String? = null,
        userId: Int = 1,
    ): DocumentInfo {
        ensureLoaded()
        return mutex.withLock {
            val id = nextId++
            val path = File(uploadDir, "${id}_$name").absolutePath
            val info =
                DocumentInfo(
                    id = id,
                    userId = userId,
                    filename = name,
                    contentType = contentType ?: "application/octet-stream",
                    filePath = path,
                    fileSize = data.size.toLong(),
                    status = "processing",
                )
            documents[id] = info
            saveBytes(path, data)
            persist()
            info
        }
    }

    /**
     * Retrieves the processing status for the document with the given id.
     *
     * @param id Document identifier.
     * @return The document's status string, or `"unknown"` if the document does not exist.
     */
    suspend fun status(id: Int): String {
        ensureLoaded()
        return mutex.withLock { documents[id]?.status ?: "unknown" }
    }

    /**
     * Mark the document with the given id as processed, set its processed timestamp, clear any error message, and persist the change.
     *
     * @param id The identifier of the document to mark as processed.
     */
    suspend fun markProcessed(id: Int) {
        ensureLoaded()
        mutex.withLock {
            val current = documents[id] ?: return@withLock
            documents[id] = current.copy(status = "processed", processedAt = Instant.now().toString(), errorMessage = null)
            persist()
        }
    }

    /**
     * Mark a document as failed and record an error message.
     *
     * Sets the document's status to `failed`, updates its `errorMessage` and `processedAt`
     * timestamp, and persists the repository. If no document exists with the given `id`,
     * the call has no effect.
     *
     * @param id The identifier of the document to mark as failed.
     * @param message The error message to attach to the document.
     */
    suspend fun markFailed(
        id: Int,
        message: String,
    ) {
        ensureLoaded()
        mutex.withLock {
            val current = documents[id] ?: return@withLock
            documents[id] = current.copy(status = "failed", errorMessage = message, processedAt = Instant.now().toString())
            persist()
        }
    }

    private suspend fun saveToDisk(
        path: String,
        content: String,
    ) {
        withContext(Dispatchers.IO) {
            runCatching {
                File(path).writeText(content)
            }.onFailure { ex ->
                logger.warn(ex) { "Failed to persist document at $path" }
            }
        }
    }

    private suspend fun saveBytes(
        path: String,
        data: ByteArray,
    ) {
        withContext(Dispatchers.IO) {
            runCatching {
                File(path).writeBytes(data)
            }.onFailure { ex ->
                logger.warn(ex) { "Failed to persist document at $path" }
            }
        }
    }

    private suspend fun ensureLoaded() {
        if (initialized) return
        mutex.withLock {
            if (initialized) return
            val file = filePath?.toFile()
            val parsed =
                if (file != null && file.exists()) {
                    withContext(Dispatchers.IO) {
                        runCatching { json.decodeFromString<List<DocumentInfo>>(file.readText()) }
                            .onFailure { ex -> logger.warn(ex) { "Failed to load documents from ${file.absolutePath}" } }
                            .getOrNull()
                    }
                } else {
                    emptyList()
                }
            if (!parsed.isNullOrEmpty()) {
                documents.clear()
                parsed.forEach { documents[it.id] = it }
                nextId = (documents.keys.maxOrNull() ?: 0) + 1
            }
            initialized = true
        }
    }

    /**
     * Persist the current in-memory documents to disk as JSON.
     *
     * If `filePath` is null this function is a no-op. Failures during write are caught and logged; they do not propagate.
     */
    private suspend fun persist() {
        val file = filePath?.toFile() ?: return
        withContext(Dispatchers.IO) {
            runCatching {
                file.parentFile?.mkdirs()
                file.writeText(json.encodeToString(documents.values.toList()))
            }.onFailure { ex ->
                logger.warn(ex) { "Failed to persist documents to ${file.absolutePath}" }
            }
        }
    }

    /**
     * Deletes all uploaded files and clears all stored document records.
     *
     * Clears the in-memory document index, resets the next document id to 1, persists the empty state, and removes files
     * from the configured upload directory.
     *
     * @return The number of files successfully deleted from the upload directory.
     */
    suspend fun dropAll(): Int {
        ensureLoaded()
        return mutex.withLock {
            val deletedCount =
                withContext(Dispatchers.IO) {
                    val files = File(uploadDir).listFiles().orEmpty()
                    val deleted = files.count { runCatching { it.delete() }.getOrDefault(false) }
                    if (deleted < files.size) {
                        logger.warn { "Deleted $deleted/${files.size} files in $uploadDir" }
                    }
                    deleted
                }
            documents.clear()
            nextId = 1
            persist()
            deletedCount
        }
    }
}

/**
 * Placeholder route groups to keep parity with the FastAPI router structure.
 * Implement real logic when backing services are available.
 */
private fun Route.authRoutes(repository: UserRepository) {
    post("/token") {
        val req = call.receive<LoginRequest>()
        val user = repository.find(req.username)
        if (user == null || user.hashedPassword != PasswordHasher.hash(req.password)) {
            call.respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid credentials"))
        } else {
            call.respond(mapOf("access_token" to TokenService.issueToken(user.username), "token_type" to "bearer"))
        }
    }
    post("/register") {
        val req = call.receive<RegisterRequest>()
        val existing = repository.find(req.username)
        if (existing != null) {
            call.respond(HttpStatusCode.Conflict, mapOf("error" to "User already exists"))
        } else {
            val user =
                User(
                    id = null,
                    username = req.username,
                    email = req.email,
                    hashedPassword = PasswordHasher.hash(req.password),
                )
            val stored = repository.add(user)
            call.respond(HttpStatusCode.Created, stored)
        }
    }
    get("/users/me") {
        val authHeader = call.request.headers[HttpHeaders.Authorization]
        val token =
            authHeader
                ?.takeIf { it.startsWith("Bearer ", ignoreCase = true) }
                ?.substring(7)
                ?.trim()
        val username = TokenService.usernameFromToken(token)
        if (username == null) {
            call.respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid or missing token"))
            return@get
        }
        val current = repository.find(username)
        if (current == null) call.respond(HttpStatusCode.NotFound) else call.respond(current)
    }
}

private fun Route.userRoutes(repository: UserRepository) {
    route("/users") {
        get("/") {
            call.respond(mapOf("users" to repository.list()))
        }
        post("/theme") {
            val req = call.receive<ThemeRequest>()
            val updated = repository.updateTheme(req.username, req.theme)
            if (updated == null) {
                call.respond(HttpStatusCode.NotFound, mapOf("error" to "User not found"))
            } else {
                call.respond(updated)
            }
        }
    }
}

private suspend fun ApplicationCall.currentUser(userRepository: UserRepository): User? {
    val authHeader = request.headers[HttpHeaders.Authorization] ?: return null
    val token =
        authHeader
            .takeIf { it.startsWith("Bearer ", ignoreCase = true) }
            ?.substring(7)
            ?.trim()
    val username = TokenService.usernameFromToken(token) ?: return null
    return userRepository.find(username)
}

private suspend fun ApplicationCall.withAuthenticatedUser(
    userRepository: UserRepository,
    block: suspend (User) -> Unit,
) {
    val user = currentUser(userRepository)
    if (user?.id == null) {
        respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid or missing token"))
        return
    }
    block(user)
}

@Serializable
private data class CreateThreadRequest(
    val title: String,
)

@Serializable
private data class UpdateThreadRequest(
    val title: String,
)

@Serializable
private data class CreateChatRequest(
    val content: String,
    val sender: String? = "user",
)

private fun Route.chatRoutes(
    userRepository: UserRepository,
    chatRepository: ChatRepository,
) {
    route("/chats") {
        get("/") {
            call.withAuthenticatedUser(userRepository) { currentUser ->
                val chats =
                    chatRepository
                        .allThreads()
                        .filter { it.userId == currentUser.id }
                        .flatMap { it.chats }
                call.respond(mapOf("chats" to chats))
            }
        }
        get("/recent") {
            call.withAuthenticatedUser(userRepository) { currentUser ->
                val threads =
                    chatRepository
                        .recentThreads()
                        .filter { it.userId == currentUser.id }
                call.respond(mapOf("threads" to threads))
            }
        }
        route("/threads") {
            get {
                call.withAuthenticatedUser(userRepository) { currentUser ->
                    val threads =
                        chatRepository
                            .allThreads()
                            .filter { it.userId == currentUser.id }
                    call.respond(mapOf("threads" to threads))
                }
            }
            post {
                val currentUser = call.currentUser(userRepository)
                if (currentUser == null || currentUser.id == null) {
                    call.respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid or missing token"))
                    return@post
                }
                val req = call.receive<CreateThreadRequest>()
                val thread = chatRepository.addThread(req.title, currentUser.id)
                call.respond(HttpStatusCode.Created, thread)
            }
            get("/{thread_uuid}") {
                val id = call.parameters["thread_uuid"]
                val thread = id?.let { chatRepository.thread(it) }
                if (thread == null) call.respond(HttpStatusCode.NotFound) else call.respond(thread)
            }
            put("/{thread_uuid}") {
                val id = call.parameters["thread_uuid"]
                val req = call.receive<UpdateThreadRequest>()
                val updated = id?.let { chatRepository.updateThreadTitle(it, req.title) }
                if (updated == null) call.respond(HttpStatusCode.NotFound) else call.respond(updated)
            }
            delete("/{thread_uuid}") {
                val id = call.parameters["thread_uuid"]
                val deleted = id?.let { chatRepository.markDeleted(it) }
                if (deleted == null) call.respond(HttpStatusCode.NotFound) else call.respond(deleted)
            }
        }
        post("/chat/{thread_uuid}") {
            val id = call.parameters["thread_uuid"]
            val req = call.receive<CreateChatRequest>()
            val currentUser = call.currentUser(userRepository)
            if (currentUser == null || currentUser.id == null) {
                call.respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid or missing token"))
                return@post
            }
            val message = id?.let { chatRepository.addChat(it, req.content, req.sender ?: "user", currentUser.id) }
            if (message == null) {
                call.respond(HttpStatusCode.NotFound, mapOf("error" to "Thread not found"))
            } else {
                call.respond(HttpStatusCode.Created, message)
            }
        }
    }
}

@Serializable
private data class UploadDocumentRequest(
    val name: String,
    val content: String,
    val contentType: String? = "text/plain",
)

@Serializable
private data class DropAllRequest(
    val confirmation: String,
)

@Serializable
private data class QueryRequest(
    val query: String,
    val mode: String? = null,
)

@Serializable
private data class KnowledgeGraphQuery(
    val query: String? = null,
    val q: String? = null,
)

@Serializable
private data class DocumentStatusResponse(
    val documentId: Int,
    val status: String,
)

@Serializable
private data class GraphNodeDto(
    val id: String,
    val properties: Map<String, String> = emptyMap(),
)

@Serializable
private data class GraphEdgeDto(
    val source: String,
    val target: String,
    val properties: Map<String, String> = emptyMap(),
)

@Serializable
private data class GraphResponse(
    val nodes: List<GraphNodeDto>,
    val edges: List<GraphEdgeDto>,
)

private fun isSupportedTextContent(contentType: ContentType?): Boolean {
    if (contentType == null) return false
    if (contentType.match(ContentType.Text.Any)) return true
    if (contentType.contentType.equals("application", ignoreCase = true)) {
        val subtype = contentType.contentSubtype.lowercase()
        if (subtype in setOf("json", "xml", "x-yaml", "yaml", "javascript", "csv")) return true
    }
    return false
}

/**
 * Registers HTTP endpoints under `/documents` for uploading text files and content, querying the RAG,
 * retrieving document metadata and status, reloading, and securely dropping all stored documents and
 * associated PathRAG storages.
 *
 * Endpoints:
 * - GET `/` : returns the authenticated user's documents.
 * - POST `/upload` : accepts a JSON payload to create a document from text and enqueues ingestion into PathRAG.
 * - POST `/upload-file` : accepts multipart file uploads (text types and select application subtypes), saves files,
 *   and enqueues ingestion; rejects unsupported media types.
 * - POST `/query` : runs a query against PathRAG and returns the answer.
 * - GET `/{document_id}` : returns a specific document for the authenticated user.
 * - GET `/{document_id}/status` : returns processing status for the specified document.
 * - POST `/reload` : acknowledges a reload request for PathRAG to recognize new documents.
 * - POST `/drop` : authenticated endpoint that, with explicit confirmation, deletes all documents, uploaded files,
 *   and instructs PathRAG to drop its storages.
 *
 * Authentication is required for endpoints that operate on or expose user-specific documents. Uploads trigger
 * asynchronous ingestion; failures mark documents as failed and are logged.
 *
 * @param repository Repository used to persist and query DocumentInfo records and uploaded files.
 * @param rag PathRAG instance used for indexing/ingesting document content and servicing queries.
 * @param userRepository Repository used to resolve and authenticate the calling user.
 */
private fun Route.documentRoutes(
    repository: DocumentRepository,
    rag: PathRAG,
    userRepository: UserRepository,
) {
    route("/documents") {
        get("/") {
            call.withAuthenticatedUser(userRepository) { currentUser ->
                val docs = repository.all()
                call.respond(mapOf("documents" to docs))
            }
        }
        post("/upload") {
            val req = call.receive<UploadDocumentRequest>()
            val currentUser = call.currentUser(userRepository)
            if (currentUser == null || currentUser.id == null) {
                call.respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid or missing token"))
                return@post
            }
            val doc = repository.add(req.name, req.content, req.contentType ?: "text/plain", currentUser.id)
            launch {
                runCatching { rag.ainsert(req.content) }
                    .onSuccess { repository.markProcessed(doc.id) }
                    .onFailure { ex ->
                        logger.warn(ex) { "Failed to ingest uploaded document ${doc.id}" }
                        repository.markFailed(doc.id, ex.message ?: "Ingestion failed")
                    }
            }
            call.respond(HttpStatusCode.Created, doc)
        }
        post("/upload-file") {
            val currentUser = call.currentUser(userRepository)
            if (currentUser == null || currentUser.id == null) {
                call.respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid or missing token"))
                return@post
            }
            val multipart = call.receiveMultipart()
            val savedDocs = mutableListOf<DocumentInfo>()
            while (true) {
                val part = multipart.readPart() ?: break
                try {
                    when (part) {
                        is PartData.FileItem -> {
                            val contentType = part.contentType
                            if (!isSupportedTextContent(contentType)) {
                                call.respond(
                                    HttpStatusCode.UnsupportedMediaType,
                                    mapOf(
                                        "error" to
                                            "Unsupported content type '${contentType ?: "unknown"}'. Only text uploads are accepted.",
                                    ),
                                )
                                return@post
                            }
                            val bytes = withContext(Dispatchers.IO) { part.provider().readBytes() }
                            val filename = part.originalFileName ?: "upload_${System.currentTimeMillis()}"
                            val saved = repository.addFile(filename, bytes, contentType?.toString(), currentUser.id)
                            savedDocs.add(saved)
                            launch {
                                runCatching {
                                    val charset = contentType?.charset() ?: StandardCharsets.UTF_8
                                    val text = String(bytes, charset)
                                    rag.ainsert(text)
                                }.onSuccess { repository.markProcessed(saved.id) }
                                    .onFailure { ex ->
                                        logger.warn(ex) { "Failed to ingest uploaded file $filename into RAG" }
                                        repository.markFailed(saved.id, ex.message ?: "Ingestion failed")
                                    }
                            }
                        }

                        else -> {}
                    }
                } finally {
                    part.dispose()
                }
            }
            if (savedDocs.isEmpty()) {
                call.respond(HttpStatusCode.BadRequest, mapOf("error" to "No file found in request"))
            } else {
                call.respond(HttpStatusCode.Created, mapOf("documents" to savedDocs))
            }
        }
        post("/query") {
            val req = call.receive<QueryRequest>()
            val mode = req.mode ?: QueryParam().mode
            val result = rag.query(req.query, QueryParam(mode = mode))
            call.respond(mapOf("answer" to result))
        }
        get("/{document_id}") {
            call.withAuthenticatedUser(userRepository) { currentUser ->
                val id = call.parameters["document_id"]?.toIntOrNull()
                val doc = id?.let { repository.get(it) }
                if (doc == null) call.respond(HttpStatusCode.NotFound) else call.respond(doc)
            }
        }
        get("/{document_id}/status") {
            call.withAuthenticatedUser(userRepository) { currentUser ->
                val id =
                    call.parameters["document_id"]?.toIntOrNull() ?: return@withAuthenticatedUser call.respond(HttpStatusCode.BadRequest)
                val doc = repository.get(id)
                if (doc == null) {
                    call.respond(HttpStatusCode.NotFound)
                } else {
                    call.respond(DocumentStatusResponse(id, doc.status))
                }
            }
        }
        post("/reload") {
            call.respond(mapOf("message" to "Reload request accepted. PathRAG will recognize new documents."))
        }
        post("/drop") {
            call.withAuthenticatedUser(userRepository) {
                val req =
                    runCatching { call.receive<DropAllRequest>() }
                        .getOrElse {
                            call.respond(HttpStatusCode.BadRequest, mapOf("error" to "Missing confirmation"))
                            return@withAuthenticatedUser
                        }
                if (req.confirmation != "DROP_ALL_DATA") {
                    call.respond(HttpStatusCode.BadRequest, mapOf("error" to "Invalid confirmation token"))
                    return@withAuthenticatedUser
                }
                val deletedFiles = repository.dropAll()
                runCatching { rag.dropAll() }.onFailure { ex -> logger.warn(ex) { "Failed to drop PathRAG storages" } }
                logger.warn { "All documents and storages dropped by user=${it.username}" }
                call.respond(mapOf("message" to "All documents and storages dropped", "deleted_files" to deletedFiles))
            }
        }
    }
}

private fun Route.knowledgeGraphRoutes(
    rag: PathRAG,
    userRepository: UserRepository,
) {
    suspend fun respondGraph(call: ApplicationCall) {
        val g = rag.graph()
        val nodeIds = g.nodes()
        val nodes =
            nodeIds.map { id ->
                GraphNodeDto(
                    id = id,
                    properties = toStringMap(g.getNode(id)),
                )
            }
        val edges =
            g
                .edges()
                .map { (u, v) ->
                    GraphEdgeDto(
                        source = u,
                        target = v,
                        properties = toStringMap(g.getEdge(u, v) ?: g.getEdge(v, u)),
                    )
                }
        call.respond(GraphResponse(nodes, edges))
    }
    route("/knowledge-graph") {
        get {
            call.withAuthenticatedUser(userRepository) { respondGraph(call) }
        }
        get("/") { call.withAuthenticatedUser(userRepository) { respondGraph(call) } } // tolerate trailing slash
        post("/query") {
            call.withAuthenticatedUser(userRepository) {
                val payload = runCatching { call.receive<KnowledgeGraphQuery>() }.getOrNull()
                val question = payload?.query ?: payload?.q
                if (question.isNullOrBlank()) {
                    call.respond(HttpStatusCode.BadRequest, mapOf("error" to "Missing 'query' in payload"))
                } else {
                    val result = rag.query(question, QueryParam(mode = "hybrid"))
                    call.respond(mapOf("answer" to result))
                }
            }
        }
    }
}

private fun toStringMap(data: Map<String, Any?>?): Map<String, String> = data?.mapValues { (_, v) -> v?.toString() ?: "" } ?: emptyMap()

private suspend fun preloadKnowledgeGraphIfEmpty(
    rag: PathRAG,
    documentRepository: DocumentRepository,
) {
    val g = rag.graph()
    val hasNodes =
        runCatching { g.nodes().isNotEmpty() }
            .onFailure { ex -> logger.warn(ex) { "Failed to inspect knowledge graph; skipping preload." } }
            .getOrDefault(false)
    if (hasNodes) return

    val docs =
        runCatching { documentRepository.all() }
            .onFailure { ex -> logger.warn(ex) { "Failed to load documents for graph preload." } }
            .getOrDefault(emptyList())
    if (docs.isEmpty()) return

    logger.info { "Knowledge graph empty; preloading from ${docs.size} documents." }
    docs.forEach { doc ->
        runCatching {
            val content =
                withContext(Dispatchers.IO) {
                    File(doc.filePath).takeIf { it.exists() }?.readText()
                }
            if (!content.isNullOrBlank()) {
                rag.ainsert(content)
            } else {
                logger.warn { "Skipping preload for document ${doc.id}; file missing or empty at ${doc.filePath}" }
            }
        }.onFailure { ex -> logger.warn(ex) { "Failed to ingest document ${doc.id} for graph preload." } }
    }
}

/**
 * Minimal .env loader that emulates python-dotenv behavior for local development.
 *
 * @property values key/value pairs loaded from disk.
 */
class EnvironmentConfig private constructor(
    private val values: Map<String, String>,
) {
    /**
     * Retrieve a value for the given key from the process environment or the loaded values map.
     *
     * @param key The name of the environment variable or map key to look up.
     * @return The corresponding value if present, or `null` if not found.
     */
    operator fun get(key: String): String? = System.getenv(key) ?: values[key]

    /**
     * Provides the configured CORS origins as a comma-separated string or "*" to allow all origins.
     *
     * @return The value of the `CORS_ORIGINS` environment variable, or "*" if not set.
     */
    fun corsOrigins(): String = this["CORS_ORIGINS"] ?: "*"

    /**
     * Returns the configured chunk size in tokens used for splitting documents.
     *
     * @return The chunk token size read from `CHUNK_TOKEN_SIZE`, or `800` if the environment variable is missing or invalid.
     */
    fun chunkTokenSize(): Int = this["CHUNK_TOKEN_SIZE"]?.toIntOrNull() ?: 800

    /**
     * Provides the overlap token size used when chunking documents.
     *
     * @return The `CHUNK_OVERLAP_TOKEN_SIZE` value parsed as an `Int`, or `120` if the environment value is missing or invalid.
     */
    fun chunkOverlapTokenSize(): Int = this["CHUNK_OVERLAP_TOKEN_SIZE"]?.toIntOrNull() ?: 120

    /**
     * MongoDB connection URI for Mongo-backed storage.
     *
     * @return The MongoDB connection URI if present, `null` otherwise.
     */
    fun mongoUri(): String? = this["MONGO_URI"]

    /**
     * MongoDB database name used for Mongo-backed storage.
     *
     * @return The configured MongoDB database name (value of `MONGO_DATABASE`), or `null` if not set.
     */
    fun mongoDatabase(): String? = this["MONGO_DATABASE"]

    companion object {
        /**
         * Create an EnvironmentConfig with no in-memory overrides so lookups come from the system environment.
         *
         * @return An EnvironmentConfig backed by an empty overrides map; values will be read from system environment variables.
         */
        fun empty() = EnvironmentConfig(emptyMap())

        /**
         * Loads environment variables from a dotenv-style file into an EnvironmentConfig.
         *
         * If the file does not exist, logs a warning and returns an EnvironmentConfig backed by the current system environment.
         * The loader ignores blank lines and lines starting with `#`. Each non-comment line is parsed as `KEY=VALUE`;
         * surrounding double quotes around values are removed and keys/values are trimmed.
         *
         * @param path Path to the dotenv-style file to load.
         * @return An EnvironmentConfig containing key/value pairs parsed from the file, or an EnvironmentConfig backed
         *         by the system environment if the file is missing.
         */
        fun load(path: Path): EnvironmentConfig {
            val file = path.toFile()
            if (!file.exists()) {
                logger.warn { "No .env file found at ${file.absolutePath}. Using system environment variables." }
                return EnvironmentConfig(emptyMap())
            }

            val pairs =
                file
                    .readLines()
                    .mapNotNull { line ->
                        val trimmed = line.trim()
                        if (trimmed.isBlank() || trimmed.startsWith("#")) return@mapNotNull null
                        val delimiterIndex = trimmed.indexOf("=")
                        if (delimiterIndex <= 0) return@mapNotNull null
                        val key = trimmed.substring(0, delimiterIndex).trim()
                        val value = trimmed.substring(delimiterIndex + 1).trim().removeSurrounding("\"")
                        key to value
                    }.toMap()

            logger.info { "Loaded environment variables from ${file.absolutePath}" }
            return EnvironmentConfig(pairs)
        }
    }
}
