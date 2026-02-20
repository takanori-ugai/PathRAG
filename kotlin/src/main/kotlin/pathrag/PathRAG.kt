package pathrag

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.runBlocking
import pathrag.base.AddonParams
import pathrag.base.BaseGraphStorage
import pathrag.base.BaseKVStorage
import pathrag.base.BaseVectorStorage
import pathrag.base.ExtraConfig
import pathrag.base.GlobalConfig
import pathrag.base.QueryParam
import pathrag.base.runBlockingMaybe
import pathrag.llm.defaultEmbeddingFunc
import pathrag.llm.ollamaComplete
import pathrag.llm.openAiComplete
import pathrag.operate.chunkingByTokenSize
import pathrag.operate.extractEntities
import pathrag.operate.kgQuery
import pathrag.storage.JsonKVStorage
import pathrag.storage.MongoGraphStorage
import pathrag.storage.MongoKVStorage
import pathrag.storage.MongoVectorStorage
import pathrag.storage.NanoVectorDBStorage
import pathrag.storage.Neo4jKVStorage
import pathrag.storage.Neo4jStorage
import pathrag.storage.Neo4jVectorStorage
import pathrag.storage.NetworkXStorage
import pathrag.utils.ResponseCache
import pathrag.utils.computeMdHashId
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * Core Kotlin implementation of PathRAG that handles ingestion and query flows.
 */
class PathRAG(
    private val workingDir: String = "./PathRAG_cache_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss")),
    private val kvStorage: String = "JsonKVStorage",
    private val vectorStorage: String = "NanoVectorDBStorage",
    private val graphStorage: String = "NetworkXStorage",
    private val chunkTokenSize: Int = 1200,
    private val chunkOverlapTokenSize: Int = 100,
    private val language: String = System.getenv("LANGUAGE") ?: "English",
    private val keywordExamples: String =
        (System.getenv("KEYWORDS_EXAMPLES") ?: "")
            .ifBlank {
                pathrag.prompt.Prompts.KEYWORDS_EXTRACTION_EXAMPLES
                    .joinToString("\n")
            },
    private val similarityCheckPrompt: String = System.getenv("SIMILARITY_CHECK_PROMPT") ?: pathrag.prompt.Prompts.SIMILARITY_CHECK,
    private val embeddingCacheConfig: Map<String, Any?> =
        mapOf(
            "enabled" to (System.getenv("EMBEDDING_CACHE_ENABLED")?.toBoolean() ?: false),
            "similarity_threshold" to (System.getenv("EMBEDDING_CACHE_SIM_THRESHOLD")?.toDoubleOrNull() ?: 0.95),
            "use_llm_check" to (System.getenv("EMBEDDING_CACHE_USE_LLM_CHECK")?.toBoolean() ?: false),
        ),
    private val highLevelKeywords: List<String> =
        System
            .getenv("HIGH_LEVEL_KEYWORDS")
            ?.split(",")
            ?.map { it.trim() }
            ?.filter { it.isNotBlank() }
            ?: emptyList(),
    private val lowLevelKeywords: List<String> =
        System
            .getenv("LOW_LEVEL_KEYWORDS")
            ?.split(",")
            ?.map { it.trim() }
            ?.filter { it.isNotBlank() }
            ?: emptyList(),
    private val clearCacheOnStart: Boolean = false,
    private val addonParams: AddonParams =
        AddonParams(
            entityTypes = System.getenv("ENTITY_TYPES")?.split(",")?.map { it.trim() } ?: emptyList(),
            language = language, // follow top-level language
            exampleNumber = System.getenv("KEYWORD_EXAMPLE_COUNT")?.toIntOrNull() ?: 3,
        ),
    private val extraConfig: ExtraConfig = ExtraConfig(),
) : AutoCloseable {
    private val logger = KotlinLogging.logger("PathRAG")
    private val llmProvider: String = System.getenv("LLM_PROVIDER")?.lowercase() ?: "openai"
    private val llmModelName: String =
        when (llmProvider) {
            "ollama" -> System.getenv("OLLAMA_MODEL") ?: "llama3"
            else -> System.getenv("OPENAI_MODEL") ?: "gpt-4o-mini"
        }

    private fun clearResponseCacheFile() {
        if (!clearCacheOnStart) return
        val cachePath = "$workingDir/llm_cache.json"
        val cacheFile = java.io.File(cachePath)
        try {
            if (cacheFile.exists()) {
                if (!cacheFile.delete()) {
                    logger.warn { "Failed to delete cache file: $cachePath" }
                }
            }
        } catch (e: SecurityException) {
            logger.warn(e) { "Security manager denied access to cache file: $cachePath" }
        }
    }

    private val embeddingFunc = defaultEmbeddingFunc()
    private val llmModelFunc: suspend (String, String?, List<Map<String, String>>, Boolean, Boolean, Int?, Any?) -> String =
        when (llmProvider) {
            "ollama" -> { prompt, system, history, keyword, stream, maxTokens, hashingKv ->
                ollamaComplete(
                    llmModelName,
                    prompt,
                    systemPrompt = system,
                    historyMessages = history,
                    keywordExtraction = keyword,
                    stream = stream,
                    maxTokens = maxTokens,
                    hashingKv = hashingKv,
                )
            }

            else -> { prompt, system, history, keyword, stream, maxTokens, hashingKv ->
                openAiComplete(
                    llmModelName,
                    prompt,
                    systemPrompt = system,
                    historyMessages = history,
                    keywordExtraction = keyword,
                    stream = stream,
                    maxTokens = maxTokens,
                    hashingKv = hashingKv,
                )
            }
        }

    private val globalConfigSnapshot =
        GlobalConfig(
            workingDir = workingDir,
            embeddingFunc = embeddingFunc,
            llmModelFunc = llmModelFunc,
            chunkTokenSize = chunkTokenSize,
            chunkOverlapTokenSize = chunkOverlapTokenSize,
            language = language,
            keywordsExamples = keywordExamples,
            embeddingCacheConfig = embeddingCacheConfig,
            addonParams = addonParams,
            llmModelName = llmModelName,
            similarityCheckPrompt = similarityCheckPrompt,
            fixedHighLevelKeywords = highLevelKeywords,
            fixedLowLevelKeywords = lowLevelKeywords,
        )

    private val llmResponseCache =
        run {
            clearResponseCacheFile()
            ResponseCache(globalConfig())
        }

    private data class CustomKgEntity(
        val entityName: String,
        val entityType: String,
        val description: String,
        val sourceId: String,
    )

    data class CustomKgChunk(
        val content: String,
        val sourceId: String? = null,
    )

    data class CustomKgEntityInput(
        val entityName: String,
        val entityType: String = "UNKNOWN",
        val description: String = "",
        val sourceId: String = "",
    )

    private data class CustomKgRelationship(
        val srcId: String,
        val tgtId: String,
        val description: String,
        val keywords: String,
        val weight: Double,
        val sourceId: String,
    )

    data class CustomKgRelationshipInput(
        val srcId: String,
        val tgtId: String,
        val description: String = "",
        val keywords: String = "",
        val weight: Double = 1.0,
        val sourceId: String = "",
    )

    data class CustomKgPayload(
        val chunks: List<CustomKgChunk> = emptyList(),
        val entities: List<CustomKgEntityInput> = emptyList(),
        val relationships: List<CustomKgRelationshipInput> = emptyList(),
    )

    /**
     * Create a key-value storage instance for the given namespace using the configured KV backend.
     *
     * @param namespace Namespace identifier used to scope stored entries.
     * @return A configured `BaseKVStorage<Map<String, Any>>` for storing document maps.
     * @throws IllegalStateException If the configured KV backend (`kvStorage`) is unknown.
     */
    private fun createKvStorage(namespace: String): BaseKVStorage<Map<String, Any>> =
        when (kvStorage) {
            "JsonKVStorage" -> JsonKVStorage(namespace, globalConfig(), embeddingFunc)
            "Neo4jKVStorage" -> Neo4jKVStorage(namespace, globalConfig())
            "MongoKVStorage" -> MongoKVStorage(namespace, globalConfig())
            else -> error("Unknown kv storage: $kvStorage")
        }

    /**
     * Selects and creates a vector storage implementation for the provided namespace using the configured backend.
     *
     * @param namespace The namespace used to scope the vector storage (e.g., a logical database/collection prefix).
     * @return A configured `BaseVectorStorage` implementation for the given namespace.
     * @throws IllegalStateException If `vectorStorage` is not a recognized backend.
     */
    private fun createVectorStorage(namespace: String): BaseVectorStorage =
        when (vectorStorage) {
            "NanoVectorDBStorage" -> NanoVectorDBStorage(namespace, globalConfig(), embeddingFunc)
            "Neo4jVectorStorage" -> Neo4jVectorStorage(namespace, globalConfig(), embeddingFunc)
            "MongoVectorStorage" -> MongoVectorStorage(namespace, globalConfig(), embeddingFunc)
            else -> error("Unknown vector storage: $vectorStorage")
        }

    /**
     * Selects and constructs a graph storage implementation according to the configured `graphStorage` type.
     *
     * @param namespace Namespace/prefix used to initialize the selected graph storage.
     * @return An instance of `BaseGraphStorage` corresponding to the configured storage type.
     * @throws IllegalStateException If `graphStorage` is not a recognized storage identifier.
     */
    private fun createGraphStorage(namespace: String): BaseGraphStorage =
        when (graphStorage) {
            "NetworkXStorage" -> NetworkXStorage(namespace, globalConfig(), embeddingFunc)
            "Neo4jStorage" -> Neo4jStorage(namespace, globalConfig())
            "MongoGraphStorage" -> MongoGraphStorage(namespace, globalConfig(), embeddingFunc)
            else -> error("Unknown graph storage: $graphStorage")
        }

    private val fullDocs: BaseKVStorage<Map<String, Any>> = createKvStorage("full_docs")
    private val textChunks: BaseKVStorage<Map<String, Any>> = createKvStorage("text_chunks")
    private var chunkEntityRelationGraph: BaseGraphStorage = createGraphStorage("chunk_entity_relation")
    private val entitiesVdb: BaseVectorStorage = createVectorStorage("entities_vdb")
    private val relationshipsVdb: BaseVectorStorage = createVectorStorage("relationships_vdb")
    private val chunksVdb: BaseVectorStorage = createVectorStorage("chunks_vdb")

    /**
     * Compose the runtime global configuration by merging the stored snapshot with additional configuration.
     *
     * The returned map contains the snapshot's keys combined with entries from `extraConfig`; keys present
     * in `extraConfig` override those from the snapshot.
     *
     * @return A map of configuration keys to values representing the merged global configuration.
     */
    private fun globalConfig(): Map<String, Any?> = globalConfigSnapshot.toMap(extraConfig.toMap())

    /**
     * Insert one or more documents into the store synchronously.
     *
     * The `stringOrStrings` argument may be a single document `String` or a collection of `String`s;
     * the input is normalized to one or more document texts and ingested into the system.
     *
     * @param stringOrStrings A single document string or a collection of document strings to insert.
     */
    fun insert(stringOrStrings: Any) = runBlockingMaybe { ainsert(stringOrStrings) }

    /**
     * Expose the underlying graph storage for inspection.
     */
    fun graph(): BaseGraphStorage = chunkEntityRelationGraph

    /**
     * Insert one or more documents: chunk their text, extract entities and relationships, and persist
     * documents, chunks, entity nodes, and relationship vectors into the configured storages.
     *
     * @param stringOrStrings A single `String` or a `Collection` of `String` values to insert. Any other
     * value is treated as empty input and no action is taken.
     * @throws IllegalStateException If embedding generation or any storage upsert fails during insertion.
     */
    @Suppress("TooGenericExceptionCaught")
    suspend fun ainsert(stringOrStrings: Any) {
        val inputs =
            when (stringOrStrings) {
                is String -> listOf(stringOrStrings)
                is Collection<*> -> stringOrStrings.filterIsInstance<String>()
                else -> emptyList()
            }
        if (inputs.isEmpty()) {
            logger.warn { "No documents provided for insertion." }
            return
        }
        val newDocs =
            inputs.associate { text ->
                val id = computeMdHashId(text.trim(), prefix = "doc-")
                id to mapOf("content" to text.trim())
            }
        val chunkMap = mutableMapOf<String, Map<String, Any>>()
        newDocs.forEach { (docKey, doc) ->
            val content = doc["content"] as String
            val chunks =
                chunkingByTokenSize(
                    content,
                    overlapTokenSize = chunkOverlapTokenSize,
                    maxTokenSize = chunkTokenSize,
                )
            chunks.forEach { chunk ->
                val content = (chunk["content"] as? String)?.trim().orEmpty()
                val id = computeMdHashId(content, prefix = "chunk-")
                chunkMap[id] = chunk + mapOf("content" to content, "full_doc_id" to docKey)
            }
        }
        try {
            chunksVdb.upsert(chunkMap)
            chunkEntityRelationGraph =
                extractEntities(
                    chunkMap,
                    chunkEntityRelationGraph,
                    entitiesVdb,
                    relationshipsVdb,
                    globalConfig(),
                )
            fullDocs.upsert(newDocs)
            textChunks.upsert(chunkMap)
        } catch (e: IllegalStateException) {
            logger.error(e) { "Failed to insert documents; embedding or storage update error occurred." }
            throw e
        }
    }

    /**
     * Insert a pre-built custom knowledge graph payload into the PathRAG storages synchronously.
     *
     * @param customKg The payload containing chunks, entities, and relationships to upsert.
     */
    fun insertCustomKg(customKg: CustomKgPayload) = runBlockingMaybe { ainsertCustomKg(customKg) }

    /**
     * Accepts a legacy Map-based custom knowledge graph payload, converts it to a strongly-typed payload, and upserts
     * its chunks, entities, and relationships into storage.
     *
     * The map is expected to contain keys "chunks", "entities", and "relationships" with values structured like:
     * - "chunks": List of maps with keys "content" (String) and optional "sourceId" (String)
     * - "entities": List of maps with keys "entityName" (String), optional "entityType" (String),
     *   optional "description" (String), and optional "sourceId" (String)
     * - "relationships": List of maps with keys "srcId" (String), "tgtId" (String),
     *   and optional "description" (String), "keywords" (String), "weight" (Number), and "sourceId" (String)
     *
     * @param customKg A legacy Map representation of a custom KG payload matching the structure described above.
     */
    @Suppress("DEPRECATION")
    @Deprecated("Use insertCustomKg(CustomKgPayload) instead")
    fun insertCustomKg(customKg: Map<String, Any?>) = runBlockingMaybe { ainsertCustomKg(customKg.toCustomKgPayload()) }

    /**
     * Upserts the provided custom knowledge-graph payload into chunk, entity, and relationship stores.
     *
     * Processes chunks, entities, and relationships from the payload and persists them into the configured
     * vector, KV, and graph storages.
     *
     * @param customKg Payload containing lists of chunks, entity inputs, and relationship inputs to upsert.
     * @throws IllegalStateException if chunk upsertion fails due to an embedding or storage error.
     */
    suspend fun ainsertCustomKg(customKg: CustomKgPayload) {
        val chunks = customKg.chunks
        val entities = customKg.entities.mapNotNull { it.toCustomEntity() }
        val relationships = customKg.relationships.mapNotNull { it.toCustomRelationship() }

        val chunkData =
            chunks.associate { chunk ->
                val content = chunk.content.trim()
                val id = computeMdHashId(content, prefix = "chunk-")
                id to mapOf("content" to content, "source_id" to chunk.sourceId.orEmpty())
            }
        if (chunkData.isNotEmpty()) {
            try {
                chunksVdb.upsert(chunkData)
                textChunks.upsert(chunkData)
            } catch (e: IllegalStateException) {
                logger.error(e) { "Failed to insert custom KG chunks; embedding or storage update error occurred." }
                throw e
            }
        }

        entities.forEach { entity ->
            val name = entity.entityName.trim('"').uppercase()
            val nodeData =
                mapOf(
                    "entity_type" to entity.entityType.ifBlank { "UNKNOWN" },
                    "description" to entity.description.ifBlank { "No description provided" },
                    "source_id" to entity.sourceId.ifBlank { "UNKNOWN" },
                )
            runCatching { chunkEntityRelationGraph.upsertNode(name, nodeData) }
                .onFailure { ex -> logger.error(ex) { "Failed to upsert node $name" } }
        }

        relationships.forEach { rel ->
            val src = rel.srcId.trim('"').uppercase()
            val tgt = rel.tgtId.trim('"').uppercase()
            val data =
                mapOf(
                    "weight" to rel.weight,
                    "description" to rel.description,
                    "keywords" to rel.keywords,
                    "source_id" to rel.sourceId.ifBlank { "UNKNOWN" },
                )
            runCatching { chunkEntityRelationGraph.upsertEdge(src, tgt, data) }
                .onFailure { ex -> logger.error(ex) { "Failed to upsert edge $src -> $tgt" } }
        }
    }

    /**
     * Convert this input into a CustomKgEntity when the input has a non-blank name.
     *
     * Trims `entityName` and returns `null` if the resulting name is blank; otherwise returns a
     * CustomKgEntity populated from this input's `entityName`, `entityType`, `description`, and `sourceId`.
     *
     * @return `CustomKgEntity` built from this input, or `null` if the trimmed `entityName` is blank.
     */
    private fun CustomKgEntityInput.toCustomEntity(): CustomKgEntity? {
        val name = entityName.trim().takeIf { it.isNotBlank() } ?: return null
        return CustomKgEntity(name, entityType, description, sourceId)
    }

    /**
     * Converts this input into a validated CustomKgRelationship.
     *
     * Returns a CustomKgRelationship when both source and target IDs are present (non-blank after trimming); otherwise returns `null`.
     *
     * @return A `CustomKgRelationship` built from this input if valid, `null` if either source or target ID is missing or blank.
     */
    private fun CustomKgRelationshipInput.toCustomRelationship(): CustomKgRelationship? {
        val src = srcId.trim().takeIf { it.isNotBlank() } ?: return null
        val tgt = tgtId.trim().takeIf { it.isNotBlank() } ?: return null
        return CustomKgRelationship(src, tgt, description, keywords, weight, sourceId)
    }

    /**
     * Converts a generic map-like payload into a CustomKgPayload.
     *
     * Parses optional "chunks", "entities", and "relationships" entries from the receiver when it is a Map.
     * - Chunks with blank or missing "content" are ignored.
     * - Entities require a non-blank "entity_name"; missing fields default to empty strings.
     * - Relationships require non-blank "src_id" and "tgt_id"; "weight" is parsed as a Double and defaults to 1.0 on parse failure.
     *
     * @return A CustomKgPayload containing parsed chunks, entities, and relationships. Returns an empty payload when the
     *         receiver is not a Map or no valid items are present.
     */
    private fun Any?.toCustomKgPayload(): CustomKgPayload {
        val map = this as? Map<*, *> ?: return CustomKgPayload()
        val chunkList =
            (map["chunks"] as? List<*>)
                ?.mapNotNull { item ->
                    (item as? Map<*, *>)?.let { chunk ->
                        val content = chunk["content"]?.toString()?.trim().orEmpty()
                        if (content.isBlank()) null else CustomKgChunk(content, chunk["source_id"]?.toString())
                    }
                }.orEmpty()
        val entities =
            (map["entities"] as? List<*>)
                ?.mapNotNull { item ->
                    (item as? Map<*, *>)?.let { ent ->
                        val name = ent["entity_name"]?.toString()?.trim().orEmpty()
                        if (name.isBlank()) {
                            null
                        } else {
                            CustomKgEntityInput(
                                entityName = name,
                                entityType = ent["entity_type"]?.toString().orEmpty(),
                                description = ent["description"]?.toString().orEmpty(),
                                sourceId = ent["source_id"]?.toString().orEmpty(),
                            )
                        }
                    }
                }.orEmpty()
        val relationships =
            (map["relationships"] as? List<*>)
                ?.mapNotNull { item ->
                    (item as? Map<*, *>)?.let { rel ->
                        val src = rel["src_id"]?.toString()?.trim().orEmpty()
                        val tgt = rel["tgt_id"]?.toString()?.trim().orEmpty()
                        if (src.isBlank() || tgt.isBlank()) {
                            null
                        } else {
                            CustomKgRelationshipInput(
                                srcId = src,
                                tgtId = tgt,
                                description = rel["description"]?.toString().orEmpty(),
                                keywords = rel["keywords"]?.toString().orEmpty(),
                                weight = rel["weight"]?.toString()?.toDoubleOrNull() ?: 1.0,
                                sourceId = rel["source_id"]?.toString().orEmpty(),
                            )
                        }
                    }
                }.orEmpty()
        return CustomKgPayload(chunkList, entities, relationships)
    }

    /**
     * Run a retrieval-augmented generation (RAG) query and return the model's response.
     *
     * @param query The user query or prompt text to execute against the knowledge graph and LLM.
     * @param param Additional query options (e.g., retrieval limits, filters, or response formatting).
     * @return The generated response text for the given query.
     */
    fun query(
        query: String,
        param: QueryParam = QueryParam(),
    ): String =
        runBlocking {
            aquery(query, param)
        }

    /**
     * Run a query against the knowledge graph and retrieval stores using the configured RAG mode and produce a textual response.
     *
     * @param query The user query to execute.
     * @param param Additional query options and controls (e.g., result limits, filters, and retrieval settings).
     * @return The generated response text for the given query.
     */
    suspend fun aquery(
        query: String,
        param: QueryParam = QueryParam(),
    ): String {
        val response =
            kgQuery(
                query,
                chunkEntityRelationGraph,
                entitiesVdb,
                relationshipsVdb,
                textChunks,
                param,
                globalConfig(),
                llmModelFunc,
                llmResponseCache,
            )
        return response
    }

    /**
     * Delete the entity with the given name and all of its relationships synchronously.
     *
     * @param entityName The name or identifier of the entity to delete; comparison is case-insensitive and the name is
     *                   normalized to uppercase.
     */
    fun deleteByEntity(entityName: String) = runBlockingMaybe { adeleteByEntity(entityName) }

    /**
     * Drop all namespaces and clear stored data for this PathRAG instance.
     */
    fun clear() = runBlockingMaybe { aclear() }

    /**
     * Drop all namespaces and clear stored data for this PathRAG instance.
     */
    suspend fun aclear() {
        fullDocs.drop()
        textChunks.drop()
        chunkEntityRelationGraph.drop()
        entitiesVdb.drop()
        relationshipsVdb.drop()
        chunksVdb.drop()
    }

    /**
     * Remove the named entity and all its relationships from the graph and associated vector stores.
     *
     * The provided `entityName` is normalized by trimming surrounding double quotes and uppercasing before deletion.
     *
     * @param entityName The entity identifier to delete (quotes will be trimmed; comparison is case-insensitive).
     */
    suspend fun adeleteByEntity(entityName: String) {
        val key = entityName.trim('"').uppercase()
        entitiesVdb.deleteEntity(key)
        relationshipsVdb.deleteRelation(key)
        chunkEntityRelationGraph.deleteNode(key)
        logger.info { "Entity '$key' and relationships deleted." }
    }

    /**
     * Remove the relationship between two entities from the graph and its associated vector entry.
     *
     * This call normalizes both identifiers to uppercase before deletion; it removes the relationship
     * record from the relationships vector store and deletes the corresponding edge from the graph.
     *
     * @param srcId Source entity identifier (will be normalized to uppercase).
     * @param tgtId Target entity identifier (will be normalized to uppercase).
     */
    fun deleteEdge(
        srcId: String,
        tgtId: String,
    ) = runBlockingMaybe { adeleteEdge(srcId, tgtId) }

    /**
     * Remove the relationship edge between two entities identified by source and target IDs.
     *
     * Source and target identifiers are normalized by trimming surrounding double quotes and converting to uppercase;
     * the relationship is removed from the relationships vector store and from the chunk-entity graph, and an informational
     * message is logged.
     *
     * @param srcId The source entity identifier (may include surrounding quotes).
     * @param tgtId The target entity identifier (may include surrounding quotes).
     */
    suspend fun adeleteEdge(
        srcId: String,
        tgtId: String,
    ) {
        val srcKey = srcId.trim('"').uppercase()
        val tgtKey = tgtId.trim('"').uppercase()
        relationshipsVdb.deleteRelationBetween(srcKey, tgtKey)
        chunkEntityRelationGraph.deleteEdge(srcKey, tgtKey)
        logger.info { "Edge '$srcKey' -> '$tgtKey' deleted." }
    }

    /**
     * Remove dangling edges and isolated nodes from the graph and corresponding vector stores.
     *
     * @return A map containing removal counts: `"removed_edges"` -> number of edges removed, `"removed_nodes"` -> number of nodes removed.
     */
    fun cleanupGraph(): Map<String, Int> = runBlockingMaybe { acleanupGraph() }

    /**
     * Remove edges that reference missing nodes and delete nodes with zero degree from the graph.
     *
     * @return A map with keys `removed_edges` and `removed_nodes` whose values are the counts of edges and nodes removed, respectively.
     */
    suspend fun acleanupGraph(): Map<String, Int> {
        var removedEdges = 0
        var removedNodes = 0

        val nodeSet = chunkEntityRelationGraph.nodes().toSet()
        val danglingEdges = chunkEntityRelationGraph.edges().filter { (s, t) -> s !in nodeSet || t !in nodeSet }
        danglingEdges.forEach { (s, t) ->
            relationshipsVdb.deleteRelationBetween(s, t)
            chunkEntityRelationGraph.deleteEdge(s, t)
            removedEdges += 1
        }

        val nodes = chunkEntityRelationGraph.nodes()
        val isolated =
            nodes.filter { node ->
                chunkEntityRelationGraph.nodeDegree(node) == 0
            }
        isolated.forEach { node ->
            entitiesVdb.deleteEntity(node)
            relationshipsVdb.deleteRelation(node)
            chunkEntityRelationGraph.deleteNode(node)
            removedNodes += 1
        }

        logger.info { "Graph cleanup removed $removedEdges dangling edges and $removedNodes isolated nodes." }
        return mapOf("removed_edges" to removedEdges, "removed_nodes" to removedNodes)
    }

    /**
     * Drop the graph and associated vector stores synchronously.
     */
    fun dropGraph() = runBlockingMaybe { adropGraph() }

    /**
     * Remove the graph storage and its associated entity and relationship vector stores.
     *
     * Drops the chunk-entity-relationship graph and the entities and relationships vector databases, and logs an
     * informational message on completion.
     */
    suspend fun adropGraph() {
        chunkEntityRelationGraph.drop()
        entitiesVdb.drop()
        relationshipsVdb.drop()
        logger.info { "Graph and associated entity/relationship vectors dropped." }
    }

    /**
     * Drop all storage namespaces (graph, vectors, and KV stores).
     */
    fun dropAll() = runBlockingMaybe { adropAll() }

    /**
     * Remove all internal PathRAG storages.
     *
     * Attempts to drop each configured storage namespace (text chunks, full documents,
     * chunk-entity-relationship graph, entities vector, relationships vector, and chunks vector).
     * Failures for individual stores are logged and do not stop the method from attempting
     * to drop the remaining stores.
     */
    suspend fun adropAll() {
        runCatching { textChunks.drop() }.onFailure { ex -> logger.warn(ex) { "Failed to drop textChunks KV" } }
        runCatching { fullDocs.drop() }.onFailure { ex -> logger.warn(ex) { "Failed to drop fullDocs KV" } }
        runCatching { chunkEntityRelationGraph.drop() }.onFailure { ex -> logger.warn(ex) { "Failed to drop graph storage" } }
        runCatching { entitiesVdb.drop() }.onFailure { ex -> logger.warn(ex) { "Failed to drop entities vector storage" } }
        runCatching { relationshipsVdb.drop() }.onFailure { ex -> logger.warn(ex) { "Failed to drop relationships vector storage" } }
        runCatching { chunksVdb.drop() }.onFailure { ex -> logger.warn(ex) { "Failed to drop chunks vector storage" } }
        logger.info { "All PathRAG storages dropped." }
    }

    /**
     * Synchronously upserts an entity into the graph and entity vector stores.
     *
     * @param entityName The name of the entity; used as the entity key (normalized).
     * @param description Optional human-readable description for the entity.
     * @param entityType Optional entity type label; defaults to "UNKNOWN".
     * @param sourceId Optional source identifier to associate with the entity for provenance.
     */
    fun upsertEntity(
        entityName: String,
        description: String = "",
        entityType: String = "UNKNOWN",
        sourceId: String? = null,
    ) = runBlockingMaybe { aupsertEntity(entityName, description, entityType, sourceId) }

    /**
     * Upserts or creates an entity node in the graph and a corresponding vector entry in the entity vector store.
     *
     * @param entityName The name or identifier of the entity to upsert.
     * @param description A textual description stored as the entity's vector content.
     * @param entityType A category or type label for the entity (defaults to "UNKNOWN").
     * @param sourceId Optional provenance or source identifier associated with the entity.
     */
    suspend fun aupsertEntity(
        entityName: String,
        description: String = "",
        entityType: String = "UNKNOWN",
        sourceId: String? = null,
    ) {
        val key = entityName.trim('"').uppercase()
        val nodeData =
            mapOf(
                "entity_type" to entityType,
                "description" to description,
                "source_id" to (sourceId ?: "UNKNOWN"),
                "entity_name" to key,
            )
        val vectorId = computeMdHashId(key, prefix = "ent-")
        runCatching { chunkEntityRelationGraph.upsertNode(key, nodeData) }
            .onFailure { ex -> logger.error(ex) { "Failed to upsert node $key" } }
        runCatching {
            entitiesVdb.upsert(
                mapOf(
                    vectorId to
                        mapOf(
                            "content" to description,
                            "entity_name" to key,
                            "source_id" to (sourceId ?: ""),
                        ),
                ),
            )
        }.onFailure { ex -> logger.error(ex) { "Failed to upsert entity vector $vectorId" } }
        logger.info { "Entity '$key' upserted." }
    }

    /**
     * Upsert an edge synchronously.
     */
    fun upsertEdge(
        srcId: String,
        tgtId: String,
        description: String = "",
        keywords: String = "",
        weight: Double = 1.0,
        sourceId: String? = null,
    ) = runBlockingMaybe { aupsertEdge(srcId, tgtId, description, keywords, weight, sourceId) }

    /**
     * Upserts an edge between two entities in the graph and creates or updates its corresponding relationship vector.
     *
     * The entity identifiers are normalized (trimmed of quoting and uppercased) before upsertion. The graph edge is
     * stored with metadata including `weight`, `description`, `keywords`, and `source_id`. A relationship vector entry
     * is created/updated with `src_id`, `tgt_id`, concatenated `content` (description + keywords), `keywords`,
     * `description`, and `source_id`.
     *
     * Failures during graph or vector upsertion are logged and do not propagate exceptions.
     *
     * @param srcId The source entity identifier.
     * @param tgtId The target entity identifier.
     * @param description Optional text describing the relationship.
     * @param keywords Optional keywords associated with the relationship.
     * @param weight Numeric weight for the relationship (default is 1.0).
     * @param sourceId Optional provenance identifier; when omitted a default placeholder is used in stored metadata.
     */
    suspend fun aupsertEdge(
        srcId: String,
        tgtId: String,
        description: String = "",
        keywords: String = "",
        weight: Double = 1.0,
        sourceId: String? = null,
    ) {
        val srcKey = srcId.trim('"').uppercase()
        val tgtKey = tgtId.trim('"').uppercase()
        val data =
            mapOf(
                "weight" to weight,
                "description" to description,
                "keywords" to keywords,
                "source_id" to (sourceId ?: "UNKNOWN"),
            )
        runCatching { chunkEntityRelationGraph.upsertEdge(srcKey, tgtKey, data) }
            .onFailure { ex -> logger.error(ex) { "Failed to upsert edge $srcKey -> $tgtKey" } }
        val relId = computeMdHashId(srcKey + tgtKey, prefix = "rel-")
        runCatching {
            relationshipsVdb.upsert(
                mapOf(
                    relId to
                        mapOf(
                            "src_id" to srcKey,
                            "tgt_id" to tgtKey,
                            "content" to (description + keywords),
                            "keywords" to keywords,
                            "description" to description,
                            "source_id" to (sourceId ?: ""),
                        ),
                ),
            )
        }.onFailure { ex -> logger.error(ex) { "Failed to upsert relationship vector $relId" } }
        logger.info { "Edge '$srcKey' -> '$tgtKey' upserted." }
    }

    /**
     * Releases resources used by PathRAG by closing any underlying storage components.
     *
     * Closes the configured storage components (fullDocs, textChunks, chunkEntityRelationGraph,
     * entitiesVdb, relationshipsVdb, chunksVdb) when they implement `AutoCloseable`.
     */
    override fun close() {
        listOf(
            fullDocs,
            textChunks,
            chunkEntityRelationGraph,
            entitiesVdb,
            relationshipsVdb,
            chunksVdb,
        ).forEach { (it as? AutoCloseable)?.close() }
    }
}
