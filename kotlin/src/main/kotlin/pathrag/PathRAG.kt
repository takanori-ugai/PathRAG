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
    private val keywordExamples: String = System.getenv("KEYWORDS_EXAMPLES") ?: "",
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

    private val llmResponseCache = ResponseCache(globalConfig())

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

    private fun createKvStorage(namespace: String): BaseKVStorage<Map<String, Any>> =
        when (kvStorage) {
            "JsonKVStorage" -> JsonKVStorage(namespace, globalConfig(), embeddingFunc)
            "Neo4jKVStorage" -> Neo4jKVStorage(namespace, globalConfig())
            "MongoKVStorage" -> MongoKVStorage(namespace, globalConfig())
            else -> error("Unknown kv storage: $kvStorage")
        }

    private fun createVectorStorage(namespace: String): BaseVectorStorage =
        when (vectorStorage) {
            "NanoVectorDBStorage" -> NanoVectorDBStorage(namespace, globalConfig(), embeddingFunc)
            "Neo4jVectorStorage" -> Neo4jVectorStorage(namespace, globalConfig(), embeddingFunc)
            "MongoVectorStorage" -> MongoVectorStorage(namespace, globalConfig(), embeddingFunc)
            else -> error("Unknown vector storage: $vectorStorage")
        }

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

    private fun globalConfig(): Map<String, Any?> = globalConfigSnapshot.toMap(extraConfig.toMap())

    /**
     * Insert documents synchronously by delegating to [ainsert].
     */
    fun insert(stringOrStrings: Any) = runBlockingMaybe { ainsert(stringOrStrings) }

    /**
     * Expose the underlying graph storage for inspection.
     */
    fun graph(): BaseGraphStorage = chunkEntityRelationGraph

    /**
     * Asynchronously insert documents, chunk them, extract entities, and populate storage.
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
     * Insert a pre-built knowledge graph payload synchronously via [ainsertCustomKg].
     */
    fun insertCustomKg(customKg: CustomKgPayload) = runBlockingMaybe { ainsertCustomKg(customKg) }

    @Suppress("DEPRECATION")
    @Deprecated("Use insertCustomKg(CustomKgPayload) instead")
    fun insertCustomKg(customKg: Map<String, Any?>) = runBlockingMaybe { ainsertCustomKg(customKg.toCustomKgPayload()) }

    /**
     * Asynchronously upsert user-provided entities/relationships and chunk metadata.
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

    private fun CustomKgEntityInput.toCustomEntity(): CustomKgEntity? {
        val name = entityName.trim().takeIf { it.isNotBlank() } ?: return null
        return CustomKgEntity(name, entityType, description, sourceId)
    }

    private fun CustomKgRelationshipInput.toCustomRelationship(): CustomKgRelationship? {
        val src = srcId.trim().takeIf { it.isNotBlank() } ?: return null
        val tgt = tgtId.trim().takeIf { it.isNotBlank() } ?: return null
        return CustomKgRelationship(src, tgt, description, keywords, weight, sourceId)
    }

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
     * Execute a query synchronously using the configured RAG mode.
     */
    fun query(
        query: String,
        param: QueryParam = QueryParam(),
    ): String =
        runBlocking {
            aquery(query, param)
        }

    /**
     * Execute a query asynchronously using the configured RAG mode.
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
     * Delete an entity and its relationships synchronously.
     */
    fun deleteByEntity(entityName: String) = runBlockingMaybe { adeleteByEntity(entityName) }

    /**
     * Delete an entity and its relationships asynchronously.
     */
    suspend fun adeleteByEntity(entityName: String) {
        val key = entityName.trim('"').uppercase()
        entitiesVdb.deleteEntity(key)
        relationshipsVdb.deleteRelation(key)
        chunkEntityRelationGraph.deleteNode(key)
        logger.info { "Entity '$key' and relationships deleted." }
    }

    /**
     * Delete an edge synchronously.
     */
    fun deleteEdge(
        srcId: String,
        tgtId: String,
    ) = runBlockingMaybe { adeleteEdge(srcId, tgtId) }

    /**
     * Delete an edge asynchronously.
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
     * Clean up dangling edges and isolated nodes synchronously.
     */
    fun cleanupGraph(): Map<String, Int> = runBlockingMaybe { acleanupGraph() }

    /**
     * Clean up dangling edges and isolated nodes asynchronously.
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
     * Drop the graph and associated vector stores asynchronously.
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
     * Drop all storage namespaces asynchronously.
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
     * Upsert a single entity synchronously.
     */
    fun upsertEntity(
        entityName: String,
        description: String = "",
        entityType: String = "UNKNOWN",
        sourceId: String? = null,
    ) = runBlockingMaybe { aupsertEntity(entityName, description, entityType, sourceId) }

    /**
     * Upsert a single entity asynchronously.
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
     * Upsert an edge asynchronously.
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
