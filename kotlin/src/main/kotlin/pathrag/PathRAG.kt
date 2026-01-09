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

    /**
         * Create a key-value storage instance for the given namespace using the configured KV backend.
         *
         * @param namespace Namespace or prefix used to scope the storage instance.
         * @return A BaseKVStorage where each stored value is a Map<String, Any>.
         * @throws IllegalStateException if the configured `kvStorage` value is not recognized.
         */
        private fun createKvStorage(namespace: String): BaseKVStorage<Map<String, Any>> =
        when (kvStorage) {
            "JsonKVStorage" -> JsonKVStorage(namespace, globalConfig(), embeddingFunc)
            "Neo4jKVStorage" -> Neo4jKVStorage(namespace, globalConfig())
            "MongoKVStorage" -> MongoKVStorage(namespace, globalConfig())
            else -> error("Unknown kv storage: $kvStorage")
        }

    /**
         * Selects and constructs the vector storage backend for the provided namespace.
         *
         * @param namespace The namespace or collection prefix used to scope the storage.
         * @return An initialized `BaseVectorStorage` implementation configured according to `vectorStorage`.
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
         * Selects and constructs the configured graph storage implementation for the given namespace.
         *
         * @param namespace Logical namespace or prefix used to scope the graph storage.
         * @return A configured `BaseGraphStorage` instance matching the `graphStorage` setting.
         * @throws IllegalStateException if the `graphStorage` configuration value is not recognized.
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
 * Produce the effective global configuration by merging the stored snapshot with runtime extras.
 *
 * @return A map containing the merged global configuration; entries from `extraConfig` override those from the snapshot.
 */
private fun globalConfig(): Map<String, Any?> = globalConfigSnapshot.toMap(extraConfig.toMap())

    /**
 * Insert one or more documents into the datastore synchronously.
 *
 * @param stringOrStrings A single document string or a collection of document strings to insert. 
 */
    fun insert(stringOrStrings: Any) = runBlockingMaybe { ainsert(stringOrStrings) }

    /**
 * Exposes the underlying graph storage used by PathRAG for inspecting and interacting with the knowledge graph.
 *
 * @return The `BaseGraphStorage` instance that backs entity and relationship data.
 */
    fun graph(): BaseGraphStorage = chunkEntityRelationGraph

    /**
     * Ingest one or more text documents into the knowledge graph and vector stores.
     *
     * Accepts a single String or a Collection of Strings; non-string inputs are ignored. Each document
     * is chunked, entities and relationships are extracted from chunks, and resulting chunks, full
     * documents, and graph updates are upserted into the configured storages.
     *
     * @param stringOrStrings A single document String or a Collection of document Strings.
     * @throws IllegalStateException If embedding generation or any storage upsert fails. 
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
 * Synchronously insert a custom knowledge-graph payload into the graph and vector stores.
 *
 * @param customKg Payload containing chunks, entities, and relationships to upsert into the graph, text, and vector storages.
 */
    fun insertCustomKg(customKg: CustomKgPayload) = runBlockingMaybe { ainsertCustomKg(customKg) }

    /**
     * Inserts a custom knowledge-graph payload by converting the provided map into the internal payload model and ingesting it.
     *
     * Converts the given map (expected to contain chunks, entities, and relationships) into a CustomKgPayload and performs the same ingestion as the strongly-typed insert path. Missing or malformed sections will be handled using the conversion defaults.
     *
     * @param customKg A map representation of a custom KG payload (chunks, entities, relationships).
     * @deprecated Use insertCustomKg(CustomKgPayload) instead.
     */
    @Suppress("DEPRECATION")
    @Deprecated("Use insertCustomKg(CustomKgPayload) instead")
    fun insertCustomKg(customKg: Map<String, Any?>) = runBlockingMaybe { ainsertCustomKg(customKg.toCustomKgPayload()) }

    /**
     * Upserts the provided custom knowledge-graph payload into chunk, entity, and relationship stores.
     *
     * Inserts chunks into the chunk vector store and chunk KV store, upserts graph nodes for entities,
     * and upserts graph edges for relationships. Individual upsert failures are logged and do not abort
     * the overall operation; entries with missing or invalid identifiers are skipped.
     *
     * @param customKg The custom knowledge-graph payload containing chunks, entities, and relationships to upsert.
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
     * Converts this input into a CustomKgEntity when the entity name is present.
     *
     * Trims `entityName` and returns `null` if the trimmed name is blank.
     *
     * @return The constructed CustomKgEntity, or `null` if the input's `entityName` is blank.
     */
    private fun CustomKgEntityInput.toCustomEntity(): CustomKgEntity? {
        val name = entityName.trim().takeIf { it.isNotBlank() } ?: return null
        return CustomKgEntity(name, entityType, description, sourceId)
    }

    /**
     * Convert this input into a CustomKgRelationship, trimming `srcId` and `tgtId`.
     *
     * @return `CustomKgRelationship` constructed from this input with trimmed `srcId` and `tgtId`, or `null` if either `srcId` or `tgtId` is blank after trimming.
     */
    private fun CustomKgRelationshipInput.toCustomRelationship(): CustomKgRelationship? {
        val src = srcId.trim().takeIf { it.isNotBlank() } ?: return null
        val tgt = tgtId.trim().takeIf { it.isNotBlank() } ?: return null
        return CustomKgRelationship(src, tgt, description, keywords, weight, sourceId)
    }

    /**
     * Parses this value as a custom knowledge-graph payload extracting chunks, entities, and relationships.
     *
     * Converts a Map with optional keys "chunks", "entities", and "relationships" into a CustomKgPayload.
     * - "chunks" expects a list of maps with "content" and optional "source_id"; blank content entries are ignored.
     * - "entities" expects a list of maps with "entity_name" (required), and optional "entity_type", "description", and "source_id"; entries with blank `entity_name` are ignored.
     * - "relationships" expects a list of maps with required "src_id" and "tgt_id", and optional "description", "keywords", "weight", and "source_id"; entries with blank source/target ids are ignored and `weight` defaults to 1.0 when missing or unparseable.
     * String values are trimmed; missing or malformed sections produce empty lists.
     *
     * @return A CustomKgPayload containing the parsed chunks, entity inputs, and relationship inputs. Defaults to an empty payload if this value is not a Map.
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
         * Run a retrieval-augmented generation (RAG) query and produce a textual response.
         *
         * @param query The input query text to execute against the knowledge graph and vector stores.
         * @param param Optional parameters that control retrieval, reranking, and generation behavior.
         * @return The model-generated response for the provided query.
         */
    fun query(
        query: String,
        param: QueryParam = QueryParam(),
    ): String =
        runBlocking {
            aquery(query, param)
        }

    /**
     * Run a retrieval-augmented generation query against the stored graph, vector DBs, and text chunks.
     *
     * @param query The natural-language query to answer.
     * @param param Additional query options controlling retrieval and generation behavior.
     * @return The generated answer as plain text.
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
     * Delete an entity and all associated relationships from the graph and vector stores.
     *
     * The provided name is normalized by trimming surrounding double quotes and converting to uppercase before deletion.
     *
     * @param entityName The entity name to delete (will be normalized by trimming `"` and uppercasing).
     */
    suspend fun adeleteByEntity(entityName: String) {
        val key = entityName.trim('"').uppercase()
        entitiesVdb.deleteEntity(key)
        relationshipsVdb.deleteRelation(key)
        chunkEntityRelationGraph.deleteNode(key)
        logger.info { "Entity '$key' and relationships deleted." }
    }

    /**
     * Synchronously delete the graph edge from the source node to the target node.
     *
     * @param srcId The source node's identifier.
     * @param tgtId The target node's identifier.
     */
    fun deleteEdge(
        srcId: String,
        tgtId: String,
    ) = runBlockingMaybe { adeleteEdge(srcId, tgtId) }

    /**
     * Remove the relationship between two entities.
     *
     * Removes the relationship vector entry and the corresponding graph edge for the specified entity IDs.
     *
     * @param srcId Identifier of the source entity.
     * @param tgtId Identifier of the target entity.
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
 * Remove dangling edges and isolated nodes from the graph and related vector stores.
 *
 * @return A map of descriptive keys to counts of removed items (for example `"edgesRemoved"` and `"nodesRemoved"` with integer values). 
 */
    fun cleanupGraph(): Map<String, Int> = runBlockingMaybe { acleanupGraph() }

    /**
     * Remove dangling edges and isolated nodes from the graph and corresponding vector stores.
     *
     * Deletes any graph edges that reference missing nodes and removes nodes with degree zero,
     * also deleting their entries from the entities and relationships vector stores.
     *
     * @return A map containing counts of removals:
     *         - `"removed_edges"`: number of dangling edges removed
     *         - `"removed_nodes"`: number of isolated nodes removed
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
     * Drops the graph storage and its associated entity and relationship vector stores.
     *
     * This permanently removes graph data and the corresponding vectors for entities and relationships.
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
     * Remove all PathRAG storage namespaces and their contents.
     *
     * Attempts to drop each configured storage (text chunks, full documents, graph, entity/relationship vectors, and chunk vectors).
     * Individual failures are logged and do not prevent other storages from being dropped; an informational message is logged when finished.
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
     * Upserts a single entity into the knowledge graph synchronously.
     *
     * Adds or updates the entity node in the graph storage and ensures a corresponding vector entry is created or replaced.
     *
     * @param entityName The name or identifier of the entity.
     * @param description A short description or metadata for the entity.
     * @param entityType The category or type of the entity (e.g., "PERSON", "ORG", "UNKNOWN").
     * @param sourceId An optional external source identifier to store with the entity metadata.
     */
    fun upsertEntity(
        entityName: String,
        description: String = "",
        entityType: String = "UNKNOWN",
        sourceId: String? = null,
    ) = runBlockingMaybe { aupsertEntity(entityName, description, entityType, sourceId) }

    /**
     * Upserts an entity node into the knowledge graph and creates or updates its vector entry.
     *
     * @param entityName The entity's name; used as the node identifier in the graph.
     * @param description Text stored in the entity's vector entry describing the entity.
     * @param entityType A label describing the entity's type (defaults to "UNKNOWN").
     * @param sourceId Optional source identifier associated with the entity; when `null` a placeholder is stored for the graph node and an empty string is stored with the vector entry.
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
     * Upserts (creates or updates) a relationship (edge) between two graph nodes.
     *
     * @param srcId Identifier of the source node.
     * @param tgtId Identifier of the target node.
     * @param description Human-readable description or metadata for the edge.
     * @param keywords Comma- or space-separated keywords associated with the edge.
     * @param weight Numeric weight or strength of the relationship.
     * @param sourceId Optional external/source identifier to associate with the edge.
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
     * Upserts an edge between two entities in the graph and creates/updates the corresponding relationship vector.
     *
     * @param srcId Identifier of the source entity.
     * @param tgtId Identifier of the target entity.
     * @param description Human-readable description of the relationship.
     * @param keywords Comma- or space-separated keywords associated with the relationship.
     * @param weight Numerical weight for the edge; higher values indicate stronger connection.
     * @param sourceId Optional upstream source identifier for provenance; may be null.
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
     * Closes all managed storage resources that implement AutoCloseable.
     *
     * Only storages that are AutoCloseable are closed; non-closeable resources are ignored.
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