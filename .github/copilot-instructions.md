# Copilot Instructions for PathRAG

PathRAG is a knowledge graph-based Retrieval-Augmented Generation (RAG) system with both Kotlin and Python components. It combines document processing, knowledge graph visualization, and LLM-based chat.

## Important: Kotlin is a Port of Python

The **Kotlin implementation** (`/kotlin/`) is a converted version of the original **Python implementation** (`/PathRAG/` and root-level Python files like `main.py`, `app.py`). The two versions implement the same core algorithms and data structures but in different languages:

- **Python** (`/PathRAG/` module): Original implementation; used for research, prototyping, and the FastAPI server layer
- **Kotlin** (`/kotlin/`): Production-grade port with improved performance, type safety, and deployment via Ktor

When making changes:
- **Bugfixes** affecting core logic (entity extraction, graph operations, queries) should be applied to **both** versions to keep them in sync
- **Kotlin-specific enhancements** (performance, concurrency) don't need Python equivalents
- **Python API layer** (`app.py`) wraps the Python PathRAG module for REST endpoints
- **Kotlin Application** provides its own Ktor-based REST API with similar endpoints

Look at the **Python PathRAG module** when understanding the original algorithm design—the Kotlin implementation mirrors its structure closely.

## Build, Test, and Lint Commands

### Kotlin (Primary Backend)

Located in `/kotlin` directory. Build system: **Gradle**

```bash
cd kotlin

# Build the project
./gradlew build

# Run tests
./gradlew test

# Run a specific test class
./gradlew test --tests pathrag.PathRAGTest

# Run a specific test method
./gradlew test --tests pathrag.PathRAGTest.upsertAndDeleteEntityAndEdge

# Run linting (ktlint)
./gradlew ktlintFormat      # Auto-fix style issues
./gradlew ktlint            # Check for style violations

# Run static analysis (detekt)
./gradlew detekt

# Build fat JAR with shadow plugin
./gradlew shadowJar

# Run a custom main class (e.g., SampleEval)
./gradlew execute -PmainClass=pathrag.SampleEvalKt

# Watch mode (useful for development)
./gradlew build --continuous
```

### Python (API & Data Processing)

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
python app.py                   # Runs on http://localhost:8000

# Run the test script
python v1_test.py
```

### UI

```bash
cd ui
# UI is a web application (check package.json for commands)
```

## High-Level Architecture

### Core Modules

**Kotlin Core (`/kotlin/src/main/kotlin/pathrag/`)**
- `PathRAG.kt`: Main orchestrator handling ingestion and query flows
- `operate/`: Chunking, entity extraction, and knowledge graph operations
- `llm/`: LLM integrations (OpenAI, Ollama) and embedding functions
- `storage/`: Pluggable storage backends (Neo4j, MongoDB, in-memory, NanoVectorDB)
- `base/`: Abstract base classes for storage interfaces and configuration
- `eval/`: Evaluation framework (RAGAS metrics, HotpotQA benchmarks)
- `prompt/`: Prompt templates for extraction and similarity checking
- `utils/`: Utilities for hashing, tokenization, caching

**Application Layer** (`Application.kt`)
- Ktor-based REST API with Swagger UI
- JWT authentication via Koin dependency injection
- CORS support for cross-origin requests

**Storage Abstraction Pattern**
Three pluggable storage layers (implemented in `/storage/`):
1. **GraphStorage**: Knowledge graph storage (Neo4j, MongoDB, NetworkX)
2. **VectorStorage**: Embeddings (NanoVectorDB, Neo4j, MongoDB)
3. **KVStorage**: Key-value pairs and metadata (JSON files, MongoDB, Neo4j)

**Python API** (`app.py`)
- FastAPI wrapper for REST endpoints
- Multi-format file processing (PDF, DOCX, PPTX, XLSX, EPUB, etc.)
- Streaming response support via WebSockets

### Data Flow

1. **Ingestion**: Documents → Chunking (token-based) → Entity extraction → Knowledge graph construction
2. **Querying**: User query → Entity identification → Graph path traversal + vector similarity → Context synthesis → LLM response
3. **Hybrid Search**: Combines vector similarity, graph relationships, and entity-centric retrieval

## Key Conventions

### Kotlin Code Style

- **ktlint**: Code formatting enforced with max line length of 140 chars
- **detekt**: Static analysis with custom rules in `config/detekt.yml`
- **Package naming**: `pathrag.*` namespace for all application code
- **Coroutines**: Heavy use of Kotlin coroutines (`runBlocking`, `async`, `launch`)
- **Extension functions**: Common for utility operations on base types

### Dependency Injection

- **Koin**: Used for dependency management in the Ktor application
- Modules are configured in the main application setup
- Avoid global state; dependencies are injected through constructors

### Storage Interface Pattern

All storage implementations follow an abstract base pattern:
```kotlin
// Example: Custom storage must extend BaseGraphStorage
abstract class BaseGraphStorage {
    abstract suspend fun ainsertEntity(...)
    abstract suspend fun aquery(...)
    // ... more abstract methods
}
```

Implementations use `asyncio`-like patterns with `a`-prefixed suspend functions for async operations.

### Configuration Management

- Environment variables for sensitive data (API keys, connection strings)
- `.env` file support via `dotenv` library
- `GlobalConfig` and `ExtraConfig` objects for application-wide settings
- Main class selection via Gradle property: `-PmainClass=fully.qualified.ClassName`

### Testing Patterns

- **MockK**: Mocking library for Kotlin (used in all test files)
- **Kotlin Test JUnit**: Standard test framework
- Tests in `/src/test/kotlin/pathrag/`
- **Cleanup**: Tests delete temporary cache directories (`PathRAG_cache_*`) in `@AfterTest`
- Storage integration tests (e.g., `Neo4jIntegrationTest`) may require external services

### Prompting & Extraction

- Extraction prompts are centralized in `prompt/Prompts.kt` (keyword extraction, similarity checking)
- LLM functions accept `ChatModel` instances and return structured responses
- Entity extraction uses configurable language and examples via environment variables

### Logging

- **kotlin-logging** with SLF4j backend (Logback)
- Always use `KotlinLogging.logger()` for logger creation
- Test logs include `showStandardStreams = true` in Gradle config for visibility

### Error Handling

- Use Kotlin's exception hierarchy; avoid silent failures
- Storage operations should throw appropriate exceptions for missing/invalid data
- API endpoints return HTTP status codes via Ktor (not custom codes)

## Development Tips

- **Entry points**: Check `Application.kt` for the main Ktor server; `main.py` for Python API
- **File structure**: Organize by feature module, not by layer (e.g., `/operate/` for all operation-related code)
- **Entity format**: Entities are normalized to uppercase for consistency across the graph
- **Token counting**: Uses `JTokkit` library; important for chunk size calculations
- **Neo4j setup**: If using Neo4j storage, ensure the driver is configured with correct connection URI in environment
- **Embedding cache**: Controlled via `EMBEDDING_CACHE_ENABLED` environment variable to reduce API calls

## MCP Server Configuration

This project is configured to use **Bash/CLI tools** for managing builds and tests. The Gradle wrapper and shell scripts are the primary interfaces for development workflows.

If your team adds **Neo4j** or **MongoDB** instances for development/testing, consider adding those MCP servers to streamline database exploration and debugging.
