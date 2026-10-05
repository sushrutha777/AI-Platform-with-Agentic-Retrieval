# Enterprise AI Platform with Agentic Retrieval

Full-stack agentic RAG application with a FastAPI backend, React frontend, LangGraph orchestration, hybrid document retrieval, web/Wikipedia tools, streaming responses, and a modular document-ingestion pipeline.

## Current architecture

The request flow is:

~~~text
User query
  -> Model Armor input scan
  -> conversation restoration
  -> LangGraph context resolver
  -> heuristic intent router
  -> direct response OR parallel retrieval
  -> context aggregation
  -> one synthesis LLM call
  -> SSE stream to React
~~~

The graph in [app/graph/](app/graph/) contains five stages:

1. 'context_resolver' resolves follow-up questions using conversation history.
2. 'router' performs zero-LLM intent classification.
3. 'parallel_retrieval' runs the selected tools concurrently with fault isolation.
4. 'context_aggregator' combines tool output and source metadata.
5. 'generate' streams the final response through the LLM gateway.

The request-facing orchestration is implemented in [app/services/chat_service.py](app/services/chat_service.py), and the HTTP route is [app/api/v1/chat.py](app/api/v1/chat.py).

## Query routing

- Greetings and farewells are answered immediately without retrieval.
- Casual questions use a direct LLM prompt.
- Knowledge questions are routed to local document search, web search, Wikipedia, or a combination.
- Temporal terms such as 'today', 'latest', 'news', and 'price' select web search.
- Encyclopedic terms such as 'who is', 'history', and 'capital of' select Wikipedia and web search.
- Document terms such as 'policy', 'PDF', 'manual', and 'according to' select local document search.
- Unclassified knowledge queries use document and web search together.

## Retrieval architecture

Local document retrieval combines:

1. Dense semantic search against Qdrant using Gemini embeddings.
2. Sparse keyword search using BM25.
3. Reciprocal Rank Fusion in [app/retriever/hybrid.py](app/retriever/hybrid.py).
4. Optional FlashRank reranking.

Qdrant runs locally using the persistent [qdrant_data/](qdrant_data/) directory, or remotely through 'QDRANT_URL' and 'QDRANT_API_KEY'.

## Project structure

~~~text
app/
  agents/          Intent routing
  api/v1/          FastAPI endpoints
  context/         Conversation history and query rewriting
  graph/           LangGraph state, nodes, and builder
  guardrails/      Google Cloud Model Armor input scanning
  llm/             LLM gateway and provider fallback
  retriever/       Dense, sparse, and hybrid retrieval
  reranker/        FlashRank and no-op rerankers
  services/        Chat orchestration and SSE streaming
  tools/           Document, web, and Wikipedia tools
ingestion_platform/
  connectors/      PDF and text connectors
  stages/          Cleaning, chunking, embedding, and indexing
  core/            Pipeline and shared models
frontend/          React/Vite chat interface
data/              Example source documents
eval/              Benchmark and evaluation scripts
tests/             Unit and integration tests
deploy/            GCP deployment scripts and guide
~~~

## Technology stack

| Area | Technology |
| --- | --- |
| Backend | Python 3.12, FastAPI, Uvicorn, Pydantic Settings |
| Orchestration | LangGraph, asyncio |
| LLM gateway | LiteLLM SDK with Gemini primary and Groq fallback |
| Embeddings | Google Gemini embeddings |
| Vector search | Qdrant |
| Sparse search | Rank-BM25 |
| Reranking | FlashRank |
| External search | Tavily or DuckDuckGo, Wikipedia |
| Frontend | React, Vite, React Markdown, Lucide React |
| Deployment | Docker Compose, Google Cloud Run, Cloud Build |

## Configuration

Create a '.env' file in the repository root. A minimal configuration is:

~~~env
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key

LLM_MODEL=gemini/gemini-3.1-flash-lite
FALLBACK_LLM_MODEL=groq/llama-3.3-70b-versatile
EMBEDDING_MODEL=models/gemini-embedding-001

QDRANT_COLLECTION_NAME=agentic_rag_documents
# Leave QDRANT_URL empty for local persistent storage.
QDRANT_URL=
QDRANT_API_KEY=

TAVILY_API_KEY=
USE_HYBRID_SEARCH=true
USE_RERANKER=true
RETRIEVAL_TOP_K=15
RERANKED_TOP_K=5

ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO
~~~

Optional integrations include LangSmith tracing, Google Cloud Speech-to-Text, and Google Cloud Model Armor. See [app/core/config.py](app/core/config.py) for all settings.

'DEBUG' must be a boolean such as 'true' or 'false'; values such as 'release' are invalid for the current Pydantic settings model.

## Run locally

### Backend

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
~~~

The backend runs at 'http://localhost:8000'; Swagger UI is at 'http://localhost:8000/docs'.

### Frontend

~~~powershell
cd frontend
npm install
npm run dev
~~~

The Vite development server normally runs at 'http://localhost:5173'.

### Docker Compose

~~~bash
docker compose up --build
~~~

Check [docker-compose.yml](docker-compose.yml) for service ports and environment variables.

## Document ingestion

The independent ingestion platform processes PDF and text files through:

~~~text
Connectors -> cleaning -> chunking -> Gemini embeddings -> Qdrant indexing
~~~

Run ingestion with:

~~~bash
python ingest.py --source ./data
python ingest.py --source ./data --reindex
python -m ingestion_platform.cli ./data
~~~

The sample documents are in [data/](data/). The indexer uses deterministic point IDs, making repeated ingestion idempotent for unchanged chunks.

## API endpoints

- 'POST /api/v1/chat/stream' — SSE stream containing progress, tokens, metadata, citations, and completion events.
- 'GET /api/v1/health' — health endpoint; readiness routes are also available under this API group.
- 'POST /api/v1/voice/transcribe' — optional speech-to-text integration.
- '/api/v1/eval/*' — evaluation endpoints when configured.

Request and response schemas are defined in [app/schemas/chat.py](app/schemas/chat.py). Use '/docs' for the live OpenAPI contract.

## Testing and evaluation

Run tests from the project virtual environment:

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q
~~~

The suite covers routing, the LangGraph pipeline, backend behavior, SSE streaming, guardrails, and evaluation metrics.

Run evaluation scripts with:

~~~bash
python eval/evaluate.py
python eval/evaluate_ragas.py
~~~

Evaluation supports an empirical lexical benchmark and can write results to [eval/ragas_report.json](eval/ragas_report.json). Results depend on configured providers, indexed documents, network access, and API keys.

## Guardrails and observability

When Model Armor is enabled, the raw latest user message is scanned before session restoration, query rewriting, routing, retrieval, or LLM execution. Blocked or unavailable scans are returned as explicit SSE events. Retrieved content is treated as untrusted evidence, not as executable instructions.

Enable LangSmith tracing with:

~~~env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=Agentic RAG
~~~

## Deployment

The repository includes:

- [Dockerfile](Dockerfile) for the backend container.
- [frontend/Dockerfile](frontend/Dockerfile) for the frontend build.
- [docker-compose.yml](docker-compose.yml) for local multi-service execution.
- [cloudbuild.yaml](cloudbuild.yaml) for Google Cloud Build.
- [deploy/deploy-gcp.ps1](deploy/deploy-gcp.ps1) and [deploy/deploy-gcp.sh](deploy/deploy-gcp.sh) for Cloud Run deployment.
- [deploy/GCP_DEPLOYMENT.md](deploy/GCP_DEPLOYMENT.md) for deployment instructions.

For production, provide provider credentials and Qdrant settings through secret management rather than committing them to '.env' or source control.

## Development notes

- External search tools can fail independently; the graph continues with successful retrieval results.
- The frontend stores conversation history locally and sends it to the stateless backend when needed.
- Provider names, model defaults, retrieval settings, and feature flags are controlled by [app/core/config.py](app/core/config.py).
