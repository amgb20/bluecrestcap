# Front-Office RAG Agent Exercise

A production-quality RAG (Retrieval-Augmented Generation) service for querying fund letters, market data, and team discussions with grounded, cited responses.

## Features

✅ **Hybrid Retrieval** - Combines BM25 keyword search with vector semantic search using Reciprocal Rank Fusion  
✅ **Multi-format Ingestion** - Parses HTML, PDF, and CSV documents with intelligent chunking  
✅ **Tool Routing** - LangChain agent routes queries to document search or price lookup tools  
✅ **Grounded Citations** - Every answer includes traceable citations in format `[source@chunk_index]`  
✅ **Observability** - Comprehensive metrics tracking (latency, tool usage, citations)  
✅ **FastAPI Service** - RESTful API with `/query` and `/health` endpoints  
✅ **Evaluation Harness** - Automated testing with metrics reporting  

## Quick Start

### 1. Setup Environment

Create a `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=rag_documents
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Qdrant (optional)

```bash
docker compose -f docker/docker-compose.yml up -d qdrant
```

### 4. Run the System

```bash
make ingest    # Ingest documents, generate embeddings, store in Qdrant
make run       # Start API server at http://localhost:8000
make test      # Run tests
make eval      # Run evaluation harness
```

## Architecture

### Core Components

- **Ingestion Pipeline** (`app/ingest.py`) - Parses HTML/PDF/CSV, chunks, embeds, stores in Qdrant
- **Hybrid Retriever** (`app/retriever.py`) - BM25 + vector search with RRF fusion
- **Price Tool** (`app/tools/prices.py`) - Queries market data from JSON
- **RAG Agent** (`app/agent.py`) - LangChain agent with tool routing
- **API Server** (`app/main.py`) - FastAPI endpoints
- **Observability** (`app/observability.py`) - Metrics and structured logging
- **Evaluation** (`eval/run_eval.py`) - Automated testing and reporting

### Tech Stack

- **LLM & Embeddings**: OpenAI (GPT-4o-mini, text-embedding-3-small)
- **Framework**: LangChain for agent orchestration
- **Vector DB**: Qdrant (cosine similarity)
- **Keyword Search**: rank-bm25 (BM25Okapi)
- **API**: FastAPI + Pydantic
- **Testing**: pytest

## Usage

### Query the API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What factors were tilted towards in Q2?"}'
```

### Example Queries

**Document Questions:**
- "What factors were tilted towards in Q2?"
- "Did the Q2 letter reference SPY?"
- "Which document discusses liquidity filters?"

**Price Questions:**
- "What is the most recent close for MSFT?"
- "Compare SPY performance to QQQ over the last 10 days."

**Hybrid Questions:**
- "According to the addendum, what SPY close should be preferred?"

## Documentation

See [USAGE.md](USAGE.md) for detailed documentation including:
- Architecture overview
- Data flow diagrams
- Configuration options
- Troubleshooting guide
- Performance optimization tips

## Key Design Decisions

1. **Hybrid Retrieval**: Combines semantic understanding (vector) with exact matching (BM25) for robust retrieval
2. **Section-based Chunking**: HTML chunks by headers, preserves semantic structure
3. **RRF Fusion**: Reciprocal Rank Fusion balances vector and keyword scores
4. **Citation System**: `[source@chunk]` format enables verification
5. **Tool Routing**: LangChain agent intelligently selects document search vs price lookup

## Evaluation Results

Run `make eval` to see:
- Success rate
- Latency (P50, P95, P99)
- Citation coverage
- Tool usage distribution
- Per-query results

## Project Structure

```
├── app/
│   ├── agent.py           # LangChain agent with tool routing
│   ├── ingest.py          # Document ingestion pipeline
│   ├── retriever.py       # Hybrid BM25 + vector retrieval
│   ├── models.py          # Pydantic data models
│   ├── observability.py   # Metrics and logging
│   ├── main.py            # FastAPI application
│   └── tools/
│       └── prices.py      # Price lookup tool
├── data/
│   ├── fund_letters/      # HTML and PDF documents
│   └── chat_logs/         # CSV chat logs
├── eval/
│   ├── queries.jsonl      # Evaluation queries
│   └── run_eval.py        # Evaluation harness
├── prompts/
│   └── answer_system.txt  # System prompt
├── requirements.txt       # Python dependencies
└── Makefile              # Common commands
```

