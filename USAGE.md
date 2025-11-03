# RAG Agent Usage Guide

This guide walks you through setting up and running the RAG agent system.

## Prerequisites

- Python 3.11+
- OpenAI API key
- Docker (optional, for Qdrant)

## Setup

### 1. Environment Configuration

Create a `.env` file in the project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration  
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=rag_documents

# Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Qdrant (Optional - uses Docker)

```bash
docker compose -f docker/docker-compose.yml up -d qdrant
```

Alternatively, you can run Qdrant locally or use Qdrant Cloud.

### 4. Ingest Documents

This will parse the HTML, PDF, and CSV files, generate embeddings, and store them in Qdrant:

```bash
make ingest
# or
python -m app.ingest
```

Expected output:
- HTML parsing with section-based chunking
- PDF parsing with paragraph chunking
- CSV parsing (one message per chunk)
- Embedding generation using OpenAI
- Storage in Qdrant vector database

## Running the Service

### Start the API Server

```bash
make run
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### POST /query
Query the RAG agent:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What factors were tilted towards in Q2?"}'
```

Response format:
```json
{
  "answer": "In Q2, the strategy maintained modest tilts to Momentum and Quality...",
  "citations": [
    {
      "doc_id": "abc123",
      "start_char": 0,
      "end_char": 200,
      "text_snippet": "...",
      "source": "q2_letter.html"
    }
  ],
  "sources": ["q2_letter.html"],
  "metadata": {
    "query_id": "uuid-here",
    "total_latency_ms": 1234.5,
    "chunks_retrieved": 5,
    "tools_used": ["document_search"]
  }
}
```

#### GET /health
Health check:

```bash
curl http://localhost:8000/health
```

## Running Evaluation

The evaluation harness runs all queries from `eval/queries.jsonl`:

```bash
make eval
# or
python eval/run_eval.py
```

This will:
1. Check API health
2. Run all evaluation queries
3. Collect metrics (latency, citations, tool usage)
4. Generate a report
5. Save results to `eval/results.json`

## Running Tests

```bash
make test
# or
pytest -q
```

## Architecture Overview

### Components

1. **Document Ingestion** (`app/ingest.py`)
   - Parses HTML (section-based), PDF (paragraph-based), CSV (message-based)
   - Generates OpenAI embeddings
   - Stores in Qdrant with rich metadata

2. **Hybrid Retriever** (`app/retriever.py`)
   - Vector search (cosine similarity via Qdrant)
   - BM25 keyword search (rank-bm25)
   - Reciprocal Rank Fusion (RRF) to combine results

3. **Tools** (`app/tools/prices.py`)
   - Price lookup from `prices_stub/prices.json`
   - Latest price, historical, and comparison queries
   - Natural language query interface

4. **Agent** (`app/agent.py`)
   - LangChain OpenAI Functions Agent
   - Tool routing (document search vs price lookup)
   - Citation extraction and response generation

5. **Observability** (`app/observability.py`)
   - Structured JSON logging
   - Metrics: latency, tool usage, retrieval quality
   - Query tracking with unique IDs

6. **API** (`app/main.py`)
   - FastAPI server
   - `/query` and `/health` endpoints
   - Request/response validation with Pydantic

7. **Evaluation** (`eval/run_eval.py`)
   - Automated testing of query performance
   - Metrics aggregation and reporting

### Data Flow

```
User Query
    ↓
FastAPI Endpoint
    ↓
RAG Agent (LangChain)
    ↓
Tool Selection
    ├─→ Document Search → Hybrid Retriever → Qdrant + BM25
    └─→ Price Lookup → PriceTool → prices.json
    ↓
Response Generation (OpenAI)
    ↓
Citation Extraction
    ↓
Response with Citations
```

## Key Features

### Hybrid Retrieval
Combines semantic (vector) and keyword (BM25) search for robust retrieval:
- Vector search captures meaning and context
- BM25 handles specific entity queries (ticker symbols, dates)
- RRF fusion balances both approaches

### Grounded Citations
Every answer includes traceable citations:
- Format: `[source_file@chunk_index]`
- Links to specific document chunks
- Enables verification and trust

### Tool Routing
Intelligent tool selection:
- Price queries → Price tool
- Document queries → Retrieval tool
- Hybrid queries → Both tools

### Observability
Comprehensive metrics:
- Query latency (retrieval, LLM, total)
- Tool usage patterns
- Citation coverage
- Success rates

## Example Queries

### Document Questions
```
"What factors were tilted towards in Q2?"
"Did the Q2 letter reference SPY?"
"Which document discusses liquidity filters?"
"What was mentioned about breadth observations?"
```

### Price Questions
```
"What is the most recent close for MSFT?"
"What was the last price for EURUSD?"
"Compare SPY performance to QQQ over the last 10 days."
```

### Hybrid Questions
```
"According to the addendum, what SPY close should be preferred for latest close questions?"
"Did we maintain a momentum tilt in Q2?"
```

## Troubleshooting

### Qdrant Connection Issues
- Ensure Qdrant is running: `docker ps | grep qdrant`
- Check QDRANT_HOST and QDRANT_PORT in `.env`

### OpenAI API Errors
- Verify OPENAI_API_KEY is set correctly
- Check API quota and rate limits

### Empty Results
- Ensure documents are ingested: `make ingest`
- Check Qdrant collection: visit `http://localhost:6333/dashboard`

### Slow Queries
- First query may be slow due to cold start
- Embedding generation adds latency
- Consider caching frequently accessed data

## Performance Tips

1. **Batch embedding generation** during ingestion
2. **Adjust chunk sizes** for your use case (currently ~500 chars)
3. **Tune top_k** for retrieval (default: 5)
4. **Use faster models** for prototyping (gpt-3.5-turbo)
5. **Cache embeddings** to avoid regeneration

## Next Steps

- Fine-tune chunking strategy for your documents
- Add more data sources
- Implement semantic caching
- Add user feedback collection
- Deploy to production (e.g., AWS, GCP)
- Add authentication and rate limiting


