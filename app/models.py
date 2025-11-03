"""Pydantic models for the RAG agent."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation reference to a document chunk."""
    doc_id: str = Field(description="Document identifier")
    start_char: int = Field(description="Start character position")
    end_char: int = Field(description="End character position")
    text_snippet: str = Field(description="Text snippet from the document")
    source: str = Field(description="Source file name")


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""
    source: str = Field(description="Source file name")
    doc_type: str = Field(description="Type of document (html, pdf, csv)")
    section_title: Optional[str] = Field(None, description="Section title if available")
    timestamp: Optional[str] = Field(None, description="Timestamp for chat logs")
    author: Optional[str] = Field(None, description="Author for chat logs")
    chunk_index: int = Field(description="Index of the chunk in the document")


class RetrievedChunk(BaseModel):
    """A retrieved chunk from the vector store."""
    id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Chunk text content")
    metadata: ChunkMetadata = Field(description="Chunk metadata")
    score: float = Field(description="Retrieval score")
    vector_score: Optional[float] = Field(None, description="Vector similarity score")
    bm25_score: Optional[float] = Field(None, description="BM25 score")


class QueryRequest(BaseModel):
    """Request model for queries."""
    query: str = Field(description="Question to answer")
    top_k: int = Field(5, description="Number of chunks to retrieve")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")


class QueryMetadata(BaseModel):
    """Metadata about query execution."""
    query_id: str = Field(description="Unique query identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    retrieval_latency_ms: Optional[float] = Field(None, description="Retrieval time in ms")
    llm_latency_ms: Optional[float] = Field(None, description="LLM generation time in ms")
    total_latency_ms: Optional[float] = Field(None, description="Total time in ms")
    chunks_retrieved: int = Field(0, description="Number of chunks retrieved")
    tools_used: List[str] = Field(default_factory=list, description="Tools invoked")
    token_count: Optional[int] = Field(None, description="Total tokens used")


class QueryResponse(BaseModel):
    """Response model for queries."""
    answer: str = Field(description="Generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Citations used")
    sources: List[str] = Field(default_factory=list, description="Source documents")
    metadata: QueryMetadata = Field(description="Query execution metadata")


class Metric(BaseModel):
    """Individual metric for observability."""
    query_id: str = Field(description="Associated query ID")
    metric_name: str = Field(description="Name of the metric")
    value: float = Field(description="Metric value")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional context")
