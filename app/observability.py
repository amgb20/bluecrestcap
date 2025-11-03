"""Observability and metrics tracking for the RAG agent."""
import json
import time
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and emits metrics for observability."""
    
    def __init__(self):
        self.metrics: List[Dict[str, Any]] = []
    
    def log_metric(self, metric_name: str, value: float, query_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a metric."""
        metric_entry = {
            "metric_name": metric_name,
            "value": value,
            "query_id": query_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.metrics.append(metric_entry)
        logger.info(json.dumps({"type": "metric", **metric_entry}))
    
    def log_query(self, query_id: str, query: str, query_type: str):
        """Log a query."""
        log_entry = {
            "type": "query",
            "query_id": query_id,
            "query": query,
            "query_type": query_type,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(json.dumps(log_entry))
    
    def log_retrieval(self, query_id: str, chunks: List[Dict[str, Any]], latency_ms: float):
        """Log retrieval results."""
        log_entry = {
            "type": "retrieval",
            "query_id": query_id,
            "chunks_count": len(chunks),
            "latency_ms": latency_ms,
            "chunk_scores": [c.get("score", 0) for c in chunks[:5]],
            "timestamp": datetime.now().isoformat()
        }
        logger.info(json.dumps(log_entry))
        self.log_metric("retrieval_latency_ms", latency_ms, query_id)
        self.log_metric("chunks_retrieved", len(chunks), query_id)
    
    def log_tool_usage(self, query_id: str, tool_name: str, success: bool, latency_ms: Optional[float] = None):
        """Log tool usage."""
        log_entry = {
            "type": "tool_usage",
            "query_id": query_id,
            "tool_name": tool_name,
            "success": success,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(json.dumps(log_entry))
    
    def log_response(self, query_id: str, answer: str, citations_count: int, total_latency_ms: float, token_count: Optional[int] = None):
        """Log final response."""
        log_entry = {
            "type": "response",
            "query_id": query_id,
            "answer_length": len(answer),
            "citations_count": citations_count,
            "total_latency_ms": total_latency_ms,
            "token_count": token_count,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(json.dumps(log_entry))
        self.log_metric("total_latency_ms", total_latency_ms, query_id)
        self.log_metric("citations_count", citations_count, query_id)
        if token_count:
            self.log_metric("token_count", token_count, query_id)
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get all collected metrics."""
        return self.metrics


@contextmanager
def timer():
    """Context manager for timing operations."""
    start = time.time()
    times = {"elapsed_ms": 0}
    try:
        yield times
    finally:
        times["elapsed_ms"] = (time.time() - start) * 1000


def generate_query_id() -> str:
    """Generate a unique query ID."""
    return str(uuid.uuid4())


# Global metrics collector instance
metrics_collector = MetricsCollector()
