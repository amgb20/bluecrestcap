"""Hybrid retrieval with BM25 and vector search."""
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

load_dotenv()


class HybridRetriever:
    """Hybrid retriever combining BM25 and vector search."""
    
    def __init__(self):
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIM", "1536"))

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set for HybridRetriever when using OpenAI embeddings.")

        self.openai_client = OpenAI(api_key=api_key)
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "rag_documents")
        
        self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        
        # Load all documents for BM25
        self.documents = []
        self.doc_metadata = []
        self._load_documents()

        # Initialize BM25 if we have documents available
        if self.documents:
            tokenized_docs = [doc.lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            self.bm25 = None
    
    def _load_documents(self):
        """Load all documents from Qdrant for BM25 indexing."""
        try:
            # Scroll through all documents in the collection
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Adjust based on your collection size
                with_payload=True,
                with_vectors=False
            )
            
            points, _ = scroll_result
            
            for point in points:
                self.documents.append(point.payload['text'])
                self.doc_metadata.append({
                    'id': point.id,
                    'metadata': point.payload.get('metadata', {})
                })
            
            print(f"Loaded {len(self.documents)} documents for BM25 indexing")
        except Exception as e:
            print(f"Warning: Could not load documents for BM25: {e}")
            self.documents = []
            self.doc_metadata = []
            self.bm25 = None
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = self.openai_client.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def vector_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        query_embedding = self.generate_embedding(query)

        try:
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
        except Exception as exc:
            print(f"Warning: Vector search failed: {exc}")
            return []
        
        results = []
        for hit in search_result:
            results.append({
                'id': hit.id,
                'text': hit.payload['text'],
                'metadata': hit.payload.get('metadata', {}),
                'score': hit.score,
                'vector_score': hit.score
            })
        
        return results
    
    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search."""
        if not self.documents or not self.bm25:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with non-zero scores
                results.append({
                    'id': self.doc_metadata[idx]['id'],
                    'text': self.documents[idx],
                    'metadata': self.doc_metadata[idx]['metadata'],
                    'score': float(scores[idx]),
                    'bm25_score': float(scores[idx])
                })
        
        return results
    
    def reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score = sum(1 / (k + rank)) for each retrieval method
        """
        # Create a map of document ID to RRF score
        rrf_scores = {}
        doc_map = {}
        
        # Add vector search scores
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
            if doc_id not in doc_map:
                doc_map[doc_id] = result
        
        # Add BM25 scores
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
            if doc_id not in doc_map:
                doc_map[doc_id] = result
        
        # Sort by RRF score
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Build final results
        fused_results = []
        for doc_id in sorted_doc_ids:
            result = doc_map[doc_id].copy()
            result['score'] = rrf_scores[doc_id]
            fused_results.append(result)
        
        return fused_results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid search (True) or just vector search (False)
            
        Returns:
            List of retrieved documents with scores and metadata
        """
        if not use_hybrid:
            # Just vector search
            return self.vector_search(query, top_k)
        
        # Hybrid search with RRF
        vector_results = self.vector_search(query, top_k=10)
        bm25_results = self.bm25_search(query, top_k=10)
        
        # Fuse results
        fused_results = self.reciprocal_rank_fusion(vector_results, bm25_results)
        
        # Return top-k
        return fused_results[:top_k]
    
    def format_chunks_for_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks as context for the LLM.
        
        Each chunk is formatted with a citation ID that can be referenced.
        """
        context_parts = []
        for idx, chunk in enumerate(chunks):
            source = chunk['metadata'].get('source', 'unknown')
            chunk_idx = chunk['metadata'].get('chunk_index', 0)
            citation_id = f"{source}@{chunk_idx}"
            
            context_parts.append(f"[{citation_id}]\n{chunk['text']}\n")
        
        return "\n---\n".join(context_parts)
