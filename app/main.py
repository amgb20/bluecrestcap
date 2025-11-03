"""FastAPI application for the RAG agent."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent import RAGAgent
from app.models import QueryRequest, QueryResponse

# Initialize FastAPI app
app = FastAPI(
    title="RAG Agent API",
    description="Front-office RAG service for fund letters and market data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG agent (singleton)
rag_agent: RAGAgent = None


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG agent on startup."""
    global rag_agent
    print("Initializing RAG agent...")
    rag_agent = RAGAgent()
    print("RAG agent ready!")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Agent API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/query (POST)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if rag_agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return {
        "status": "healthy",
        "agent": "ready"
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Process a query and return an answer with citations.
    
    Args:
        request: QueryRequest with the question
        
    Returns:
        QueryResponse with answer, citations, and metadata
    """
    if rag_agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        response = rag_agent.query(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
