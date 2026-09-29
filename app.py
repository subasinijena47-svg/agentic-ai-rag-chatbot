from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from src.graph import build_rag_graph

app = FastAPI(
    title="Agentic AI RAG API",
    description="Strict RAG Chatbot powered by LangGraph, Pinecone, and OpenAI",
    version="1.0.0"
)

# Initialize graph lazily or on startup
graph = None

@app.on_event("startup")
def startup_event():
    global graph
    try:
        index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
        graph = build_rag_graph(index_name=index_name)
    except Exception as e:
        print(f"Warning during startup: {e}")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    final_answer: str
    retrieved_context: List[str]
    confidence_score: float

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Agentic AI RAG Chatbot API",
        "endpoints": {
            "POST /chat": "Submit a question to the RAG system",
            "GET /docs": "Interactive Swagger API documentation"
        }
    }

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    global graph
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    if graph is None:
        index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
        graph = build_rag_graph(index_name=index_name)

    initial_state = {
        "question": request.query,
        "context": [],
        "answer": "",
        "score": 0.0
    }
    
    result = graph.invoke(initial_state)
    
    return QueryResponse(
        final_answer=result["answer"],
        retrieved_context=result["context"],
        confidence_score=result["score"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
