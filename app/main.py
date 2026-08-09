import time
import uuid
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import (
    Workspace, WorkspaceCreate, DocumentUpload, DocumentChunk,
    SearchQuery, SearchResultChunk, GroundedAnswerRequest,
    GroundedAnswerResponse, HealthCheckResponse, SystemLogEntry
)
from app.ingestion.chunker import chunk_text
from app.retrieval.hybrid_reranker import HybridRetriever
from app.generation.citation_validator import generate_grounded_answer
from app.observability.logger import logger_instance

app = FastAPI(
    title="AI Search + Evaluation Platform API",
    description="Hybrid retrieval, citation validation, and evaluation platform backend",
    version="0.1.0"
)

# Enable CORS for local React dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory storage
workspaces_db: Dict[str, Workspace] = {}
retriever = HybridRetriever()

# Initialize default workspace
default_ws = Workspace(id="ws_default", name="Company Policies", description="Default company policies workspace")
workspaces_db[default_ws.id] = default_ws

# Seed default documents for immediate testing
sample_text_1 = """
Employees can work remotely up to 3 days per week with manager approval. 
Remote work requests must be submitted through the HR portal at least 2 weeks in advance. 
Security policy requires all remote connections to use the company corporate VPN with multi-factor authentication.
"""
sample_text_2 = """
Annual paid time off (PTO) accrual rate is 20 days per year for full-time employees. 
Unused PTO up to 5 days can be rolled over to the next calendar year. 
Sick leave requests should be logged into the HR portal on the day of absence.
"""
chunks_1 = chunk_text(sample_text_1, "doc_hr_1", default_ws.id, "employee_handbook.pdf")
chunks_2 = chunk_text(sample_text_2, "doc_hr_2", default_ws.id, "hr_faq.md")
retriever.index_chunks(chunks_1 + chunks_2)


@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    return HealthCheckResponse()


@app.post("/api/v1/workspaces", response_model=Workspace)
def create_workspace(ws: WorkspaceCreate):
    workspace = Workspace(name=ws.name, description=ws.description)
    workspaces_db[workspace.id] = workspace
    return workspace


@app.get("/api/v1/workspaces", response_model=List[Workspace])
def list_workspaces():
    return list(workspaces_db.values())


@app.post("/api/v1/documents", response_model=Dict[str, Any])
def upload_document(doc: DocumentUpload):
    if doc.workspace_id not in workspaces_db:
        raise HTTPException(status_code=404, detail="Workspace not found")

    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    chunks = chunk_text(doc.content, doc_id, doc.workspace_id, doc.filename)
    retriever.index_chunks(chunks)

    return {
        "doc_id": doc_id,
        "filename": doc.filename,
        "chunks_created": len(chunks),
        "workspace_id": doc.workspace_id
    }


@app.post("/api/v1/search", response_model=List[SearchResultChunk])
def search_documents(query: SearchQuery):
    if query.search_type == "bm25":
        results = retriever.bm25_search(query.query, query.workspace_id)
        res_chunks = []
        for chunk_id, score in results[:query.top_k]:
            c = retriever.chunks_db[chunk_id]
            res_chunks.append(SearchResultChunk(
                chunk_id=c.id, doc_id=c.doc_id, content=c.content,
                score=round(score, 4), bm25_score=round(score, 4), vector_score=0.0,
                metadata=c.metadata
            ))
        return res_chunks
    elif query.search_type == "vector":
        results = retriever.vector_search(query.query, query.workspace_id)
        res_chunks = []
        for chunk_id, score in results[:query.top_k]:
            c = retriever.chunks_db[chunk_id]
            res_chunks.append(SearchResultChunk(
                chunk_id=c.id, doc_id=c.doc_id, content=c.content,
                score=round(score, 4), bm25_score=0.0, vector_score=round(score, 4),
                metadata=c.metadata
            ))
        return res_chunks
    else: # Hybrid
        return retriever.hybrid_search(query.query, query.workspace_id, top_k=query.top_k)


@app.post("/api/v1/ask", response_model=GroundedAnswerResponse)
def ask_question(req: GroundedAnswerRequest):
    start_time = time.time()
    request_id = f"req_{uuid.uuid4().hex[:8]}"

    # Retrieve chunks
    chunks = retriever.hybrid_search(req.query, req.workspace_id, top_k=req.top_k)

    # Generate answer with citations
    answer, citations, citation_valid, confidence = generate_grounded_answer(req.query, chunks)

    latency_ms = (time.time() - start_time) * 1000.0
    tokens_used = len(req.query.split()) + sum(len(c.content.split()) for c in chunks) + len(answer.split())
    cost_usd = tokens_used * 0.000002

    # Log request to observability logger
    logger_instance.log_request(
        request_id=request_id,
        query=req.query,
        retrieval_method=req.search_type,
        num_chunks_retrieved=len(chunks),
        num_chunks_used=min(len(chunks), 2),
        llm_provider="openai",
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
        citation_valid=citation_valid
    )

    return GroundedAnswerResponse(
        answer=answer,
        citations=citations,
        retrieved_chunks=chunks,
        citation_valid=citation_valid,
        confidence=confidence,
        request_id=request_id,
        latency_ms=round(latency_ms, 2),
        tokens_used=tokens_used,
        cost_usd=round(cost_usd, 6)
    )


@app.get("/api/v1/logs", response_model=List[SystemLogEntry])
def get_system_logs(limit: int = 50):
    return logger_instance.get_logs(limit=limit)
