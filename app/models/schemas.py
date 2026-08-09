from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import time
import uuid

class WorkspaceCreate(BaseModel):
    name: str = Field(..., json_schema_extra={"example": "Company Policies"})
    description: Optional[str] = Field(None, json_schema_extra={"example": "HR, IT, and Security policies documentation"})

class Workspace(BaseModel):
    id: str = Field(default_factory=lambda: f"ws_{uuid.uuid4().hex[:8]}")
    name: str
    description: Optional[str] = None
    created_at: float = Field(default_factory=time.time)

class DocumentUpload(BaseModel):
    workspace_id: str
    filename: str
    content: str
    file_type: str = Field("txt", json_schema_extra={"example": "pdf, md, txt"})

class DocumentChunk(BaseModel):
    id: str
    doc_id: str
    workspace_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchQuery(BaseModel):
    workspace_id: str
    query: str
    top_k: int = 5
    search_type: str = Field("hybrid", json_schema_extra={"example": "hybrid, vector, bm25"})

class SearchResultChunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Citation(BaseModel):
    id: int
    chunk_id: str
    doc_id: str
    source_filename: str
    excerpt: str
    valid: bool = True

class GroundedAnswerRequest(BaseModel):
    workspace_id: str
    query: str
    search_type: str = "hybrid"
    top_k: int = 5

class GroundedAnswerResponse(BaseModel):
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[SearchResultChunk]
    citation_valid: bool
    confidence: str = Field("high", json_schema_extra={"example": "high, low"})
    request_id: str
    latency_ms: float
    tokens_used: int
    cost_usd: float

class HealthCheckResponse(BaseModel):
    status: str = "ok"
    database: str = "connected"
    vector_store: str = "connected"
    llm_provider: str = "available"

class SystemLogEntry(BaseModel):
    request_id: str
    query: str
    retrieval_method: str
    num_chunks_retrieved: int
    num_chunks_used: int
    llm_provider: str
    latency_ms: float
    tokens_used: int
    cost_usd: float
    citation_valid: bool
    error: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
