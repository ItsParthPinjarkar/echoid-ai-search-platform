from fastapi.testclient import TestClient
from app.main import app
from app.ingestion.chunker import clean_text, chunk_text
from app.retrieval.hybrid_reranker import HybridRetriever
from app.generation.citation_validator import validate_citations
from app.models.schemas import SearchResultChunk

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"

def test_clean_and_chunk_text():
    raw_text = "  Hello   world! \r\n This is a test paragraph.  "
    cleaned = clean_text(raw_text)
    assert "Hello world!" in cleaned

    chunks = chunk_text(cleaned, "doc_test", "ws_test", "test.txt", chunk_size=10, overlap=2)
    assert len(chunks) >= 1
    assert chunks[0].doc_id == "doc_test"

def test_hybrid_retriever():
    retriever = HybridRetriever()
    chunks = chunk_text("Python is a popular programming language for AI and machine learning.", "doc_py", "ws_1", "py.txt")
    retriever.index_chunks(chunks)

    results = retriever.hybrid_search("programming language", "ws_1", top_k=5)
    assert len(results) >= 1
    assert results[0].doc_id == "doc_py"

def test_citation_validator():
    chunk = SearchResultChunk(
        chunk_id="chunk_doc1_0",
        doc_id="doc1",
        content="Remote work is allowed up to 3 days per week.",
        score=0.9
    )
    answer = "Employees can work remotely up to 3 days [chunk_doc1_0]."
    citations, valid = validate_citations(answer, [chunk])
    assert valid is True
    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk_doc1_0"

def test_ask_endpoint():
    payload = {
        "workspace_id": "ws_default",
        "query": "What is the remote work policy?",
        "search_type": "hybrid",
        "top_k": 3
    }
    response = client.post("/api/v1/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert data["citation_valid"] is True
    assert data["tokens_used"] > 0
