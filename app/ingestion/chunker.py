import re
import hashlib
from typing import List
from app.models.schemas import DocumentChunk

def clean_text(text: str) -> str:
    """Normalize text by removing excess whitespace and non-standard control characters."""
    if not text:
        return ""
    # Normalize newline breaks
    text = re.sub(r'\r\n', '\n', text)
    # Remove control characters except standard whitespace
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def chunk_text(
    text: str,
    doc_id: str,
    workspace_id: str,
    filename: str,
    chunk_size: int = 300,
    overlap: int = 50
) -> List[DocumentChunk]:
    """
    Split text into overlapping chunks and build metadata structures.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    words = cleaned.split(" ")
    chunks: List[DocumentChunk] = []

    step = max(1, chunk_size - overlap)
    chunk_idx = 0
    seen_hashes = set()

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunk_str = " ".join(chunk_words).strip()
        if not chunk_str:
            continue

        # Content hash deduplication
        chunk_hash = hashlib.md5(chunk_str.encode("utf-8")).hexdigest()
        if chunk_hash in seen_hashes:
            continue
        seen_hashes.add(chunk_hash)

        chunk_id = f"chunk_{doc_id}_{chunk_idx}"
        chunk = DocumentChunk(
            id=chunk_id,
            doc_id=doc_id,
            workspace_id=workspace_id,
            content=chunk_str,
            chunk_index=chunk_idx,
            metadata={
                "filename": filename,
                "word_count": len(chunk_words),
                "content_hash": chunk_hash,
                "start_word_index": i,
            }
        )
        chunks.append(chunk)
        chunk_idx += 1

    return chunks
