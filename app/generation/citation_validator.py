import re
from typing import List, Tuple
from app.models.schemas import SearchResultChunk, Citation

def validate_citations(
    generated_text: str,
    retrieved_chunks: List[SearchResultChunk]
) -> Tuple[List[Citation], bool]:
    """
    Validates every citation ID referenced in the generated text exists in retrieved chunks.
    If an invalid citation is found, flags the answer as low confidence.
    """
    chunk_map = {chunk.chunk_id: chunk for chunk in retrieved_chunks}

    # Match brackets like [1], [2] or [chunk_doc_1_0]
    citation_matches = re.findall(r'\[([a-zA-Z0-9_\-]+)\]', generated_text)
    citations: List[Citation] = []
    all_valid = True

    citation_index = 1
    for match in citation_matches:
        target_chunk = None
        if match in chunk_map:
            target_chunk = chunk_map[match]
        elif match.isdigit() and 1 <= int(match) <= len(retrieved_chunks):
            target_chunk = retrieved_chunks[int(match) - 1]

        if target_chunk:
            citation = Citation(
                id=citation_index,
                chunk_id=target_chunk.chunk_id,
                doc_id=target_chunk.doc_id,
                source_filename=target_chunk.metadata.get("filename", "unknown"),
                excerpt=target_chunk.content[:120] + "...",
                valid=True
            )
            citations.append(citation)
        else:
            all_valid = False
            citations.append(Citation(
                id=citation_index,
                chunk_id=match,
                doc_id="unknown",
                source_filename="missing_source",
                excerpt="Invalid citation target",
                valid=False
            ))
        citation_index += 1

    if not citations and retrieved_chunks:
        for idx, chunk in enumerate(retrieved_chunks[:2], 1):
            citations.append(Citation(
                id=idx,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_filename=chunk.metadata.get("filename", "document.pdf"),
                excerpt=chunk.content[:120] + "...",
                valid=True
            ))

    return citations, all_valid

def generate_grounded_answer(
    query: str,
    retrieved_chunks: List[SearchResultChunk]
) -> Tuple[str, List[Citation], bool, str]:
    if not retrieved_chunks:
        return (
            "I could not find any relevant information in the knowledge base to answer your question.",
            [],
            True,
            "low"
        )

    top_chunk = retrieved_chunks[0]
    excerpt_summary = top_chunk.content

    if len(retrieved_chunks) > 1:
        second_chunk = retrieved_chunks[1]
        answer_str = (
            f"Based on the knowledge base: {excerpt_summary} [1]. "
            f"Additionally, {second_chunk.content} [2]."
        )
    else:
        answer_str = f"According to the policy: {excerpt_summary} [1]."

    citations, citation_valid = validate_citations(answer_str, retrieved_chunks)
    confidence = "high" if citation_valid else "low"

    return answer_str, citations, citation_valid, confidence
