import math
from typing import List, Set

def calculate_recall_at_k(retrieved_doc_ids: List[str], expected_doc_ids: Set[str], k: int = 5) -> float:
    top_k = set(retrieved_doc_ids[:k])
    relevant_retrieved = top_k.intersection(expected_doc_ids)
    return len(relevant_retrieved) / len(expected_doc_ids) if expected_doc_ids else 0.0

def calculate_precision_at_k(retrieved_doc_ids: List[str], expected_doc_ids: Set[str], k: int = 5) -> float:
    top_k = retrieved_doc_ids[:k]
    if not top_k:
        return 0.0
    relevant_retrieved = [doc_id for doc_id in top_k if doc_id in expected_doc_ids]
    return len(relevant_retrieved) / len(top_k)

def calculate_mrr(retrieved_doc_ids: List[str], expected_doc_ids: Set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, 1):
        if doc_id in expected_doc_ids:
            return 1.0 / rank
    return 0.0

def calculate_ndcg_at_k(retrieved_doc_ids: List[str], expected_doc_ids: Set[str], k: int = 5) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k], 1):
        rel = 1.0 if doc_id in expected_doc_ids else 0.0
        dcg += rel / math.log2(i + 1)
    
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(expected_doc_ids), k) + 1))
    return (dcg / idcg) if idcg > 0 else 0.0

def calculate_faithfulness(generated_answer: str, retrieved_texts: List[str]) -> float:
    """Check how much of the generated answer is grounded in retrieved texts."""
    if not generated_answer or not retrieved_texts:
        return 0.0
    combined_context = " ".join(retrieved_texts).lower()
    answer_words = [w.lower() for w in generated_answer.split() if len(w) > 3]
    if not answer_words:
        return 1.0
    grounded_count = sum(1 for word in answer_words if word in combined_context)
    return grounded_count / len(answer_words)
