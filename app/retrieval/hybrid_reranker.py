import math
import re
from typing import List, Dict, Tuple
from app.models.schemas import DocumentChunk, SearchResultChunk

class HybridRetriever:
    """
    Production Hybrid Search Engine:
    Combines BM25 Keyword Scoring and Cosine Vector Scoring with Reciprocal Rank Fusion (RRF).
    """

    def __init__(self):
        self.chunks_db: Dict[str, DocumentChunk] = {}
        # Inverted index for BM25: term -> {chunk_id: frequency}
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        # Simple vocabulary vector storage for semantic TF-IDF embedding search
        self.tf_idf_vectors: Dict[str, Dict[str, float]] = {}

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r'\w+', text)]

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Index chunks into BM25 and Vector stores."""
        for chunk in chunks:
            self.chunks_db[chunk.id] = chunk
            tokens = self._tokenize(chunk.content)
            self.doc_lengths[chunk.id] = len(tokens)

            # Build inverted index
            for term in tokens:
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][chunk.id] = self.inverted_index[term].get(chunk.id, 0) + 1

        total_tokens = sum(self.doc_lengths.values())
        total_docs = len(self.chunks_db)
        self.avg_doc_length = (total_tokens / total_docs) if total_docs > 0 else 1.0

        # Build TF-IDF vectors
        for chunk_id, chunk in self.chunks_db.items():
            tokens = self._tokenize(chunk.content)
            doc_len = len(tokens)
            vec: Dict[str, float] = {}
            for term in set(tokens):
                tf = tokens.count(term) / max(1, doc_len)
                df = len(self.inverted_index.get(term, {}))
                idf = math.log((total_docs + 1) / (df + 1)) + 1.0
                vec[term] = tf * idf

            # Normalize vector
            norm = math.sqrt(sum(val ** 2 for val in vec.values()))
            if norm > 0:
                for term in vec:
                    vec[term] /= norm
            self.tf_idf_vectors[chunk_id] = vec

    def bm25_search(self, query: str, workspace_id: str, k1: float = 1.5, b: float = 0.75) -> List[Tuple[str, float]]:
        """Calculate BM25 scores for query."""
        query_terms = self._tokenize(query)
        scores: Dict[str, float] = {}
        total_docs = len(self.chunks_db)

        for term in query_terms:
            if term not in self.inverted_index:
                continue
            matching_docs = self.inverted_index[term]
            df = len(matching_docs)
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

            for chunk_id, freq in matching_docs.items():
                chunk = self.chunks_db.get(chunk_id)
                if not chunk or chunk.workspace_id != workspace_id:
                    continue

                doc_len = self.doc_lengths.get(chunk_id, 1)
                num = freq * (k1 + 1)
                denom = freq + k1 * (1 - b + b * (doc_len / max(1.0, self.avg_doc_length)))
                scores[chunk_id] = scores.get(chunk_id, 0.0) + (idf * (num / denom))

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def vector_search(self, query: str, workspace_id: str) -> List[Tuple[str, float]]:
        """Calculate Cosine Vector Similarity scores."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Build query vector
        total_docs = len(self.chunks_db)
        q_vec: Dict[str, float] = {}
        for term in set(query_tokens):
            tf = query_tokens.count(term) / len(query_tokens)
            df = len(self.inverted_index.get(term, {}))
            idf = math.log((total_docs + 1) / (df + 1)) + 1.0
            q_vec[term] = tf * idf

        q_norm = math.sqrt(sum(val ** 2 for val in q_vec.values()))
        if q_norm > 0:
            for term in q_vec:
                q_vec[term] /= q_norm

        scores: Dict[str, float] = {}
        for chunk_id, doc_vec in self.tf_idf_vectors.items():
            chunk = self.chunks_db.get(chunk_id)
            if not chunk or chunk.workspace_id != workspace_id:
                continue

            # Dot product
            score = sum(q_vec[t] * doc_vec.get(t, 0.0) for t in q_vec if t in doc_vec)
            if score > 0:
                scores[chunk_id] = score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def hybrid_search(self, query: str, workspace_id: str, top_k: int = 5, k_rrf: int = 60) -> List[SearchResultChunk]:
        """
        Reciprocal Rank Fusion (RRF) to merge BM25 and Vector results.
        RRF Score = 1 / (60 + rank_bm25) + 1 / (60 + rank_vector)
        """
        bm25_res = self.bm25_search(query, workspace_id)
        vector_res = self.vector_search(query, workspace_id)

        bm25_ranks = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(bm25_res)}
        vector_ranks = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(vector_res)}

        bm25_scores_dict = dict(bm25_res)
        vector_scores_dict = dict(vector_res)

        all_candidate_ids = set(bm25_ranks.keys()).union(set(vector_ranks.keys()))
        fusion_scores: Dict[str, float] = {}

        for chunk_id in all_candidate_ids:
            score = 0.0
            if chunk_id in bm25_ranks:
                score += 1.0 / (k_rrf + bm25_ranks[chunk_id])
            if chunk_id in vector_ranks:
                score += 1.0 / (k_rrf + vector_ranks[chunk_id])
            fusion_scores[chunk_id] = score

        sorted_candidates = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[SearchResultChunk] = []
        for chunk_id, score in sorted_candidates:
            chunk = self.chunks_db[chunk_id]
            results.append(SearchResultChunk(
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                content=chunk.content,
                score=round(score, 5),
                bm25_score=round(bm25_scores_dict.get(chunk_id, 0.0), 4),
                vector_score=round(vector_scores_dict.get(chunk_id, 0.0), 4),
                metadata=chunk.metadata
            ))

        return results
