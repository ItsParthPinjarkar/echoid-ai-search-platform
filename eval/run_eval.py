import json
import yaml
import time
import sys
from pathlib import Path
from typing import Dict, Any

from app.models.schemas import GroundedAnswerRequest
from app.main import ask_question, retriever
from eval.metrics import (
    calculate_recall_at_k, calculate_precision_at_k,
    calculate_mrr, calculate_ndcg_at_k, calculate_faithfulness
)

def run_evaluation_suite():
    eval_dir = Path(__file__).parent
    dataset_path = eval_dir / "golden_dataset.json"
    config_path = eval_dir / "eval_config.yaml"

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    thresholds = config.get("thresholds", {})

    total_queries = len(golden_dataset)
    recalls = []
    precisions = []
    mrrs = []
    ndcgs = []
    faithfulness_scores = []
    citation_accuracies = []
    latencies = []

    print("=" * 60)
    print("AI Search + Evaluation Platform - Evaluation Benchmark")
    print(f"Dataset: {dataset_path.name} | Queries: {total_queries}")
    print("=" * 60)

    for item in golden_dataset:
        q_id = item["id"]
        question = item["question"]
        expected_docs = set(item["relevant_doc_ids"])

        start_time = time.time()
        req = GroundedAnswerRequest(workspace_id="ws_default", query=question, search_type="hybrid", top_k=5)
        response = ask_question(req)
        latency = (time.time() - start_time) * 1000.0

        retrieved_doc_ids = [c.doc_id for c in response.retrieved_chunks]

        recall = calculate_recall_at_k(retrieved_doc_ids, expected_docs, k=5)
        precision = calculate_precision_at_k(retrieved_doc_ids, expected_docs, k=5)
        mrr = calculate_mrr(retrieved_doc_ids, expected_docs)
        ndcg = calculate_ndcg_at_k(retrieved_doc_ids, expected_docs, k=5)

        retrieved_texts = [c.content for c in response.retrieved_chunks]
        faithfulness = calculate_faithfulness(response.answer, retrieved_texts)

        citation_acc = 1.0 if response.citation_valid else 0.0

        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)
        ndcgs.append(ndcg)
        faithfulness_scores.append(faithfulness)
        citation_accuracies.append(citation_acc)
        latencies.append(latency)

        print(f"Query [{q_id}]: '{question[:40]}...' -> Recall@5: {recall:.2f} | Latency: {latency:.1f}ms")

    avg_recall = sum(recalls) / total_queries if total_queries else 0.0
    avg_precision = sum(precisions) / total_queries if total_queries else 0.0
    avg_mrr = sum(mrrs) / total_queries if total_queries else 0.0
    avg_ndcg = sum(ndcgs) / total_queries if total_queries else 0.0
    avg_faithfulness = sum(faithfulness_scores) / total_queries if total_queries else 0.0
    avg_citation_acc = sum(citation_accuracies) / total_queries if total_queries else 0.0

    latencies.sort()
    p95_index = int(0.95 * len(latencies))
    p95_latency = latencies[min(p95_index, len(latencies) - 1)]

    print("\n" + "=" * 60)
    print("FINAL EVALUATION METRICS REPORT")
    print("-" * 60)
    print(f"Retrieval Recall@5:      {avg_recall:.4f}  (Min Threshold: {thresholds.get('recall_at_5_min', 0.75)})")
    print(f"Retrieval Precision@5:   {avg_precision:.4f}")
    print(f"Retrieval MRR:           {avg_mrr:.4f}")
    print(f"Retrieval nDCG@5:        {avg_ndcg:.4f}")
    print(f"Answer Faithfulness:     {avg_faithfulness:.4f}  (Min Threshold: {thresholds.get('faithfulness_min', 0.85)})")
    print(f"Citation Accuracy:       {avg_citation_acc:.4f}  (Min Threshold: {thresholds.get('citation_accuracy_min', 0.90)})")
    print(f"p95 Latency:             {p95_latency:.2f}ms  (Max Threshold: {thresholds.get('p95_latency_ms_max', 1500)}ms)")
    print("=" * 60)

    # Threshold checks
    failures = []
    if avg_recall < thresholds.get('recall_at_5_min', 0.75):
        failures.append(f"Recall@5 ({avg_recall:.2f}) < Min ({thresholds.get('recall_at_5_min')})")
    if avg_citation_acc < thresholds.get('citation_accuracy_min', 0.90):
        failures.append(f"Citation Accuracy ({avg_citation_acc:.2f}) < Min ({thresholds.get('citation_accuracy_min')})")
    if avg_faithfulness < thresholds.get('faithfulness_min', 0.85):
        failures.append(f"Faithfulness ({avg_faithfulness:.2f}) < Min ({thresholds.get('faithfulness_min')})")
    if p95_latency > thresholds.get('p95_latency_ms_max', 1500):
        failures.append(f"p95 Latency ({p95_latency:.1f}ms) > Max ({thresholds.get('p95_latency_ms_max')}ms)")

    if failures:
        print("\n[FAIL] EVALUATION FAILED THRESHOLD CHECKS:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print("\n[PASS] ALL EVALUATION METRICS PASSED THRESHOLD CHECKS!")
        sys.exit(0)

if __name__ == "__main__":
    run_evaluation_suite()
