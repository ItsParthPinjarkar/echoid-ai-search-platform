import time
from typing import List
from app.models.schemas import SystemLogEntry

class ObservabilityLogger:
    """In-memory trace and metric logger."""
    def __init__(self):
        self.logs: List[SystemLogEntry] = []

    def log_request(
        self,
        request_id: str,
        query: str,
        retrieval_method: str,
        num_chunks_retrieved: int,
        num_chunks_used: int,
        llm_provider: str,
        latency_ms: float,
        tokens_used: int,
        cost_usd: float,
        citation_valid: bool,
        error: str = None
    ) -> SystemLogEntry:
        entry = SystemLogEntry(
            request_id=request_id,
            query=query,
            retrieval_method=retrieval_method,
            num_chunks_retrieved=num_chunks_retrieved,
            num_chunks_used=num_chunks_used,
            llm_provider=llm_provider,
            latency_ms=round(latency_ms, 2),
            tokens_used=tokens_used,
            cost_usd=round(cost_usd, 6),
            citation_valid=citation_valid,
            error=error,
            timestamp=time.time()
        )
        self.logs.append(entry)
        return entry

    def get_logs(self, limit: int = 50) -> List[SystemLogEntry]:
        return sorted(self.logs, key=lambda x: x.timestamp, reverse=True)[:limit]

logger_instance = ObservabilityLogger()
