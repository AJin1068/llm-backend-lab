import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal, Protocol

from taxsplit.llm.base import CallKind

logger = logging.getLogger("taxsplit.llm.calls")

CallStatus = Literal["ok", "error", "timeout", "cancelled"]


@dataclass(frozen=True)
class LlmCall:
    """LLM 호출 1회의 기록. PR 8에서 Postgres llm_calls 표의 한 줄이 된다."""

    kind: CallKind
    provider: str
    model: str
    status: CallStatus
    streamed: bool
    latency_ms: int
    created_at: datetime
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    # 단가를 모르는 모델이면 None
    cost_usd: Decimal | None = None
    stop_reason: str | None = None
    error_type: str | None = None
    status_code: int | None = None
    request_id: str | None = None


class LlmCallRepository(Protocol):
    def add(self, call: LlmCall) -> None: ...


class InMemoryLlmCallRepository:
    """list에 쌓는 구현. Postgres 구현은 PR 8에서 같은 모양으로 붙인다."""

    def __init__(self) -> None:
        self.items: list[LlmCall] = []

    def add(self, call: LlmCall) -> None:
        self.items.append(call)


def log_call(call: LlmCall) -> None:
    """같은 내용을 JSON 한 줄로도 남긴다. 장애 때 로그에서 바로 찾기 위해서다."""
    record = asdict(call)
    record["created_at"] = call.created_at.isoformat()
    record["cost_usd"] = None if call.cost_usd is None else str(call.cost_usd)
    logger.info(json.dumps(record, ensure_ascii=False))
