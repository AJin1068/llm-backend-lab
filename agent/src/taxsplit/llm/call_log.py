import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, Protocol

from taxsplit.llm.base import CallKind

logger = logging.getLogger("taxsplit.llm.calls")

type CallStatus = Literal["ok", "error", "timeout", "cancelled"]


@dataclass(frozen=True)
class LlmCall:
    """LLM 호출 1회의 기록. 프롬프트·답 본문과 API 키는 넣지 않는다."""

    kind: CallKind
    provider: str
    model: str
    streamed: bool
    status: CallStatus
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: Decimal | None = None  # 단가를 모르는 모델이면 None
    stop_reason: str | None = None
    error_type: str | None = None
    status_code: int | None = None
    request_id: str | None = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_log_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["id"] = str(self.id)
        data["kind"] = self.kind.value
        data["created_at"] = self.created_at.isoformat()
        data["cost_usd"] = None if self.cost_usd is None else str(self.cost_usd)
        return data


class LlmCallRepository(Protocol):
    """호출 기록 저장소. Postgres 구현은 PR 8(#17)."""

    async def add(self, call: LlmCall) -> None: ...


class InMemoryLlmCallRepository:
    def __init__(self) -> None:
        self.items: list[LlmCall] = []

    async def add(self, call: LlmCall) -> None:
        self.items.append(call)


def log_call(call: LlmCall) -> None:
    """같은 내용을 JSON 한 줄로 남긴다."""
    logger.info(json.dumps(call.to_log_dict(), ensure_ascii=False))
