from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol


class CallKind(StrEnum):
    """호출 종류. 종류마다 모델을 따로 설정하고, 기록에서 종류별 비용·지연을 나눠 본다."""

    MAIN_AGENT = "main_agent"  # 메인 대화
    PDF_EXTRACT = "pdf_extract"  # PDF 추출
    SIM_USER = "sim_user"  # 평가 하네스의 가상 사용자
    VERIFIER = "verifier"  # 서비스 안 검증 에이전트
    JUDGE = "judge"  # 채점 (GPT 연결은 PR 11)


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMRequest:
    """한 번의 호출 요청. model·max_tokens를 비우면 설정값으로 채운다."""

    kind: CallKind
    messages: Sequence[Message]
    system: str | None = None
    model: str | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0  # 캐시 쓰기 (5분 캐시)
    cache_read_input_tokens: int = 0  # 캐시 읽기


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: Usage
    stop_reason: str | None
    request_id: str | None = None


@dataclass(frozen=True)
class TextDelta:
    """스트리밍 중 받은 글자 조각."""

    text: str


@dataclass(frozen=True)
class StreamEnd:
    """스트리밍의 마지막 이벤트. 모은 답과 토큰 수가 들어 있다."""

    response: LLMResponse


type StreamEvent = TextDelta | StreamEnd


@dataclass
class LLMError(Exception):
    """LLM 호출 실패. usage는 실패 전까지 알게 된 토큰 수(스트리밍 도중 실패 등)."""

    message: str
    status_code: int | None = None
    request_id: str | None = None
    usage: Usage = field(default_factory=Usage)

    def __str__(self) -> str:
        return self.message


class LLMTimeoutError(LLMError):
    """시간 초과 (연결, 조각 사이 대기, 전체 마감 중 하나)."""


class LLMClient(Protocol):
    """LLM 호출 인터페이스. 공급자별 구현(fake, anthropic)과 기록 래퍼가 같은 모양이다."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """답을 한 번에 받는다."""
        ...

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """답을 조각(TextDelta)으로 받고, 성공하면 마지막에 StreamEnd 하나를 받는다."""
        ...
