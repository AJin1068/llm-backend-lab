from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol


class CallKind(StrEnum):
    """호출 종류. 모든 LLM 호출에 붙고, 종류마다 모델을 따로 설정하고 비용을 나눠 본다."""

    MAIN_AGENT = "main_agent"  # 메인 대화 (도구 호출 포함)
    PDF_EXTRACT = "pdf_extract"  # 간소화 PDF에서 공제 항목 뽑기
    SIM_USER = "sim_user"  # 평가 하네스의 가상 사용자
    VERIFIER = "verifier"  # 서비스 안 검증 에이전트
    JUDGE = "judge"  # 채점. GPT 연결은 PR 11


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMRequest:
    kind: CallKind
    messages: list[Message]
    system: str | None = None
    # 비우면 설정값(호출 종류별 모델, llm_max_tokens)으로 채운다.
    model: str | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    request_id: str | None = None


@dataclass(frozen=True)
class TextDelta:
    """스트리밍 중 도착한 글자 조각."""

    text: str


@dataclass(frozen=True)
class StreamDone:
    """스트리밍의 마지막 이벤트. 토큰 사용량이 여기서 확정된다."""

    response: LLMResponse


StreamEvent = TextDelta | StreamDone


class LLMError(Exception):
    """LLM 호출 실패. 회사별 SDK 오류를 이 종류로 바꿔서 바깥에 SDK 타입이 새지 않게 한다."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMTimeoutError(LLMError):
    """connect/read/total 중 하나를 넘김."""


class LLMClient(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse: ...

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """TextDelta를 여러 번, 마지막에 StreamDone을 한 번 낸다."""
        ...
