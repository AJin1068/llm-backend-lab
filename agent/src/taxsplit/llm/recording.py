import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import replace
from typing import Protocol

from taxsplit.llm.base import (
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    StreamEnd,
    StreamEvent,
    Usage,
)
from taxsplit.llm.call_log import CallStatus, LlmCall, LlmCallRepository, log_call
from taxsplit.llm.pricing import cost_usd


class ProviderClient(LLMClient, Protocol):
    provider: str


class ModelSettings(Protocol):
    llm_max_tokens: int

    def model_for(self, kind): ...


class IncompleteStreamError(LLMError):
    """스트리밍이 마지막 이벤트(StreamEnd) 없이 끝났다."""


class RecordingLLMClient:
    """공급자 클라이언트를 감싼다. 모델·max_tokens를 설정값으로 채우고, 호출 1회마다 기록 1건.

    "호출 1회" = complete/stream 1회. 성공, 오류, 시간 초과, 스트리밍 도중 실패,
    소비자가 중간에 닫음(aclose)·취소 모두 정확히 1건이다. 기록은 이 클래스 한 곳에서만 남긴다.
    """

    def __init__(
        self,
        inner: ProviderClient,
        repo: LlmCallRepository,
        settings: ModelSettings,
    ) -> None:
        self._inner = inner
        self._repo = repo
        self._settings = settings

    @property
    def provider(self) -> str:
        return self._inner.provider

    def _resolve(self, request: LLMRequest) -> LLMRequest:
        return replace(
            request,
            model=request.model or self._settings.model_for(request.kind),
            max_tokens=request.max_tokens or self._settings.llm_max_tokens,
        )

    async def _record(
        self,
        request: LLMRequest,
        started: float,
        *,
        streamed: bool,
        response: LLMResponse | None = None,
        exc: BaseException | None = None,
    ) -> None:
        status: CallStatus
        usage = Usage()
        error_type = status_code = request_id = stop_reason = None
        model = request.model or ""
        if response is not None:
            status = "ok"
            usage = response.usage
            stop_reason = response.stop_reason
            request_id = response.request_id
            model = response.model or model
        elif isinstance(exc, LLMTimeoutError):
            status = "timeout"
        elif isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
            status = "cancelled"
        else:
            status = "error"
        if exc is not None:
            error_type = type(exc).__name__
            if isinstance(exc, LLMError):
                usage = exc.usage
                status_code = exc.status_code
                request_id = exc.request_id

        call = LlmCall(
            kind=request.kind,
            provider=self.provider,
            model=model,
            streamed=streamed,
            status=status,
            latency_ms=round((time.perf_counter() - started) * 1000),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cost_usd=cost_usd(self.provider, model, usage),
            stop_reason=stop_reason,
            error_type=error_type,
            status_code=status_code,
            request_id=request_id,
        )
        log_call(call)
        await self._repo.add(call)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        request = self._resolve(request)
        started = time.perf_counter()
        try:
            response = await self._inner.complete(request)
        except BaseException as exc:
            await self._record(request, started, streamed=False, exc=exc)
            raise
        await self._record(request, started, streamed=False, response=response)
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        request = self._resolve(request)
        started = time.perf_counter()
        recorded = False  # 마지막 이벤트 직후 닫혀도 두 번 기록하지 않게 하는 표시
        try:
            # aclosing: 소비자가 중간에 닫으면 안쪽 스트림(HTTP 연결)도 함께 닫는다
            async with aclosing(self._inner.stream(request)) as events:
                async for event in events:
                    if isinstance(event, StreamEnd):
                        recorded = True
                        await self._record(request, started, streamed=True, response=event.response)
                    yield event
            if not recorded:
                raise IncompleteStreamError("스트리밍이 마지막 이벤트 없이 끝남")
        except BaseException as exc:
            if not recorded:
                recorded = True
                await self._record(request, started, streamed=True, exc=exc)
            raise
