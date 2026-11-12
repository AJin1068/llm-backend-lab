import time
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import UTC, datetime

from taxsplit.llm.base import (
    CallKind,
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    StreamDone,
    StreamEvent,
)
from taxsplit.llm.call_log import CallStatus, LlmCall, LlmCallRepository, log_call
from taxsplit.llm.pricing import PriceTable


class RecordingLLMClient:
    """실제 클라이언트를 감싸서 (1) 모델·max_tokens를 설정값으로 채우고
    (2) 호출 1회마다 기록을 정확히 1건 남긴다. 성공·실패·중간 취소 모두 1건이다."""

    def __init__(
        self,
        inner: LLMClient,
        provider: str,
        repo: LlmCallRepository,
        prices: PriceTable,
        model_for: Callable[[CallKind], str],
        default_max_tokens: int,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._inner = inner
        self._provider = provider
        self._repo = repo
        self._prices = prices
        self._model_for = model_for
        self._default_max_tokens = default_max_tokens
        self._clock = clock

    def _fill(self, request: LLMRequest) -> LLMRequest:
        return replace(
            request,
            model=request.model or self._model_for(request.kind),
            max_tokens=request.max_tokens or self._default_max_tokens,
        )

    def _record(
        self,
        request: LLMRequest,
        started: float,
        streamed: bool,
        status: CallStatus,
        response: LLMResponse | None = None,
        error: BaseException | None = None,
    ) -> None:
        assert request.model is not None
        usage = response.usage if response else None
        model = response.model if response else request.model
        call = LlmCall(
            kind=request.kind,
            provider=self._provider,
            model=model,
            status=status,
            streamed=streamed,
            latency_ms=int((self._clock() - started) * 1000),
            created_at=datetime.now(UTC),
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            cache_read_input_tokens=usage.cache_read_input_tokens if usage else 0,
            cache_creation_input_tokens=usage.cache_creation_input_tokens if usage else 0,
            cost_usd=self._prices.cost(self._provider, model, usage) if usage else None,
            stop_reason=response.stop_reason if response else None,
            error_type=type(error).__name__ if error else None,
            status_code=getattr(error, "status_code", None),
            request_id=response.request_id if response else None,
        )
        self._repo.add(call)
        log_call(call)

    @staticmethod
    def _status_of(error: BaseException) -> CallStatus:
        if isinstance(error, LLMTimeoutError):
            return "timeout"
        if isinstance(error, LLMError):
            return "error"
        # 소비자가 스트림을 중간에 닫음(GeneratorExit)·작업 취소(CancelledError) 등
        return "cancelled"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        request = self._fill(request)
        started = self._clock()
        try:
            response = await self._inner.complete(request)
        except BaseException as e:
            self._record(request, started, False, self._status_of(e), error=e)
            raise
        self._record(request, started, False, "ok", response=response)
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        request = self._fill(request)
        started = self._clock()
        recorded = False
        try:
            async for event in self._inner.stream(request):
                if isinstance(event, StreamDone):
                    # 기록을 먼저 남기고 마지막 이벤트를 넘긴다.
                    # 소비자가 StreamDone을 받은 뒤 바로 닫아도 기록은 "ok" 1건이다.
                    self._record(request, started, True, "ok", response=event.response)
                    recorded = True
                    yield event
                    return
                yield event
        except BaseException as e:
            if not recorded:
                self._record(request, started, True, self._status_of(e), error=e)
            raise
        # StreamDone 없이 끝난 스트림은 계약 위반이다.
        error = LLMError("스트림이 StreamDone 없이 끝남")
        self._record(request, started, True, "error", error=error)
        raise error
