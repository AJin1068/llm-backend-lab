import asyncio
from collections.abc import AsyncIterator

from taxsplit.llm.base import (
    LLMError,
    LLMRequest,
    LLMResponse,
    StreamEnd,
    StreamEvent,
    TextDelta,
    Usage,
)


class FakeLLMClient:
    """정해진 답을 돌려주는 가짜 클라이언트. 기본 provider라 키 없이 앱·테스트가 돈다.

    테스트용 옵션:
      error           부르면 바로 이 오류를 낸다
      fail_after      스트리밍에서 조각 n개를 보낸 뒤 error(없으면 LLMError)를 낸다
      skip_end        스트리밍에서 마지막 StreamEnd 없이 끝난다
      delay           조각마다 기다리는 시간(초)
    """

    provider = "fake"

    def __init__(
        self,
        reply: str = "가짜 답입니다.",
        *,
        chunks: list[str] | None = None,
        usage: Usage | None = None,
        error: LLMError | None = None,
        fail_after: int | None = None,
        skip_end: bool = False,
        delay: float = 0.0,
    ) -> None:
        self.reply = reply
        self.chunks = chunks if chunks is not None else [reply]
        self.usage = usage or Usage(input_tokens=10, output_tokens=5)
        self.error = error
        self.fail_after = fail_after
        self.skip_end = skip_end
        self.delay = delay
        self.requests: list[LLMRequest] = []

    def _response(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text="".join(self.chunks),
            model=request.model or "fake",
            usage=self.usage,
            stop_reason="end_turn",
            request_id="fake-request",
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self._response(request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        if self.error is not None and self.fail_after is None:
            raise self.error
        for i, chunk in enumerate(self.chunks):
            if self.fail_after is not None and i == self.fail_after:
                raise self.error or LLMError("가짜 스트리밍 도중 실패")
            if self.delay:
                await asyncio.sleep(self.delay)
            yield TextDelta(chunk)
        if self.skip_end:
            return
        yield StreamEnd(self._response(request))
