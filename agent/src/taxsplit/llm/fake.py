from collections.abc import AsyncIterator

from taxsplit.llm.base import (
    LLMError,
    LLMRequest,
    LLMResponse,
    StreamDone,
    StreamEvent,
    TextDelta,
    Usage,
)


class FakeLLMClient:
    """네트워크 없이 정해진 답을 돌려주는 클라이언트. 기본 provider이고 테스트에서 쓴다."""

    def __init__(
        self,
        text: str = "fake response",
        usage: Usage | None = None,
        error: LLMError | None = None,
        fail_after_chunks: int | None = None,
    ) -> None:
        self.text = text
        self.usage = usage or Usage(input_tokens=10, output_tokens=5)
        self.error = error
        # 스트리밍 도중 실패를 흉내 낼 때: 이 수만큼 조각을 낸 뒤 error를 던진다.
        self.fail_after_chunks = fail_after_chunks
        self.requests: list[LLMRequest] = []

    def _response(self, request: LLMRequest) -> LLMResponse:
        assert request.model is not None
        return LLMResponse(
            text=self.text, model=request.model, usage=self.usage, stop_reason="end_turn"
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self._response(request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        if self.error is not None and self.fail_after_chunks is None:
            raise self.error
        for i, word in enumerate(self.text.split(" ")):
            if self.fail_after_chunks is not None and i >= self.fail_after_chunks:
                assert self.error is not None
                raise self.error
            yield TextDelta(word if i == 0 else " " + word)
        yield StreamDone(self._response(request))
