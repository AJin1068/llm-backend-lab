import asyncio
from collections.abc import AsyncIterator

import anthropic
import httpx2

from taxsplit.llm.base import (
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    StreamEnd,
    StreamEvent,
    TextDelta,
    Usage,
)


def _usage(raw: object) -> Usage:
    return Usage(
        input_tokens=getattr(raw, "input_tokens", None) or 0,
        output_tokens=getattr(raw, "output_tokens", None) or 0,
        cache_creation_input_tokens=getattr(raw, "cache_creation_input_tokens", None) or 0,
        cache_read_input_tokens=getattr(raw, "cache_read_input_tokens", None) or 0,
    )


def _to_llm_error(exc: Exception, usage: Usage | None = None) -> LLMError:
    usage = usage or Usage()
    if isinstance(exc, TimeoutError):
        return LLMTimeoutError("LLM 호출 전체 시간 초과", usage=usage)
    # 스트리밍 도중의 HTTP 계층 오류는 SDK가 감싸지 않고 그대로 올라올 수 있다
    if isinstance(exc, httpx2.TimeoutException):
        return LLMTimeoutError("LLM 응답 대기 시간 초과", usage=usage)
    # APITimeoutError는 APIConnectionError의 하위라 먼저 본다
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMTimeoutError("LLM 호출 시간 초과", usage=usage)
    if isinstance(exc, anthropic.APIStatusError):
        return LLMError(
            f"LLM API 오류 ({exc.status_code}, {type(exc).__name__})",
            status_code=exc.status_code,
            request_id=exc.request_id,
            usage=usage,
        )
    if isinstance(exc, (anthropic.APIConnectionError, httpx2.HTTPError)):
        return LLMError("LLM API 연결 실패", usage=usage)
    return LLMError(f"LLM 호출 실패 ({type(exc).__name__})", usage=usage)


_HANDLED = (anthropic.APIError, httpx2.HTTPError, TimeoutError)


class AnthropicLLMClient:
    """공식 Anthropic SDK 직접 호출 (ADR-0001).

    - SDK 자동 재시도는 끈다(max_retries=0). 재시도 정책은 우리 코드(PR 9·17)가 맡는다.
    - connect/read 타임아웃은 HTTP 계층에, total(전체 마감)은 asyncio로 따로 건다.
      read는 조각 사이 대기만 재므로, 조각이 조금씩 계속 오면 total이 끊는다.
    """

    provider = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 60.0,
        total_timeout: float = 120.0,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._total_timeout = total_timeout
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=httpx2.Timeout(read_timeout, connect=connect_timeout),
            http_client=http_client,
        )

    def _params(self, request: LLMRequest) -> dict:
        if request.model is None or request.max_tokens is None:
            raise ValueError(
                "model과 max_tokens가 채워진 요청이어야 한다 (RecordingLLMClient 경유)"
            )
        params: dict = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.system is not None:
            params["system"] = request.system
        return params

    async def complete(self, request: LLMRequest) -> LLMResponse:
        params = self._params(request)
        try:
            async with asyncio.timeout(self._total_timeout):
                message = await self._client.messages.create(**params)
        except _HANDLED as exc:
            raise _to_llm_error(exc) from exc
        text = "".join(block.text for block in message.content if block.type == "text")
        return LLMResponse(
            text=text,
            model=message.model,
            usage=_usage(message.usage),
            stop_reason=message.stop_reason,
            request_id=getattr(message, "_request_id", None),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        params = self._params(request)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._total_timeout
        usage = Usage()
        model = request.model
        stop_reason: str | None = None
        parts: list[str] = []
        request_id: str | None = None

        try:
            # 전체 마감은 "다음 이벤트를 기다리는 동안"에만 건다. yield를 감싸면
            # 소비자 쪽 코드가 취소되는 문제가 생긴다.
            async with asyncio.timeout_at(deadline):
                sdk_stream = await self._client.messages.create(**params, stream=True)
            request_id = sdk_stream.response.headers.get("request-id")
            async with sdk_stream:
                events = aiter(sdk_stream)
                while True:
                    async with asyncio.timeout_at(deadline):
                        try:
                            event = await anext(events)
                        except StopAsyncIteration:
                            break
                    match event.type:
                        case "message_start":
                            model = event.message.model
                            usage = _usage(event.message.usage)
                        case "content_block_delta" if event.delta.type == "text_delta":
                            parts.append(event.delta.text)
                            yield TextDelta(event.delta.text)
                        case "message_delta":
                            stop_reason = event.delta.stop_reason
                            # message_delta의 usage는 누적값. 출력 토큰은 여기서 확정된다
                            delta_usage = event.usage
                            usage = Usage(
                                input_tokens=delta_usage.input_tokens or usage.input_tokens,
                                output_tokens=delta_usage.output_tokens or usage.output_tokens,
                                cache_creation_input_tokens=(
                                    delta_usage.cache_creation_input_tokens
                                    or usage.cache_creation_input_tokens
                                ),
                                cache_read_input_tokens=(
                                    delta_usage.cache_read_input_tokens
                                    or usage.cache_read_input_tokens
                                ),
                            )
                        case _:
                            pass
        except _HANDLED as exc:
            # 실패 전까지 받은 토큰(message_start의 입력 토큰 등)은 남긴다.
            # 그래야 비용이 0으로 잡히지 않는다
            raise _to_llm_error(exc, usage) from exc

        yield StreamEnd(
            LLMResponse(
                text="".join(parts),
                model=model or "",
                usage=usage,
                stop_reason=stop_reason,
                request_id=request_id,
            )
        )
