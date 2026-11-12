import asyncio
from collections.abc import AsyncIterator

import anthropic
from anthropic.types import Message as AnthropicMessage

from taxsplit.llm.base import (
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    StreamDone,
    StreamEvent,
    TextDelta,
    Usage,
)


class AnthropicLLMClient:
    """Anthropic 공식 SDK를 직접 부른다 (ADR-0001: 에이전트 프레임워크 없음)."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        connect_timeout: float,
        read_timeout: float,
        total_timeout: float,
        http_client: anthropic.DefaultAsyncHttpxClient | None = None,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            # SDK Timeout의 첫 값은 write·pool 기본값. 호출 전체 상한은 total로 따로 건다.
            timeout=anthropic.Timeout(read_timeout, connect=connect_timeout, read=read_timeout),
            # SDK 자동 재시도를 끈다. 재시도하면 "호출 1회 = 기록 1건"이 흐려지고,
            # 재시도 정책은 우리 코드(PR 9 도구 호출, PR 17 게이트웨이)가 정한다.
            max_retries=0,
            http_client=http_client,
        )
        self._total_timeout = total_timeout

    @staticmethod
    def _params(request: LLMRequest) -> dict:
        params: dict = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.system is not None:
            params["system"] = request.system
        return params

    @staticmethod
    def _to_response(message: AnthropicMessage, request_id: str | None) -> LLMResponse:
        u = message.usage
        return LLMResponse(
            # thinking 블록 등은 빼고 글자 답만 모은다.
            text="".join(b.text for b in message.content if b.type == "text"),
            model=message.model,
            usage=Usage(
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cache_read_input_tokens=u.cache_read_input_tokens or 0,
                cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
            ),
            stop_reason=message.stop_reason,
            request_id=request_id,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            async with asyncio.timeout(self._total_timeout):
                message = await self._client.messages.create(**self._params(request))
        except (TimeoutError, anthropic.APITimeoutError) as e:
            raise LLMTimeoutError(f"LLM 호출 시간 초과: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"LLM API 오류: {e.message}", status_code=e.status_code) from e
        except anthropic.APIError as e:
            raise LLMError(f"LLM 호출 실패: {e}") from e
        return self._to_response(message, message._request_id)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        try:
            async with asyncio.timeout(self._total_timeout):
                async with self._client.messages.stream(**self._params(request)) as s:
                    async for text in s.text_stream:
                        yield TextDelta(text)
                    message = await s.get_final_message()
                    request_id = s.request_id
        except (TimeoutError, anthropic.APITimeoutError) as e:
            raise LLMTimeoutError(f"LLM 스트리밍 시간 초과: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"LLM API 오류: {e.message}", status_code=e.status_code) from e
        except anthropic.APIError as e:
            raise LLMError(f"LLM 스트리밍 실패: {e}") from e
        yield StreamDone(self._to_response(message, request_id))
