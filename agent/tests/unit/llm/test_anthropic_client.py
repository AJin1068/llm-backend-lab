"""SDK를 진짜로 부르되, 네트워크 대신 가짜 전송 계층(MockTransport)이 답한다."""

import asyncio
import json

import anthropic
import httpx2
import pytest

from taxsplit.llm.anthropic_client import AnthropicLLMClient
from taxsplit.llm.base import (
    CallKind,
    LLMError,
    LLMRequest,
    LLMTimeoutError,
    Message,
    StreamDone,
    TextDelta,
)

BASE_URL = "http://gateway.test"

REQ = LLMRequest(
    kind=CallKind.MAIN_AGENT,
    messages=[Message("user", "누가 공제받는 게 유리해?")],
    system="너는 연말정산 도우미다.",
    model="claude-sonnet-5-5",
    max_tokens=100,
)

MESSAGE_JSON = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "아내 쪽이 유리해요."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 20,
        "output_tokens": 8,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 0,
    },
}


def make(handler, total_timeout: float = 5.0) -> tuple[AnthropicLLMClient, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request)

    client = AnthropicLLMClient(
        api_key="test-key",
        base_url=BASE_URL,
        connect_timeout=1.0,
        read_timeout=1.0,
        total_timeout=total_timeout,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(record)),
    )
    return client, seen


def test_complete_sends_to_base_url_and_maps_usage():
    client, seen = make(
        lambda r: httpx2.Response(200, json=MESSAGE_JSON, headers={"request-id": "req_1"})
    )

    resp = asyncio.run(client.complete(REQ))

    assert str(seen[0].url) == f"{BASE_URL}/v1/messages"
    body = json.loads(seen[0].content)
    assert body["model"] == "claude-sonnet-5-5"
    assert body["max_tokens"] == 100
    assert body["system"] == "너는 연말정산 도우미다."
    assert resp.text == "아내 쪽이 유리해요."
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (20, 8)
    assert resp.usage.cache_read_input_tokens == 3
    assert resp.request_id == "req_1"


def test_api_error_becomes_llm_error_without_sdk_retry():
    client, seen = make(
        lambda r: httpx2.Response(
            529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
        )
    )

    with pytest.raises(LLMError) as exc:
        asyncio.run(client.complete(REQ))

    assert exc.value.status_code == 529
    assert len(seen) == 1  # SDK 자동 재시도 꺼짐


def test_transport_timeout_becomes_llm_timeout():
    def slow(request):
        raise httpx2.ReadTimeout("read timeout", request=request)

    client, _ = make(slow)

    with pytest.raises(LLMTimeoutError):
        asyncio.run(client.complete(REQ))


def test_total_timeout_cuts_long_call():
    class Slow(httpx2.AsyncBaseTransport):
        async def handle_async_request(self, request):
            await asyncio.sleep(1)
            return httpx2.Response(200, json=MESSAGE_JSON)

    client = AnthropicLLMClient(
        api_key="k",
        base_url=BASE_URL,
        connect_timeout=1.0,
        read_timeout=5.0,
        total_timeout=0.05,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=Slow()),
    )

    with pytest.raises(LLMTimeoutError):
        asyncio.run(client.complete(REQ))


def sse(events: list[tuple[str, dict]]) -> bytes:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


STREAM_BODY = sse(
    [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {**MESSAGE_JSON, "content": [], "stop_reason": None},
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "아내 "},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "쪽"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 8},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
)


def test_stream_yields_deltas_then_done_with_usage():
    client, _ = make(
        lambda r: httpx2.Response(
            200,
            content=STREAM_BODY,
            headers={"content-type": "text/event-stream", "request-id": "req_2"},
        )
    )

    async def run():
        return [e async for e in client.stream(REQ)]

    events = asyncio.run(run())

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["아내 ", "쪽"]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.response.text == "아내 쪽"
    assert done.response.usage.output_tokens == 8
    assert done.response.stop_reason == "end_turn"
    assert done.response.request_id == "req_2"
