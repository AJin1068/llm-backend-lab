from taxsplit.core.config import Settings
from taxsplit.llm.base import LLMClient
from taxsplit.llm.call_log import LlmCallRepository
from taxsplit.llm.fake import FakeLLMClient
from taxsplit.llm.pricing import PriceTable
from taxsplit.llm.recording import RecordingLLMClient


def build_llm_client(settings: Settings, repo: LlmCallRepository) -> LLMClient:
    """설정의 provider에 맞는 클라이언트를 만들고 기록 래퍼로 감싼다."""
    inner: LLMClient
    if settings.llm_provider == "anthropic":
        # SDK import는 실제로 쓸 때만 한다.
        from taxsplit.llm.anthropic_client import AnthropicLLMClient

        key = settings.anthropic_api_key
        inner = AnthropicLLMClient(
            api_key=key.get_secret_value() if key else None,
            base_url=settings.llm_base_url,
            connect_timeout=settings.llm_connect_timeout,
            read_timeout=settings.llm_read_timeout,
            total_timeout=settings.llm_total_timeout,
        )
    else:
        inner = FakeLLMClient()
    return RecordingLLMClient(
        inner=inner,
        provider=settings.llm_provider,
        repo=repo,
        prices=PriceTable.load(),
        model_for=settings.model_for,
        default_max_tokens=settings.llm_max_tokens,
    )
