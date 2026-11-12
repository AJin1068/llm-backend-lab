from taxsplit.core.config import Settings
from taxsplit.llm.anthropic_client import AnthropicLLMClient
from taxsplit.llm.call_log import LlmCallRepository
from taxsplit.llm.fake import FakeLLMClient
from taxsplit.llm.recording import ProviderClient, RecordingLLMClient


def build_provider_client(settings: Settings) -> ProviderClient:
    match settings.llm_provider:
        case "fake":
            return FakeLLMClient()
        case "anthropic":
            if settings.anthropic_api_key is None:
                raise ValueError("LLM_PROVIDER=anthropic이면 ANTHROPIC_API_KEY가 필요하다")
            return AnthropicLLMClient(
                api_key=settings.anthropic_api_key.get_secret_value(),
                base_url=settings.llm_base_url,
                connect_timeout=settings.llm_connect_timeout,
                read_timeout=settings.llm_read_timeout,
                total_timeout=settings.llm_total_timeout,
            )


def build_llm_client(settings: Settings, repo: LlmCallRepository) -> RecordingLLMClient:
    """서비스가 쓰는 LLM 클라이언트. 항상 기록 래퍼로 감싸서 돌려준다."""
    return RecordingLLMClient(build_provider_client(settings), repo, settings)
