from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from taxsplit.llm.base import CallKind


class Settings(BaseSettings):
    """환경변수와 .env에서 읽는 앱 설정. Spring의 @ConfigurationProperties 역할."""

    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    app_env: Literal["local", "test", "prod"] = "local"

    database_url: str

    # --- LLM ---
    # 기본은 fake: 키가 없어도 앱과 테스트가 돌고, 실수로 돈이 나가지 않는다.
    llm_provider: Literal["fake", "anthropic"] = "fake"
    anthropic_api_key: SecretStr | None = None
    # 비우면 SDK 기본 주소. 게이트웨이(PR 16~)가 생기면 이 값만 게이트웨이 주소로 바꾼다.
    llm_base_url: str | None = None
    # 타임아웃 3종(초). connect: 서버에 붙기까지, read: 응답 조각 사이 대기, total: 호출 전체.
    llm_connect_timeout: float = 5.0
    llm_read_timeout: float = 60.0
    llm_total_timeout: float = 120.0
    llm_max_tokens: int = 4096

    # 호출 종류별 모델. 지정이 없으면 llm_model_default를 쓴다 (ADR-0006: 모델 이름은 설정값).
    llm_model_default: str = "claude-sonnet-5-5"
    llm_model_main_agent: str | None = None
    llm_model_pdf_extract: str | None = None
    llm_model_sim_user: str | None = None
    llm_model_verifier: str | None = None
    llm_model_judge: str | None = None

    def model_for(self, kind: CallKind) -> str:
        return getattr(self, f"llm_model_{kind.value}") or self.llm_model_default


@lru_cache
def get_settings() -> Settings:
    # 한 번만 읽고 재사용 (싱글톤). 테스트에서 값을 바꾸면 get_settings.cache_clear() 필요
    return Settings()
