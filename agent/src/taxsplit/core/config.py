from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from taxsplit.llm.base import CallKind

DEFAULT_MODEL = "claude-sonnet-5-5"


class Settings(BaseSettings):
    """환경변수와 .env에서 읽는 앱 설정. Spring의 @ConfigurationProperties 역할."""

    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    app_env: Literal["local", "test", "prod"] = "local"

    database_url: str

    # 키 없이도 서버·테스트가 돌고 돈이 나가지 않도록 기본은 fake
    llm_provider: Literal["fake", "anthropic"] = "fake"
    # SecretStr: 로그·repr에 키가 찍히지 않게
    anthropic_api_key: SecretStr | None = None
    # 비우면 SDK 기본 주소. 게이트웨이가 생기면 이 값만 게이트웨이 주소로 바꾼다
    llm_base_url: str | None = None

    # 운영값(세법 숫자 아님). 실제 응답 시간을 본 뒤 조정한다
    llm_connect_timeout: float = 5.0  # 연결은 원래 빨라야 함
    llm_read_timeout: float = 60.0  # 응답 조각(chunk) 사이 최대 대기
    llm_total_timeout: float = 120.0  # 요청 전체 마감 (스트리밍 포함)
    llm_max_tokens: int = 4096  # 한 번 답의 출력 상한 = 비용 상한

    # 호출 종류별 모델. 비우면 llm_model_default를 쓴다 (ADR-0006: 모델 이름은 설정값)
    llm_model_default: str = DEFAULT_MODEL
    llm_model_main_agent: str = ""
    llm_model_pdf_extract: str = ""
    llm_model_sim_user: str = ""
    llm_model_verifier: str = ""
    llm_model_judge: str = ""

    @field_validator("anthropic_api_key", "llm_base_url", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        # .env.example처럼 `LLM_BASE_URL=`로 비워 두면 None(=SDK 기본값)으로 본다
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def model_for(self, kind: CallKind) -> str:
        """호출 종류에 쓸 모델 이름. 종류별 값이 비어 있으면 기본 모델."""
        specific: str = getattr(self, f"llm_model_{kind.value}")
        return specific.strip() or self.llm_model_default.strip() or DEFAULT_MODEL


@lru_cache
def get_settings() -> Settings:
    # 한 번만 읽고 재사용 (싱글톤). 테스트에서 값을 바꾸면 get_settings.cache_clear() 필요
    return Settings()
