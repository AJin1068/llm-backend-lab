from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경변수와 .env에서 읽는 앱 설정. Spring의 @ConfigurationProperties 역할."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["local", "test", "prod"] = "local"

    database_url: str = ""


@lru_cache
def get_settings() -> Settings:
    # 한 번만 읽고 재사용 (싱글톤). 테스트에서 값을 바꾸면 get_settings.cache_clear() 필요
    return Settings()
