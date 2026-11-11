import pytest
from pydantic import ValidationError

from llm_backend_lab.core.config import Settings


def test_settings_fails_without_database_url(monkeypatch):
    monkeypatch.delenv(
        "DATABASE_URL", raising=False
    )  # raising=False: 원래 없어도 에러 내지 말고 넘어가
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
