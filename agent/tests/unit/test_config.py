import pytest
from pydantic import ValidationError

from taxsplit.core.config import Settings


def test_settings_fails_without_database_url(monkeypatch):
    monkeypatch.delenv(
        "DATABASE_URL", raising=False
    )  # raising=False: 원래 없어도 에러 내지 말고 넘어가
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_llm_defaults_to_fake_without_key():
    s = Settings(_env_file=None)
    assert s.llm_provider == "fake"
    assert s.anthropic_api_key is None


def test_model_per_kind_falls_back_to_default(monkeypatch):
    from taxsplit.llm.base import CallKind

    monkeypatch.setenv("LLM_MODEL_DEFAULT", "claude-sonnet-5-5")
    monkeypatch.setenv("LLM_MODEL_PDF_EXTRACT", "claude-haiku-5-5")
    s = Settings(_env_file=None)
    assert s.model_for(CallKind.PDF_EXTRACT) == "claude-haiku-5-5"
    assert s.model_for(CallKind.MAIN_AGENT) == "claude-sonnet-5-5"


def test_api_key_is_hidden_in_repr(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    s = Settings(_env_file=None)
    assert "sk-secret" not in repr(s)
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "sk-secret"
