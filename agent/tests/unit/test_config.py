import pytest
from pydantic import ValidationError

from taxsplit.core.config import Settings
from taxsplit.llm.base import CallKind


@pytest.fixture
def clean_llm_env(monkeypatch):
    # 개발자 셸에 LLM_* 값이 있어도 테스트 결과가 같게
    import os

    for name in list(os.environ):
        if name.startswith(("LLM_", "ANTHROPIC_")):
            monkeypatch.delenv(name)


def test_settings_fails_without_database_url(monkeypatch):
    monkeypatch.delenv(
        "DATABASE_URL", raising=False
    )  # raising=False: 원래 없어도 에러 내지 말고 넘어가
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_llm_defaults(clean_llm_env):
    s = Settings(_env_file=None)
    assert s.llm_provider == "fake"
    assert s.anthropic_api_key is None
    assert s.llm_base_url is None
    assert all(s.model_for(kind) == "claude-sonnet-5-5" for kind in CallKind)


def test_model_per_call_kind_from_env(clean_llm_env, monkeypatch):
    monkeypatch.setenv("LLM_MODEL_DEFAULT", "claude-opus-5-5")
    monkeypatch.setenv("LLM_MODEL_PDF_EXTRACT", "claude-haiku-5-5")
    monkeypatch.setenv("LLM_MODEL_SIM_USER", "claude-haiku-5-5")
    monkeypatch.setenv("LLM_MODEL_VERIFIER", "")  # 비우면 기본 모델
    s = Settings(_env_file=None)

    assert s.model_for(CallKind.PDF_EXTRACT) == "claude-haiku-5-5"
    assert s.model_for(CallKind.SIM_USER) == "claude-haiku-5-5"
    assert s.model_for(CallKind.VERIFIER) == "claude-opus-5-5"
    assert s.model_for(CallKind.MAIN_AGENT) == "claude-opus-5-5"


def test_blank_default_model_falls_back_to_code_default(clean_llm_env, monkeypatch):
    monkeypatch.setenv("LLM_MODEL_DEFAULT", "")
    assert Settings(_env_file=None).model_for(CallKind.JUDGE) == "claude-sonnet-5-5"


def test_blank_key_and_base_url_are_none(clean_llm_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("LLM_BASE_URL", " ")
    s = Settings(_env_file=None)
    assert s.anthropic_api_key is None
    assert s.llm_base_url is None


def test_api_key_hidden_in_repr(clean_llm_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    s = Settings(_env_file=None)
    assert "sk-ant-secret" not in repr(s)
    assert s.anthropic_api_key.get_secret_value() == "sk-ant-secret"
