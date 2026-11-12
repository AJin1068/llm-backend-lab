from decimal import Decimal

from taxsplit.llm.base import Usage
from taxsplit.llm.pricing import cost_usd, load_prices


def test_sums_each_token_kind():
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    # Sonnet 5.5: 입력 2 + 캐시 쓰기 2.50 + 캐시 읽기 0.10 + 출력 10
    assert cost_usd("anthropic", "claude-sonnet-5-5", usage) == Decimal("14.60")


def test_small_call():
    usage = Usage(input_tokens=1200, output_tokens=300)
    # 1200 × 2/1M + 300 × 10/1M = 0.0024 + 0.003
    assert cost_usd("anthropic", "claude-sonnet-5-5", usage) == Decimal("0.0054")


def test_unknown_model_is_none_not_zero():
    usage = Usage(input_tokens=10, output_tokens=10)
    assert cost_usd("anthropic", "claude-unknown-9", usage) is None
    assert cost_usd("fake", "claude-sonnet-5-5", usage) is None


def test_haiku_long_prompt_boundary_counts_cache_tokens():
    at_threshold = Usage(input_tokens=60_000, cache_read_input_tokens=40_000)
    over = Usage(input_tokens=60_000, cache_read_input_tokens=40_001)
    # 정확히 10만은 기본 단가: 60000 × 0.10/1M + 40000 × 0.01/1M
    assert cost_usd("anthropic", "claude-haiku-5-5", at_threshold) == Decimal("0.0064")
    # 10만을 넘으면 모든 토큰이 비싼 단가: 60000 × 0.50/1M + 40001 × 0.05/1M
    assert cost_usd("anthropic", "claude-haiku-5-5", over) == Decimal("0.03200005")


def test_every_price_has_source_and_checked_date():
    import tomllib
    from pathlib import Path

    import taxsplit.llm.pricing as pricing

    raw = tomllib.loads(Path(pricing.__file__).with_name("pricing.toml").read_text())
    for models in raw.values():
        for name, table in models.items():
            assert table["source"].startswith("https://"), name
            assert table["checked"], name
    assert ("anthropic", "claude-sonnet-5-5") in load_prices()
