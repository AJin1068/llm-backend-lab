from decimal import Decimal

from taxsplit.llm.base import Usage
from taxsplit.llm.pricing import PriceTable

prices = PriceTable.load()


def test_sonnet_cost_counts_all_token_kinds():
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )
    # 2 + 10 + 0.10 + 2.50 (pricing.toml)
    assert prices.cost("anthropic", "claude-sonnet-5-5", usage) == Decimal("14.6")


def test_haiku_uses_long_context_rates_only_above_threshold():
    at_threshold = Usage(input_tokens=90_000, cache_read_input_tokens=10_000)
    above = Usage(input_tokens=90_001, cache_read_input_tokens=10_000)
    assert prices.cost("anthropic", "claude-haiku-5-5", at_threshold) == (
        Decimal(90_000) * Decimal("0.10") + Decimal(10_000) * Decimal("0.01")
    ) / Decimal(1_000_000)
    assert prices.cost("anthropic", "claude-haiku-5-5", above) == (
        Decimal(90_001) * Decimal("0.50") + Decimal(10_000) * Decimal("0.05")
    ) / Decimal(1_000_000)


def test_unknown_model_has_no_cost_instead_of_zero():
    assert prices.cost("anthropic", "no-such-model", Usage(input_tokens=1)) is None
    assert prices.cost("fake", "claude-sonnet-5-5", Usage(input_tokens=1)) is None
