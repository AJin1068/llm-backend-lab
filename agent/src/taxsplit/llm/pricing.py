import tomllib
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from taxsplit.llm.base import Usage

_PRICING_FILE = Path(__file__).with_name("pricing.toml")
_PER = Decimal(1_000_000)


@dataclass(frozen=True)
class Rates:
    """USD / 100만 토큰."""

    input: Decimal
    cache_write: Decimal
    cache_read: Decimal
    output: Decimal


@dataclass(frozen=True)
class ModelPrice:
    base: Rates
    long_prompt_threshold: int | None = None
    long_prompt: Rates | None = None


def _rates(table: dict) -> Rates:
    return Rates(
        input=Decimal(table["input"]),
        cache_write=Decimal(table["cache_write"]),
        cache_read=Decimal(table["cache_read"]),
        output=Decimal(table["output"]),
    )


def load_prices(path: Path = _PRICING_FILE) -> dict[tuple[str, str], ModelPrice]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    prices: dict[tuple[str, str], ModelPrice] = {}
    for provider, models in data.items():
        for model, table in models.items():
            long_table = table.get("long_prompt")
            prices[(provider, model)] = ModelPrice(
                base=_rates(table),
                long_prompt_threshold=table.get("long_prompt_threshold"),
                long_prompt=_rates(long_table) if long_table else None,
            )
    return prices


@lru_cache
def _default_prices() -> dict[tuple[str, str], ModelPrice]:
    return load_prices()


def cost_usd(
    provider: str,
    model: str,
    usage: Usage,
    prices: dict[tuple[str, str], ModelPrice] | None = None,
) -> Decimal | None:
    """토큰 종류별 단가 × 토큰 수의 합(USD). 단가표에 없는 모델은 None."""
    price = (prices if prices is not None else _default_prices()).get((provider, model))
    if price is None:
        return None

    rates = price.base
    prompt_tokens = (
        usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
    )
    if (
        price.long_prompt is not None
        and price.long_prompt_threshold is not None
        and prompt_tokens > price.long_prompt_threshold
    ):
        rates = price.long_prompt

    total = (
        rates.input * usage.input_tokens
        + rates.cache_write * usage.cache_creation_input_tokens
        + rates.cache_read * usage.cache_read_input_tokens
        + rates.output * usage.output_tokens
    )
    return total / _PER
