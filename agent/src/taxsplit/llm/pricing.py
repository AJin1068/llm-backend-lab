import tomllib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from taxsplit.llm.base import Usage

_DEFAULT_PATH = Path(__file__).with_name("pricing.toml")
_PER = Decimal(1_000_000)


@dataclass(frozen=True)
class Rates:
    input: Decimal
    output: Decimal
    cache_write: Decimal
    cache_read: Decimal

    @classmethod
    def from_toml(cls, raw: dict) -> Rates:
        # 단가는 문자열로 적고 Decimal로 읽는다. float로 읽으면 0.1 같은 값에 오차가 붙는다.
        return cls(**{k: Decimal(raw[k]) for k in ("input", "output", "cache_write", "cache_read")})


@dataclass(frozen=True)
class ModelPrice:
    base: Rates
    long_context: Rates | None = None
    long_context_threshold: int | None = None

    def rates_for(self, usage: Usage) -> Rates:
        if self.long_context is None or self.long_context_threshold is None:
            return self.base
        prompt = (
            usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
        )
        return self.long_context if prompt > self.long_context_threshold else self.base


class PriceTable:
    """회사·모델별 단가 파일을 읽어 호출 1건의 비용(USD)을 계산한다."""

    def __init__(self, prices: dict[tuple[str, str], ModelPrice]) -> None:
        self._prices = prices

    @classmethod
    def load(cls, path: Path = _DEFAULT_PATH) -> PriceTable:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        prices: dict[tuple[str, str], ModelPrice] = {}
        for provider, models in raw.items():
            for model, entry in models.items():
                long = entry.get("long_context")
                prices[(provider, model)] = ModelPrice(
                    base=Rates.from_toml(entry),
                    long_context=Rates.from_toml(long) if long else None,
                    long_context_threshold=entry.get("long_context_threshold"),
                )
        return cls(prices)

    def cost(self, provider: str, model: str, usage: Usage) -> Decimal | None:
        """단가를 모르는 모델이면 None. 0으로 적으면 '공짜'로 오해하니 비워 둔다."""
        price = self._prices.get((provider, model))
        if price is None:
            return None
        r = price.rates_for(usage)
        total = (
            usage.input_tokens * r.input
            + usage.output_tokens * r.output
            + usage.cache_creation_input_tokens * r.cache_write
            + usage.cache_read_input_tokens * r.cache_read
        )
        return total / _PER
