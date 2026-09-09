"""Parent-owned runtime accounting for bounded agent and tool execution."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from decimal import Decimal
from typing import Literal, TypeAlias

from pydantic import Field

from .models import ContractModel, RunLimits

BudgetFailureCode: TypeAlias = Literal[
    "cost_limit",
    "image_byte_limit",
    "invalid_usage",
    "model_request_limit",
    "output_token_limit",
    "repair_limit",
    "timeout",
    "tool_call_limit",
]
DependencyFailureCode: TypeAlias = Literal[
    "image_tool_unavailable",
    "model_unavailable",
    "retrieval_unavailable",
]


class BudgetExceeded(RuntimeError):
    """Report a stable budget failure without echoing request content."""

    def __init__(self, code: BudgetFailureCode) -> None:
        self.code = code
        super().__init__(code)


class DependencyUnavailable(RuntimeError):
    """Report an unavailable runtime dependency without retaining provider text."""

    def __init__(self, code: DependencyFailureCode) -> None:
        self.code = code
        super().__init__(code)


class BudgetSnapshot(ContractModel):
    """Immutable, replay-safe usage observed at a point in one run."""

    model_requests: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    image_bytes: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost_usd: Decimal = Field(ge=0)
    repairs: int = Field(ge=0, le=1)
    elapsed_ms: int = Field(ge=0)


class RunBudget:
    """Own cumulative run usage; child operations receive only a shared scope."""

    def __init__(
        self,
        limits: RunLimits,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limits = limits
        self._monotonic = monotonic
        self._started_at = monotonic()
        self._model_requests = 0
        self._tool_calls = 0
        self._image_bytes = 0
        self._output_tokens = 0
        self._estimated_cost_usd = Decimal("0")
        self._repairs = 0

    @property
    def limits(self) -> RunLimits:
        return self._limits

    def check_timeout(self) -> None:
        if self._monotonic() - self._started_at >= self._limits.timeout_seconds:
            raise BudgetExceeded("timeout")

    def remaining_seconds(self) -> float:
        elapsed = self._monotonic() - self._started_at
        remaining = self._limits.timeout_seconds - elapsed
        if remaining <= 0:
            raise BudgetExceeded("timeout")
        return remaining

    @staticmethod
    def _positive_count(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BudgetExceeded("invalid_usage")
        return value

    def _consume_count(
        self,
        *,
        current: int,
        amount: int,
        limit: int,
        code: BudgetFailureCode,
    ) -> int:
        self.check_timeout()
        amount = self._positive_count(amount)
        if current + amount > limit:
            raise BudgetExceeded(code)
        return current + amount

    def consume_model_requests(self, count: int = 1) -> None:
        self._model_requests = self._consume_count(
            current=self._model_requests,
            amount=count,
            limit=self._limits.max_model_requests,
            code="model_request_limit",
        )

    def consume_tool_calls(self, count: int = 1) -> None:
        self._tool_calls = self._consume_count(
            current=self._tool_calls,
            amount=count,
            limit=self._limits.max_tool_calls,
            code="tool_call_limit",
        )

    def consume_image_bytes(self, count: int) -> None:
        self._image_bytes = self._consume_count(
            current=self._image_bytes,
            amount=count,
            limit=self._limits.max_image_bytes,
            code="image_byte_limit",
        )

    def consume_output_tokens(self, count: int) -> None:
        self._output_tokens = self._consume_count(
            current=self._output_tokens,
            amount=count,
            limit=self._limits.max_output_tokens,
            code="output_token_limit",
        )

    def consume_estimated_cost(self, amount: Decimal | float | int) -> None:
        self.check_timeout()
        try:
            decimal_amount = amount if isinstance(amount, Decimal) else Decimal(str(amount))
        except (ArithmeticError, ValueError):
            raise BudgetExceeded("invalid_usage") from None
        if not decimal_amount.is_finite() or decimal_amount < 0:
            raise BudgetExceeded("invalid_usage")
        if decimal_amount == 0:
            return
        limit = Decimal(str(self._limits.max_estimated_cost_usd))
        if self._estimated_cost_usd + decimal_amount > limit:
            raise BudgetExceeded("cost_limit")
        self._estimated_cost_usd += decimal_amount

    def consume_repair(self) -> None:
        self.check_timeout()
        repair_limit = min(1, self._limits.max_model_requests - 1)
        if self._repairs + 1 > repair_limit:
            raise BudgetExceeded("repair_limit")
        self._repairs += 1

    def child_scope(self) -> ToolBudgetScope:
        return ToolBudgetScope(self)

    def snapshot(self) -> BudgetSnapshot:
        elapsed = self._monotonic() - self._started_at
        elapsed_ms = max(0, math.floor(elapsed * 1000))
        return BudgetSnapshot(
            model_requests=self._model_requests,
            tool_calls=self._tool_calls,
            image_bytes=self._image_bytes,
            output_tokens=self._output_tokens,
            estimated_cost_usd=self._estimated_cost_usd,
            repairs=self._repairs,
            elapsed_ms=elapsed_ms,
        )


class ToolBudgetScope:
    """Narrow view that charges a parent's tool/image budget without replacing it."""

    def __init__(self, parent: RunBudget) -> None:
        self.__parent = parent

    @property
    def limits(self) -> RunLimits:
        return self.__parent.limits

    def check_timeout(self) -> None:
        self.__parent.check_timeout()

    def remaining_seconds(self) -> float:
        return self.__parent.remaining_seconds()

    def consume_tool_call(self) -> None:
        self.__parent.consume_tool_calls()

    def consume_image_bytes(self, count: int) -> None:
        self.__parent.consume_image_bytes(count)

    def snapshot(self) -> BudgetSnapshot:
        return self.__parent.snapshot()
