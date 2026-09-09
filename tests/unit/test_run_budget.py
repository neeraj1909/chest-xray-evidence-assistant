from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from chest_xray_evidence_assistant.models import RunLimits
from chest_xray_evidence_assistant.runtime import BudgetExceeded, RunBudget


@pytest.mark.parametrize(
    ("consume", "limit_value", "code"),
    [
        ("consume_model_requests", 2, "model_request_limit"),
        ("consume_tool_calls", 3, "tool_call_limit"),
        ("consume_image_bytes", 100, "image_byte_limit"),
        ("consume_output_tokens", 10, "output_token_limit"),
    ],
)
def test_each_count_budget_fails_closed_at_its_parent_limit(
    consume: str,
    limit_value: int,
    code: str,
) -> None:
    limits = RunLimits(
        max_model_requests=2,
        max_tool_calls=3,
        max_image_bytes=100,
        max_output_tokens=10,
    )
    budget = RunBudget(limits)
    method = getattr(budget, consume)

    method(limit_value)
    with pytest.raises(BudgetExceeded) as caught:
        method(1)

    assert caught.value.code == code
    assert str(caught.value) == code


def test_cost_budget_uses_decimal_accounting() -> None:
    budget = RunBudget(RunLimits(max_estimated_cost_usd=0.25))

    budget.consume_estimated_cost(Decimal("0.10"))
    budget.consume_estimated_cost(Decimal("0.15"))

    with pytest.raises(BudgetExceeded, match="^cost_limit$"):
        budget.consume_estimated_cost(Decimal("0.001"))

    assert budget.snapshot().estimated_cost_usd == Decimal("0.25")


def test_only_one_repair_is_available_when_two_model_requests_are_allowed() -> None:
    budget = RunBudget(RunLimits(max_model_requests=2))

    budget.consume_repair()

    with pytest.raises(BudgetExceeded, match="^repair_limit$"):
        budget.consume_repair()
    assert budget.snapshot().repairs == 1


def test_child_scope_shares_usage_and_cannot_replace_parent_limits() -> None:
    limits = RunLimits(max_tool_calls=1, max_image_bytes=100)
    budget = RunBudget(limits)
    child = budget.child_scope()

    child.consume_tool_call()
    child.consume_image_bytes(40)

    assert child.snapshot() == budget.snapshot()
    assert budget.snapshot().tool_calls == 1
    assert budget.snapshot().image_bytes == 40
    assert child.limits is limits
    assert not hasattr(child, "child_scope")
    assert not hasattr(child, "reset")
    with pytest.raises(ValidationError):
        child.limits.max_tool_calls = 3  # type: ignore[misc]
    with pytest.raises(BudgetExceeded, match="^tool_call_limit$"):
        child.consume_tool_call()


def test_timeout_uses_parent_deadline_and_stable_failure_code() -> None:
    now = [100.0]
    budget = RunBudget(
        RunLimits(timeout_seconds=0.5),
        monotonic=lambda: now[0],
    )
    child = budget.child_scope()

    now[0] = 100.49
    child.check_timeout()
    now[0] = 100.5

    with pytest.raises(BudgetExceeded, match="^timeout$"):
        child.check_timeout()


@pytest.mark.parametrize(
    ("method", "value"),
    [
        ("consume_model_requests", 0),
        ("consume_tool_calls", -1),
        ("consume_image_bytes", -1),
        ("consume_output_tokens", -1),
        ("consume_estimated_cost", Decimal("NaN")),
    ],
)
def test_invalid_usage_deltas_are_rejected(method: str, value: object) -> None:
    budget = RunBudget(RunLimits())

    with pytest.raises(BudgetExceeded, match="^invalid_usage$"):
        getattr(budget, method)(value)
