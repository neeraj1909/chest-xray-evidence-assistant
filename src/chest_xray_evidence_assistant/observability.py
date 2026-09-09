"""In-memory handoff for one complete redacted run trace."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, model_validator

from .models import ContractModel, Identifier, RunEventKind, RunTrace, Sha256Digest


@dataclass(slots=True)
class RunTraceCollector:
    """Capture one terminal trace without changing success/error semantics."""

    _trace: RunTrace | None = None

    @property
    def trace(self) -> RunTrace | None:
        return self._trace

    def record(self, trace: RunTrace) -> None:
        if self._trace is not None:
            raise RuntimeError("trace collector already contains a terminal trace")
        self._trace = trace


class RedactionProbe(ContractModel):
    name: Identifier
    value_sha256: Sha256Digest
    representations_checked: int = Field(ge=1, le=3)


class TraceRedactionReport(ContractModel):
    """Reproducible evidence that named sensitive probes are absent from traces."""

    schema_version: Literal[1] = 1
    trace_count: int = Field(gt=0)
    event_count: int = Field(gt=0)
    event_kinds: tuple[RunEventKind, ...] = Field(min_length=1)
    trace_sha256: Sha256Digest
    probes: tuple[RedactionProbe, ...] = Field(min_length=1)
    violations: tuple[Identifier, ...]
    passed: bool

    @model_validator(mode="after")
    def bind_passed_to_violations(self) -> TraceRedactionReport:
        if self.passed != (not self.violations):
            raise ValueError("redaction result does not match its violations")
        if len({probe.name for probe in self.probes}) != len(self.probes):
            raise ValueError("redaction probe names must be unique")
        return self


def _probe_representations(value: str | bytes) -> tuple[bytes, set[str]]:
    if isinstance(value, str):
        raw = value.encode("utf-8")
        if not raw:
            raise ValueError("redaction probes cannot be empty")
        return raw, {value.casefold()}
    if not value:
        raise ValueError("redaction probes cannot be empty")
    representations = {
        value.hex().casefold(),
        base64.b64encode(value).decode("ascii").casefold(),
    }
    try:
        representations.add(value.decode("utf-8").casefold())
    except UnicodeDecodeError:
        pass
    return value, representations


def audit_trace_redaction(
    traces: Sequence[RunTrace],
    probes: Mapping[str, str | bytes],
) -> TraceRedactionReport:
    """Check exact sensitive probes without retaining their raw values."""

    if not traces or not probes:
        raise ValueError("redaction audit requires traces and probes")
    serialized = json.dumps(
        [trace.model_dump(mode="json") for trace in traces],
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    searchable = serialized.casefold()
    probe_records: list[RedactionProbe] = []
    violations: list[str] = []
    for name, value in sorted(probes.items()):
        raw, representations = _probe_representations(value)
        probe_records.append(
            RedactionProbe(
                name=name,
                value_sha256=hashlib.sha256(raw).hexdigest(),
                representations_checked=len(representations),
            )
        )
        if any(representation in searchable for representation in representations):
            violations.append(name)
    event_kinds = tuple(sorted({event.kind for trace in traces for event in trace.events}))
    return TraceRedactionReport(
        trace_count=len(traces),
        event_count=sum(len(trace.events) for trace in traces),
        event_kinds=event_kinds,
        trace_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        probes=tuple(probe_records),
        violations=tuple(violations),
        passed=not violations,
    )
