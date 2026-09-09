"""Typed contracts for the Chest X-ray Evidence Assistant."""

from .model_ports import DeterministicModelAdapter, ModelAdapter
from .models import (
    AgentTraceEvent,
    ContractModel,
    EvidenceRequest,
    ImageAsset,
    ImageLocator,
    NormalizedBoundingBox,
    QuestionContext,
    RunLimits,
    RunTrace,
    SourceEvidence,
    VisualEvidence,
    VisualResponse,
)
from .observability import RunTraceCollector, TraceRedactionReport, audit_trace_redaction
from .runtime import (
    BudgetExceeded,
    BudgetSnapshot,
    DependencyUnavailable,
    RunBudget,
    ToolBudgetScope,
)

__all__ = [
    "AgentTraceEvent",
    "BudgetExceeded",
    "BudgetSnapshot",
    "ContractModel",
    "DeterministicModelAdapter",
    "DependencyUnavailable",
    "EvidenceRequest",
    "ImageAsset",
    "ImageLocator",
    "NormalizedBoundingBox",
    "ModelAdapter",
    "QuestionContext",
    "RunLimits",
    "RunBudget",
    "RunTrace",
    "RunTraceCollector",
    "SourceEvidence",
    "ToolBudgetScope",
    "TraceRedactionReport",
    "VisualEvidence",
    "VisualResponse",
    "audit_trace_redaction",
]
