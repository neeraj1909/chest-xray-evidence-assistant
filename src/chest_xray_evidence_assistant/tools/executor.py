"""Budgeted dispatch and replay for the exact allow-listed tool set."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, cast

from pydantic import BaseModel

from ..models import Sha256Digest, TraceValue
from ..retrieval.records import ScoredChunk
from ..runtime import BudgetExceeded, BudgetSnapshot, DependencyUnavailable, RunBudget
from .contracts import (
    CropImageCall,
    CropImageResult,
    GetImageMetadataCall,
    ImageMetadataResult,
    ImageToolRejected,
    RetrieveReferenceCall,
    ToolCall,
    ToolCallRecord,
    ToolCallRejected,
    ToolRecordName,
)
from .ports import ImageToolsPort, ReferenceRetrievalPort
from .registry import INITIAL_TOOL_REGISTRY, ToolRegistry

ToolExecutionFailureCode: TypeAlias = Literal[
    "image_tool_unavailable",
    "image_not_authorized",
    "invalid_tool_result",
    "replay_mismatch",
    "retrieval_unavailable",
    "tool_execution_failed",
]
ToolResult: TypeAlias = CropImageResult | ImageMetadataResult | tuple[ScoredChunk, ...]


class ToolExecutionRejected(RuntimeError):
    """Expose only a stable dispatch failure code."""

    def __init__(self, code: ToolExecutionFailureCode) -> None:
        self.code = code
        super().__init__(code)


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=str)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return {"value_type": type(value).__name__}


def _digest(value: Any) -> Sha256Digest:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_arguments(call: ToolCall | None) -> dict[str, TraceValue]:
    if isinstance(call, CropImageCall):
        box = call.arguments.box
        return {
            "image_id": call.arguments.image_id,
            "x_min": box.x_min,
            "y_min": box.y_min,
            "x_max": box.x_max,
            "y_max": box.y_max,
        }
    if isinstance(call, GetImageMetadataCall):
        return {"image_id": call.arguments.image_id}
    if isinstance(call, RetrieveReferenceCall):
        return {
            "query_sha256": _digest(call.arguments.query),
            "top_k": call.arguments.top_k,
        }
    return {}


def _safe_result(result: ToolResult) -> dict[str, TraceValue]:
    if isinstance(result, CropImageResult):
        return {
            "source_image_id": result.source_image_id,
            "output_image_id": result.image.image_id,
            "input_sha256": result.input_sha256,
            "output_sha256": result.output_sha256,
            "width_px": result.image.width_px,
            "height_px": result.image.height_px,
            "byte_size": result.image.byte_size,
        }
    if isinstance(result, ImageMetadataResult):
        return {
            "image_id": result.image.image_id,
            "input_sha256": result.input_sha256,
            "output_sha256": result.output_sha256,
            "width_px": result.image.width_px,
            "height_px": result.image.height_px,
            "byte_size": result.image.byte_size,
        }
    chunk_ids = [item.chunk.chunk_id for item in result]
    document_ids = [item.chunk.document_id for item in result]
    return {
        "result_count": len(result),
        "chunk_ids_sha256": _digest(chunk_ids),
        "document_ids_sha256": _digest(document_ids),
    }


def _record_name(name: str, call: ToolCall | None) -> ToolRecordName:
    if call is not None:
        return call.name
    if name in INITIAL_TOOL_REGISTRY.names:
        return cast(ToolRecordName, name)
    return "unrecognized"


def _dependency_failure_code(call: ToolCall | None) -> ToolExecutionFailureCode:
    if isinstance(call, RetrieveReferenceCall):
        return "retrieval_unavailable"
    if isinstance(call, CropImageCall | GetImageMetadataCall):
        return "image_tool_unavailable"
    return "tool_execution_failed"


def _replay_id(
    *,
    run_sha256: str,
    sequence: int,
    name: ToolRecordName,
    arguments_sha256: str,
) -> str:
    return _digest(
        {
            "arguments_sha256": arguments_sha256,
            "name": name,
            "run_sha256": run_sha256,
            "sequence": sequence,
        }
    )


def _stable_record(record: ToolCallRecord) -> dict[str, Any]:
    payload = record.model_dump(mode="json")
    payload["budget_before"]["elapsed_ms"] = 0
    payload["budget_after"]["elapsed_ms"] = 0
    return payload


class BoundedToolExecutor:
    """Dispatch tools sequentially under one immutable parent run budget."""

    def __init__(
        self,
        *,
        budget: RunBudget,
        image_tools: ImageToolsPort,
        run_sha256: Sha256Digest,
        allowed_image_ids: frozenset[str],
        reference_retriever: ReferenceRetrievalPort | None = None,
        registry: ToolRegistry = INITIAL_TOOL_REGISTRY,
    ) -> None:
        self._scope = budget.child_scope()
        self._image_tools = image_tools
        self._reference_retriever = reference_retriever
        self._registry = registry
        self._run_sha256 = run_sha256
        self._allowed_image_ids = set(allowed_image_ids)
        self._records: list[ToolCallRecord] = []
        self._authorized_retrieval_results: dict[str, ScoredChunk] = {}

    @property
    def records(self) -> tuple[ToolCallRecord, ...]:
        return tuple(self._records)

    @property
    def authorized_retrieval_results(self) -> tuple[ScoredChunk, ...]:
        return tuple(self._authorized_retrieval_results.values())

    def _append_record(
        self,
        *,
        requested_name: str,
        call: ToolCall | None,
        arguments_sha256: str,
        result: ToolResult | None,
        failure_code: str | None,
        status: Literal["succeeded", "rejected", "failed"],
        budget_before: BudgetSnapshot,
    ) -> ToolCallRecord:
        sequence = len(self._records)
        name = _record_name(requested_name, call)
        record = ToolCallRecord(
            sequence=sequence,
            replay_id=_replay_id(
                run_sha256=self._run_sha256,
                sequence=sequence,
                name=name,
                arguments_sha256=arguments_sha256,
            ),
            name=name,
            status=status,
            arguments_sha256=arguments_sha256,
            result_sha256=_digest(result) if result is not None else None,
            failure_code=failure_code,
            safe_arguments=_safe_arguments(call),
            safe_result=_safe_result(result) if result is not None else {},
            budget_before=budget_before,
            budget_after=self._scope.snapshot(),
        )
        self._records.append(record)
        return record

    async def _dispatch(self, call: ToolCall) -> ToolResult:
        if isinstance(call, CropImageCall):
            if call.arguments.image_id not in self._allowed_image_ids:
                raise ToolExecutionRejected("image_not_authorized")
            result = await self._image_tools.crop_image(call.arguments)
            self._scope.consume_image_bytes(result.image.byte_size)
            self._allowed_image_ids.add(result.image.image_id)
            return result
        if isinstance(call, GetImageMetadataCall):
            if call.arguments.image_id not in self._allowed_image_ids:
                raise ToolExecutionRejected("image_not_authorized")
            return await self._image_tools.get_image_metadata(call.arguments)
        if self._reference_retriever is None:
            raise ToolExecutionRejected("retrieval_unavailable")
        results = await self._reference_retriever.retrieve_reference(call.arguments)
        if (
            not isinstance(results, tuple)
            or len(results) > call.arguments.top_k
            or any(not isinstance(result, ScoredChunk) for result in results)
        ):
            raise ToolExecutionRejected("invalid_tool_result")
        chunk_ids = [result.chunk.chunk_id for result in results]
        expected_query_sha256 = hashlib.sha256(call.arguments.query.encode("utf-8")).hexdigest()
        if (
            len(chunk_ids) != len(set(chunk_ids))
            or [result.rank for result in results] != list(range(1, len(results) + 1))
            or any(result.query_sha256 != expected_query_sha256 for result in results)
            or len({result.index_fingerprint for result in results}) > 1
        ):
            raise ToolExecutionRejected("invalid_tool_result")
        self._authorized_retrieval_results.update(
            (result.chunk.chunk_id, result) for result in results
        )
        return results

    async def execute(self, name: str, arguments: Mapping[str, object]) -> ToolResult:
        """Validate, charge, dispatch, and record one tool attempt."""

        arguments_sha256 = _digest(arguments)
        budget_before = self._scope.snapshot()
        call: ToolCall | None = None
        try:
            self._scope.consume_tool_call()
            call = self._registry.validate_call(name, arguments)
            async with asyncio.timeout(self._scope.remaining_seconds()):
                result = await self._dispatch(call)
        except TimeoutError:
            error = BudgetExceeded("timeout")
            self._append_record(
                requested_name=name,
                call=call,
                arguments_sha256=arguments_sha256,
                result=None,
                failure_code=error.code,
                status="rejected",
                budget_before=budget_before,
            )
            raise error from None
        except (
            BudgetExceeded,
            ImageToolRejected,
            ToolCallRejected,
            ToolExecutionRejected,
        ) as error:
            self._append_record(
                requested_name=name,
                call=call,
                arguments_sha256=arguments_sha256,
                result=None,
                failure_code=error.code,
                status="rejected",
                budget_before=budget_before,
            )
            raise
        except DependencyUnavailable:
            error = ToolExecutionRejected(_dependency_failure_code(call))
            self._append_record(
                requested_name=name,
                call=call,
                arguments_sha256=arguments_sha256,
                result=None,
                failure_code=error.code,
                status="failed",
                budget_before=budget_before,
            )
            raise error from None
        except Exception:
            error = ToolExecutionRejected("tool_execution_failed")
            self._append_record(
                requested_name=name,
                call=call,
                arguments_sha256=arguments_sha256,
                result=None,
                failure_code=error.code,
                status="failed",
                budget_before=budget_before,
            )
            raise error from None

        self._append_record(
            requested_name=name,
            call=call,
            arguments_sha256=arguments_sha256,
            result=result,
            failure_code=None,
            status="succeeded",
            budget_before=budget_before,
        )
        return result

    async def replay(
        self,
        expected: ToolCallRecord,
        name: str,
        arguments: Mapping[str, object],
    ) -> ToolCallRecord:
        """Re-execute a call and verify stable evidence apart from wall time."""

        try:
            await self.execute(name, arguments)
        except (BudgetExceeded, ImageToolRejected, ToolCallRejected, ToolExecutionRejected):
            pass
        actual = self._records[-1]
        if _stable_record(actual) != _stable_record(expected):
            raise ToolExecutionRejected("replay_mismatch")
        return actual
