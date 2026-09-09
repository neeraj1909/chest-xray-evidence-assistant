"""Fixed registry for validating model-requested tool calls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, cast

from pydantic import ValidationError

from ..models import ContractModel
from .contracts import (
    CropImageArguments,
    CropImageCall,
    GetImageMetadataArguments,
    GetImageMetadataCall,
    RetrieveReferenceArguments,
    RetrieveReferenceCall,
    ToolCall,
    ToolCallRejected,
    ToolName,
)


@dataclass(frozen=True, slots=True)
class _ToolDefinition:
    name: ToolName
    description: str
    arguments_type: type[ContractModel]
    call_type: type[ContractModel]


_DEFINITIONS: Final = (
    _ToolDefinition(
        name="crop_image",
        description="Create a bounded crop from one server-owned image.",
        arguments_type=CropImageArguments,
        call_type=CropImageCall,
    ),
    _ToolDefinition(
        name="get_image_metadata",
        description="Read verified safe metadata for one server-owned image.",
        arguments_type=GetImageMetadataArguments,
        call_type=GetImageMetadataCall,
    ),
    _ToolDefinition(
        name="retrieve_reference",
        description="Search the authorized reference corpus with a bounded query.",
        arguments_type=RetrieveReferenceArguments,
        call_type=RetrieveReferenceCall,
    ),
)


class ToolRegistry:
    """Expose and validate only the project's three initial tool contracts."""

    __slots__ = ()

    @property
    def names(self) -> tuple[ToolName, ...]:
        return tuple(definition.name for definition in _DEFINITIONS)

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        """Return deterministic model-facing schemas for the allow-listed tools."""

        return tuple(
            {
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.arguments_type.model_json_schema(),
            }
            for definition in _DEFINITIONS
        )

    def validate_call(
        self,
        name: str,
        arguments: Mapping[str, object],
    ) -> ToolCall:
        """Return a typed call or a redacted, fail-closed rejection."""

        definition = next(
            (candidate for candidate in _DEFINITIONS if candidate.name == name),
            None,
        )
        if definition is None:
            raise ToolCallRejected("unknown_tool")

        try:
            call = definition.call_type.model_validate({"name": name, "arguments": arguments})
        except ValidationError:
            raise ToolCallRejected("invalid_arguments") from None
        return cast(ToolCall, call)


INITIAL_TOOL_REGISTRY = ToolRegistry()
