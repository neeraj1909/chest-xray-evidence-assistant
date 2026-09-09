"""Explicit, opt-in hosted/local model construction."""

from __future__ import annotations

from dataclasses import dataclass
from os import environ
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from pydantic_ai.models import Model

ProviderName = Literal["openai", "ollama"]


class ProviderConfigError(ValueError):
    """Raised when live-provider configuration is unsafe or incomplete."""


@dataclass(frozen=True, slots=True)
class LiveModelConfig:
    """Non-secret provider settings; credentials remain environment-only."""

    provider: ProviderName
    model: str
    base_url: str | None = None

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()
        if provider not in {"openai", "ollama"}:
            raise ProviderConfigError("CXR_PROVIDER must be 'openai' or 'ollama'")

        model = self.model.strip()
        if not model:
            raise ProviderConfigError("CXR_MODEL is required")

        base_url = self.base_url
        if base_url is not None:
            base_url = _normalize_url("provider base URL", base_url)
        if provider == "ollama" and base_url is None:
            raise ProviderConfigError("OLLAMA_BASE_URL is required")

        object.__setattr__(self, "provider", cast(ProviderName, provider))
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", base_url)

    @classmethod
    def from_env(cls) -> LiveModelConfig:
        provider = environ.get("CXR_PROVIDER", "openai").strip().lower()
        model = environ.get("CXR_MODEL", "").strip()
        base_url = (
            _read_url("OPENAI_BASE_URL")
            if provider == "openai"
            else _read_url("OLLAMA_BASE_URL", required=provider == "ollama")
        )
        config = cls(
            provider=cast(ProviderName, provider),
            model=model,
            base_url=base_url,
        )
        config.validate_environment()
        return config

    def validate_environment(self) -> None:
        """Check required secrets without copying them into configuration."""

        if self.provider == "openai" and not environ.get("OPENAI_API_KEY", "").strip():
            raise ProviderConfigError("OPENAI_API_KEY is required")


def _normalize_url(name: str, value: str) -> str:
    normalized = value.strip()
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ProviderConfigError(
            f"{name} must be an HTTP(s) URL without credentials, query, or fragment"
        )
    return normalized.rstrip("/")


def _read_url(name: str, *, required: bool = False) -> str | None:
    value = environ.get(name, "").strip()
    if not value:
        if required:
            raise ProviderConfigError(f"{name} is required")
        return None
    return _normalize_url(name, value)


def build_live_model(config: LiveModelConfig | None = None) -> Model:
    """Build a provider model only for an explicitly enabled live smoke."""

    if environ.get("CXR_LIVE_SMOKE", "").strip() != "1":
        raise ProviderConfigError("set CXR_LIVE_SMOKE=1 to enable live providers")

    selected = config or LiveModelConfig.from_env()
    selected.validate_environment()

    try:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.ollama import OllamaProvider
        from pydantic_ai.providers.openai import OpenAIProvider
    except ImportError as exc:
        raise ProviderConfigError(
            "install the live dependency group with the OpenAI extra"
        ) from exc

    if selected.provider == "openai":
        provider = OpenAIProvider(base_url=selected.base_url)
    else:
        provider = OllamaProvider(base_url=selected.base_url)

    return OpenAIChatModel(selected.model, provider=provider)
