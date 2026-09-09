from __future__ import annotations

import pytest

from chest_xray_evidence_assistant.providers import (
    LiveModelConfig,
    ProviderConfigError,
    build_live_model,
)


@pytest.fixture(autouse=True)
def clear_live_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CXR_LIVE_SMOKE",
        "CXR_MODEL",
        "CXR_PROVIDER",
        "OLLAMA_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_openai_config_is_normalized_without_copying_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CXR_PROVIDER", " OpenAI ")
    monkeypatch.setenv("CXR_MODEL", " gpt-4.1-mini ")
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-value")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1/")

    config = LiveModelConfig.from_env()

    assert config == LiveModelConfig(
        provider="openai",
        model="gpt-4.1-mini",
        base_url="https://example.test/v1",
    )
    assert "test-secret-value" not in repr(config)


@pytest.mark.parametrize(
    ("environment", "expected_message"),
    [
        ({"CXR_PROVIDER": "unknown", "CXR_MODEL": "model"}, "CXR_PROVIDER"),
        ({"CXR_PROVIDER": "openai", "OPENAI_API_KEY": "key"}, "CXR_MODEL"),
        ({"CXR_PROVIDER": "openai", "CXR_MODEL": "model"}, "OPENAI_API_KEY"),
        ({"CXR_PROVIDER": "ollama", "CXR_MODEL": "model"}, "OLLAMA_BASE_URL"),
    ],
)
def test_incomplete_or_unknown_provider_config_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    expected_message: str,
) -> None:
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ProviderConfigError, match=expected_message):
        LiveModelConfig.from_env()


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://example.test/v1",
        "https://user:password@example.test/v1",
        "https://example.test/v1?api_key=secret",
        "https://example.test/v1#fragment",
        "not-a-url",
    ],
)
def test_provider_url_rejects_credentials_queries_and_invalid_urls(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
) -> None:
    monkeypatch.setenv("CXR_PROVIDER", "openai")
    monkeypatch.setenv("CXR_MODEL", "model")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    with pytest.raises(ProviderConfigError, match=r"HTTP\(s\) URL"):
        LiveModelConfig.from_env()


def test_model_construction_requires_explicit_live_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    config = LiveModelConfig(provider="openai", model="gpt-4.1-mini")

    with pytest.raises(ProviderConfigError, match="CXR_LIVE_SMOKE=1"):
        build_live_model(config)


def test_model_construction_revalidates_environment_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CXR_LIVE_SMOKE", "1")
    config = LiveModelConfig(provider="openai", model="gpt-4.1-mini")

    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY"):
        build_live_model(config)


def test_live_group_constructs_openai_model_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("openai")
    from pydantic_ai.models.openai import OpenAIChatModel

    monkeypatch.setenv("CXR_LIVE_SMOKE", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")

    model = build_live_model(LiveModelConfig(provider="openai", model="gpt-4.1-mini"))

    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "gpt-4.1-mini"
