from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Sequence

from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from chest_xray_evidence_assistant.baselines import (
    OFFLINE_BASELINE_MODEL,
    capture_one_shot_baseline,
    load_baseline_case,
    serialize_baseline,
)
from chest_xray_evidence_assistant.providers import (
    LiveModelConfig,
    ProviderConfigError,
    build_live_model,
)

ROOT = Path(__file__).resolve().parents[1]
OFFLINE_RESPONSE = ROOT / "tests" / "fixtures" / "responses" / "answered.json"


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a fingerprinted one-shot/no-tool baseline artifact."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the explicitly configured live provider instead of the offline fake.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Artifact path (defaults below artifacts/baselines).",
    )
    return parser.parse_args(arguments)


def _default_output(live: bool) -> Path:
    filename = "one-shot-live.json" if live else "one-shot-offline.json"
    return ROOT / "artifacts" / "baselines" / filename


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    output_path = args.output or _default_output(args.live)

    if args.live and os.environ.get("CXR_LIVE_SMOKE", "").strip() != "1":
        print(json.dumps({"status": "disabled", "error_code": "live_baseline_not_enabled"}))
        return 2

    try:
        if args.live:
            config = LiveModelConfig.from_env()
            model = build_live_model(config)
            mode = "live"
            provider = config.provider
            model_name = config.model
            models.ALLOW_MODEL_REQUESTS = True
        else:
            models.ALLOW_MODEL_REQUESTS = False
            response_payload = json.loads(OFFLINE_RESPONSE.read_text(encoding="utf-8"))
            model = TestModel(
                custom_output_args=response_payload,
                model_name=OFFLINE_BASELINE_MODEL,
            )
            mode = "offline_fake"
            provider = "test"
            model_name = OFFLINE_BASELINE_MODEL

        request, image_bytes = load_baseline_case(ROOT)
        artifact = asyncio.run(
            capture_one_shot_baseline(
                request,
                image_bytes,
                model=model,
                mode=mode,
                provider=provider,
                model_name=model_name,
            )
        )
    except ProviderConfigError:
        print(json.dumps({"status": "failed", "error_code": "live_config_invalid"}))
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_code": f"baseline_{type(exc).__name__}",
                }
            )
        )
        return 1

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialize_baseline(artifact), encoding="utf-8")
    except OSError:
        print(json.dumps({"status": "failed", "error_code": "baseline_output_error"}))
        return 1
    print(
        json.dumps(
            {
                "status": "succeeded",
                "mode": artifact.mode,
                "output": str(output_path),
                "configuration_sha256": artifact.configuration_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
