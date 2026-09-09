from __future__ import annotations

import asyncio
import json
import os

from chest_xray_evidence_assistant.agent import run_evidence_request
from chest_xray_evidence_assistant.baselines import load_baseline_case
from chest_xray_evidence_assistant.models import EvidenceRequest
from chest_xray_evidence_assistant.providers import (
    LiveModelConfig,
    ProviderConfigError,
    build_live_model,
)


def load_smoke_case() -> tuple[EvidenceRequest, bytes]:
    return load_baseline_case()


def main() -> int:
    if os.environ.get("CXR_LIVE_SMOKE", "").strip() != "1":
        print(json.dumps({"status": "disabled", "error_code": "live_smoke_not_enabled"}))
        return 2

    try:
        config = LiveModelConfig.from_env()
        model = build_live_model(config)
        request, image_bytes = load_smoke_case()
        response = asyncio.run(run_evidence_request(request, image_bytes, model=model))
    except ProviderConfigError:
        print(json.dumps({"status": "failed", "error_code": "live_config_invalid"}))
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_code": f"live_{type(exc).__name__}",
                }
            )
        )
        return 1

    print(
        json.dumps(
            {
                "provider": config.provider,
                "model": config.model,
                "status": response.status,
                "confidence": response.confidence,
                "visual_evidence_count": len(response.visual_evidence),
                "source_evidence_count": len(response.source_evidence),
                "trace_events": len(response.trace),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
