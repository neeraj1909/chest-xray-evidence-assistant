# CXR-AgentBench-v0

This directory contains an owned, deterministic, non-clinical benchmark. All
40 PNGs are programmatically generated grayscale patterns; they contain no
patient data and are licensed as synthetic project fixtures.

Regenerate the JSONL, manifest, and images from the repository root:

```bash
uv run python scripts/generate_benchmark.py
```

The manifest binds every image and the JSONL file by SHA-256, and tests require
the generator output to match the committed bytes. Development,
validation, and test partitions are assigned by image identity and unique
source group, never by prompt alone. The prompts test observable behavior,
tool use, retrieval provenance, contradictory input, and safe abstention. They
do not measure clinical accuracy. Observation ground truth is an actual
left-to-right pixel-intensity gradient that the offline policy decodes before
answering; it is not inferred from the manifest label.
