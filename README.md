# Chest X-ray Evidence Assistant

A small, non-diagnostic text-and-vision assistant that produces structured,
evidence-linked observations for one allowed chest X-ray and one text question.

## Goal

Accept one synthetic, de-identified, demo, or appropriately licensed image and
a question, then return a typed response containing:

- observations and uncertainty;
- evidence tied to image regions;
- optional reference evidence with source provenance; and
- one of `answered`, `needs_clarification`, or `abstain`.

The default path is offline and deterministic. Live model providers are
optional, and invalid or insufficient inputs must fail closed.

## Architecture

```mermaid
flowchart TB
    U[User: upload one image and ask one question] --> UI[Minimal local UI or API]

    subgraph Input[Input boundary]
        UI --> REG[Asset registry]
        REG --> CHECK[Validate media type, size, hash, license, and PHI boundary]
        CHECK --> REQ[EvidenceRequest: image, question, context, and limits]
    end

    subgraph Runtime[Application runtime]
        REQ --> AGENT[Single Pydantic AI multimodal agent]
        AGENT --> ROUTER{Allow-listed tool registry: 0 to 3 calls}
        ROUTER --> CROP[crop_image: normalized crop]
        ROUTER --> META[get_image_metadata: hash, dimensions, media type]
        ROUTER --> RETRIEVE[retrieve_reference: bounded text query]
        CROP --> AGENT
        META --> AGENT
        RETRIEVE --> AGENT
        AGENT --> RAW[Candidate VisualResponse]
    end

    subgraph Evidence[Evidence and safety]
        CORPUS[Small reference corpus] --> INDEX[BM25 or in-memory index]
        INDEX --> RETRIEVE
        RAW --> VERIFY[Deterministic evidence and safety verifier]
        VERIFY --> OUT[Validated VisualResponse]
        OUT --> UI
    end

    subgraph Quality[Trace and evaluation]
        AGENT -.-> TRACE[Redacted trace: versions, hashes, tools, budgets, outcome]
        VERIFY -.-> TRACE
        TRACE --> EVAL[Pydantic Evals and deterministic graders]
        EVAL --> BENCH[CXR-AgentBench-v0 reports]
    end
``` 

The request flow is:

1. Validate the uploaded image and question.
2. Register the image with a server-owned ID and bounded metadata.
3. Run one Pydantic AI multimodal agent.
4. Let the agent call at most three typed tools: crop/zoom, image metadata, and
   reference retrieval.
5. Return tool results to the agent, then pass its candidate response to the
   deterministic verifier.
6. Verify evidence locators, source provenance, confidence, status, and runtime
   limits before returning a `VisualResponse`.
7. Store only redacted trace metadata for evaluation and reproducibility.

The current code implements the contract and fixture layers. The agent, tools,
retrieval, verifier, UI, and deployment layers will be added incrementally.

## Contracts

`src/chest_xray_evidence_assistant/models.py` defines the provider-neutral
interfaces:

- `EvidenceRequest` and `QuestionContext` for user input;
- `ImageAsset` for bounded image metadata and provenance;
- `VisualEvidence` and `ImageLocator` for grounded image claims;
- `SourceEvidence` for document and URL provenance;
- `RunLimits` for tool, model, image, output, timeout, and cost budgets; and
- `VisualResponse` for answers, clarification, abstention, uncertainty, and
  trace metadata.

`data/fixtures/manifest.json` records the hash, dimensions, size, origin, and
intended use of every committed synthetic fixture.

## Project status

Implemented:

- strict Pydantic contracts;
- deterministic synthetic image fixtures;
- fixture manifest and file verification; and
- deterministic fake response tests.

Planned:

- one opt-in multimodal agent;
- the three bounded tools;
- provenance-preserving retrieval;
- deterministic safety verification;
- a minimal local UI; and
- benchmark, evaluation, and Docker smoke workflows.

## Safety scope

This project does not diagnose disease, prescribe treatment, access PACS/EHR
systems, process real patient records, or make external clinical decisions.
The initial release uses no real clinical data. A second agent, segmentation
model, fourth tool, or EHR integration requires a separate evaluation before it
is added.

## Development

```bash
uv run --group dev pytest -q
uv run --group dev ruff format --check .
uv run --group dev ruff check .
uv run --group dev python -m compileall -q src tests scripts
uv run --group dev python scripts/generate_fixtures.py
```

The detailed staged implementation plan is maintained in the project planning
capsule outside this repository.
