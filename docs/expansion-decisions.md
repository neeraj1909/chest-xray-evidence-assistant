# Expansion decisions

Decision date: 2026-09-10

The current bounded single-agent architecture stays. Segmentation, a critic
agent, external benchmark transfer, and EHR integration are deferred. The
machine-checked record is
[`expansion-decisions.json`](expansion-decisions.json).

## Evidence boundary

The project uses the owned, synthetic `CXR-AgentBench-v0`: 40 images and 120
prompts. The comparison below is the 48-prompt validation+test subset, with case
set SHA-256
`2fee50d77fcda85f7830c837662aa4165b3a47f2ae28da35a711db5ba8b63b13`.
Every configuration uses the same cases, seeds, limits, prompts, rubric,
response schema, retrieval index, and corpus.

The source is the committed
[`matrix-summary.json`](../tests/fixtures/evals/matrix-summary.json), SHA-256
`80c20ae5ed7d641033354e1823bfd86942d0699b0c341374a858454e13fc2fdd`.
This is a scripted offline harness. Its latency, token, and USD fields are
deterministic estimates, not live measurements. Its scores test wiring and
evaluation behavior, not clinical quality.

## Keep the bounded single agent

| Configuration | Task success | Tool-required success | Safety pass | Unsupported claims | High-severity failures | p95 ms | Tool calls | Output tokens | Estimated USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One-shot VLM | 0.6250 | 0.0000 | 1.0000 | 0.0000 | 0 | 16 | 0 | 1,719 | 0 |
| VLM + image tools | 0.8125 | 0.5000 | 1.0000 | 0.0000 | 0 | 25 | 9 | 2,052 | 0 |
| Full bounded agent | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 35 | 18 | 3,264 | 0 |

Against one-shot, the full agent gains 0.375 task success and 1.0
tool-required success. Against the image-tool baseline, it gains 0.1875 and
0.5 respectively. Neither comparison shows a scripted safety regression. The
cost of the full path is also explicit: +19 ms estimated p95 and +1,545
estimated output tokens over one-shot; +10 ms and +1,212 tokens over the
image-tool baseline.

Decision: keep the existing full bounded agent. This accepts the complexity
already present—one agent, three allow-listed tools, deterministic verification,
and fixed budgets. It does not authorize another tool or agent.

## Defer visual grounding or segmentation

No approved mask dataset, verified dataset/model license pair, selected model
artifact, local resource measurement, held-out IoU/measurement report, or
segmentation-specific safety report exists in this project.

Reconsider only after those artifacts exist, thresholds are pre-registered,
and the capability fits by replacing a current tool or by separately approving
a revised tool budget. A mask must remain visual evidence; it cannot become a
diagnosis or bypass provenance, abstention, confidence, or response validation.

## Defer an independent critic agent

The scripted held-out full-agent run has 1.0 task and safety pass rates, zero
unsupported claims, and zero high-severity failures. It therefore exposes no
residual failure set that justifies a critic experiment. No actor-versus-critic
ablation or live overhead measurement exists either.

Reconsider only after a reproducible residual-failure set shows deterministic
verification is insufficient. The actor-only and actor-plus-critic comparison
must use identical cases and demonstrate a positive task or safety gain without
regressing the frozen matrix. A critic remains read-only and receives no tool,
permission, or external-action authority.

## Defer external benchmark transfer

ChestAgentBench, AgentClinic, and MedAgentBench are candidates, not approved
inputs. This project has not verified an exact version's access terms, licenses,
data boundary, or action-schema compatibility, and it has not executed an
external report.

Reconsider one exact benchmark at a time after approval and review. Run it in a
separate sandbox, exclude incompatible tasks explicitly, and report transfer
results separately from `CXR-AgentBench-v0`. No EHR, PACS, patient-system, or
clinical-action integration is authorized.

## Next scope

Stabilize the local MVP. If the user explicitly supplies environment-only
configuration and authorizes possible provider cost, validate one live provider
through the existing smoke path. That evidence would close the remaining live
provider boundary without changing the architecture.
