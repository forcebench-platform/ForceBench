# Validation scope

`validation/public_report.json` is the release's machine-readable record
of tested source hashes, environment versions, asset pins, commands, coverage
and limitations. This document defines what those checks establish. Simulator
coverage and model diagnostics are separate entries in that report.

## Completed source and CPU corrections

The source corrections include:

- **Nut assets:** restored the deployed authored wrappers; pinned both external
  source USD hashes; verified one root rigid body, inactive source joints,
  normalization/frames and unchanged source SDF collision geometry.
- **CoM outcome:** aligned the success predicate with correct placement,
  release and rest. Stationary inspection remains an observation diagnostic
  rather than an extra success gate.
- **Stove geometry:** the old test incorrectly compared the knob with the
  original Stove root center. The corrected test inspects the actual compact,
  single-center visual asset. The actual SPEC position was not changed.
- **Faucet Direction:** new evaluations use the paper's one-sided signed angle
  threshold, with release, clearance and rest requirements retained. The
  original narrow-band scorer remains explicitly available for archived
  records. Historical table values are not rescored or overwritten.
- **Stove Press collection:** trajectory truncation and collection-quality
  checks now share the same per-clone post-action completion snapshot, rather
  than combining that trajectory with later batch-end measurements. This
  establishes consistent judging time points, not a completed 200-episode
  expert regeneration.

The complete CPU suite's observed counts, source state and skip explanations
are recorded in the public report.

## Asset validation

The offline importer verified and copied all 300 selected runtime files
(2,300,568,689 bytes) into an independent directory. An independent audit found
zero missing, different or unexpected files and no symlink payload dependence.
The installed OmniPBR and OmniGlass materials were checked separately.

All 97 raw authored dependency gaps remain recorded. Their source-bound
reviews cover renderer-provided MDL modules/materials, thumbnail branches
outside the imported default prim, unselected physics variants and a disabled
PaintTool branch. This is scoped coverage for the eight pinned default scenes,
rather than a claim that every upstream authoring variant is complete.

The single-center Stovetop builder independently reproduced the deployed
binary SHA256. Conveyor and warehouse-rack recipes preserve their recorded
input/output hashes but have not yet received a fresh rebuild check because
the original input caches were absent. Source/asset checks establish file
identity and transformation evidence, not redistribution rights.

## Simulator execution matrix

A simulator check must initialize the actual PhysX backend and RTX renderer,
step environments, produce finite state/action/wrench arrays, capture both
224×224 RGB views, and write checked output records. Hold actions are bounded
execution diagnostics; their success/failure counts are not policy scores.

| Default task | Simulator execution result |
|---|---|
| Weight Sort | Passed: 2 environments, 10 hold steps each |
| CoM Sort | Passed: 2 environments, 10 hold steps each |
| Faucet Turn | Passed: 2 environments, 10 hold steps each |
| Faucet Direction | Passed: 2 environments, 10 hold steps each |
| Stove Turn | Passed: 2 environments, 10 hold steps each |
| Stove Press | Passed: 2 environments, 10 hold steps each |
| Drawer Pull | Passed: 2 environments, 10 hold steps each |
| Nut Thread | Passed: 2 environments, 10 hold steps each |

All eight tasks passed on the final source and independently prepared asset
snapshot at seeds 300/301. Weight Sort and Nut Thread additionally passed one-
environment checks at seed 300: ten checks, eighteen episodes and 180 control
steps in total. State/action/joint/wrench arrays were finite, both RGB views
were nonuniform, and simulator logs contained no error lines. The background
room is visual-only; its inherited physics schemas, joints and collision groups
are removed before clone replication while task bodies remain outside that
subtree. The exact case and source hashes are in the public report.

## Model execution matrix

All ten systems passed bounded GPU execution checks using the packaged
numerical code and explicit real dataset/normalization/base-parameter inputs.
The nine OpenPI systems used recorded Faucet Turn data with real pi0/pi0.5
base parameters, three optimizer updates, finite nonzero gradients, recorded
observation inference and loopback OpenPI websocket exchange. Global batch was
32 on one device, except ForceVLA's batch 64 on two devices. Diagnostic data
loading used `num_workers=0`; no 10,000-update production checkpoint save/restore
was tested for these nine systems.

| Paper system | Bounded GPU result |
|---|---|
| pi0.5 No Force | Passed: 3 updates, recorded inference, loopback websocket |
| Token Fusion | Passed: 3 updates, recorded inference, loopback websocket |
| Prompt Fusion | Passed: 3 updates, recorded inference, loopback websocket |
| Cross-Attention | Passed: 3 updates, recorded inference, loopback websocket |
| Token Fusion + Pred | Passed: 3 updates, recorded inference, loopback websocket |
| Temporal Prompt | Passed: 3 updates, recorded inference, loopback websocket |
| Dual-Path Fusion | Passed: 3 updates, recorded inference, loopback websocket |
| pi0 | Passed: 3 updates, recorded inference, loopback websocket |
| ForceVLA | Passed: 3 updates, recorded inference, loopback websocket |
| ForceFlow | Passed: 2-update checks, recorded inference, native websocket, checkpoint reload |

ForceFlow used the recorded 200-episode Weight Sort dataset (91,368 frames),
including a full normalizer-boundary comparison. It strictly loaded the
selected 100,000-step checkpoint without changing it. A two-update diagnostic
at batch 2, a separate two-update native training CLI check, and a newly saved
checkpoint's strict reload/inference passed. Native direct and websocket
actions agreed exactly with shape `[32,7]`. These short checks do not establish
full-training convergence or historical paper checkpoint identity.

A separate fresh Python 3.11 environment passed the pi0.5/F4 three-update,
inference and two-request websocket check without inheriting existing site
packages. Git transport failed for the pinned LeRobot dependency; the official
archive at the same commit was used, with the other locked versions unchanged.
This verifies the controlled pi0.5 environment and F4, not fresh installations
of every model family.

A representative real-model simulator loop also passed: one Faucet Turn
environment at seed 300 executed eight control steps from a GPU pi0.5 base
model's actual `[50,7]` prediction through `openpi_ws`. It used real simulator
RGB/state, explicit Faucet normalization and a fresh server environment, with
no hold-action bypass and no task-training updates. This checks the action
pipeline; it is not complete-task success or a trained-policy score.

Installation, model execution, transport and a simulator-policy rollout are
distinct checks. A passed model-only diagnostic is not an end-to-end simulator
policy evaluation. Report each tested layer and any omitted checkpoint stage.

## Paper values and new outputs

`results/paper_reference.json` is an immutable transcription of the submitted
paper tables. `configs/paper_protocol.json` records the target protocol.
`tools/verify_paper_reference.py` validates those reference artifacts; it does
not rerun a policy.

New runs must record their own source, assets, dataset, normalization,
checkpoint, seed and scoring identities. Preserve first simultaneous full
success, fixed post-success stability and batch-end state as different fields.
Batch-end settling is not a uniform per-episode stability interval.

No software smoke result, successful asset import or partial historical recount
is a claim of complete paper reproduction. Final historical model-to-data and
scorer bindings require their own evidence. Real-world reference results do
not establish a validated real-robot deployment in this simulation release.

## Simulator package identity

The supported public installation target is the official Isaac Lab
`3.0.0b2.post1` wheel, paired with Isaac Sim 6.0.1.0. Core source, extension
configuration and application-file comparisons are recorded in the public
report. Wheel-specific extension initializers relocate package directories;
the inherited Factory camera patches and task environment implementations
are not loaded. Kit may load the corresponding agent metadata packages.

The copied reference tree has no Git metadata, so its historical upstream
commit remains unverified. A wheel source comparison and isolated package
execution check do not claim a complete clean simulator installation or
reproduction of the paper scores. See [the installation guide](getting_started_main.md).
