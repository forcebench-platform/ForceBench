# Distribution notes

ForceBench's own source code and authored USD reference/primitive layers are
distributed under the MIT license in `LICENSE`. Third-party source, models,
weights, datasets and assets retain their separate terms. The root license
does not replace upstream notices or grant rights to separately acquired
materials.

## Included upstream code

Preserved notices are stored beside the corresponding source:

- OpenPI variants: their `LICENSE` and, where present, `LICENSE_GEMMA.txt` files.
- ForceVLA variants and bundled Flaxformer: their respective license files.
- ForceFlow and CleanDiffuser: `baselines/forceflow/LICENSE` and
  `baselines/forceflow/CleanDiffuser/LICENCE.txt`.
- Other dependencies: their upstream notices remain applicable when installed.

Source inclusion does not grant permission to redistribute separately
downloaded foundation-model weights, tokenizer assets or trained checkpoints.

## External simulation assets

The source distribution includes 17 authored USDA reference/primitive layers.
It excludes the referenced third-party model and texture payloads. The
300-file runtime inventory in `assets/asset_manifest.json` is an identity and
dependency record for separately acquired files.

| Material | Distribution boundary |
|---|---|
| NVIDIA Factory/FORGE, SimReady, Isaac and residential-scene materials | Obtain and use under the applicable upstream terms. Referenced binaries are not bundled. |
| Lightwheel Kitchen sources | Retain the upstream CC BY-NC 4.0 terms and attribution. The upstream text is preserved at `assets/lightwheel_kitchen/LICENSE.txt`; source meshes and textures remain external. |
| Poly Haven textures | Obtain separately and retain their applicable source terms and attribution information. |
| ArtVIP RADMANSO model | The historical record has not established redistribution permission. No model payload is supplied or licensed by this repository. |
| Historical `franka_mimic.usd` export | The deployed bytes are pinned, but their complete upstream custody/permission chain is unconfirmed. The filename does not establish a redistribution grant. |
| Isaac Sim renderer materials and MDL SDK modules | Supplied by the independently installed runtime. They are checked as runtime requirements and are not copied into the asset bundle. |

Authored wrappers and derivation scripts do not license the external geometry
they reference or process. `tools/prepare_assets.py` imports a user-provided
asset tree after checksum checks; it does not download materials, accept new
asset terms or authorize redistribution. See [the asset guide](docs/assets.md).

## Scope of published evidence

Paper tables are labeled reference values. New execution checks are described
in `validation/public_report.json` and [the validation guide](docs/validation_scope.md).
The ten passed model diagnostics establish bounded software execution;
task-by-task simulator coverage is reported separately. Neither changes the
licensing boundary or establishes reproduction of published success rates.
The distribution excludes private handoff records, credentials, internal
evaluation archives, raw datasets and trained model payloads.
