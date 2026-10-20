# Distribution notes

This is a source-only snapshot. It contains no third-party scene/texture
payloads, datasets, trained model weights, private evaluation logs or internal
handoff archives. Their absence does not grant permission to redistribute them.

Three small authored USD wrapper/support files are included. The faucet and
stove wrappers reference the external Lightwheel meshes; the mesh and texture
payloads themselves are absent. The original Lightwheel license is retained
in `assets/lightwheel_kitchen/LICENSE.txt` for reference.

## First-party code

ForceBench's own source code is distributed under the MIT license in `LICENSE`.
Third-party code and assets retain their separate licenses. This grant does
not cover separately acquired datasets, assets or model weights.

## Bundled upstream source

Upstream notices remain beside the corresponding files:

- OpenPI variants: `LICENSE` and, where present, `LICENSE_GEMMA.txt` under their
  runtime directories.
- ForceVLA runtime variants and their bundled `flaxformer`: their respective
  `LICENSE` files.
- ForceFlow and CleanDiffuser: `baselines/forceflow/LICENSE` and
  `baselines/forceflow/CleanDiffuser/LICENCE.txt`.

These notices cover their own upstream materials. The root license does not
replace them, and source inclusion does not license separately downloaded
weights.

## External assets

Task code refers to processed NVIDIA SimReady/Isaac, Lightwheel Kitchen,
ArtVIP, and residential-scene assets. The direct paths are enumerated in
`forcebench/asset_requirements.json`. A complete redistributable asset recipe is
not yet provided. In particular, the ArtVIP and residential asset distribution
scope is unresolved. Acquire and use upstream content under its original
terms; do not assume the benchmark's code license covers those files.
