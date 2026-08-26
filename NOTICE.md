# NOTICE (draft)

ForceBench's own code (task definitions, controller, policy adapters, eval
harness, and the small "normalized wrapper" USDs it authors itself) is
released under [LICENSE](./LICENSE) for non-commercial research use.

Several tasks depend on 3D assets whose *own* copyright is held by third
parties. **ForceBench does not redistribute those parties' original asset
files.** It redistributes only (a) its own derived wrapper/normalization code
and (b) a documented pointer + checksum so a user who has independently
obtained the asset (through the vendor's own official channel, under that
vendor's own terms) can verify they have the right file and drop it into the
expected path. Per-task status below.

**IMPORTANT — this table is a draft assembled from the project's internal
provenance notes and has not had a legal review.** Two rows are marked
`UNCONFIRMED` because the extraction agent did not independently verify the
license of that source; do not treat this table as a legal clearance until
those are resolved.

| Task | 3rd-party asset(s) | Source | License (as currently understood) | Status |
|---|---|---|---|---|
| `weight_sort` | `flatbox_a02` | NVIDIA Omniverse SimReady catalog | NVIDIA Omniverse/Isaac asset terms (not open-source; redistribution of the binary itself is restricted) | Wrapper code only; binary must come from NVIDIA's own catalog |
| `rattle_sort` | `flatbox_a07` (`SimReady_Containers_Shipping_01_NVD`) | NVIDIA Omniverse SimReady catalog (downloaded on L20) | Same as above | Wrapper code only |
| `stove_turn_on` | `Stovetop012` | Lightwheel Kitchen, commit `10b0f7a8...` | **CC BY-NC 4.0** (non-commercial) | Wrapper + derived single-center mesh only; `LICENSE.txt` from the Lightwheel release is included verbatim in `assets/lightwheel_kitchen/` |
| `faucet_turn` | `Sink054`, `KitchenRoom.usd` | Lightwheel Kitchen, same release as above | **CC BY-NC 4.0** | Wrapper code only |
| `radmanso_drawer_sort_table` | ArtVIP RADMANSO 3-drawer cabinet | ArtVIP asset pack | **UNCONFIRMED** — license not yet verified by this project | Do not redistribute the binary until confirmed |
| `peg_insert` | `peg_8mm.usd`, `hole_8mm.usd`, `franka_mimic.usd` | "self-contained factory assets already present on franka2" per internal provenance notes | **UNCONFIRMED** — likely NVIDIA Factory/Isaac-derived given the `factory_*` source naming, but the chain of custody before it landed on the "franka2" machine was not independently traced | Do not redistribute the binary until confirmed |
| `gear_mesh` | `factory_gear_{base,medium,small,large}.usd` | NVIDIA Isaac Lab 2.3.2 FORGE deployment (`isaac-forge` on L20), image digest `sha256:388dbc8...` | NVIDIA Isaac Sim/Omniverse EULA (accepted via `ACCEPT_EULA=Y` at deployment time; not an open redistribution license) | Wrapper code only; user must run their own FORGE/Isaac Lab deployment to obtain the source files |
| `nut_thread` | `factory_nut_m16.usd`, `factory_bolt_m16.usd` | Same FORGE deployment as above | Same as gear_mesh | Wrapper code only |

## What ForceBench actually ships for each task

- Task logic (`spec.py`, `runtime.py`, success predicates, randomization
  ranges) — ForceBench's own code, MIT-equivalent under [LICENSE](./LICENSE).
- `assets/normalized/<task>/*.usda` wrapper files — ForceBench-authored glue
  (root rigid body, Support/Grasp/Mate frames, scale normalization) that
  *references* the third-party source file by path but does not embed it.
- `assets/SHA256SUMS` — checksums of both ForceBench's own derived files and
  (where legally shippable) the exact upstream file ForceBench was built
  against, so a user can confirm their independently-obtained copy matches.
- `scripts/download_assets.sh` — points to each vendor's official
  distribution channel per the table above; does not itself host or proxy any
  restricted binary.

## Open items before a v1.0 tag

1. Confirm ArtVIP's license for the RADMANSO cabinet asset used by
   `radmanso_drawer_sort_table`.
2. Trace the actual upstream source/license of `peg_8mm.usd` /
   `hole_8mm.usd` / `franka_mimic.usd` beyond "was already on franka2" --
   right now that's a location, not a provenance chain.
3. Get the missing 6 of 9 SHA256 entries into `assets/SHA256SUMS` (gear/nut/
   bolt files) -- flagged separately in `PROVENANCE_GAPS.md` from the code
   extraction.
