# Nut Thread assets

The repository includes the two ForceBench-authored reference wrappers at
`assets/normalized/nut_thread/{nut,bolt}.usda`. They preserve the collision
meshes in the external NVIDIA source USD files; they contain no source meshes
or textures. The authored wrappers use the repository's MIT license. Obtain
the source geometry separately under its applicable upstream terms.

The pinned sources are self-contained Factory/FORGE exports from the recorded
Isaac Lab 2.3.2 / Isaac Sim 5.1 deployment. Files from another asset release
must not be substituted based only on their names. The source filenames and
SHA256 values are:

| Local name | Upstream export name | SHA256 |
|---|---|---|
| `nut_m16.usd` | `factory_nut_m16.usd` | `8ff4fa10680ab09b780e9ba1f468eadc06ab3b3a8557c196bc2fa02f799677d4` |
| `bolt_m16.usd` | `factory_bolt_m16.usd` | `e822eb329d3ab2aad9bae57a405c60539ecea12bfa4d1af5d3e3609f1fe1f654` |

After obtaining those source exports, install them into your external asset
directory with the wrappers:

```bash
python tools/prepare_nut_assets.py --source-dir /path/to/factory --asset-dir /path/to/forcebench-assets
python tools/prepare_nut_assets.py --asset-dir /path/to/forcebench-assets --check-only
```

With a Python environment providing `pxr`, also check the composed collision
geometry and rigid-body structure:

```bash
python tools/verify_nut_asset_composition.py --asset-dir /path/to/forcebench-assets
```

The installer accepts either local or upstream export filenames, verifies both
source checksums before copying, and refuses to overwrite different existing
files. It performs no download, flattening or geometry conversion. Its output
is a JSON preflight report. Set `FORCEBENCH_ASSET_DIR` to the resulting asset
directory when launching the simulator.

The required layout is:

```text
forcebench-assets/
  nut_m16.usd
  bolt_m16.usd
  normalized/nut_thread/nut.usda
  normalized/nut_thread/bolt.usda
```

The wrappers normalize the support plane, leave one root rigid body, remove
the source articulation and fixed joint, and add `Support`, `Grasp` and `Mate`
frames. The nut source is translated by `(0, 0, -0.01)` m; the bolt source
translation is zero. Both referenced collision meshes retain `sdf`:
13,089 source nut vertices and 22,395 source bolt vertices. A USD composition
check confirmed these properties against the pinned source files on
2026-10-21. This check does not by itself validate contact behavior during a
complete episode.

The recovered historical general-purpose builder has a later
`convexDecomposition` option for Nut. Regenerating the wrappers with that
option would change the checked collision contract. The release therefore
distributes the pinned deployed wrappers and uses the installer above.
Their checksums and normalization are recorded in
`assets/normalized/nut_thread/source_manifest.json`.
