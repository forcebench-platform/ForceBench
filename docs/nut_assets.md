# Nut Thread assets

The repository includes the two ForceBench-authored reference wrappers at
`assets/normalized/nut_thread/{nut,bolt}.usda`. They preserve the collision
meshes in the external NVIDIA source USD files; they contain no source meshes
or textures. The authored wrappers use the repository's MIT license. Obtain
the source geometry separately under its applicable upstream terms.

Use the unified downloader to obtain the official Factory sources and the
other Nut Thread scene dependencies:

```bash
python tools/download_assets.py --source-dir /path/to/asset-source --task nut_thread
```

The official raw exports and the historical deployed exports have identical
complete authored USD content. The byte hashes differ. The runtime and both
installers accept only these explicitly audited choices:

| Local name | Historical deployed SHA256 | Official Isaac 5.1 raw SHA256 |
|---|---|---|
| `nut_m16.usd` | `8ff4fa10680ab09b780e9ba1f468eadc06ab3b3a8557c196bc2fa02f799677d4` | `f212a7f5664843abab3ec7d556ebf68ca99d4ad0880121978a45e445ad59431a` |
| `bolt_m16.usd` | `e822eb329d3ab2aad9bae57a405c60539ecea12bfa4d1af5d3e3609f1fe1f654` | `29bf74da441b636e02982721c2a8103f0035561b0beb45072fed5cdfc5c81cf0` |

Official filenames are `factory_nut_m16.usd` and `factory_bolt_m16.usd` under
NVIDIA's Isaac 5.1 `Isaac/IsaacLab/Factory/` root. The registry records their
exact URLs and expected sizes. No local flattening is needed. Complete typed
comparisons found zero differences across 57/60 authored paths and 201/207
fields. See [the verification record](../assets/nut_asset_equivalence_20261027.json).

官方原文件与历史导出文件的完整 USD 内容一致，安装器和运行时均保留精确白名单。
无需用近似螺纹、改变 SDF 碰撞或关闭哈希检查。

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

The installer accepts either local or upstream export filenames, verifies the approved
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
