# Provenance ledger gaps

`assets/ASSET_PROVENANCE.md` (as copied into this staging tree) states that
"All nine borrowed USDs retain explicit provenance and their deployed hashes
are recorded in `assets/SHA256SUMS`." That claim is incorrect for the copy of
`assets/SHA256SUMS` shipped in the source ForceBench tree (and therefore in
this extraction, which copies that ledger verbatim per scope: text/hash files
only, no binaries).

## What the doc claims is tracked (the "nine borrowed USDs")

1. `peg_8mm.usd` (from `factory_peg_8mm.usd`)
2. `hole_8mm.usd` (from `factory_hole_8mm.usd`)
3. `franka_mimic.usd` (from `franka_mimic.usd`)
4. `gear_base.usd` (from `factory_gear_base.usd`)
5. `gear_medium.usd` (from `factory_gear_medium.usd`)
6. `gear_small.usd` (from `factory_gear_small.usd`)
7. `gear_large.usd` (from `factory_gear_large.usd`)
8. `nut_m16.usd` (from `factory_nut_m16.usd`)
9. `bolt_m16.usd` (from `factory_bolt_m16.usd`)

## What `assets/SHA256SUMS` actually contains, of that list

Only 3 of the 9 are present: `peg_8mm.usd`, `hole_8mm.usd`, `franka_mimic.usd`.

(Note: `assets/SHA256SUMS` also has an entry for `peg_8mm_convex.usd`, which is
a ForceBench-derived asset, not one of the nine "borrowed" files, so it doesn't
count against or for this gap either way.)

## Missing entries (6 of 9) — all six gear/nut/bolt FORGE-derived files

- `gear_base.usd`
- `gear_medium.usd`
- `gear_small.usd`
- `gear_large.usd`
- `nut_m16.usd`
- `bolt_m16.usd`

None of these six appear anywhere in `assets/SHA256SUMS`, despite
`ASSET_PROVENANCE.md` asserting all nine are recorded there.

## Computed hashes for the six missing files (ready to slot in)

The actual `.usd` binaries are small (all well under 1.5 MB each) and were
directly reachable read-only on L20 at `/home/pc/ForceBench/assets/`, so rather
than just flagging the gap, `sha256sum` was run against them in place (no
binaries were copied into this staging tree — code/text only, per scope).
Whoever wires up the asset-hosting pipeline for these six files can drop these
lines directly into `assets/SHA256SUMS` once the binaries are actually staged
for distribution:

```
e55ace31759c493a159ae94840bbbb389bfc1eba396c58f77696368e4d5638aa  gear_base.usd
a86713f9cb518216456b1b9ab7fd159bce7cc083fce0afca2374fb40cb739b13  gear_medium.usd
f62c054f1f2ddfe4c6e96d2c6bcb5b9bcbc7cc87cb4c35d5e32d0d7c6e5367bf  gear_small.usd
9e92af584fa258670b7f61c3b81cdda3a74cf8d906fd44640209c6d01be0c377  gear_large.usd
8ff4fa10680ab09b780e9ba1f468eadc06ab3b3a8557c196bc2fa02f799677d4  nut_m16.usd
e822eb329d3ab2aad9bae57a405c60539ecea12bfa4d1af5d3e3609f1fe1f654  bolt_m16.usd
```

(Source: `/home/pc/ForceBench/assets/{gear_base,gear_medium,gear_small,gear_large,nut_m16,bolt_m16}.usd`
on L20, hashed 2026-08-26. File sizes ranged ~342 KB to ~1.24 MB — small enough
that hashing in place was safe and cheap; not verified against the original
FORGE/Factory source hashes cited in `ASSET_PROVENANCE.md`, only against the
bytes currently deployed on L20.)

## Also worth knowing

`ASSET_PROVENANCE.md` cites two additional raw *source* hashes (for the small
and large flanking gears, before ForceBench's own flattening step) that are
themselves not represented in `SHA256SUMS` under any name — they describe an
upstream/pre-flatten artifact, not the deployed file, so they're a separate,
smaller gap not counted above:

- small gear source: `c3284806005543e88006f10a2ee43454f4ceb4f43c72be810654c41afc46434c`
- large gear source: `c0c494b877f3548cc1d74759e14f115c1ba548c53f5000e2337445a7eca01739`
