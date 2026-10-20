#!/usr/bin/env python3
"""Verify the source payload against RELEASE_MANIFEST.json without a GPU."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    issues = []
    names = set()
    for row in manifest["files"]:
        name = PurePosixPath(row["path"])
        if (name.is_absolute() or ".." in name.parts or "\\" in row["path"]
                or ":" in row["path"] or row["path"] in names):
            issues.append({"path": row["path"], "issue": "invalid or duplicate path"})
            continue
        names.add(row["path"])
        path = ROOT.joinpath(*name.parts)
        disk = Path("\\\\?\\" + str(path.resolve())) if os.name == "nt" else path
        if path.is_symlink() or not disk.is_file() or not path.resolve().is_relative_to(ROOT):
            issues.append({"path": row["path"], "issue": "missing or indirect file"})
            continue
        content = disk.read_bytes()
        if len(content) != row["bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
            issues.append({"path": row["path"], "issue": "content differs"})
    print(json.dumps({"verified": not issues, "payload_files": len(names), "issues": issues}, indent=2))
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
