#!/usr/bin/env python3
"""Resolve the formal evaluation step limit from task metadata."""

from __future__ import annotations

import argparse

from coribench.registry import get_task_manifest


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("task")
args = parser.parse_args()

manifest = get_task_manifest(args.task)
manifest.validate()
print(manifest.evaluation_max_steps)
