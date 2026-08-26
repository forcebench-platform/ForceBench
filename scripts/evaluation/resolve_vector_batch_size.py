#!/usr/bin/env python3
"""Resolve the validated vector-evaluation batch size from task scene metadata."""

from __future__ import annotations

import argparse

from coribench.registry import get_task_manifest
from coribench.scene_profiles import get_scene_profile


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("task")
args = parser.parse_args()

manifest = get_task_manifest(args.task)
profile = get_scene_profile(manifest.scene_profile)
profile.validate()
print(profile.evaluation_num_envs)
