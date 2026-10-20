#!/usr/bin/env python3
"""Check paper-table transcription with exact decimal arithmetic.

This checks reference data, not experimental reproduction or release readiness.
No input is rewritten. Decimal ROUND_HALF_UP preserves the manuscript's display
rounding (for example 22.125 -> 22.13), unlike Python's default round().
"""
from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_float=Decimal)


def audit(reference, leaderboard, protocol):
    errors = []
    means = []
    methods = [method["id"] for method in reference["methods"]]
    if len(methods) != 10 or len(set(methods)) != 10:
        errors.append("Reference must contain ten unique methods.")
    if "internvla" in " ".join(methods).lower():
        errors.append("InternVLA is outside the submitted-paper reference.")
    for study, task_count, method_count, episodes, total_key in (
        ("simulation", 8, 10, 200, "total_evaluation_episodes"),
        ("real_world", 5, 8, 20, "total_evaluation_trials"),
    ):
        data = reference[study]
        tasks = [task["id"] for task in data["tasks"]]
        rows = data["results"]
        row_ids = [row["method_id"] for row in rows]
        if len(tasks) != task_count or len(set(tasks)) != task_count:
            errors.append(f"{study}: expected {task_count} unique tasks.")
        if len(rows) != method_count or len(set(row_ids)) != method_count:
            errors.append(f"{study}: expected {method_count} unique method rows.")
        if set(row_ids) - set(methods):
            errors.append(f"{study}: results contain unknown methods.")
        if study == "simulation" and set(row_ids) != set(methods):
            errors.append("simulation: result method set differs from method list.")
        episode_key = "episodes_per_task_method" if study == "simulation" else "trials_per_task_method"
        if data[episode_key] != episodes:
            errors.append(f"{study}: expected {episodes} episodes per cell.")
        if data[total_key] != task_count * method_count * episodes:
            errors.append(f"{study}: total evaluation count is inconsistent.")
        for row in rows:
            rates = [Decimal(str(value)) for value in row["success_rates_percent"]]
            if len(rates) != task_count:
                errors.append(f"{study}/{row['method_id']}: wrong task count.")
                continue
            for task, rate in zip(tasks, rates):
                count = rate * episodes / 100
                if rate < 0 or rate > 100 or count != count.to_integral_value():
                    errors.append(f"{study}/{row['method_id']}/{task}: invalid episode-grid rate {rate}.")
            exact = sum(rates) / task_count
            display = exact.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            reported = Decimal(str(row["mean_success_percent"]))
            if reported != display:
                errors.append(f"{study}/{row['method_id']}: mean {reported} != {display}.")
            means.append({"study": study, "method_id": row["method_id"],
                          "exact_mean": str(exact), "display_mean": str(display),
                          "reported_mean": str(reported)})
    # Compare meaningful table content, including ordering and protocol; unrelated
    # descriptive metadata may evolve without changing the submitted tables.
    for key in ("methods", "simulation", "real_world"):
        if leaderboard.get(key) != reference.get(key):
            errors.append(f"leaderboard.{key} differs from the frozen paper reference.")
    protocol_tasks = protocol.get("tasks", [])
    display_ids = [task.get("display_id", task.get("id")) for task in protocol_tasks]
    aliases = {"rattle_sort": "com_sort", "faucet_force_turn": "faucet_direction",
               "stove_turn_on": "stove_turn", "stove_press_turn": "stove_press",
               "radmanso_drawer_sort_table": "drawer_pull"}
    display_ids = [aliases.get(task, task) for task in display_ids]
    expected_ids = [task["id"] for task in reference["simulation"]["tasks"]]
    if display_ids != expected_ids:
        errors.append("Protocol task order/aliases differ from paper reference.")
    if protocol.get("method_ids") != methods:
        errors.append("Protocol method order differs from paper reference.")
    for name, expected in (("physics_hz", 60), ("control_hz", 15),
                           ("max_control_steps", 1500), ("demonstrations_per_task", 200)):
        if protocol.get(name) != expected:
            errors.append(f"Protocol {name} must be {expected} for this reference.")
    for name, expected in (("training_seeds", {"start": 0, "stop_exclusive": 200}),
                           ("evaluation_seeds", {"start": 300, "stop_exclusive": 500})):
        if protocol.get(name) != expected:
            errors.append(f"Protocol {name} differs from paper reference.")
    return {"check": "paper_reference_consistency", "passed": not errors,
            "scope": "Reference transcription and arithmetic only; not release readiness or reproduced performance.",
            "simulation_cells": 80, "real_world_cells": 40,
            "rounding": "Decimal ROUND_HALF_UP to two decimal places",
            "errors": errors, "means": means}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=root / "results/paper_reference.json")
    parser.add_argument("--leaderboard", type=Path, default=root / "results/leaderboard.json")
    parser.add_argument("--protocol", type=Path, default=root / "configs/paper_protocol.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit(load(args.reference), load(args.leaderboard), load(args.protocol))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"check": "paper_reference_consistency", "passed": False,
                  "errors": [f"Malformed or unreadable input: {exc}"]}
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
