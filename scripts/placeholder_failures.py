#!/usr/bin/env python3
"""Count placeholder answers in the HumanEval per-task records.

Reproduces the placeholder counts quoted in Section 9.1 of the manuscript and in
the response letter. Under the raw completion prompt, a Qwen2.5-Coder Instruct
checkpoint sometimes answers with a comment such as "# Your code here" and no
executable statement. This script reads the per-task records of the A800
HumanEval+ regeneration pass and, for every failed task, checks whether the
completion contains any executable line.

Usage (from the package root):
    python scripts/placeholder_failures.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "a800-results"
CELLS = [
    ("a800-1", "P01"), ("a800-2", "P04"), ("a800-2", "P05"), ("a800-2", "P06"),
    ("a800-2", "P07"), ("a800-2", "P08"), ("a800-2", "P09"),
    ("a800-1", "P10"), ("a800-1", "P11"), ("a800-1", "P12"),
    ("a800-2", "P14"), ("a800-2", "P16"), ("a800-2", "P17"),
    ("a800-1", "F01"),
]


def has_code(completion):
    return any(line.strip() and not line.strip().startswith("#")
               for line in completion.splitlines())


def main():
    print(f"{'cell':5} {'model':32} {'prec':5} {'pass@1':>7} {'placeholder':>11} {'points':>6}")
    for server, cell in CELLS:
        d = ROOT / server / "heplus"
        meta = json.loads((d / f"{cell}_meta.json").read_text(encoding="utf-8"))
        gen = {}
        for line in (d / f"{cell}.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            gen[rec["task_id"]] = rec["solution_list"][0]
        ev = json.loads((d / f"{cell}.evalplus-samples_eval_results.json").read_text(encoding="utf-8"))["eval"]
        n = len(ev)
        passed = sum(v[0]["base_status"] == "pass" for v in ev.values())
        stubs = sum(1 for tid, v in ev.items()
                    if v[0]["base_status"] != "pass" and not has_code(gen[tid]))
        print(f"{cell:5} {meta['model']:32} {meta['precision']:5} {100 * passed / n:7.2f} "
              f"{stubs:11d} {100 * stubs / n:6.1f}")


if __name__ == "__main__":
    main()
