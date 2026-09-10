#!/usr/bin/env python3
"""Build LCB post-2024-11-01 slice OFFLINE from /root/lcb-dataset (release_v6 = 6 JSONL increments).
Outputs data/lcb_tasks.jsonl (gen side) + data/lcb_eval_samples.jsonl (judge side, official
CodeGenerationProblem.get_evaluation_sample format @28fef95). Asserts N=288 + manifest match."""
import base64, csv, json, pickle, zlib
from pathlib import Path

SRC, OUT, CUTOFF = Path("/root/lcb-dataset"), Path("/root/efficient-codegen-exp/data"), "2024-11-01"
FILES = ["test.jsonl","test2.jsonl","test3.jsonl","test4.jsonl","test5.jsonl","test6.jsonl"]

rows = []
for f in FILES:
    for line in open(SRC/f, encoding="utf-8"):
        r = json.loads(line)
        if r["contest_date"] >= CUTOFF:      # ISO dates: lexicographic == chronological
            rows.append((f, r))
assert len(rows) == 288, f"expected 288, got {len(rows)}"
assert len({r["question_id"] for _, r in rows}) == 288
per_src = {}
for f, r in rows: per_src[f] = per_src.get(f, 0) + 1
assert per_src == {"test5.jsonl": 113, "test6.jsonl": 175}, per_src

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT/"lcb_tasks.jsonl","w",encoding="utf-8") as ft, open(OUT/"lcb_eval_samples.jsonl","w",encoding="utf-8") as fe:
    for f, r in rows:
        pub = json.loads(r["public_test_cases"])
        try:
            priv = json.loads(r["private_test_cases"])
        except Exception:   # official decode chain: base64 -> zlib -> pickle -> json
            priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(r["private_test_cases"].encode("utf-8")))))
        meta = json.loads(r["metadata"]) if r["metadata"] else {}
        starter = r["starter_code"] or ""
        ft.write(json.dumps({"task_id": r["question_id"], "question_title": r["question_title"],
            "question_content": r["question_content"], "starter_code": starter,
            "has_starter": bool(starter), "platform": r["platform"], "difficulty": r["difficulty"],
            "contest_date": r["contest_date"], "src_file": f}, ensure_ascii=False)+"\n")
        fe.write(json.dumps({"task_id": r["question_id"], "difficulty": r["difficulty"],
            "platform": r["platform"], "has_starter": bool(starter), "n_tests": len(pub)+len(priv),
            "input_output": json.dumps({"inputs":  [t["input"]  for t in pub+priv],
                                        "outputs": [t["output"] for t in pub+priv],
                                        "fn_name": meta.get("func_name", None)})}, ensure_ascii=False)+"\n")

man = {row["question_id"] for row in csv.DictReader(open(SRC/"lcb_post_2024-11-01.csv", encoding="utf-8"))}
assert man == {r["question_id"] for _, r in rows}, "manifest mismatch"
# starter/fn_name consistency: leetcode <=> starter <=> fn_name
chk = [json.loads(l) for l in open(OUT/"lcb_eval_samples.jsonl", encoding="utf-8")]
assert sum(1 for c in chk if c["has_starter"]) == 111
assert sum(1 for c in chk if json.loads(c["input_output"])["fn_name"]) == 111
print("OK: 288 tasks + 288 eval samples ->", OUT)
