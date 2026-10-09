"""Merge the 19 audited studies into the 122-study corpus and recompute every corpus statistic.

Inputs : hub primary-studies.csv + reporting-compliance.json (read-only), audit-A.json, audit-B.json.
Outputs: primary-studies-141.csv, qa-141.json, stats-141.json (all in this directory), and a printed report.

The per-study 8-item reporting audit of the 122 studies is not archived, so its counts are
back-calculated from the published percentages (N=122 and per stage); ambiguous items are
recomputed under every candidate count.
"""
import csv, json, statistics, itertools
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
import os
HUB = Path(os.environ.get("SLR_HUB_DATA", str(Path(__file__).resolve().parent / "inputs")))
rows = list(csv.DictReader(open(HUB / "primary-studies.csv")))
qa = json.load(open(HUB / "reporting-compliance.json"))
new = json.load(open(HERE / "audit-A.json", encoding="utf-8")) + json.load(open(HERE / "audit-B.json", encoding="utf-8"))
assert len(rows) == 122 and len(new) == 19, (len(rows), len(new))
for s_ in new:  # agents store codes and QA as {code/score, justification/evidence}
    for f in ("primary_categories", "secondary_categories"):
        s_[f] = [c["code"] if isinstance(c, dict) else c for c in s_.get(f) or []]
    for f in ("qa1", "qa2", "qa3", "qa4"):
        if isinstance(s_[f], dict):
            s_[f] = s_[f]["score"]
    s_["scope_flag"] = s_.get("scope_flag") or ""

SCHEME = {r["Code"] if "Code" in r else list(r.values())[0]: r for r in csv.DictReader(open(HUB / "classification-scheme.csv"))}


def code(c):
    return c.strip().split("-")[0].strip()


def fmt_codes(codes):
    # follow the CSV spelling "3b-EarlyExit"; reuse the suffix seen in the existing rows
    seen = {}
    for r in rows:
        for c in (r["Primary Categories"] + ";" + r["Secondary Categories"]).split(";"):
            if c.strip():
                seen.setdefault(code(c), c.strip())
    return "; ".join(seen.get(code(c), code(c)) for c in codes)


def stage(c):
    c = code(c)
    if c[:2] in {"2c", "3l"} or c.startswith("1"):
        return "RQ1"
    if c.startswith("2"):
        return "RQ2"
    if c.startswith("3"):
        return "RQ4" if c[:2] in {"3g", "3h", "3i", "3m"} else "RQ3"
    if c.startswith("4"):
        return "RQ5"


RQMAP = {"RQ1": "RQ1", "RQ2": "RQ2", "RQ3": "RQ3", "RQ4": "RQ4", "RQ5": "RQ5"}
out = [dict(r) for r in rows]
for i, s in enumerate(sorted(new, key=lambda s: s["key"])):
    pc = [code(c) for c in s["primary_categories"]]
    sc = [code(c) for c in s.get("secondary_categories", [])]
    out.append({
        "ID": f"S{123 + i:03d}", "Key": s["key"], "Title": s["title"], "Year": str(s["year"]),
        "Venue": s["venue"] if s["venue_type"] != "preprint" else "arXiv preprint",
        "Venue Type": s["venue_type"], "Source": "search-update",
        "RQ": "; ".join(sorted({stage(c) for c in pc})),
        "Primary Categories": fmt_codes(pc), "Secondary Categories": fmt_codes(sc),
        "Scope Flags": s.get("scope_flag", ""), "Brief Rationale": s["rationale"],
    })
    qa[s["key"]] = {"qa1_empirical_evaluation": s["qa1"], "qa2_baseline_comparison": s["qa2"],
                    "qa3_experimental_config": s["qa3"], "qa4_reproducibility_artifacts": s["qa4"],
                    "qa_total": s["qa1"] + s["qa2"] + s["qa3"] + s["qa4"]}
N = len(out)
with open(HERE / "primary-studies-141.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(out)
json.dump(qa, open(HERE / "qa-141.json", "w"), indent=1)

R = {}
stg = {r["Key"]: sorted({stage(c) for c in r["Primary Categories"].replace(",", ";").split(";") if c.strip() and stage(c)}) for r in out}
peer = {r["Key"]: r["Venue Type"] != "preprint" for r in out}
R["N"] = N
R["stage_counts"] = {q: sum(q in v for v in stg.values()) for q in ["RQ1", "RQ2", "RQ3", "RQ4", "RQ5"]}
combos = Counter(tuple(v) for v in stg.values())
R["multi"] = sum(len(v) > 1 for v in stg.values())
R["max_span"] = max(len(v) for v in stg.values())
R["singles"] = {k[0]: n for k, n in combos.items() if len(k) == 1}
R["pairs"] = {"+".join(k): n for k, n in combos.most_common() if len(k) > 1}
R["intersection_range"] = (min(combos.values()), max(combos.values()))
R["years"] = dict(sorted(Counter(r["Year"] for r in out).items()))
R["years_peer"] = dict(sorted(Counter(r["Year"] for r in out if peer[r["Key"]]).items()))
R["venue_types"] = dict(Counter(r["Venue Type"] for r in out))
R["n_peer"] = sum(peer.values())
R["peer_share"] = round(100 * R["n_peer"] / N)
R["preprint_share"] = round(100 * (N - R["n_peer"]) / N)
R["peer_share_by_stage"] = {q: f"{sum(peer[k] for k, v in stg.items() if q in v)}/{R['stage_counts'][q]}="
                            f"{round(100 * sum(peer[k] for k, v in stg.items() if q in v) / R['stage_counts'][q])}%"
                            for q in R["stage_counts"]}
R["sources"] = dict(Counter(r["Source"] for r in out))
R["scope_flags"] = dict(Counter(r["Scope Flags"] for r in out))

VENUE_NORM = {"International Conference on Automated Software Engineering": "ASE",
              "International Symposium on Software Testing and Analysis": "ISSTA",
              "IEEE Transactions on Software Engineering": "TSE", "IEEE Trans. Software Eng.": "TSE",
              "ACM Trans. Softw. Eng. Methodol.": "TOSEM",
              "ACM Transactions on Software Engineering and Methodology": "TOSEM",
              "Conference on Empirical Methods in Natural Language Processing": "EMNLP",
              "Proc. ACM Softw. Eng.": "PACMSE"}
R["venues"] = Counter()
for r in out:
    if peer[r["Key"]]:
        v = r["Venue"]
        v = VENUE_NORM.get(v, v)
        if "PACMSE" in v or "Proc. ACM Softw" in v:
            v = "PACMSE"
        R["venues"][v] += 1
R["venues"] = R["venues"].most_common(15)

# primary-category leaders
cat = Counter()
cat_peer = Counter()
for r in out:
    for c in {code(c) for c in r["Primary Categories"].replace(",", ";").split(";") if c.strip()}:
        cat[c] += 1
        cat_peer[c] += peer[r["Key"]]
R["top_categories"] = cat.most_common(8)
R["top_categories_peer"] = cat_peer.most_common(8)
R["empirical_4b"] = (cat_peer["4b"], cat["4b"])

FIND = {"F1": ["1a", "1b", "2d"], "F2": ["2a", "2b"], "F3": ["2c", "2f"], "F4": ["3a", "3b", "3j", "3k", "3o"],
        "F5": ["3d", "3e", "3l"], "F6": ["3g", "3h", "3i", "3m"], "F7": ["4a", "4b"]}
R["findings"] = {}
for f, cs in FIND.items():
    sup = [r["Key"] for r in out if {code(c) for c in r["Primary Categories"].replace(",", ";").split(";") if c.strip()} & set(cs)]
    p = sum(peer[k] for k in sup)
    R["findings"][f] = (len(sup), p, round(100 * p / len(sup)))
R["prompt_compression_3d"] = (cat_peer["3d"], cat["3d"])
R["speculative_3a"] = (cat_peer["3a"], cat["3a"])
R["multiagent_3k"] = (cat_peer["3k"], cat["3k"])

tot = [qa[r["Key"]]["qa_total"] for r in out]
R["qa"] = {"mean": round(statistics.mean(tot), 2), "median": statistics.median(tot), "min": min(tot), "max": max(tot),
           "dist": dict(sorted(Counter(tot).items(), reverse=True))}
R["qa_old"] = {"mean": round(statistics.mean(qa[r["Key"]]["qa_total"] for r in rows), 2)}

# ---- 8-item audit: back-calculated 122 counts + 19 audited
ITEMS = ["correctness", "model_name", "hardware", "latency", "memory", "serving_config", "monetary", "energy"]
OLD_PCT = {"correctness": 96, "model_name": 97, "hardware": 69, "latency": 48, "memory": 19,
           "serving_config": 39, "monetary": 20, "energy": 7}


def cands(pct, n):
    return [k for k in range(n + 1) if round(100 * k / n + 1e-9) == pct or int(100 * k / n + 0.5) == pct]


old = {it: cands(p, 122) for it, p in OLD_PCT.items()}
# latency and memory are pinned by the peer/preprint split (59%/40% of 49/73; 31%/11%)
old["latency"] = [29 + 29]
old["memory"] = [15 + 8]
FAM = {"latency": "latency_throughput", "memory": "memory", "energy": "energy", "monetary": "monetary"}


def flag(s, it):
    # the four efficiency columns follow the generation-process metric families, so runtime of the
    # generated code (GenEffCode benchmarks) does not count as latency
    if it in FAM and "efficiency_metric_families" in s:
        return FAM[it] in s["efficiency_metric_families"]
    v = s["audit"][it]
    return bool(v.get("value")) if isinstance(v, dict) else bool(v)


newc = {it: sum(flag(s, it)
                for s in new) for it in ITEMS}
R["audit_old_candidates"] = old
R["audit_new_counts"] = newc
R["audit_141"] = {it: sorted({f"{k + newc[it]}/{N}={round(100 * (k + newc[it]) / N)}%" for k in old[it]}) for it in ITEMS}


# per-stage (Lat, Mem, Energy, Money) back-calculated from tab:compliance-stage
OLD_STAGE = {"RQ1": (25, 16, 8, 0, 32), "RQ2": (24, 42, 29, 4, 29), "RQ3": (59, 56, 12, 8, 8),
             "RQ4": (24, 58, 38, 8, 33), "RQ5": (21, 38, 38, 29, 10)}
R["stage_audit"] = {}
for q, (n, *pcts) in OLD_STAGE.items():
    ns = [s for s in new if q in {stage(c) for c in s["primary_categories"]}]
    cells = []
    for it, p in zip(["latency", "memory", "energy", "monetary"], pcts):
        cs = cands(p, n)
        add = sum(flag(s, it) for s in ns)
        nn = n + len(ns)
        cells.append("/".join(sorted({str(round(100 * (k + add) / nn)) for k in cs})))
    R["stage_audit"][q] = (n + len(ns), cells)

# metric depth: 32/41/49 of 122 (26/34/40%) counts efficiency metric families beyond the four audit
# columns (the four columns alone cannot reach 49 studies with two or more), so the new studies are
# counted over the broad family list recorded by the audit agents
depth = Counter(min(len(s.get("efficiency_metric_families", [])), 2) for s in new)
R["depth_new"] = dict(depth)
R["depth_141"] = {"none": round(100 * (32 + depth[0]) / N), "one": round(100 * (41 + depth[1]) / N),
                  "two_plus": round(100 * (49 + depth[2]) / N)}
lm = sum(flag(s, "latency") and flag(s, "memory") for s in new)
R["lat_mem_141"] = [f"{k + lm}/{N}={round(100 * (k + lm) / N)}%" for k in cands(15, 122)]
lmon = sum(flag(s, "latency") and flag(s, "monetary") for s in new)
R["lat_money_new"] = lmon
pp = [s for s in new if s["venue_type"] != "preprint"]
pr = [s for s in new if s["venue_type"] == "preprint"]
R["lat_peer_pre"] = (f"{29 + sum(flag(s, 'latency') for s in pp)}/{R['n_peer']}", f"{29 + sum(flag(s, 'latency') for s in pr)}/{N - R['n_peer']}")
R["mem_peer_pre"] = (f"{15 + sum(flag(s, 'memory') for s in pp)}/{R['n_peer']}", f"{8 + sum(flag(s, 'memory') for s in pr)}/{N - R['n_peer']}")
R["humaneval_mbpp_new"] = sum(bool(s["uses_humaneval_or_mbpp"]) for s in new)
R["multi_repo_new"] = sum(bool(s["multilingual_or_repo_level"]) for s in new)

json.dump(R, open(HERE / "stats-141.json", "w"), indent=1, default=str)
for k, v in R.items():
    print(f"{k}: {v}")
