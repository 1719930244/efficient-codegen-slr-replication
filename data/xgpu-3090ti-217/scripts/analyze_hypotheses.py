#!/usr/bin/env python3
"""预注册假设分析脚本（H1–H5）

依据 docs/hypothesis-preregistration.md 的预注册判定准则，对
~/efficient-codegen-exp/results/{composition,pareto}/run{N}/<config>.json
（以及 energy_round/ 能耗补测轮）做逐假设检验，输出 Markdown 报告。

设计要点
- 只读结果 JSON，不修改任何实验数据。
- 缺失数据的假设标注「数据未齐」，绝不抛错中断。
- 检验：配对 t（以轮次为配对单元，见方法说明）+ Wilcoxon 符号秩双报告；
  Holm-Bonferroni 跨假设校正；报告 Cohen's d。
- 仅依赖 numpy/scipy/pandas；statsmodels 可选（有则线性混合模型辅助，
  无则降级为轮次级配对检验并在报告中说明）。

用法:
    python analyze_hypotheses.py --results-root ~/efficient-codegen-exp/results \
        --out hypothesis-report.md
    # 数据分散在多台机器时，可把各自 results 树同步到本地后传多个根目录:
    python analyze_hypotheses.py --results-root /data/a800-1/results /data/a800-2/results \
        --out hypothesis-report.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf  # type: ignore
    HAS_STATSMODELS = True
except Exception:  # pragma: no cover - 取决于环境
    smf = None
    HAS_STATSMODELS = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── 常量 ──────────────────────────────────────────────────

SIZE_ORDER = {"0.5B": 0, "1.5B": 1, "3B": 2, "7B": 3, "14B": 4, "32B": 5}
SMALL_SIZES = {"0.5B", "1.5B", "3B"}
LARGE_SIZES = {"14B", "32B"}

# 预注册判定阈值
H3_SUBADD_THRESHOLD = 0.90   # R_obs/R_pred < 0.90 → 支持次可加
H3_INDEP_THRESHOLD = 0.95    # ≥ 0.95 → 推翻
H5_MISMATCH_THRESHOLD = 0.50  # 不一致比例 ≥50% → 支持

# 预注册的平台映射（run1–3 V100，run4–5 A800；能耗轮在 A800）
def platform_for_run(run_label) -> str:
    if run_label == "energy":
        return "A800(能耗轮)"
    if isinstance(run_label, int):
        return "A800"  # 2026-09-05修正:本结果树全产自A800,V100数据遗失未并入
    return "未知"

DATA_NOT_READY = "数据未齐"


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class HypothesisResult:
    hid: str
    title: str
    verdict: str                 # 支持 / 推翻 / 部分推翻 / 数据未齐 / 不确定
    sections: list = field(default_factory=list)   # markdown 片段
    pvalues: dict = field(default_factory=dict)    # key -> 单侧 p（供 Holm）
    notes: list = field(default_factory=list)


# ── 数据加载 ──────────────────────────────────────────────

def _parse_size(model_name: str) -> str:
    """Qwen2.5-Coder-7B-Instruct → 7B"""
    if not isinstance(model_name, str):
        return ""
    parts = model_name.replace("Qwen2.5-Coder-", "").replace("-Instruct", "")
    return parts.strip("-")


def _extract_experiment_and_run(rel_parts) -> tuple | None:
    """把相对路径段映射为 (experiment, run_label)。

    支持:
      composition/run3/C01.json      -> ("composition", 3)
      pareto/run2/P07.json           -> ("pareto", 2)
      energy_round/composition/C01.json -> ("composition", "energy")
      energy_round/pareto/P01.json      -> ("pareto", "energy")
    """
    parts = [p.lower() for p in rel_parts]
    if parts and parts[0] == "energy_round":
        if len(parts) >= 3 and parts[1] in ("composition", "pareto"):
            return parts[1], "energy"
        return None
    if len(parts) >= 3 and parts[0] in ("composition", "pareto"):
        run_dir = parts[1]
        if run_dir.startswith("run"):
            try:
                return parts[0], int(run_dir[3:])
            except ValueError:
                return None
    return None


RESULT_COLUMNS = [
    "experiment", "run_label", "platform", "config_id", "model_name", "size",
    "precision", "decoding", "sampling", "pass_at_1", "mean_total_time_ms",
    "p50_total_time_ms", "p95_total_time_ms", "mean_tokens_per_sec",
    "peak_gpu_memory_mb", "mean_energy_j_per_request", "mean_energy_j_per_token",
    "num_problems", "details_nonempty", "path",
]


def load_results(roots: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    """扫描所有根目录，返回 (长表 DataFrame, 加载日志)。"""
    rows, log = [], []
    seen = set()
    n_bad = 0
    for root in roots:
        if not root.exists():
            log.append(f"[警告] 结果根目录不存在: {root}")
            continue
        for path in sorted(root.rglob("*.json")):
            if path.name == "summary.json":
                continue
            try:
                rel = path.relative_to(root).parts
            except ValueError:
                continue
            er = _extract_experiment_and_run(rel)
            if er is None:
                continue
            experiment, run_label = er
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception as e:
                n_bad += 1
                log.append(f"[警告] 无法解析 {path}: {e}")
                continue
            config_id = d.get("config_id") or path.stem
            key = (experiment, run_label, config_id)
            if key in seen:  # 多根目录重复，保留先出现的
                continue
            seen.add(key)
            rows.append({
                "experiment": experiment,
                "run_label": run_label,
                "platform": platform_for_run(run_label),
                "config_id": config_id,
                "model_name": d.get("model_name", ""),
                "size": _parse_size(d.get("model_name", "")),
                "precision": d.get("precision", ""),
                "decoding": d.get("decoding", ""),
                "sampling": d.get("sampling", ""),
                "pass_at_1": d.get("pass_at_1", np.nan),
                "mean_total_time_ms": d.get("mean_total_time_ms", np.nan),
                "p50_total_time_ms": d.get("p50_total_time_ms", np.nan),
                "p95_total_time_ms": d.get("p95_total_time_ms", np.nan),
                "mean_tokens_per_sec": d.get("mean_tokens_per_sec", np.nan),
                "peak_gpu_memory_mb": d.get("peak_gpu_memory_mb", np.nan),
                "mean_energy_j_per_request": d.get("mean_energy_j_per_request", -1.0),
                "mean_energy_j_per_token": d.get("mean_energy_j_per_token", -1.0),
                "num_problems": d.get("num_problems", 0),
                "details_nonempty": bool(d.get("details")),
                "path": str(path),
            })
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if n_bad:
        log.append(f"[警告] 共 {n_bad} 个 JSON 解析失败，已跳过。")
    return df, log


def inventory_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_未发现任何结果文件。_"
    lines = []
    for exp in ("composition", "pareto"):
        sub = df[df.experiment == exp]
        if sub.empty:
            lines.append(f"- **{exp}**: 无数据")
            continue
        runs = sorted(
            sub.run_label.unique(),
            key=lambda r: (r == "energy", r if r != "energy" else 0),
        )
        per_run = []
        for r in runs:
            rr = sub[sub.run_label == r]
            per_run.append(
                f"run{r}({len(rr)}配置)" if r != "energy" else f"能耗轮({len(rr)}配置)"
            )
        lines.append(f"- **{exp}**: {len(sub)} 条记录，轮次: {', '.join(per_run)}")
    plats = df.groupby("platform").size()
    lines.append("- **平台**: " + ", ".join(f"{k}: {v} 条" for k, v in plats.items()))
    if not df.empty and not df.details_nonempty.any():
        lines.append(
            "- **任务级 details**: 所有 JSON 的 `details` 均为空 → "
            "配对检验降级为以『轮次』为配对单元（详见方法说明）。"
        )
    return "\n".join(lines)


# ── 统计工具 ──────────────────────────────────────────────

def _onesided_p(p_two: float, statistic: float, direction: int) -> float:
    """direction=+1 表示备择为 >0。"""
    if np.isnan(p_two):
        return np.nan
    if (statistic > 0 and direction > 0) or (statistic < 0 and direction < 0):
        return p_two / 2.0
    return 1.0 - p_two / 2.0


def analyze_paired_diff(diff: np.ndarray, direction: int = +1) -> dict:
    """对配对差值数组做单样本 t + Wilcoxon + Cohen's d。

    direction: 备择方向（+1: 均值 > 0）。
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    out = dict(n=n, mean=float(np.mean(diff)) if n else np.nan,
               sd=float(np.std(diff, ddof=1)) if n >= 2 else np.nan,
               t=np.nan, p_t=np.nan, w=np.nan, p_w=np.nan, cohens_d=np.nan)
    if n == 0:
        return out
    if out["sd"] and out["sd"] > 0:
        out["cohens_d"] = out["mean"] / out["sd"]
    if n >= 2 and out["sd"] and out["sd"] > 0:
        res = stats.ttest_1samp(diff, 0.0)
        out["t"], out["p_t"] = float(res.statistic), _onesided_p(res.pvalue, res.statistic, direction)
    nz = diff[diff != 0]
    if len(nz) >= 1:
        try:
            alt = "greater" if direction > 0 else "less"
            wres = stats.wilcoxon(diff, alternative=alt, zero_method="wilcox")
            out["w"], out["p_w"] = float(wres.statistic), float(wres.pvalue)
        except Exception:
            pass
    return out


def holm_bonferroni(pvals: dict[str, float]) -> dict[str, float]:
    """返回校正后 p（键与输入一致）。NaN 保留为 NaN，不参与校正。"""
    items = [(k, p) for k, p in pvals.items() if p is not None and not np.isnan(p)]
    adj = {k: np.nan for k in pvals}
    if not items:
        return adj
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    running_max = 0.0
    for i, (k, p) in enumerate(items):
        running_max = max(running_max, (m - i) * p)
        adj[k] = min(running_max, 1.0)
    return adj


def fmt(x, spec=".3f", na="—"):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return na
    return format(x, spec)


def verdict_line(h: HypothesisResult) -> str:
    icon = {"支持": "✅ 支持", "推翻": "❌ 推翻", "部分推翻": "⚠️ 部分推翻",
            "数据未齐": "⏳ 数据未齐", "不确定": "❓ 不确定"}.get(h.verdict, h.verdict)
    return f"**判定：{icon}**"


def stat_block(res: dict, alpha: float, adj_p_t: float, adj_p_w: float) -> list[str]:
    lines = [
        f"- 配对轮次数 n = {res['n']}，差值均值 = {fmt(res['mean'])}，SD = {fmt(res['sd'])}",
        f"- 配对 t 检验：t = {fmt(res['t'], '.3f')}，单侧 p = {fmt(res['p_t'], '.4f')}"
        f"（Holm 校正后 {fmt(adj_p_t, '.4f')}）",
        f"- Wilcoxon 符号秩：W = {fmt(res['w'], '.1f')}，单侧 p = {fmt(res['p_w'], '.4f')}"
        f"（Holm 校正后 {fmt(adj_p_w, '.4f')}）",
        f"- Cohen's d = {fmt(res['cohens_d'])}",
    ]
    if res["n"] < 6:
        lines.append(f"- ⚠️ n={res['n']} ≤ 5：轮次级样本量过小，检验功效严重受限，结果需谨慎解读。"
                     "（本脚本使用单侧检验，n=5 时精确单侧最小 p=1/32≈0.031 仍可达 p<0.05，"
                     "显著性并非不可达；问题在于功效与稳健性，而非显著性天花板。）")
    return lines


# ── 数据提取辅助 ──────────────────────────────────────────

def comp_runs_with(df: pd.DataFrame, configs: list[str]) -> list:
    """返回同时包含全部指定配置的正式轮次（排除能耗轮）。"""
    sub = df[(df.experiment == "composition") & (df.run_label != "energy")]
    runs = []
    for r, g in sub.groupby("run_label"):
        if set(configs).issubset(set(g.config_id)):
            runs.append(r)
    return sorted(runs)


def comp_metric_by_run(df: pd.DataFrame, config: str, metric: str, runs) -> np.ndarray:
    vals = []
    sub = df[(df.experiment == "composition") & (df.config_id == config)]
    for r in runs:
        row = sub[sub.run_label == r]
        vals.append(float(row[metric].iloc[0]) if len(row) else np.nan)
    return np.asarray(vals, dtype=float)


# 预注册为 6 尺寸（含 32B）；run_pareto.py 尺寸表只有 5 尺寸（无 32B）。
PREREG_SIZES = ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]


def split_runs_by_platform(runs) -> dict:
    """按预注册平台映射把整数轮次分组（run1–3=V100，run4–5=A800）。"""
    groups: dict = {}
    for r in runs:
        if isinstance(r, int):
            groups.setdefault("V100" if r <= 3 else "A800", []).append(r)
    return groups


def platform_direction_check(runs, diff, desc: str) -> tuple[list, bool]:
    """预注册 §2.5：分平台检查配对差值的方向一致性。

    返回 (说明行列表, 是否方向冲突)。方向相反时按预注册不得合并两平台轮次。
    仅单平台有数据时不构成冲突。
    """
    diff = np.asarray(diff, dtype=float)
    groups = split_runs_by_platform(runs)
    if len(groups) < 2:
        plat = next(iter(groups), None)
        note = (f"- 平台一致性（§2.5）：当前共同轮次均位于 {plat}，无跨平台合并问题。"
                if plat else "- 平台一致性（§2.5）：无有效整数轮次。")
        return [note], False
    lines, means = [], {}
    for plat in sorted(groups):
        pruns = groups[plat]
        idx = [runs.index(r) for r in pruns]
        m = float(np.nanmean(diff[idx])) if idx else np.nan
        means[plat] = m
        lines.append(f"- {plat}（{', '.join('run' + str(r) for r in pruns)}）："
                     f"{desc}均值 = {fmt(m, '.1f')}")
    valid_nonzero = [m for m in means.values() if not np.isnan(m) and m != 0]
    conflict = len({m > 0 for m in valid_nonzero}) > 1
    if conflict:
        lines.append("- ⚠️ 两平台效应方向相反：按预注册 §2.5 不得合并两平台轮次，"
                     "池化检验仅作参考，不据其出具判定。")
    else:
        lines.append("- 两平台方向一致（或仅单平台有数据），池化检验照常报告。")
    return lines, conflict


# ── H1 解码策略主导 ───────────────────────────────────────

LAT = "p50_total_time_ms"  # 预注册主指标: wall-clock 延迟 p50

def test_h1(df: pd.DataFrame, alpha: float) -> HypothesisResult:
    h = HypothesisResult("H1", "解码策略主导假设（7B：speculative 收益 > 降一档量化收益）", DATA_NOT_READY)
    runs = comp_runs_with(df, ["C01", "C02", "C04"])
    if len(runs) < 2:
        h.sections.append(
            f"主检验需要 C01/C02/C04 至少 2 个共同轮次，当前仅 {len(runs)} 轮（{runs}）。"
            f"判定：**{DATA_NOT_READY}**。")
        return h
    c01 = comp_metric_by_run(df, "C01", LAT, runs)
    c02 = comp_metric_by_run(df, "C02", LAT, runs)
    c04 = comp_metric_by_run(df, "C04", LAT, runs)
    d_dec = c01 - c04    # Δ_dec = lat(C01) − lat(C04)
    d_quant = c01 - c02  # Δ_quant = lat(C01) − lat(C02)
    # H1 预测 Δ_dec > Δ_quant ⇔ (Δ_dec − Δ_quant) = (C02 − C04) > 0
    diff = d_dec - d_quant
    res = analyze_paired_diff(diff, direction=+1)

    tbl = ["| run | 平台 | lat(C01) | lat(C02) | lat(C04) | Δ_dec | Δ_quant |",
           "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(runs):
        tbl.append(f"| run{r} | {platform_for_run(r)} | {fmt(c01[i], '.0f')} | {fmt(c02[i], '.0f')} | "
                   f"{fmt(c04[i], '.0f')} | {fmt(d_dec[i], '.0f')} | {fmt(d_quant[i], '.0f')} |")
    h.sections.append("主检验（INT8 侧，延迟单位 ms，指标 p50）：\n\n" + "\n".join(tbl))
    h.pvalues["H1:main_t"] = res["p_t"]
    h.pvalues["H1:main_w"] = res["p_w"]
    h.notes.append(("main_res", res))  # stat_block 在 Holm 校正后由 finalize 渲染
    h.notes.append(("d_dec_mean", float(np.mean(d_dec))))
    h.notes.append(("d_quant_mean", float(np.mean(d_quant))))

    # INT4 侧交叉验证（C01/C03/C04）
    runs4 = comp_runs_with(df, ["C01", "C03", "C04"])
    if len(runs4) >= 2:
        a03 = comp_metric_by_run(df, "C03", LAT, runs4)
        a04 = comp_metric_by_run(df, "C04", LAT, runs4)
        # Δ_dec > Δ_quant4 ⇔ (C01−C04) > (C01−C03) ⇔ (C03 − C04) > 0
        res4 = analyze_paired_diff(a03 - a04, direction=+1)
        h.sections.append(
            f"INT4 侧交叉验证（C01/C03/C04，n={res4['n']}）：Δ_dec−Δ_quant4 均值 = "
            f"{fmt(res4['mean'], '.0f')} ms，单侧 p(t) = {fmt(res4['p_t'], '.4f')}，"
            f"单侧 p(Wilcoxon) = {fmt(res4['p_w'], '.4f')}，Cohen's d = {fmt(res4['cohens_d'])}。"
            "（注：该交叉验证为跨平台池化、探索性。）")
        h.pvalues["H1:int4_t"] = res4["p_t"]
    else:
        h.sections.append("INT4 侧交叉验证（C01/C03/C04）：共同轮次不足 2，数据未齐（不影响主判定）。")

    # 预注册 §2.5：分平台方向一致性检查（冲突时不合并轮次）
    plat_lines, conflict = platform_direction_check(runs, diff, "Δ_dec−Δ_quant")
    h.sections.append("**平台方向一致性（预注册 §2.5）**")
    h.sections.extend(plat_lines)
    if conflict:
        h.pvalues.clear()  # 不合并 → 池化 p 值（含 INT4 交叉）不参与 Holm 族
        h.notes.append(("platform_conflict", True))

    return h


# ── H2 量化×采样交互 ──────────────────────────────────────

def test_h2(df: pd.DataFrame, alpha: float) -> HypothesisResult:
    h = HypothesisResult("H2", "量化×采样交互假设（INT4 下 adaptive 的 pass@1 损失更大）", DATA_NOT_READY)
    runs = comp_runs_with(df, ["C01", "C03", "C07", "C09"])
    if len(runs) < 2:
        h.sections.append(
            f"需要 C01/C03/C07/C09 至少 2 个共同轮次，当前仅 {len(runs)} 轮（{runs}）。"
            f"判定：**{DATA_NOT_READY}**。")
        return h
    p = {c: comp_metric_by_run(df, c, "pass_at_1", runs) for c in ("C01", "C03", "C07", "C09")}
    I = (p["C07"] - p["C09"]) - (p["C01"] - p["C03"])
    res = analyze_paired_diff(I, direction=+1)

    # ⚠️ 指标语义警示（已核对 eval_humaneval.run_benchmark：adaptive 分支 passed=any(passed_list)）
    h.sections.append(
        "- ⚠️ **指标可比性警示**：adaptive 配置（C07/C09）JSON 中的 `pass_at_1` 实为 "
        "`any(passed_list)`——每题最多 10 次采样中至少通过一次的概率（≈pass@10）；"
        "greedy 配置（C01/C03）才是单次采样的真 pass@1。交互对比 [C07−C09]−[C01−C03] "
        "把两种不同估计量放在一起，违背预注册『pass@1（无偏估计）』的主指标定义，"
        "交互估计受污染，本节结论需显著降级解读。")
    m_c07, m_c01 = float(np.nanmean(p["C07"])), float(np.nanmean(p["C01"]))
    h.sections.append(
        f"- 数据示例：跨轮平均 pass(C07) = {fmt(m_c07, '.1f')}% vs pass(C01) = {fmt(m_c01, '.1f')}%，"
        "二者差距主要来自多次采样（≤10 次取或）而非模型能力差异。adaptive 配置的单样本真 pass@1 "
        "无法从现有汇总 JSON 恢复（`details` 为空），故无法给出可比口径的交互估计。")

    tbl = ["| run | 平台 | pass(C01) | pass(C03) | pass(C07) | pass(C09) | 交互对比 I |",
           "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(runs):
        tbl.append(f"| run{r} | {platform_for_run(r)} | {fmt(p['C01'][i], '.2f')} | {fmt(p['C03'][i], '.2f')} | "
                   f"{fmt(p['C07'][i], '.2f')} | {fmt(p['C09'][i], '.2f')} | {fmt(I[i], '.2f')} |")
    h.sections.append("2×2 子阵（standard 解码，单位 %；注意 C07/C09 实为 ≤10 次采样的 any-pass 率）：\n\n"
                      + "\n".join(tbl))
    # 可比口径参考（仅描述性）：greedy 内部真 pass@1 的量化效应
    d_greedy = p["C01"] - p["C03"]
    h.sections.append(
        f"- 可比口径参考（仅描述性）：greedy 侧真 pass@1 量化效应 C01−C03 跨轮均值 = "
        f"{fmt(float(np.nanmean(d_greedy)), '.2f')} 个百分点"
        "（该对比两侧均为单样本真 pass@1，口径可比）。")
    h.pvalues["H2:t"] = res["p_t"]
    h.pvalues["H2:w"] = res["p_w"]
    h.notes.append(("main_res", res))  # stat_block 在 Holm 校正后由 finalize 渲染

    # 预注册 §2.5：分平台方向一致性检查（冲突时不合并轮次）
    plat_lines, conflict = platform_direction_check(runs, I, "交互对比 I")
    h.sections.append("**平台方向一致性（预注册 §2.5）**")
    h.sections.extend(plat_lines)
    if conflict:
        h.pvalues.clear()  # 不合并 → 池化 p 值不参与 Holm 族
        h.notes.append(("platform_conflict", True))

    return h


# ── H3 叠加边际递减 ───────────────────────────────────────

def test_h3(df: pd.DataFrame, alpha: float) -> HypothesisResult:
    h = HypothesisResult("H3", "叠加边际递减假设（C12 实测提速 < 乘法模型预测，阈值 0.90）", DATA_NOT_READY)
    runs = comp_runs_with(df, ["C01", "C03", "C04", "C07", "C12"])
    if len(runs) < 1:
        h.sections.append(
            "需要 C01/C03/C04/C07/C12 至少 1 个共同轮次，当前为 0。"
            f"判定：**{DATA_NOT_READY}**。")
        return h
    rows = []
    ratios = []
    for r in runs:
        lat = {c: float(comp_metric_by_run(df, c, LAT, [r])[0])
               for c in ("C01", "C03", "C04", "C07", "C12")}
        if any(v <= 0 or np.isnan(v) for v in lat.values()):
            continue
        r_quant = lat["C01"] / lat["C03"]
        r_dec = lat["C01"] / lat["C04"]
        r_samp = lat["C07"] / lat["C01"]   # adaptive 相对 greedy 的延迟倍率（fp16/standard）
        R_pred = r_quant * r_dec * r_samp
        R_obs = lat["C01"] / lat["C12"]
        ratio = R_obs / R_pred if R_pred > 0 else np.nan
        ratios.append(ratio)
        rows.append((r, r_quant, r_dec, r_samp, R_pred, R_obs, ratio))
    if not ratios:
        h.sections.append(f"有效轮次为 0。判定：**{DATA_NOT_READY}**。")
        return h

    # ⚠️ 计量单位警示（已核对 eval_humaneval.adaptive_sampling：total_time_ms 逐样本累加）
    h.sections.append(
        "**⚠️ 计量单位警示（先读）**：adaptive 配置 C07/C12 的 `p50_total_time_ms` 是每题最多 "
        "10 次采样的累计墙钟时间，而 C01/C03/C04 是单次生成时间，两类计量单位不可直接相比。"
        "因此 r_samp 主要反映采样次数而非单样本减速；即使 r_quant、r_dec 均 ≥1，R_obs/R_pred "
        "也会被机械性压低——实测 run1：r_samp≈5.3、R_obs≈0.107、比值≈0.02≪0.90；要使比值达到 "
        "≥0.95 的推翻阈值需量化+投机合计约 45 倍提速，物理上不可能。故按预注册字面公式给出的"
        "判定（无论支持与否）对『叠加是否饱和』没有信息量，仅作公式忠实记录；实质性结论需单样本"
        "归一化，但 `details` 为空、每题实际采样次数不可恢复，无法精确归一化。")

    tbl = ["| run | 平台 | r_quant(C01/C03) | r_dec(C01/C04) | r_samp(C07/C01) | R_pred | R_obs(C01/C12) | R_obs/R_pred |",
           "|---|---|---|---|---|---|---|---|"]
    for (r, rq, rd, rs, rp, ro, rr) in rows:
        tbl.append(f"| run{r} | {platform_for_run(r)} | {fmt(rq)} | {fmt(rd)} | {fmt(rs)} | "
                   f"{fmt(rp)} | {fmt(ro)} | {fmt(rr)} |")
    mean_ratio = float(np.mean(ratios))
    h.sections.append("（延迟指标 p50；注意 r_samp 混入了采样次数，单位与其余列不一致，见上方警示）\n\n"
                      + "\n".join(tbl))
    # 单样本归一化敏感性：设 C07/C12 平均采样次数同为 k∈[3,10]（early-stop 最少 3 次、最多 10 次），
    # 则归一化后 R_obs 与 R_pred 同除以 k 的效应抵消为 ratio × k²。
    h.sections.append(
        f"- 单样本归一化敏感性（设 C07/C12 平均采样次数相同，k∈[3,10]）：归一化后比值 ≈ "
        f"R_obs/R_pred × k² ∈ [{fmt(mean_ratio * 9.0)}, {fmt(mean_ratio * 100.0)}]。"
        "区间宽且依赖不可辨识的 k，无法据此得出实质结论。")
    if any(rq < 1 or rd < 1 for (_, rq, rd, rs, rp, ro, rr) in rows):
        h.sections.append(
            "- ⚠️ 平台警示：存在 r_quant < 1 或 r_dec < 1 的轮次（相应单项技术在当前数据中"
            "并未提速）。乘法基线仍按预注册公式计算，次可加判定只依赖 R_obs 相对 R_pred "
            "的比值，但『叠加饱和』的机理解释在该平台需谨慎。")
    h.sections.append(f"- 跨轮平均 R_obs/R_pred = **{fmt(mean_ratio)}**"
                      f"（阈值：< {H3_SUBADD_THRESHOLD} 支持次可加；≥ {H3_INDEP_THRESHOLD} 推翻）")
    if len(ratios) >= 2:
        t_res = stats.ttest_1samp(np.asarray(ratios), H3_SUBADD_THRESHOLD)
        p_less = _onesided_p(t_res.pvalue, -t_res.statistic, +1)  # 检验 ratio < 0.90
        h.sections.append(f"- 辅助（探索性）单样本 t：ratio 与 0.90 比较，t = {fmt(t_res.statistic)}，"
                          f"单侧 p( ratio<0.90 ) = {fmt(p_less, '.4f')}")
        h.pvalues["H3:t"] = p_less
    if mean_ratio < H3_SUBADD_THRESHOLD:
        h.verdict = "支持"
    elif mean_ratio >= H3_INDEP_THRESHOLD:
        h.verdict = "推翻"
    else:
        h.verdict = "不确定"
        h.sections.append("- ⚠️ 均值落在 [0.90, 0.95) 区间，预注册未定义该区间的判定，记为不确定。")
    if h.verdict in ("支持", "推翻"):
        h.sections.append("- ⚠️ 上述判定按预注册字面公式给出，但受计量单位混杂影响（见本节开头警示），"
                          "『支持』由测量口径机械性导致、『推翻』分支实际不可达，均不构成对叠加饱和"
                          "与否的实质性证据；H3 实质结论待单样本归一化数据后方可给出。")
    h.notes.append(("mean_ratio", mean_ratio))
    return h


# ── H4 尺寸调节效应（帕累托前沿构成）───────────────────────

def pareto_frontier(points: list[dict]) -> list[dict]:
    """pass@1 越大越好、延迟越小越好的非支配集。"""
    frontier = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if (q["pass_at_1"] >= p["pass_at_1"] and q["lat"] <= p["lat"]
                    and (q["pass_at_1"] > p["pass_at_1"] or q["lat"] < p["lat"])):
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    return frontier


def _cfgs_from_group(g: pd.DataFrame) -> list:
    """把子集的贪心记录按 (precision, decoding, sampling) 聚合为前沿候选点。"""
    cfgs = []
    for (prec, dec, samp), gg in g.groupby(["precision", "decoding", "sampling"]):
        cfgs.append({
            "label": f"{prec}/{dec}",
            "precision": prec, "decoding": dec,
            "pass_at_1": float(gg["pass_at_1"].mean()),
            "lat": float(gg[LAT].mean()),
            "n_runs": len(gg),
        })
    return cfgs


def _frontier_families(cfgs: list) -> frozenset:
    """前沿的支配家族集合：quant（量化）/ dec（投机）/ base（基线）。"""
    fr = pareto_frontier(cfgs)
    return frozenset(
        "dec" if c["decoding"] == "speculative"
        else ("quant" if c["precision"] in ("int8", "int4") else "base")
        for c in fr)


def test_h4(df: pd.DataFrame, alpha: float) -> HypothesisResult:
    h = HypothesisResult("H4", "尺寸调节效应假设（≤3B 量化支配前沿，≥14B 投机解码支配前沿）", DATA_NOT_READY)
    pdf = df[(df.experiment == "pareto") & (df.run_label != "energy") & (df.sampling == "greedy")]
    # 7B 处并入 composition 的 speculative/greedy 配置（pass@1 可比：均为贪心输出）
    # 注意：列名 size 与 DataFrame.size 属性冲突，必须用 ["size"] 访问
    cdf = df[(df["experiment"] == "composition") & (df["run_label"] != "energy")
             & (df["sampling"] == "greedy") & (df["decoding"] == "speculative")
             & (df["size"] == "7B")]
    src = pd.concat([pdf, cdf], ignore_index=True)
    if src.empty:
        h.sections.append(f"无 Pareto/Composition 贪心解码数据。判定：**{DATA_NOT_READY}**。")
        return h
    if not cdf.empty:
        h.sections.append(
            "- ⚠️ 数据合并声明：为补齐 7B 的 speculative 贪心配置，把 composition 实验的记录并入 "
            "pareto 前沿。两类实验可能跑在不同 GPU/平台（V100/A800），延迟标尺未必可比；"
            "仅 pass@1 可比（均为贪心输出），跨实验延迟比较需谨慎解读。")

    sizes = sorted(src["size"].unique(), key=lambda s: SIZE_ORDER.get(s, 99))
    rev_lines = ["| 尺寸 | 前沿配置 | 量化型 | 解码型 | 基线(fp16/std) | 该尺寸有无speculative数据 |",
                 "|---|---|---|---|---|---|"]
    reversals, small_ok, large_ok, large_assessable = [], True, True, False
    any_spec_anywhere = False
    per_size = {}
    for s in sizes:
        g = src[src["size"] == s]
        cfgs = _cfgs_from_group(g)
        if len(cfgs) < 2:
            continue
        fr = pareto_frontier(cfgs)
        has_spec = any(c["decoding"] == "speculative" for c in cfgs)
        any_spec_anywhere = any_spec_anywhere or has_spec
        n_q = sum(1 for c in fr if c["decoding"] != "speculative" and c["precision"] in ("int8", "int4"))
        n_d = sum(1 for c in fr if c["decoding"] == "speculative")
        n_b = len(fr) - n_q - n_d
        per_size[s] = (fr, has_spec, n_q, n_d)
        rev_lines.append(
            f"| {s} | {', '.join(c['label'] for c in fr)} | {n_q} | {n_d} | {n_b} | "
            f"{'有' if has_spec else '无'} |")
        if s in SMALL_SIZES:
            if has_spec and n_d > n_q:
                small_ok = False
                reversals.append(f"{s}（小尺寸却由解码型支配前沿）")
        elif s in LARGE_SIZES:
            if has_spec:
                large_assessable = True
                if n_d <= n_q:
                    large_ok = False
                    reversals.append(f"{s}（大尺寸前沿由量化型支配，与预测相反）")
    h.sections.append("\n".join(rev_lines))

    # 预注册 6 尺寸中未出现在数据里的（区分『从未实施』与『数据未齐』）
    obs_sizes = set(src["size"])
    missing_prereg = [s for s in PREREG_SIZES if s not in obs_sizes]
    if missing_prereg:
        extra = ("其中 32B 属预注册 6 尺寸之一，但 run_pareto.py 尺寸表中不存在（从未实施），"
                 "属数据收集偏离预注册，与『缺能耗数据』性质不同。" if "32B" in missing_prereg else "")
        h.sections.append(f"- 预注册尺寸未出现在数据中：{'、'.join(missing_prereg)}。{extra}")

    # §2.5 平台敏感性：前沿点为跨平台轮次均值；两平台均有数据的尺寸分平台核对支配家族
    multi_plat_sizes, plat_notes = [], []
    for s in sizes:
        g = src[src["size"] == s]
        plats = sorted(set(g["platform"]))
        if len(plats) < 2:
            continue
        multi_plat_sizes.append(s)
        fams = {}
        for plat in plats:
            cfgs_p = _cfgs_from_group(g[g["platform"] == plat])
            fams[plat] = _frontier_families(cfgs_p) if len(cfgs_p) >= 2 else None
        fv = {plat: f for plat, f in fams.items() if f}
        if len(fv) >= 2 and len(set(fv.values())) > 1:
            plat_notes.append("- ⚠️ " + s + " 分平台前沿支配家族不一致："
                              + "；".join(f"{plat}={'/'.join(sorted(f))}" for plat, f in sorted(fv.items()))
                              + "，跨平台平均前沿掩盖平台差异，该尺寸的支配结论需分平台解读。")
    if multi_plat_sizes:
        h.sections.append("- ⚠️ 平台提示：上表前沿点为跨平台轮次均值（"
                          + "、".join(multi_plat_sizes)
                          + " 混合了 V100 run1–3 与 A800 run4–5），两平台延迟标尺可能不同；"
                          "按 §2.5 合并前应检查一致性，分平台核对如下：")
        h.sections.extend(plat_notes if plat_notes else
                          ["- 各平台前沿支配家族一致（或某平台配置数 <2 无法单独计算）。"])

    if not per_size:
        h.sections.append(f"可评估尺寸不足。判定：**{DATA_NOT_READY}**。")
        return h

    trivial_small = [s for s in sorted(SMALL_SIZES, key=lambda x: SIZE_ORDER[x])
                     if s in per_size and not per_size[s][1]]
    absent_small = [s for s in sorted(SMALL_SIZES, key=lambda x: SIZE_ORDER[x])
                    if s not in obs_sizes]

    detail = []
    if reversals:
        h.verdict = "部分推翻"
        detail.append("出现反向支配：" + "；".join(reversals) + "。按预注册报告边界条件。")
    elif not any_spec_anywhere:
        h.verdict = DATA_NOT_READY
        detail.append("全部尺寸均无投机解码配置数据，无法检验『≥14B 解码支配』一侧；"
                      "现有数据仅能说明 ≤3B 前沿由量化配置构成（与预测一致，但属平凡情形）。")
    elif not large_assessable:
        h.verdict = DATA_NOT_READY
        detail.append("小尺寸侧证据：" + ("符合量化支配预测。" if small_ok else "存在异常。")
                      + " 但 ≥14B 无投机解码数据，大尺寸侧无法检验。")
        if trivial_small or absent_small:
            detail.append("⚠️ 且小尺寸 " + "、".join(trivial_small + absent_small)
                          + " 无 speculative 配置数据，其『量化支配』平凡成立，不构成对小尺寸侧的实质检验。")
    elif small_ok and large_ok:
        h.verdict = "支持"
        detail.append("≤3B 量化支配、≥14B 解码支配，且无例外反转。")
        if trivial_small or absent_small:
            detail.append("⚠️ 但小尺寸 " + "、".join(trivial_small + absent_small)
                          + " 根本没有 speculative 配置数据，其『量化支配』平凡成立，"
                          "不构成对小尺寸侧假设的实质检验；『支持』结论的强度限于有 spec 对比的尺寸。")
    else:
        h.verdict = "推翻"
        detail.append("趋势方向不符合预注册。")
    h.sections.append("- " + "\n- ".join(detail) if detail else "")
    return h


# ── H5 能耗–延迟不一致 ────────────────────────────────────

def test_h5(df: pd.DataFrame, alpha: float) -> HypothesisResult:
    h = HypothesisResult("H5", "能耗–延迟不一致假设（≥50% 尺寸上两者最优配置不同）", DATA_NOT_READY)
    pdf = df[(df.experiment == "pareto") & (df.sampling == "greedy") & (df.decoding == "standard")]
    if pdf.empty:
        h.sections.append(f"无 Pareto（standard/greedy）数据。判定：**{DATA_NOT_READY}**。")
        return h
    edf = pdf[pdf.mean_energy_j_per_request >= 0]
    if edf.empty:
        h.sections.append(
            "所有 Pareto 记录的 `mean_energy_j_per_request` 均为 -1（未测能耗）。"
            f"判定：**{DATA_NOT_READY}**（需 A800 能耗轮 data）。")
        return h

    h.sections.append("（口径：**延迟与能耗的 argmin 均取自同一批有能耗测量的记录**（能耗轮，A800），"
                      "避免旧口径中延迟用跨平台全量记录、能耗仅用能耗轮记录造成的假性不一致；"
                      "不一致应由配置差异而非平台差异解释。）")
    sizes = sorted(pdf["size"].unique(), key=lambda s: SIZE_ORDER.get(s, 99))
    tbl = ["| 尺寸 | 延迟最优配置 | 延迟 p50(ms) | 能耗最优配置 | 能耗/任务(J) | 是否一致 |",
           "|---|---|---|---|---|---|"]
    n_eval, n_mismatch = 0, 0
    ref_lines = []
    for s in sizes:
        g_en = edf[edf["size"] == s]
        # 同源：延迟与能耗都在能耗测量记录内取 argmin
        lat_agg = g_en.groupby(["precision", "decoding"])[LAT].mean()
        en_agg = g_en.groupby(["precision", "decoding"])["mean_energy_j_per_request"].mean()
        if lat_agg.empty or en_agg.empty:
            tbl.append(f"| {s} | — | — | — | — | 该尺寸无能耗数据 |")
            continue
        lat_best = lat_agg.idxmin()
        en_best = en_agg.idxmin()
        n_eval += 1
        same = lat_best == en_best
        if not same:
            n_mismatch += 1
        tbl.append(f"| {s} | {lat_best[0]}/{lat_best[1]} | {fmt(lat_agg.min(), '.0f')} | "
                   f"{en_best[0]}/{en_best[1]} | {fmt(en_agg.min(), '.2f')} | "
                   f"{'一致' if same else '**不一致**'} |")
        # 参考：跨平台全量记录的延迟 argmin（仅提示差异来源，不作判定依据）
        lat_agg_all = pdf[pdf["size"] == s].groupby(["precision", "decoding"])[LAT].mean()
        if not lat_agg_all.empty and lat_agg_all.idxmin() != lat_best:
            ref_lines.append(f"- 参考：{s} 在跨平台全量记录中的延迟 argmin 为 "
                             f"{'/'.join(lat_agg_all.idxmin())}，与同源口径不同"
                             "（差异来自平台/轮次标尺，不改变判定）。")
    h.sections.append("\n".join(tbl))
    h.sections.extend(ref_lines)
    per_cfg_n = edf.groupby(["size", "config_id"]).size()
    if (per_cfg_n <= 1).any():
        h.sections.append(f"- ⚠️ 存在尺寸×配置仅 {int(per_cfg_n.min())} 条能耗记录的情形"
                          "（能耗轮每配置通常只跑 1 次），argmin 对单次测量噪声敏感，无重复可平均。")
    if n_eval == 0:
        h.sections.append(f"无尺寸可同时评估能耗与延迟。判定：**{DATA_NOT_READY}**。")
        return h
    ratio = n_mismatch / n_eval
    h.sections.append(f"- 可评估尺寸 {n_eval} 个，其中不一致 {n_mismatch} 个，"
                      f"比例 = **{fmt(ratio, '.1%')}**（阈值 ≥ {H5_MISMATCH_THRESHOLD:.0%}）。")
    if n_eval < len(sizes):
        h.sections.append(f"- ⚠️ {len(sizes) - n_eval} 个尺寸缺能耗数据，未计入分母。")
    if "32B" not in set(pdf["size"]):
        h.sections.append("- ⚠️ 32B 为预注册 6 尺寸之一，但 run_pareto.py 尺寸表中不存在（从未实施），"
                          "属数据收集偏离预注册，与『缺能耗数据』性质不同，已在方法说明中单列。")
    h.verdict = "支持" if ratio >= H5_MISMATCH_THRESHOLD else "推翻"
    h.notes.append(("ratio", ratio))
    return h


# ── 可选混合效应模型（statsmodels）────────────────────────

PLATFORM_FE_NOTE = (
    "注：预注册 §2.2 将『平台』列为固定效应，但平台由轮次编号完全决定"
    "（run1–3=V100，run4–5=A800），与轮次随机截距完全共线，无法同时纳入同一模型"
    "（客观限制）；平台一致性问题改按预注册 §2.5 在 H1/H2 的分平台方向一致性检查中处理"
    "（见相应小节）。")


def mixed_models_section(df: pd.DataFrame) -> str:
    if not HAS_STATSMODELS:
        return ("**线性混合模型**：当前环境未安装 statsmodels，按预注册分析计划第 2 条降级为"
                "轮次级配对检验（上文已报告）。如需混合模型请 `pip install statsmodels` 后重跑。\n\n"
                + PLATFORM_FE_NOTE)
    out = []
    try:
        runs = comp_runs_with(df, ["C01", "C02", "C04"])
        rows = []
        for r in runs:
            for c in ("C01", "C02", "C04"):
                v = comp_metric_by_run(df, c, LAT, [r])[0]
                rows.append({"run": f"run{r}", "config": c, "lat": v})
        if len(runs) >= 3:
            m = smf.mixedlm("lat ~ C(config, Treatment(reference='C01'))",
                            pd.DataFrame(rows), groups=pd.DataFrame(rows)["run"])
            fit = m.fit(reml=False)
            out.append("**H1 辅助混合模型**（延迟 ~ 配置，轮次为随机截距）：")
            out.append("```")
            out.append(str(fit.summary().tables[1]))
            out.append("```")
        else:
            out.append("**H1 辅助混合模型**：共同轮次 <3，随机效应不可靠，跳过。")
    except Exception as e:
        out.append(f"**H1 辅助混合模型**：拟合失败（{type(e).__name__}: {e}），已降级。")
    try:
        runs = comp_runs_with(df, ["C01", "C03", "C07", "C09"])
        rows = []
        for r in runs:
            for c, q, s in (("C01", "fp16", "greedy"), ("C03", "int4", "greedy"),
                            ("C07", "fp16", "adaptive"), ("C09", "int4", "adaptive")):
                v = comp_metric_by_run(df, c, "pass_at_1", [r])[0]
                rows.append({"run": f"run{r}", "quant": q, "sampling": s, "pass1": v})
        if len(runs) >= 3:
            d2 = pd.DataFrame(rows)
            m = smf.mixedlm("pass1 ~ C(quant) * C(sampling)", d2, groups=d2["run"])
            fit = m.fit(reml=False)
            out.append("**H2 辅助混合模型**（pass@1 ~ 精度×采样，轮次为随机截距）：")
            out.append("```")
            out.append(str(fit.summary().tables[1]))
            out.append("```")
        else:
            out.append("**H2 辅助混合模型**：共同轮次 <3，随机效应不可靠，跳过。")
    except Exception as e:
        out.append(f"**H2 辅助混合模型**：拟合失败（{type(e).__name__}: {e}），已降级。")
    out.append(PLATFORM_FE_NOTE)
    return "\n\n".join(out)


# ── 主流程 ────────────────────────────────────────────────

def finalize_h1(h: HypothesisResult, adj: dict, alpha: float):
    if "main_res" in dict(h.notes):
        notes = dict(h.notes)
        res = notes["main_res"]
        if notes.get("platform_conflict"):
            h.verdict = "不确定"
            h.sections.append(
                "- 判定依据：§2.5 平台方向冲突，按预注册不合并轮次；"
                "池化检验不再作为判定依据，待平台内轮次补齐后分平台检验。")
            return
        point_ok = res["mean"] > 0  # Δ_dec − Δ_quant > 0
        pt, pw = adj.get("H1:main_t", np.nan), adj.get("H1:main_w", np.nan)
        h.sections.extend(stat_block(res, alpha, pt, pw))
        sig = (not np.isnan(pt) and pt < alpha) or (not np.isnan(pw) and pw < alpha)
        h.verdict = "支持" if (point_ok and sig) else "推翻"
        extra = "" if point_ok else "（点估计方向与预测相反：Δ_dec ≤ Δ_quant）"
        h.sections.append(
            f"- 判定依据：点估计 Δ_dec>Δ_quant {'成立' if point_ok else '不成立'}{extra}；"
            f"Holm 校正后 p(t)={fmt(pt, '.4f')}、p(Wilcoxon)={fmt(pw, '.4f')}，α={alpha}。")
        dd, dq = dict(h.notes).get("d_dec_mean", np.nan), dict(h.notes).get("d_quant_mean", np.nan)
        if (not np.isnan(dd) and dd <= 0) or (not np.isnan(dq) and dq <= 0):
            h.sections.append(
                f"- ⚠️ 平台警示：当前数据中 Δ_dec 均值 = {fmt(dd, '.0f')} ms、Δ_quant 均值 = "
                f"{fmt(dq, '.0f')} ms，存在 ≤0 的情形（相应技术相对基线并未提速、甚至更慢）。"
                "判定仍按预注册字面准则（Δ_dec > Δ_quant 的相对大小）执行，"
                "但『解码带来的吞吐提升』的原始陈述在该平台不成立，解读时务必注意。")


def finalize_h2(h: HypothesisResult, adj: dict, alpha: float):
    if "main_res" in dict(h.notes):
        notes = dict(h.notes)
        res = notes["main_res"]
        if notes.get("platform_conflict"):
            h.verdict = "不确定"
            h.sections.append(
                "- 判定依据：§2.5 平台方向冲突，按预注册不合并轮次；"
                "池化交互估计不再作为判定依据（且该估计量本身受 any-pass 口径污染，见本节开头警示）。")
            return
        point_ok = res["mean"] > 0  # I > 0
        pt, pw = adj.get("H2:t", np.nan), adj.get("H2:w", np.nan)
        h.sections.extend(stat_block(res, alpha, pt, pw))
        sig = (not np.isnan(pt) and pt < alpha) or (not np.isnan(pw) and pw < alpha)
        h.verdict = "支持" if (point_ok and sig) else "推翻"
        h.sections.append(
            f"- 判定依据：交互对比 I 均值{'>' if point_ok else '≤'}0；"
            f"Holm 校正后 p(t)={fmt(pt, '.4f')}、p(Wilcoxon)={fmt(pw, '.4f')}，α={alpha}。"
            "（注意：该判定基于受污染的估计量——C07/C09 为 ≤10 次采样的 any-pass 率，"
            "与 C01/C03 的单样本真 pass@1 不同源，见本节开头警示。）")


def main():
    ap = argparse.ArgumentParser(description="预注册假设分析（H1–H5）")
    ap.add_argument("--results-root", nargs="+", required=True,
                    help="结果根目录（可多个），应包含 composition/ pareto/ [energy_round/] 子目录")
    ap.add_argument("--out", required=True, help="Markdown 报告输出路径")
    ap.add_argument("--alpha", type=float, default=0.05, help="显著性水平（默认 0.05）")
    args = ap.parse_args()

    roots = [Path(p).expanduser() for p in args.results_root]
    df, load_log = load_results(roots)

    hs = []
    for fn in (test_h1, test_h2, test_h3, test_h4, test_h5):
        try:
            hs.append(fn(df, args.alpha))
        except Exception as e:  # 任何假设失败都不中断整体
            name = fn.__name__.replace("test_", "").upper()
            bad = HypothesisResult(name, f"{name}（分析过程异常）", DATA_NOT_READY)
            bad.sections.append(f"分析过程出现异常（已按预注册标注为数据未齐，不中断报告）："
                                f"`{type(e).__name__}: {e}`")
            hs.append(bad)

    # Holm-Bonferroni（跨假设的全部预注册检验）
    all_p = {}
    for h in hs:
        all_p.update(h.pvalues)
    adj = holm_bonferroni(all_p)
    for h in hs:
        if h.hid == "H1":
            finalize_h1(h, adj, args.alpha)
        elif h.hid == "H2":
            finalize_h2(h, adj, args.alpha)

    rep = []
    rep.append("# 预注册假设检验报告（H1–H5）\n")
    rep.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rep.append(f"- 结果根目录：{', '.join(str(r) for r in roots)}")
    rep.append(f"- 显著性水平 α = {args.alpha}；多重比较校正：Holm-Bonferroni")
    rep.append(f"- 延迟主指标：p50 wall-clock（ms）；statsmodels {'可用' if HAS_STATSMODELS else '不可用（已降级）'}")
    rep.append("\n## 数据盘点\n")
    rep.append(inventory_table(df))
    if load_log:
        rep.append("")
        rep.extend(load_log)
    rep.append("\n### 方法说明（偏离预注册之处，先于结果声明）")
    rep.append(
        "- 各结果 JSON 的 `details` 字段为空，任务级(164题)配对不可得；"
        "本脚本以**轮次**为配对单元（每配置每轮一个聚合观测，n≤5），"
        "功效低于预注册设想（820 观测/配置）。预注册判定准则不变。")
    rep.append("- 平台映射按预注册：run1–3=V100，run4–5=A800；能耗轮视为 A800。"
               "H1/H2 配对检验前按预注册 §2.5 做分平台方向一致性检查，方向相反则不合并轮次"
               "（相应假设判定为『不确定』，池化结果仅作参考）；H4 前沿附分平台敏感性核对。")
    rep.append("- **指标口径警示**：adaptive 配置（C07–C12）JSON 中的 `pass_at_1` 实为 ≤10 次采样的 "
               "any-pass 率（≈pass@10），`p50_total_time_ms` 为逐样本累计墙钟时间；与 greedy 配置的"
               "单样本指标不同源、不可直接比较（详见 H2/H3 各节警示）。")
    rep.append("- **32B 缺失声明**：预注册为 6 尺寸（含 32B），但 run_pareto.py 尺寸表仅 5 尺寸，"
               "32B 从未实施，属数据收集偏离预注册（与『缺能耗数据』性质不同），相关假设分母自动缩减。")
    rep.append("- Holm-Bonferroni 作用于含显著性检验的假设族："
               "H1(主t/Wilcoxon、INT4交叉t)、H2(t/Wilcoxon)、H3(辅助t)。"
               "H3 阈值判定、H4/H5 结构性判定不产生 p 值；§2.5 平台冲突的检验不进入该族。")

    rep.append("\n## 结果一览\n")
    rep.append("| 假设 | 判定 |")
    rep.append("|---|---|")
    for h in hs:
        rep.append(f"| {h.hid} {h.title} | {h.verdict} |")

    for h in hs:
        rep.append(f"\n## {h.hid} {h.title}\n")
        rep.append(verdict_line(h))
        rep.append("")
        rep.extend([s for s in h.sections if s])

    if all_p:
        rep.append("\n## Holm-Bonferroni 校正明细\n")
        rep.append("| 检验 | 原始单侧 p | 校正后 p | 是否显著 |")
        rep.append("|---|---|---|---|")
        for k in sorted(all_p):
            p0, pa = all_p[k], adj[k]
            sig = "是" if (not np.isnan(pa) and pa < args.alpha) else "否"
            rep.append(f"| {k} | {fmt(p0, '.4f')} | {fmt(pa, '.4f')} | {sig} |")

    rep.append("\n## 混合效应模型（预注册分析计划第 2 条）\n")
    rep.append(mixed_models_section(df))

    rep.append("\n## 功效与负面结果声明")
    rep.append("- 轮次级配对样本量 ≤5，显著≠重要、不显著≠无效应；"
               "请结合 Cohen's d 与预注册第 4 条功效说明解读。")
    rep.append("- 依预注册第 6 条：被推翻的假设将如实成文。")

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print(f"报告已写入: {out_path}")
    for h in hs:
        print(f"  {h.hid}: {h.verdict}")


if __name__ == "__main__":
    main()
