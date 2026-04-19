# Efficient CodeGen Survey — Empirical Experiment Plan

> 目标：为 TOSEM SLR 增加实证实验，验证跨技术组合的交互效应，绘制效率-质量 Pareto 前沿。

## 1. 实验环境

| 项目 | 配置 |
|------|------|
| GPU | 4× Tesla V100-SXM2-32GB |
| 内存 | 226 GB |
| CPU | 20 核 |
| CUDA | 12.4 (via conda pytorch-cuda) |
| PyTorch | 2.5.1 |
| Python | 3.10 (micromamba env `exp`) |
| 项目目录 | `~/efficient-codegen-exp/` |
| 模型来源 | ModelScope (HuggingFace 不可达) |

连接方式：`ssh -p 6000 szw@47.102.98.150`（FRP 隧道）

## 2. 模型

所有模型均为 Qwen2.5-Coder-Instruct 系列：

| 模型 | 参数量 | FP16 显存 | 用途 |
|------|--------|----------|------|
| Qwen2.5-Coder-0.5B-Instruct | 0.5B | ~1 GB | 草稿模型 + Pareto |
| Qwen2.5-Coder-1.5B-Instruct | 1.5B | ~3 GB | Pareto |
| Qwen2.5-Coder-3B-Instruct | 3B | ~6 GB | Pareto |
| Qwen2.5-Coder-7B-Instruct | 7B | ~14 GB | 主模型 + Pareto |
| Qwen2.5-Coder-14B-Instruct | 14B | ~28 GB | Pareto |

模型下载源：`https://www.modelscope.cn/models/Qwen/Qwen2.5-Coder-{size}-Instruct`

## 3. 基准测试

| 基准 | 题目数 | 任务类型 |
|------|--------|---------|
| HumanEval | 164 | 函数级 Python 代码生成 |

## 4. 实验 1：Technique Composition（技术组合交互效应）

### 4.1 研究问题

**RQ-Exp**: 不同效率优化技术组合后，效果是加性的、超加性的、还是存在干扰？

### 4.2 技术因子

| 维度 | 水平 | 对应论文分类 | 实现工具 |
|------|------|-------------|---------|
| A. 模型精度 | FP16 / INT8 / INT4(GPTQ) | RQ4 Post-Training Quantization | bitsandbytes, auto-gptq |
| B. 解码策略 | 标准自回归 / 投机解码 | RQ3 Speculative Decoding | transformers `assistant_model` |
| C. 采样优化 | 贪心(n=1) / 自适应采样(n=10, 置信度早停) | RQ3 Adaptive Sampling | 自实现 |

### 4.3 配置矩阵（3×2×2 = 12 配置）

| 编号 | 精度 | 解码 | 采样 |
|------|------|------|------|
| C01 | FP16 | 标准 | 贪心 |
| C02 | INT8 | 标准 | 贪心 |
| C03 | INT4 | 标准 | 贪心 |
| C04 | FP16 | 投机 | 贪心 |
| C05 | INT8 | 投机 | 贪心 |
| C06 | INT4 | 投机 | 贪心 |
| C07 | FP16 | 标准 | 自适应 |
| C08 | INT8 | 标准 | 自适应 |
| C09 | INT4 | 标准 | 自适应 |
| C10 | FP16 | 投机 | 自适应 |
| C11 | INT8 | 投机 | 自适应 |
| C12 | INT4 | 投机 | 自适应 |

主模型：Qwen2.5-Coder-7B-Instruct
草稿模型（投机解码）：Qwen2.5-Coder-0.5B-Instruct

### 4.4 测量指标

| 类别 | 指标 | 单位 |
|------|------|------|
| 质量 | pass@1 | % |
| 质量 | pass@10（仅自适应采样） | % |
| 延迟 | ms/token（平均每 token 生成时间） | ms |
| 延迟 | 总生成时间（mean / p50 / p95） | ms |
| 吞吐 | tokens/sec | tok/s |
| 内存 | 峰值 GPU 显存 | MB |
| 成本 | 总生成 token 数 | count |

### 4.5 关键科学问题

1. INT4 量化是否降低投机解码的草稿接受率？
2. 量化 + 投机解码的加速是加性还是有干扰？
3. 自适应采样 + 量化是否保持 pass@1 不变？
4. 全组合（C12）相比基线（C01）的综合效率提升倍数是多少？

### 4.6 执行方式

- 每个配置运行 HumanEval 全部 164 题
- 贪心采样: temperature=0, n=1
- 自适应采样: temperature=0.8, n=10, 使用 pass@1/pass@10 估计
- max_new_tokens=512
- 每个配置运行 3 次取平均（稳定延迟测量）
- GPU 0-1 用于此实验

## 5. 实验 2：Pareto Frontier（效率-质量权衡前沿）

### 5.1 配置矩阵（5×3 = 15 配置）

| 模型 | FP16 | INT8 | INT4 |
|------|------|------|------|
| Qwen2.5-Coder-0.5B-Instruct | P01 | P02 | P03 |
| Qwen2.5-Coder-1.5B-Instruct | P04 | P05 | P06 |
| Qwen2.5-Coder-3B-Instruct | P07 | P08 | P09 |
| Qwen2.5-Coder-7B-Instruct | P10 | P11 | P12 |
| Qwen2.5-Coder-14B-Instruct | P13 | P14 | P15 |

### 5.2 测量指标

与实验 1 相同（贪心采样，n=1）

### 5.3 输出图表

1. **Pareto 图 A**: latency (ms) vs pass@1 (%)
2. **Pareto 图 B**: peak GPU memory (GB) vs pass@1 (%)
3. **Pareto 图 C**: throughput (tokens/s) vs pass@1 (%)
4. 每个点标注模型名+精度，连接 Pareto 前沿线

### 5.4 执行方式

- 贪心采样: temperature=0, n=1
- max_new_tokens=512
- 每个配置运行 3 次取平均
- GPU 2-3 用于此实验（可与实验 1 并行）

### 5.5 GPU 调度（04-17 更新）

原计划 pareto 3 轮串行占 GPU3（估计 37.5h），实际与 composition 三轮并行占满 GPU0-2 冲突。
现已改为增量切换：
- GPU3 Pareto Run1 继续跑（已完成 P01-P04，跑到 P05）
- Composition Run1 C12 完成 → watcher 自动启动 Pareto Run2 到 GPU0
- Composition Run2 C12 完成 → watcher 自动启动 Pareto Run3 到 GPU1
- Watcher 脚本: `~/efficient-codegen-exp/scripts/launch-pareto-parallel.sh`，日志 `launch-pareto-parallel.log`
- 标志文件 `.pareto_{2,3}_launched` 防重复启动
- 原 pareto 串行调度器 (bash -c '...run1 && ...run2 && ...run3') 已 kill，确保不会在 run1 结束后重新占用 GPU3
- 切换后 pareto 总耗时预估从 37.5h 降到 ~13h

## 6. 实验 3：Energy Round（能耗补测轮，04-17 新增）

### 6.1 背景

原 `eval_humaneval.py` 未包含 NVML 能耗采样代码。运行中的 composition/pareto 三轮实验都不会产出 energy 数据；已完成 C01-C08 × 3 轮、P01-P13 × run1 无法补回。

### 6.2 实现

`scripts/eval_humaneval.py` 打补丁（04-17 02:05）：
- 模块级 `read_energy_mj(device)` 使用 `pynvml.nvmlDeviceGetTotalEnergyConsumption`（V100 driver 570.195.03 实际支持）
- `GenerationResult.energy_mj` 新增字段（-1 为不支持）
- `BenchmarkResult` 新增 `total_energy_j / mean_energy_j_per_request / mean_energy_j_per_token`
- `generate_completion` 前后 `torch.cuda.synchronize()` 后读取 NVML 计数器，差值为本次生成的能耗（mJ）
- 测试：4 个 GPU live 功率读数 94-116W，符合 V100 推理工作负载

### 6.3 能耗补测轮配置（27 配置 × 1 pass）

- 脚本: `scripts/run_energy_round.py`
- 跑完 12 composition（C01-C12, 7B）+ 15 pareto（P01-P15, 0.5B-14B × FP16/INT8/INT4）各一次
- GPU: cuda:3（pareto run1 完成后由 watcher 自动占用）
- 预计时间: composition ~4h + pareto ~6h = ~10h
- 输出: `results/energy_round/{composition,pareto}/C*.json` / `P*.json`

### 6.4 Watcher 触发链

`scripts/launch-energy-round.sh`（PID 186356，日志 `launch-energy-round.log`）：
- 条件 1：`results/pareto/run1/P15.json` 出现 → GPU3 空闲 → 启动 composition 能耗轮
- 条件 2：composition 能耗轮 C12 + pareto run3 P15 完成 → 启动 pareto 能耗轮（cuda:3）
- 标志文件 `.energy_comp_launched` / `.energy_pareto_launched`

### 6.5 论文呈现

- **方法段**（sec-rq6.tex line 13）：保留 NVML `nvmlDeviceGetTotalEnergyConsumption` 描述，**补注 energy 只取单轮测量**（区别于 pass@1/lat/mem 的 3 轮均值±std）
- **Table 2**（composition 12 行）最后一列 `Energy (J/req)`
- **Table 4**（pareto 15 行）最后一列 `Energy (J/req)`
- **Figure Pareto (d)** Energy vs pass@1 子图
- **Isolated effects 段** 新增 FP16/INT8/INT4 能耗对比一段
- **Finding 8** 加一句"virtual downscaling 同时带来每 token 能耗下降"

## 6. 论文中的位置

- **实验结果** → Discussion 新增子节 "Empirical Composition Analysis"
- **Pareto 图** → Discussion "Unified cost models" 建议的实证支撑
- **新增 Finding** → 关于技术组合交互效应的发现

## 7. 文件结构

```
experiments/
├── EXPERIMENT-PLAN.md          # 本文件
├── scripts/
│   ├── download_models.py      # 模型下载脚本
│   ├── run_composition.py      # 实验 1 主脚本
│   ├── run_pareto.py           # 实验 2 主脚本
│   ├── eval_humaneval.py       # HumanEval 评测核心
│   └── plot_results.py         # 可视化脚本
├── results/
│   ├── composition/            # 实验 1 结果
│   └── pareto/                 # 实验 2 结果
└── figures/                    # 生成的图表
```
