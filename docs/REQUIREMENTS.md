# Companion Website 需求文档

## 目标

为 `efficient-codegen-slr-replication` 仓库构建 GitHub Pages 前端，作为论文的 companion website，在 cover letter 中引用。

## 部署方式

- GitHub Pages，source 设为 `docs/` 目录
- 仓库: `https://github.com/1719930244/efficient-codegen-slr-replication`
- 最终 URL: `https://1719930244.github.io/efficient-codegen-slr-replication/`

## 技术栈

- 单页 HTML + 内嵌 CSS/JS（无构建步骤）
- Chart.js CDN 做可视化图表
- 自定义可搜索/可筛选表格
- 响应式设计

## 数据源

所有数据来自 `data/` 目录，内嵌为 JSON：

| 文件 | 内容 |
|------|------|
| `data/primary-studies.csv` | 122 篇 primary study 的元数据（ID, Key, Title, Year, Venue, RQ, Categories 等） |
| `data/classification-scheme.csv` | 24 个分类 → study keys 映射 |
| `data/statistics.json` | 年份/RQ/分类分布统计 |
| `data/reporting-compliance.json` | 每篇论文的 QA 评分 |
| `data/by-rq/rq{1-4}-studies.csv` | 按 RQ 分的子集 |

## 页面结构

### 1. Hero / 概览区

- 论文标题: "Towards Efficient LLM-Based Code Generation: A Systematic Review"
- 关键数字: 122 primary studies, 26,265 records screened, 7 databases, 5 RQs, 7 findings
- 链接: 论文 PDF（待定）、GitHub 仓库、引用 BibTeX

### 2. PRISMA 流程图

可视化展示筛选流程:
```
26,265 raw → 22,118 dedup → 3,810 T&A → 89 full-text
  + 18 snowball + 18 supplementary → 125 → 122 (去重)
```

### 3. 统计可视化

- 年份分布柱状图: 2020(1), 2022(1), 2023(14), 2024(35), 2025(52), 2026(19)
- RQ 分布饼图/柱状图: RQ1:10, RQ2:29, RQ3:87, RQ4:21（25 篇跨 RQ）
- 分类分布水平条形图: 24 个子类按数量排序
- Reporting compliance 雷达图或条形图: correctness 94%, latency 54%, memory 19%, energy 8% 等

### 4. 分类法 (Taxonomy) 交互展示

- 按生命周期阶段组织: Data Preparation → Training → Inference → Evaluation
- 可展开/折叠的树形结构
- 点击某个分类高亮对应的论文
- 推理层按 cost model 组织: Cost_per_token / N_tokens / K_calls

### 5. 论文列表（可搜索/可筛选表格）

列: ID, Title (链接到 arXiv/DOI), Year, Venue, RQ, Categories, QA Score
筛选器:
- 按 RQ 筛选（多选）
- 按年份范围筛选
- 按分类筛选
- 全文搜索
- 按 Venue Type 筛选（conference / journal / preprint）

### 6. Key Findings 区

展示 7 个 findings 的摘要卡片:
- Finding 1: Quality Dominates Quantity
- Finding 2: Minimal Parameter Updates Suffice
- Finding 3: What to Distill Matters More Than How
- Finding 4: Adaptive Resource Allocation Outperforms Uniform
- Finding 5: Code Structure Enables Unique Optimizations
- Finding 6: Deployment Compounds with Inference but Rarely Evaluated Together
- Finding 7: Efficiency Evaluation Lags Behind Optimization

### 7. Footer

- 论文引用信息
- 联系方式
- License: CC BY 4.0
- "Last updated" 时间戳

## 设计风格

- 学术风格，简洁专业
- 配色: 深蓝/白为主，辅以灰色
- 不用花哨动画，重信息密度
- 参考: paperswithcode.com, connected-papers.com 的学术网站风格

## Cover Letter 中的引用

cover letter 已写入 Overleaf (`/home/szw/school/efficient-codegen-survey/cover-letter.tex`)，其中 companion website 的 URL 需要在网站部署后更新为实际地址。当前 cover letter 中未写死 URL。

## 待确认

- [ ] 论文 PDF 链接（arXiv 预印本？）
- [ ] 是否需要 BibTeX 引用按钮
- [ ] 是否需要深色模式
- [ ] 论文标题的 arXiv/DOI 链接格式（Key 中包含 arXiv ID，可自动生成链接）
