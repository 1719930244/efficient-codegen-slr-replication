# 3 月空档补检报告（1007）

## 问题
七个数据库于 2026-03-06 检索；检索更新的窗口是 2026-04-01 至 08-31（S2 一批 04-01..05-28，OpenAlex 一批 04-01..09-07，arXiv 一批 04-01..08-31，见复现包 data/monthly-updates/UPDATE-LOG.md）。2026-03-07 至 03-31 共 25 天没有任何一批覆盖。GPT 第三轮指出了这一点，核对后属实。

## 做法（与检索更新同口径）
- OpenAlex：复现包标准脚本 scripts/monthly_update_openalex.py --start 2026-03-07 --end 2026-03-31。Group A 命中 5,954，B∧C 布尔过滤后 561，标题严格提名 116（已对 122 篇语料与历次批次按标题去重）。
- arXiv：沿用 lit-arxiv/arxiv_search.py，窗口改为 202603070000..202603312359。原始 150，布尔 54，标题严格 12，新提名 12。
- 合并去重 118 条，与 141 篇语料重合 0 条。OpenAlex 摘要另行补取，5 条无摘要。
- 筛选：gpt-6-astra 与 qwen3-max 各筛一遍，判定一致 99/118；有任一筛选器未排除的 38 条由 Claude 逐条裁决（adjudication.json）。**作者需要复核。**

## 结果
- 严格合格：0 条。
- 待全文核查：1 条，即 G000「Divide and conquer: Optimizing code Chain-of-Thought in Small Language Model」（EAAI 2026，DOI 10.1016/j.engappai.2026.114360，2026-03-07 发表）。摘要不公开。OpenAlex 关键词包括 CoT 推理、代码生成、小语言模型、资源高效模型、边缘部署。作者 Guang Yang 等，与语料 S014（COTTON，TSE 2024）同一课题组。按先例很可能合格，IC4 要看全文才能确认，需要机构访问权限。
- 边界项：15 条，均为通用效率方法、代码生成只是评测任务之一，按审计与 4–8 月更新的同一口径不计。
- 排除：22 条。另有 80 条两个筛选器一致排除。

## 写进稿件（红字，占位待填）
- §3.2 检索段末加两句：补检覆盖 3 月 7 日至 31 日，得到 118 篇候选，符合全部标准的为 [AUTHOR INPUT: none, pending the full text of one record]。
- 信 AE#1 第二段：同样两句，占位为 \todo。
- 若 G000 全文核查合格：语料 141 → 142（若同时加入审计 3 篇则 145），全部计数重算，PRISMA 图要加补检框。若不合格：占位填 none，图不变。
