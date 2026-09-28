# 调研与探索轮次记录（2026-09-28，一天六轮）

> 用途：把本日的调研探索沉淀为一处索引，作为后续优化的方向依据。
> 每轮记录：问题 → 结论 → 落盘位置 → 对后续工作的含义。

## 轮次索引

| # | 问题 | 结论 | 落盘 |
|---|---|---|---|
| 1 | 全库有没有前后不一致/矛盾 | 20 项发现（15 确认/5 部分），全部修复 | `runs/AUDIT_INCONSISTENCY_FINDINGS.md`、`runs/codex_audit_review.log`、PROGRESS_BRANCH §11 |
| 2 | 实验结果能否支撑论文 | workshop 定位可支撑；"learned policy 有效"不可主张 | 会话分析（outline 修订落地） |
| 3 | 补充实验 | 独立流复现 G3 1.296、LOO +0.188、Bonferroni 分级 | PROGRESS_BRANCH §13、outline §5.2/§4.7/limitation 5/12 |
| 4 | 状态/动作设计要不要改 | 文献谱系定位 + codex 裁决 KEEP_FROZEN_QUALIFY | `state_action_design_survey.md`、`runs/codex_design_review.log`、§14 |
| 5 | 富状态到底有没有用 | 实现后 A/B：E1 探针 5/5 阴性、E2 LLM 对齐无差 2.1× token | 分支 `exp/lenz-state-jev-presentation`、`runs/EXP_LENZ_JEV_REPORT.md`、§15 |
| 6 | 大小模型交互 + 信息充分性的前沿；经典 BO 的既有答案 | 四线谱系 + 经典三件套（后验/acquisition/GP-Hedge/熵价值）+ **最近邻 LMABO (ICLR 2026)** | `model_interaction_survey.md`（含线 0 补充） |

## 关键认知（六轮收敛出的定位）

1. **经典 BO 早已回答"状态/信息/规则"**：后验=充分统计量、acquisition=
   解析决策规则、GP-Hedge=在线组合路由、ES/PES/MES=信息价值的熵形式。
   我们的 P2 是同一价值的信息论量纲换算（效用单位 vs nats），P3 是组合线
   假设掉的退化情形。理论已对接（`docs/THEORY.md` "Relation to classical
   theory"）。
2. **四条 LLM 交互线全部假设门控有收益可赚**——阴性前沿 + 定理解释是
   稀缺对照证据，这是论文的定位句。
3. **信息充分性的归因反转**：不是 agent 缺信息，是面板 D 上限低
   （E1/E2 实证 + P2 理论）。
4. **压缩状态无损失**（本面板、手工特征粒度）：分支 A/B 闭环。

## 后续优化方向（按优先级）

### P0 —— 投稿前必须
1. **GP-Hedge 基线**：8 动作菜单 ≈ 离散化 portfolio；BO 审稿人必问
   "为什么不用在线自适应组合"。实现 GP-Hedge-over-8-actions（在线
   Hedge 权重按 terminal gain 更新）或 related work 明确论证差异。
2. **引用并划界 LMABO（ICLR 2026）**：最近邻。划界点：acquisition 层
   vs 策略层；零样本 vs 反事实监督；无成本治理 vs 成本前沿（阴性）。
3. **codex 复审 outline**：三次大改（RLCD 整合、复现实验、措辞修正/
   经典对接）之后的版本未再过评审。

### P1 —— 投稿前可选 / 期刊版必做
4. 信息论视角加固 E2/P2 的讨论（THEORY.md 已对接一半；正文写作时
   展开 D 与熵降的关系）。
5. **locked family eval**：需扩 backend allowlist（冻结决策，单独评审），
   解除 §4.7 analysis-rerun 的家族复用半边。
6. **外部 baseline**（SMAC3 / HEBO / Sobol 锚点；期刊版 P0）。

### P2 —— 期刊版（换面板后）
7. 新面板（≥1 HPO 任务 + ≥1 工程/仿真问题）上**第一道诊断 = 重跑分支
   两脚本**（`exp_lenz_gate2_probe.py` / `exp_lenz_vs_frozen_llm.py`，
   数据集无关）；若翻阳 → 启用 sidecar 特征（Q-T6 方案 b）/ reconfigure
   类动作（裁决 VIABLE_LATER 项）。
8. Sara 式 probe 的成本治理扩展（计算动作 token 成本入前沿度量）。
9. 长程/随机目标面板上重开"learned policy 闭环价值"问题（A(s)>0 区域
   存在的面板才是学习策略的主场）。

## 引用纪律（沿 ref 库惯例）

✅ 已核实：Sara/lenz 全文、SwiftSage (NeurIPS 23)、GP-Hedge (UAI 11)、
PES/MES/ES、RouteLLM、FrugalGPT、Conformal Arbitrage、Robots-that-Ask
(CoRL 23)、经典 AIA 三篇、LMABO（OpenReview+ICLR 26 proceedings）。
⚠️ 待核实（多为 2026 preprint）：HiL-Bench、Agentic Abstention、C3PO、
System-1.x、adaptive retrieval 各篇、ESP/Vasconcelos 2022 细节。
