# 2024+ 注释版文献清单（按方法论支线组织）

> 2026-09-27 整理。每条包含：出处链接、一句话内容、**支撑本项目哪个部分**、核实状态。
> 经典锚点文献（<2024）只保留无法被新文献替代的，集中在最后一节。

---

## 一、Agentic BO 主线（决定论文定位，related work 的骨架）

### 1. Agentic Bayesian Optimization through Surrogate-Augmented Autoresearch ✅
- Brunzema, Tiao, Le, De Angeli, Xuan, Gligorijevic（Toyota Research Institute / Meta），2026
- https://arxiv.org/html/2608.00316v1
- **内容**：提出 agentic BO 范式——LLM agent 是 BO 环的中央决策者（自由探查/提点/
  改配置/停止），贝叶斯后端提供统计机器。两个组件：Sara（推理 agent）+ lenz
  （模块化 BoTorch/GPyTorch 后端，结构化 CLI，trial log 是唯一事实源）。
- **基准与评估**：合成函数（Branin、Ackley 10/20D、Hartmann 6D、约束 Hartmann、
  双目标 GP 样本路径）、LCBench HPO、化学反应产率（Suzuki–Miyaura 等）、
  neural scaling law；防记忆化的不透明接口；基线 Sobol/Ax/LLAMBO/Centaur；
  simple regret 曲线（10 seeds 中位数 + IQR 带）、hypervolume、Mann–Whitney 检验。
- **对我们的关键事实**：成本只报告不优化（token 用量在附录 C.2；带计算成本的
  meta-MDP 列为附录 E 的 future work）；无学习策略、无反事实监督。
  **这正是本项目 RQ3（校准路由 + 非劣效成本前沿）填的空白。**
- **与项目的既有联系**：Phase 9.6 已引用此文区分 Sara/lenz 架构与 Jev 适配器。

### 2. LLAMBO: Large Language Models to Enhance Bayesian Optimization ✅
- Liu et al.（van der Schaar 组），ICLR 2024，https://arxiv.org/abs/2402.03921
- **内容**：LLM 进 BO 环的开山工作——LLM 同时担任 surrogate、warm-start 提议器、
  候选导航器。
- **支撑**：related work 必引；是我们"LLM 当决策者"叙事的起点；2026 锚点论文
  的对比基线（其结论：无 surrogate 时 LLAMBO 有时不如随机采样）。

### 3. Can LLMs Beat Classical Hyperparameter Optimization Algorithms? ⚠️
- 2026，https://arxiv.org/abs/2603.24647
- **内容**：autoresearch 测试床上对比经典 HPO 与 LLM 方法；含 LLAMBO 复现保真度
  批判（OptunaHub 移植与原实现差异显著影响性能，附录 A.9 有逐项对比表）。
- **支撑**：我们"typed action 空间 + 冻结提示 + 确定性 RF = 设计层面的稳健性"
  的论据；也支撑我们对 LLM 非确定性的实验处理（冻结 seed、mock 对照等）。

### 4. GPTOpt: Towards Efficient LLM-Based Black-Box Optimization ✅
- 2025，https://arxiv.org/html/2510.25404v1（OpenReview 有对应条目）
- **内容**：微调 LLM 直接做连续黑盒优化，BBOB 风格数值基准。
- **支撑**：LLM-as-optimizer 参照系；说明"让 LLM 直接优化"与我们的"LLM 做
  meta 决策、被成本治理"是两条不同路线。

### 5. AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization ⚠️
- 2026，https://arxiv.org/pdf/2602.20133
- **内容**：LLM 引导的进化式零阶优化，统一搜索强度/资源分配/策略生成。
- **支撑**：related work；自适应策略生成方向的近作。

### 6. MALBO: Optimizing LLM-Based Multi-Agent Teams via Bayesian Optimization ⚠️
- Sabbatella, 2025（arXiv，编号待核实）
- **内容**：反方向——用 BO 优化 LLM 多 agent 团队配置。
- **支撑**：related work 的"双向融合"完整性，可选引用。

### 7. 综述与地图 ✅
- "LLMs for Bayesian Optimization in Scientific Domains" 综述（2025，ACL Anthology
  有条目；搜索摘要提及，编号待核实）
- LLM4OPT 收录库：https://github.com/ishmael233/LLM4OPT
  （区分"高层"用途——算法选择与编排——与"低层"组件设计；写作时用于定位术语）

---

## 二、路由与成本线（RQ3 选择性推理的方法论背景）

> 共同范式："小/便宜模型先行，低置信度升级到强模型"。注意：这些工作全部在
> QA/推理任务上；**没有一个在昂贵黑盒优化的决策环里做校准路由 + 非劣效成本前沿**。
> 这是我们 RQ3 的差异点，写 related work 时要显式说出来。

### 8. Adaptive LLM Routing with Test-Time Optimal Compute ✅
- ICML 2025 poster：https://icml.cc/virtual/2025/poster/43788
- 按查询动态分配到不同成本/质量的模型；质量-成本权衡的正式化。

### 9. Certainty-Based Adaptive Routing for Efficient LLM/MLLM Reasoning ✅
- arXiv:2505.15154（"Prolonged Reasoning Is Not All You Need"），约 41 引用
- 基于确定性的路由避免浪费性长推理——与我们 ConfidenceRouter 的置信门同构。

### 10. Route-To-Reason (RTR) ⚠️
- arXiv:2505.19435 — 同时选择模型与推理策略。

### 11. Select-then-Route (StR) ⚠️
- EMNLP 2025 Industry：https://aclanthology.org/2025.emnlp-industry.28
- 两阶段：先选小模型池，再池内路由。

### 12. A Unified Approach to Routing and Cascading for LLMs ✅
- ICML：https://icml.cc （编号待核实）
- 路由与级联的统一理论；大模型池 + 准确成本/质量估计下级联路由最优。

### 13. Dynamic Model Routing and Cascading for Efficient LLM Inference ✅
- 2026 综述：https://arxiv.org/html/2603.04445v2
- 系统分析 SOTA 路由/级联方法；related work 收尾引用。

### 14. 历史锚点（<2024，保留）
- RouteLLM（2024，~95% 质量 + 85% 查询给便宜模型）；FrugalGPT（2023，级联降本）。
- 引用时标注为"该方向起源"，正文重心放 8–13。

---

## 三、反事实监督 / OPE 线（RQ1 方法论的术语接口）

> 目的：把我们的 "counterfactual rollout supervision" 显式接入 off-policy
> evaluation (OPE) 术语体系，让审稿人能对号入座。

### 15. Off-Policy Evaluation and Counterfactual Methods 综述 ✅
- 2025：https://arxiv.org/html/2501.05278v1
- logged data 上学习/改进策略的反事实估计器综述——我们用保存的 rollout 分支
  重打效用分即是此范式。

### 16. Safe Bayesian Optimization with Counterfactual Policies ⚠️
- arXiv:2607.05620 — 估计基线策略的反事实结果（带不确定性）做安全优化。

### 17. Counterfactual Credit Guided Bayesian Optimization (CCGBO) ⚠️
- OpenReview：https://openreview.net/forum?id=Mn6w2M5Nvd
- 显式量化设计变量的反事实贡献；BO 内的反事实推理近作。

---

## 四、经典统计锚点（<2024，2024+ 无替代范式，保留引用）

| 文献 | 支撑点 |
|---|---|
| Agarwal et al., "Statistical Precipice", NeurIPS 2021 | seed 聚类 bootstrap CI、IQM、W/T/L——我们全部统计报告的处方 |
| Demšar, JMLR 2006 | 跨问题比较原则（勿逐问题 t 检验拼结论） |
| Ross, Gordon & Bagnell, AISTATS 2011 | 误差复合理论——解释我们单步遗憾赢 3/5 与闭环不赢固定规则的分叉 |
| Guo et al., ICML 2017 | temperature scaling——`TemperatureCalibrator` 的原始出处 |
| Dudík et al., ICML 2011 / Precup et al., ICML 2000 | OPE 奠基（第 15 条综述会引用它们，正文引综述即可） |
| Eggensperger et al., NeurIPS 2013 | 合成面板 + 固定预算的 BO 基准惯例 |
| Piaggio et al., BMJ 2012（CONSORT 非劣效扩展） | RQ3 预设标准（≤0.10σ margin）= 非劣效 margin 设计 |
| Nosek et al., PNAS 2018 | 预设定验收标准（preregistration 精神） |

---

## 核对清单（引用前逐项完成）

- [ ] 下载每篇 PDF 核对标题/作者/年份/venue（尤其 ⚠️ 条目与 arXiv 编号）
- [ ] 补全 MALBO、CCGBO、ICML 路由论文的准确 arXiv/DOI
- [ ] 确认 Sara/lenz 是否有公开代码/后端可作外部基线（lenz 的 CLI 描述见 §一.1）
- [ ] 检查 BBOB/COCO 2025 workshop 论文集中是否有 LLM 参赛方法（GPTOpt 之外的）
