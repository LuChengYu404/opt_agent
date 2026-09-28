# 状态/动作设计谱系调研（agentic BO 及近邻）

> 2026-09-28 整理。目的：回答"本项目的 OptimizationState（15 维过程标量）
> 与 8-action 封闭决策空间，相对文献处于什么位置；是否/如何需要调整"。
> 供 codex 评审与 related-work 写作使用。
> 核实标记：✅ = 本次经全文抓取或检索确认设计细节；⚠️ = 检索摘要确认，引用前需对 PDF。

## 一、谱系总览

各家分布在两条轴上：
- **状态轴**：原始轨迹（GPTOpt/LLAMBO）→ 后验上下文+候选集（Volpp）→
  过程统计量（TempoRL/DAC、本项目）→ 全量 (数据, 配置, 指令, 审议史)（Sara）。
- **动作轴**：直接选下一个 x（GPTOpt/LLAMBO/Volpp）→ 策略级 typed 动作
  （本项目）→ 开放机器控制命令（Sara reconfigure/probe）→ 程序空间（FunBO）。

**本项目的位置：状态最压缩 + 动作居中（策略层）+ 唯一逐决策反事实监督。**

## 二、逐家细节

### 1. Sara/lenz — Agentic BO（Brunzema et al., arXiv:2608.00316, 2026）✅
（全文抓取核对；本项目最近邻，ref 库主条目）

- **动作** 𝒜 = 𝒜_comp ∪ 𝒜_eval ∪ {stop}：
  - `probe`：只读查询后端（预测、acquisition 分数、诊断、Pareto、trial 历史）；
  - `reconfigure`：改配置 c_t（换 acquisition/surrogate、收紧搜索域、改目标/约束）；
  - `propose`：向后端要候选 x̃，agent 采纳/精修/**否决**，可先在不同配置下比较假想候选；
  - `evaluate`：提交 x，唯一推进预算 t 的动作；
  - `stop`。
  计算类动作不消耗评估预算（消耗 token，**只报告不优化**，附录 E 才憧憬 meta-MDP 训练）。
- **状态** s_t^(j) = (𝒟_t 观测对, c_t 配置, 𝒦_t 外生指令上下文, ℋ_t 审议历史含推理轨迹)。
- metalevel MDP 形式化；通用 LLM 直接当策略，无学习。
- **同构点**：计算/评估动作二分 ↔ 本项目 `n_new_samples ∈ {0,1,2,3}` 记账；
  四元组状态是呈现本项目状态设计的现成语言（本项目 = (𝒟,c) 的投影摘要，
  c 冻结、ℋ 退化为 last_action、无 𝒦）。
- **关键差异**：开放命令空间 → 无法做反事实监督与预设定 gate。

### 2. LLAMBO（Liu et al., ICLR 2024, arXiv:2402.03921）✅
- **状态**：prompt 即状态——任务描述 + 搜索空间 + 预算 + **全部 (配置, 值) 历史的表格渲染**
  + 任务元数据先验。
- **动作**：warm-start 初始配置；surrogate (均值, 不确定性) 预测；navigator 提议
  **下一个 x**；"eliciting temperature/propensity" 提示词控制探索-利用。
- 复现研究（arXiv:2511.18891, 2025）：其判别式 surrogate 作为单任务回归器弱于
  GP/SMAC，收益主要来自跨任务语义先验——**富状态≠高信号**的一个证据点。

### 3. GPTOpt（arXiv:2510.25404, 2025）✅
- **状态**：原始 (x_i, y_i) 轨迹直接作输入；**动作**：输出下一个 x。
- 微调小型 LLM（非 prompt），生成优化数据训练，BBOB/COCO 基准。
- "LLM 即优化器"路线：状态最原始、动作最底层、与本项目的 meta 决策层不同层。

### 4. Volpp et al. — Meta-Learning Acquisition Functions（ICLR 2021, arXiv:1904.02642）✅
- **状态**：输入 + 观测 + GP 后验上下文 + 候选点集，注意力编码，context-agnostic
  （跨任务迁移）；**动作**：从候选集选下一个 x；RL 以 regret 训练。
- 形式上最像本项目"从固定菜单挑"——但菜单是 x、监督是 RL 而非反事实 oracle。

### 5. TempoRL / DAC 线（Biedenkapp et al., ICML 2021, arXiv:2106.05262）✅
- **状态**：优化器 trace 统计（loss 曲线、特征统计）；**动作**：显式扩充 **skip 动作**
  ——学"何时出手"（behavior Q + skip Q）。
- 本项目"零成本动作（stop/retrain）+ 何时路由 System-2"的问题原型与文献锚点。

### 6. FunBO（DeepMind/Oxford, arXiv:2406.04824, 2024）✅
- 不在逐步决策层：FunSearch 进化 **acquisition 函数代码**，LLM 提议程序变异；
  动作空间 = 程序空间，离线策略搜索。支持带 context features 的任务。

### 7. 其他相关（一句话，未深挖）⚠️
- OptFormer（Chen et al., 2022）：transformer 优化器，轨迹输入、下一 x 分布输出。
- Maraval et al. transformer meta-BO（Volpp 引文线）：上下文编码 + 候选打分。
- AISTATS 2024 "Timing as an Action"：TempoRL 后续，何时观测/出手。
- BO-METS（Wang et al., NeurIPS 2023）：meta-learned transfer BO（DISCUSS Q-S2 已引）。

## 三、对比表（论文 related work 可直接改造）

| 工作 | 状态 | 动作 | 监督/策略 | 记忆 |
|---|---|---|---|---|
| Sara/lenz | (𝒟, c, 𝒦, ℋ) 全量 | probe/reconfigure/propose/evaluate/stop（开放） | 无学习 | 完整审议史 |
| LLAMBO | prompt：任务+全历史表+元数据 | 下一 x（+温度倾向） | zero-shot prompt | 全历史 |
| GPTOpt | 原始 (x,y) 轨迹 | 下一 x | 微调 | 全轨迹 |
| Volpp | GP 后验上下文+候选集 | 候选集中选 x | RL (regret) | 上下文编码 |
| TempoRL/DAC | trace 统计特征 | 干预 or skip | Q-learning | 特征统计 |
| FunBO | — | acquisition 代码（程序空间） | 演化搜索 | — |
| **本项目** | **15 维过程标量（5 frozen 块）** | **8 个 typed 策略动作** | **反事实 oracle 逐决策标签** | 仅 last_action |

## 四、对本项目的三点含义（待 codex 裁决）

1. **表达力 ↔ 可测性权衡**：封闭 8-action 空间是反事实监督 + Gate 2-5 +
   非劣效成本前沿的前提；Sara 的开放空间无法做这些。卖点而非缺陷。
2. **"信号薄"结论的表示限定**：P2 的 D 上限与状态表示无关（oracle 标签直算），
   天花板结论安全；但 **Gate 2（可学性）的阴性是"在 15 维过程状态上"的阴性**
   ——文献中富状态（全轨迹）设计理论上可提取更细模式（LLAMBO 复现研究
   同时表明富状态也带来噪声与成本）。Gate 2 措辞应加 granularity 限定。
3. **候选调整方向**（成本递增）：
   - (a) 冻结现状 + 措辞限定（零重跑成本）；
   - (b) state 加 optional sidecar（Q-T6 方案 b：`Optional[ProblemFeatures]`，
     向后兼容，数据集需重生成或双轨）；
   - (c) state 升级到轨迹级（向 Sara/LLAMBO 靠拢，监督/gate 全部重做）；
   - (d) action 空间扩容（加 reconfigure 类动作，oracle rollout 语义需重定义）；
   - (e) 任何改动推迟到期刊版，workshop 版按 (a)。

## 五、codex 裁决（2026-09-28，runs/codex_design_review.log）

**VERDICT: KEEP_FROZEN_QUALIFY**（状态/动作定义冻结，只修措辞）。

- 选项裁定：(a) 冻结+措辞限定 VIABLE_NOW；(b) sidecar VIABLE_LATER（留给
  期刊版 HPO/工程任务）；(c) 轨迹级状态 NOT_RECOMMENDED（本面板无已证收益、
  清空证据库、4/5 家族 D 上限依旧低）；(d) 扩动作（reconfigure 类）
  VIABLE_LATER（需重定义 oracle rollout 语义）；(e) workshop 按 (a) 出、
  期刊版再评估 VIABLE_NOW。
- 无损改动：**None**——没有任何状态/动作定义变更能在不推翻现有 artifact 的
  前提下带来足够收益。
- 六条 claim 措辞修正（已落实到 outline）：P2 的 D 需注明范围（固定 8-action
  菜单+采样状态+rollout utility，不含闭环 terminal value/扩容动作）；
  P3 是条件命题（"tested routers 未赚回成本"+P3 为充分条件解释，非实证证明）；
  schedule 是 sufficiency（标题从 "Reduces to" 改为 "Subsumes"）；封闭菜单=
  "exhaustive labels over our predefined menu"（不排除别处反事实比较）；
  Gate 2 保持粒度限定；Gate 4/首动作标签 "not established"（非 "unattainable"）。
