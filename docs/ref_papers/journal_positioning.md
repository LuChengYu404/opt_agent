# 期刊论文定位：创新性主张与完整性缺口

> 2026-09-27 整理。目标：从 workshop paper 升级为期刊论文（目标档次参考
> JMLR / JAIR / ML journal / 或 BO 强相关的期刊）。本文回答两个问题：
> **相对 2024–2026 最近邻工作，我们的创新点是什么**；**期刊审稿人会要求
> 补什么**。

---

## 一、与最近邻工作的差异定位

### vs. Sara/lenz（Brunzema et al., arXiv:2608.00316, 2026）——最重要的对照

| 维度 | Sara/lenz (2026) | 本项目 |
|---|---|---|
| 决策形式 | LLM agent 自由探查/提点/改配置/停止（open-ended） | 8 个 typed action 的封闭决策空间 |
| 策略来源 | 系统 prompt 指令，无学习 | 反事实 oracle 监督 → 训练的校准 System-1（RF） |
| LLM 成本 | **只报告不优化**（token 记录在附录 C.2） | **选择性 System-2：置信度路由 + 预设非劣效 margin 的成本前沿** |
| 统计处理 | 10 seeds 中位数 + IQR + Mann–Whitney | 配对 seed bootstrap CI、预设定验收标准、诚实阴性结果 |
| 计算成本入模 | future work（附录 E meta-MDP） | **已做**（RQ3 全网格，含 10-seed 确认） |

**核心论点：我们正面回答了该文留下的开放问题（计算成本何时值得花）。**
他们证明了 LLM 能坐进决策环；我们回答"什么时候值得让它坐进来"。

### vs. LLAMBO（ICLR 2024）

- LLAMBO 把 LLM 放进环里（当 surrogate/提议器）但无成本治理；2026 复现研究
  （arXiv:2603.24647）指出其结果对实现细节高度敏感。
- 我们的对应设计：typed action 空间 + 冻结提示 + 确定性 RF + mock/real 双模式
  对照——稳健性是设计出来的，不是调出来的。

### vs. GPTOpt / AdaEvolve（2025–2026）

- 它们让 LLM **本身当优化器**；我们研究 **meta 决策层**上 LLM 推理何时值得花
  （决策对象是"下一步采取什么优化策略"，不是"下一个 x"）。层面不同，可互补。

### 我们独有的框架贡献

**Decision Landscape（adaptive-value 三 regime 分类）**：Flat / Heavy-tailed /
Frequent-opportunity。在上述所有近邻工作中都没有对应物——这是"在哪类问题上
值得用学习策略/LLM"的先验刻画，直接解释我们的阴性结果（Flat 家族上 LLM/学习
策略赚不回成本）。

---

## 二、一句话卖点（摘要/引言用）

> Agentic Bayesian Optimization 已经证明 LLM 能坐进优化决策环；本文回答
> **"它什么时候值得坐进来"**：我们用反事实 oracle 监督训练校准的 System-1，
> 以预设定非劣效 margin 在五个优化家族上给出首个优化决策环内的质量-成本
> 前沿——结论是诚实的阴性：冻结 System-1（0 次 LLM 调用）在 5/5 家族上
> 支配 always-LLM，且没有任何置信度/随机路由策略能赚回成本；机制分析表明
> LLM 的价值在于"全局→局部"的调度结构而非逐步选择。我们同时给出
> decision-landscape 刻画（三类 regime）与跨家族/跨维度迁移的系统性阴性结果。

**阴性结果为什么是贡献而非弱点**：agentic BO 文献（Sara/lenz、LLAMBO 及其
复现研究）普遍只报告 LLM 参与的正向故事；第一个系统性的成本前沿阴性 + 机制
解释 + 预设定标准，是该领域目前缺失的对照证据。写作时要主动把阴性结果
框定为"成本治理的 benchmark 时刻"。

---

## 三、期刊级完整性缺口（按优先级，审稿人视角）

### P0 — 基准面（最大风险）

- 现状：5 个合成 BOCoDe 家族 + 少量 IOH。Sara/lenz 用了 HPO（LCBench）+
  化学反应 + scaling law；审稿人会照此标准要求。
- 必补：
  1. **BBOB/COCO 全套或 IOH 更广覆盖**（IOH backend 已有，扩函数/维度即可）；
  2. **≥1 个 HPO 任务**（LCBench 或 Optuna benchmarking 类）；
  3. **≥1 个真实工程/仿真问题**——Phase 14 不能再是占位符；若车辆悬架仿真器
     不可得，用 BBOB-exp / 工程参数辨识类问题垫，并在 limitations 说明。

### P0 — 外部基线

- 现状：全部内部基线（Rule/Surrogate/Local/Jev/LLM）。
- 必补：**Sobol/Ax/SMAC3 或 HEBO**（经典优化器锚点）+ **LLAMBO**
  （OptunaHub 移植版，注意引用 2603.24647 的保真度差异表）。
- 可选：GPTOpt 作为"微调 LLM 优化器"参照。

### P1 — 缺一个"赢"的 regime

- 全阴性对 workshop 够、对期刊需至少一个正向结果。最有希望的路径：
  **把 LLM 的全局→局部调度蒸馏成可学习策略**——机制检查里确定性
  global-then-local 基线已在 dev seeds 上赢过 LLM（见
  EXP_FINAL_GOAL_CLOSED_LOOP_REPORT 机制检查段）；下一步在多峰问题族上展示
  learned schedule > 固定规则，即"从 LLM 轨迹蒸馏调度先验"的完整故事
  （这也是 Sara/lenz future work 未做的）。

### P1 — 统计完整化

- 多重比较校正（Benjamini–Hochberg 或按家族分层）；全实验统一 ≥10 seeds
  （RQ3 已到，RQ1/RQ2 对齐）；报告 IQM（Agarwal 2021 处方）。
- σ_family 等预设定量的定义先于实验写进脚本（本次 10-seed 复核发现原报告
  未记录 σ 定义——流程教训写进论文可加分）。

### P2 — 保持并显式声称的强项

- 385 unit tests、数据集 SHA-256 溯源、预设定验收标准、阴性结果全保留、
  符号错误主动修正——写进 reproducibility 段落，是对审稿人的差异化卖点。
- Phase 14 工程迁移：若 P0 补了一个工程问题，该段落从"占位符"转为真实结果。

---

## 四、写作映射（结构 → 文献支撑）

| 论文段落 | 支撑文献 |
|---|---|
| §1 引言 | agentic BO 叙事：LLAMBO → 复现批判 → Sara/lenz → 我们的成本前沿 |
| §2 Related work | papers_2024plus.md 四条线各一段 |
| §3 方法 | OPE 术语接口（arXiv:2501.05278）+ temperature scaling（Guo 2017）+ 非劣效 margin（CONSORT 扩展） |
| §5 RQ3 | 路由线 8–13 + 阴性结果框定 |
| §6/§7 Limitations | Ross & Bagnell 2011（单步 vs 闭环分叉）+ 基准面/多重比较 |
| 附录 | Agarwal 2021 统计处方、预设定标准全文 |

相关文件：`docs/workshop_paper_outline.md`（现有 8 页骨架，可直接扩展）；
`runs/EXP_RQ3_REAL_LLM_COST_FRONTIER_REPORT.md`（10-seed 修正版结论）；
`PROGRESS.md`（三 RQ 现状）。
