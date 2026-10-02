# opt_agent_v3 — 待评审问题清单

> **目的**：把当前项目里**还没拍板、需外部评审**的问题集中起来，方便后续单独交给专家与前沿模型评审。
>
> 写作原则：
> - 第 1 节（背景与已完成事项）只记事实，每一条都可在仓库内被 `git log` / 测试 / 运行产物印证。
> - 第 2 节（科学问题）与第 3 节（技术问题）都明确写出**为什么现在卡住 + 候选方向 + 评审要回答什么**，避免空泛问题。
> - 项目核心研究方向不变（见 `MASTER_PLAN.md` §2 与 `project-opt-agent-v3` 记忆）：Decision-Centric Adaptive Optimization Agent，3 个 RQ（Learnability / Generalization / Selective Reasoning）。

## 0. 2026-09-27 更新（评审前必读）

本清单起草于 Phase 9.0 前后（2026-09-23 左右）。此后的实验已实证回答了其中若干问题；评审时请以下列更新为准，原始问题保留在下方作为上下文：

- **Q-S1 / Q-S5 / Q-T1 / Q-T2（utility 与 Gate 4）— 部分回答**：Phase 8.6C 诊断出 horizon 是主导因素（h3 → h10 翻转约 50% 的 top action）；8.6D/8.6E/8.6F 完成长程标签、闭环学生模型与首动作/延续解耦诊断。20-seed 复现（EXP-RQ1.2）后 Gate 3 PASS（adjusted_lift 1.294，CI [0.848, 1.954]）、Gate 4 stop-gain 5/5、uniform-gain 4/5、behavior-gain 1/5；Gate 2 仍 FAIL。
- **Q-S3（RQ3 收益如何量化）— 已回答（阴性）**：真 deepseek-chat 成本前沿已跑完（450 条配对轨迹，5 家族 × 10 seeds × 9 策略，零 fallback）。冻结 RF System-1（0 LLM 调用）在 5/5 家族上持平或优于 always-LLM；无任何置信度阈值或随机率路由 arm 的 CI 显著优于 System-1-only。前沿是退化的。见 `runs/EXP_RQ3_REAL_LLM_COST_FRONTIER_REPORT.md`（注意顶部符号修正声明：初版 3-seed 报告的 Hartmann 结论是符号反转错误）。
- **Q-S4（可发表性）— 已定位**：论文大纲成形（`docs/workshop_paper_outline.md`，codex 5 轮评审 APPROVE），定位 workshop paper：框架 + RQ1 Gate 3/5 + RQ2 部分阳性 + RQ3 阴性成本前沿（含 RQ2.2 维度迁移阴性）。
- **Q-T4（Router 信号）— 置信度信号已证伪**：RF softmax 最大值几乎不超过 0.25，所有置信度阈值几乎把全部步骤路由给 System-2；随机率基线不优于 System-1-only。entropy / OOD 联动仍未测。
- **Q-T5（hard routing；真 LLM 是否入常规路径）— 是**：EXP-RQ3 的全部 LLM arm 均为真 deepseek-chat 后端闭环运行，零 fallback。
- **Q-S2 / Q-T6（工程迁移 suspension）— 仍未跑**：环境中无工程仿真器，Phase 14 仍是占位符。
- **Q-T3（Laya 评估）— 部分回答**：Phase 9.5 完成真实 Laya zero-shot 5×5 基准（action collapse：600/625 选 error_guided_sampling）。
- §1.3 gate 表与 §1.4 项目位置为起草时快照；当前口径见 `PROGRESS.md` 顶部。

---

## 1. 背景与已完成事项

### 1.1 项目定位

- **研究主线**：把工程优化过程建模为 `s_t → a_t → backend → s_{t+1}`，研究"面对当前优化状态，下一步采取什么优化策略"。
- **3 个研究问题**（`MASTER_PLAN.md` §2、`project-opt-agent-v3` 记忆）：
  - RQ1 Learnability：optimization strategy selection 能否从 trajectory 学到稳定可学的 `π(a|s)`？
  - RQ2 Generalization：decision policy 能否跨问题迁移（数值 → 未见数值 → 工程）？
  - RQ3 Selective Reasoning：calibrated System-1/System-2 routing 能否在降低 LLM cost 的同时保留优化性能？
- **复用 vs 自研边界**（`project-opt-agent-v3-research-core` 记忆）：
  - 自研：OptimizationState 表示、Counterfactual Rollout、Decision Supervision `P*(a|s)`、Decision Dataset、Calibration/OOD、Router、Cross-problem Transfer。
  - 复用：BOCoDe（benchmark）、BoTorch（GP/BO）、Laya（System-1 原型）、NanoJev（dataset schema/split）、Tau（System-2 harness）、Frontier-Eng（评测哲学）。

### 1.2 已完成代码与实证产物

代码规模：39 个 src 模块（~6,000 行），27 个 test 文件（~6,200 行），13 个 entry-point script。

| 阶段 | 模块/产物 | 状态 |
|---|---|---|
| Phase 0–1 | `UnifiedBackend`（BOCoDe + BoTorch，3 个 problem：Branin-2D / Hartmann-6D / Ackley-10D） | ✅ 19/19 测试 |
| Phase 2 | `OptimizationState`（5 块 frozen dataclass：sampling / surrogate / optimization / resource / history） | ✅ 9/9 测试 |
| Phase 3 | 8 个 typed actions + `execute_action` + param schema 校验 | ✅ 18/18 测试 |
| Phase 4 | `DecisionLoop`（reset/step/run + Trajectory + TerminationReason） | ✅ 63/63 测试 |
| Phase 5–6 | `Checkpoint`（GP+RNG+history 恢复）+ `CounterfactualRollout` + `oracle_test` | ✅ 105/105 测试 |
| Phase 7 | `compute_utility`（4-term）+ `DatasetGenerator` + JSONL + DecisionRecord schema | ✅ 117/117 测试 |
| Phase 8 | Empirical dataset（22 records）+ STOP-dominance bias 发现 | ✅ 125/125 测试 |
| Phase 8.5 | Utility bias 修复（boundary_sampling bug、scales、w_stop_opportunity） | ✅ 136/136 测试 |
| Phase 8.6A | Scale wiring fix（eval/fit=total_budget、opt=std(init)）；Gate 3 +0.74 PASS | ✅ 145/145 测试 |
| Phase 8.6B | `gate2_stability` + `oracle_test_from_record` + replay metadata；machinery ready | ✅ 151/151 测试 |
| Phase 9.0 | `LayaPolicy` + `ConfidenceRouter` + `FallbackSystem2` + `HybridPolicy` + calibration metrics | ✅ 250/250 测试 |

实证产物（部分）：
- `runs/phase8_6b_h3_10seed.jsonl` — 226 records，3 problems × 10 seeds × horizon 3
- `runs/laya_branin_seed0_budget10.json` — Laya 真实 zero-shot 推理（5 轮决策）
- `runs/phase9_1_smoke_deepseek.json` / `phase9_1_smoke_fallback.json` — Hybrid end-to-end smoke
- `runs/phase9_baseline_smoke.json` — MLP baseline 对照（label agreement，**非**优化性能）

### 1.3 当前 gate 状态（PROGRESS.md 口径）

| Gate | 状态 | 证据 |
|---|---|---|
| 1. Boundary / action tests | ✅ | Phase 3 + 8 测试稳定 |
| 2. Multi-seed ≥10 stability | ❌ FAIL | Branin mid top_action_share=0.8 但其他 (problem, phase) 组 <60% |
| 3. Early exploration mass > STOP | ✅ PASS (point estimate) | adjusted_lift = +1.67；尚无 seed-clustered CI |
| 4. Terminal optimization advantage | ❌ FAIL | Branin：top-label vs behavior mean gain = -0.026, bootstrap 95% [-0.049, -0.010] |
| 5. Affine rescaling | 未测 | machinery 已有，未跑 |

### 1.4 项目当前所处位置

- 架构闭环与代码骨架完整（state → action → backend → state，HybridPolicy wired）。
- RQ1 实证 gate 没通过：`gold_probs` 当前不能可靠反映哪个 action 在 terminal regret 上更好（Gate 4 Branin FAIL）。
- Phase 9 supervised training 因此被 gate（PROGRESS.md 明确："Phase 9 supervision training remains gated by the terminal optimization check"）。
- 已经能交出去：① zero-shot System-1 reference（Laya 真推理）② Hybrid Agent workflow ③ DecisionDataset pipeline（生成 + replay + oracle）。

---

## 2. 科学问题（待评审）

> 这部分问题是关于"研究方向本身是否成立、应当如何框定"。

### Q-S1. RQ1 的实证标准是否应当调整？

**问题**：当前用 `terminal normalized best_value` 作为 gate 4 的判定标准，要求 gold top action 在 paired full-budget continuation 中显著优于 STOP / uniform / behavior。但 Branin 上 mean gain = -0.026，top-label < behavior。这说明当前 utility 公式产生的 `gold_probs` 不能可靠区分 action，还是说明"top-label 在 terminal 上一定优于 behavior"本身就不是合理目标？

**为什么卡住**：
- 走"修 utility 让 gate 4 通过"是默认路径（Phase 8.7+），但需要一个先验保证修了 utility 之后 gate 4 真的会过。
- 走"放宽 gate 标准"风险：等于把"decision supervision 是否有效"这个 RQ1 的核心问题回避掉。
- 学术上需要回答：counterfactual rollout 得到的 `gold_probs` 究竟应当拟合什么目标？terminal regret？per-step regret？incremental Δbest？

**候选方向**：
- (a) 维持 terminal regret 目标，深入修 utility（terminal-relative Δopt + 多 horizon 验证）。
- (b) 切换到 incremental Δbest（更短 horizon、更可分），gate 4 改用 single-step Δbest。
- (c) 把 RQ1 重新定义为"decision policy 能否学到比 behavior 显著好的 terminal outcome"——把 behavior 当 lower bound 而不是 uniform。

**评审需要回答**：
- 在 Bayesian Optimization 文献里，"learned acquisition policy"的 ground-truth target 一般是怎么定义的？参考 Yue & Kontar (AISTATS 2020)、Lee et al. (UAI 2020)、Zhang & Chen (arXiv:2507.06529)。
- terminal regret 作为 BO 决策 target 是否合理？还是 BO literature 主要关心 per-step / cumulative improvement？
- 如果 ground-truth target 本身不是 terminal regret，那当前 Phase 7 utility + counterfactual rollout 的整套设计是否需要重构？

---

### Q-S2. RQ2 的"跨问题迁移"是否仍以 BOCoDe 数值问题为基准？

**问题**：`scripts/run_engineering_transfer.py` 已存在，但还没真正跑过工程问题。RQ2 写的是"train numerical → unseen numerical → unseen engineering"，但工程问题（例如 suspension）的 state schema 含义与数值 benchmark 完全不同（spring stiffness / damping 等物理量是否应进入 state 还是保持 optimization-process-only？）。

**为什么卡住**：
- 当前 `OptimizationState` 严格只描述 optimization process（coverage / uncertainty / stagnation / budget_ratio），不依赖具体物理量。这是有意的设计（§27 硬约束：state 描述 optimization process 而非物理变量），但也意味着工程迁移几乎零成本——同时也意味着"跨问题迁移"可能 trivial 平凡通过。
- 如果工程问题的 OptimizationState 完全和数值问题一致，那 RQ2 是否真的验证了"迁移"？还是只是"同一套 process descriptor 在不同 objective 上都能 work"？

**候选方向**：
- (a) 维持当前 process-only state；RQ2 改为"跨 objective 同一 process descriptor 的 transfer"，强调 *representation* 的不变性而非 *semantic* 的不变性。
- (b) 引入 problem-specific feature block（可选 sidecar），让 RQ2 验证"加 problem feature vs 不加"的差异。
- (c) 选一个真实工程 problem（suspension）跑一次 transfer learning，看 ablation 是否能区分两种 state 表示。

**评审需要回答**：
- RQ2 的"transfer"在研究叙事上需要多强？是 state-level transfer、policy-level transfer、还是 task-level transfer？
- 现有 literature（meta-learning for BO, e.g. Wang et al. NeurIPS 2023 "BO-METS")的 transfer benchmark 怎么设计？

---

### Q-S3. RQ3 的"selective reasoning"应当如何量化收益？

**问题**：当前 RQ3 框架（HybridPolicy + Router + FallbackSystem2）已有，但 smoke run 没法量化收益。"降低 LLM cost 同时保留性能"需要一个 baseline 对照，目前还没设计。

**为什么卡住**：
- 没有真 LLM baseline（DeepSeek 接口存在但只在 smoke 路径），无法计算"LLM-only"的总 token cost。
- 没有 ablation：high-tau routing（多用 System-1）vs low-tau routing（多用 System-2）的 Pareto 曲线还没画。
- 没有 reference policy 对照：纯 Laya zero-shot vs 纯 rule policy vs Hybrid 在 closed-loop 上的 terminal regret 差异还没测。

**候选方向**：
- (a) 先做 closed-loop ablation（zero-shot Laya / Rule / Hybrid τ∈{0.3,0.5,0.7}），画出 regret–cost Pareto。
- (b) RQ3 改 framing：把 Selective Reasoning 的目标从"regret–cost Pareto"收紧为"在固定 budget 下达到某 regret 阈值所需的 LLM call 次数"。
- (c) 把 RQ3 拆成两段：先证 calibration（System-1 confidence 是否可靠）→ 再证 selective routing（基于 calibration 的 routing 是否省钱）。

**评审需要回答**：
- Selective Reasoning 的 standard benchmark 在 literature 里长什么样？参考 Azure/GPT "cascades"、DeepMind "FrugalGPT" 等。
- 对于一个 BO decision policy，"省钱"的合理单位是 LLM call 次数、token 数、还是 wall time？
- calibration 在 routing 里是不是必要条件？还是可以被 raw probability 直接用？

---

### Q-S4. Phase 9 当前的"框架落地、不监督训练"是否构成可发表贡献？

**问题**：当前 Phase 9.0 deliverable = HybridPolicy skeleton + LayaPolicy adapter + smoke run。RQ1 gate 没通过意味着没有 learned System-1，没法做 supervised training 的对照。

**为什么卡住**：
- 投系统/工程类 venue：架构 + ablation + RQ3 Pareto 曲线就够了（即使 RQ1 没 pass），但 narrative 要改成"hybrid decision framework"。
- 投 ML/RL 类 venue：必须有 learned policy + transfer evidence，RQ1 必须 pass。
- 投 BO/optimization venue：state representation + counterfactual rollout 本身可能就是一个 contribution（不需要 learned policy）。

**候选方向**：
- (a) 把 Phase 9 定位成"system contribution"：架构 + zero-shot System-1 + RQ3 ablation，RQ1 留 future work。
- (b) 把 Phase 9 定位成"method contribution"：必须先解决 Phase 8.7+ utility，让 RQ1 通过，再做 supervised training。
- (c) 拆成两篇：① state representation + counterfactual rollout 作为 optimization-decision benchmark；② hybrid agent 作为 application。

**评审需要回答**：
- 现有 venue（AAAI / NeurIPS / ICLR / IEEE TEVC / Engineering Optimization）对"BO + decision policy"这个组合的接受门槛？
- 如果 utility 短期修不好，Phase 9 还能不能产出可发表的东西？
- 项目当前更接近"system paper"还是"method paper"？

---

### Q-S5. Utility misspecification 是真问题还是被夸大了？

**问题**：codex 多次标记"utility misspecification → reward hacking"为 deferred to Phase 8.7+。但当前 Gate 4 FAIL（top-label < behavior）是否一定意味着 utility mis-spec？也可能是 counterfactual rollout 的 continuation policy 太弱、或 horizon 太短、或 top-action 本身就是 STOP。

**为什么卡住**：
- Branin mid top_action_share=0.8（gold stable = retrain_surrogate），说明 label consistency 没问题；FAIL 出在"该 top action 在 terminal 上不如 behavior"。
- 这是 label–outcome gap，不是 label–label inconsistency。
- 可能原因：① utility 权重错了 ② continuation policy 太弱（horizon=3 太短）③ STOP 应该是 mid phase 的正确答案，行为 policy 在 mid phase 偶然比 retrain_surrogate 好。

**候选方向**：
- (a) 控制变量诊断：固定 utility 改 continuation policy（uniform / rule / oracle）看 Gate 4。
- (b) 控制变量诊断：固定 continuation policy 改 utility（去掉 w_stop_opportunity / 改 w_optimization）看 Gate 4。
- (c) 区分 "label consistency" vs "label validity" 两个独立指标，分别报告。

**评审需要回答**：
- utility misspecification 的检测有没有 literature-supported diagnostic？例如 inverse reward design、preference-based reward learning。
- "label–outcome gap" 在 imitation learning 里通常怎么诊断？
- 是否应该把 utility 替换为 learned value function（用 BO 真实 outcome 做监督）？

---

## 3. 技术问题（待评审）

> 这部分问题是关于"实现层具体怎么改"，但每一条都对应一个科学判断。

### Q-T1. Phase 8.7+ utility 改动的范围如何划定？

**问题**：当前 `compute_utility` 公式：
```
U(a|s) = w_optimization · Δopt / opt_scale
       - w_evaluation · n_new_samples / eval_scale
       - w_fit · n_model_fits / fit_scale
       - w_stop_opportunity · remaining_budget_ratio
```
默认权重 `(1.0, 1.0, 1.0, 0.75)`，scale defaults 已修复（Phase 8.6A）。Gate 4 FAIL 仍存在。

**为什么卡住**：
- 选项 A：在 4-term 框架内调权重（局部改动），但 codex 已警告 utility 4-term 本质上 mis-specified。
- 选项 B：换成 terminal-regret 直接做 utility（更接近 gate 4 测量的东西），但需要每个 (s, a) 都跑完整 rollout → 成本高。
- 选项 C：把 utility 替换为 learned value function（用 BO 真实 outcome 监督），但需要先收集 outcome data。

**候选方向**：
- (a) 先跑 ablations：固定 utility 改 continuation policy，看 Gate 4 是否仍 FAIL。
- (b) 增加 horizon sweep（1 / 3 / 5 / 10），看 Gate 4 FAIL 是否随 horizon 变化。
- (c) 直接用 terminal Δbest 替代 4-term utility（per (s, a) full-budget rollout），让 gold_probs 直接拟合 terminal outcome。

**评审需要回答**：
- utility 公式 vs terminal value，哪个更鲁棒？
- learned value function 在样本量 ~220 records 下能不能稳定学到？
- horizon 对 decision supervision 的影响，文献支持是什么？

---

### Q-T2. counterfactual rollout 的 continuation policy 应当如何选择？

**问题**：`CounterfactualRollout` 在跑 (s, a) 时，后续 horizon 步用什么 policy？现在用同 gold_probs 的 policy（行为 policy），但这意味着 (s, a) 的 gold utility 依赖未来 policy 选择。

**为什么卡住**：
- 选项 A：on-policy（continuation = 当前生成 dataset 的 policy），简单但 label 与 policy 耦合。
- 选项 B：uniform random continuation，独立于 policy，但和真实使用场景不符。
- 选项 C：oracle continuation（用最好的 action，但 oracle 不知道是哪个），需要假设 perfect knowledge。
- 选项 D：mixed policy（固定比例 random + best-of-K），稳定但引入额外超参。

**候选方向**：
- (a) ablation：在同一 dataset 上分别用 on-policy / uniform / rule / best-of-3 continuation，看 Gate 4 差异。
- (b) 引入 explicit continuation policy 作为 dataset 元数据，方便后续 rerun。
- (c) 把 "continuation policy" 作为 decision supervision 的 orthogonal axis 单独报告。

**评审需要回答**：
- BO rollout 文献里 continuation policy 一般怎么选？
- on-policy label 在 RLHF / RL 里叫什么、有没有已知的偏差？
- 是否应该用 multi-policy rollout 取平均，得到 policy-robust label？

---

### Q-T3. LayaPolicy 当前 zero-shot 决策质量如何评估？

**问题**：`runs/laya_branin_seed0_budget10.json` 显示 Laya 在 5 轮决策里全部选 `error_guided_sampling`，best_value 一直停在 `-3.545`。但这只是 1 个 seed × 1 个 problem 的 smoke。

**为什么卡住**：
- 没 systematic benchmark：Laya 在 3 problems × 10 seeds × 不同 budget 上的 closed-loop 结果都没跑。
- 没 comparison baseline：纯 Laya zero-shot vs RulePolicy vs uniform random 在 closed-loop 上谁好？
- Laya 没经过 BO 任务 fine-tune，理论上 zero-shot 不应该懂 LogEI 之类的 domain-specific 概念。

**候选方向**：
- (a) 跑 Laya closed-loop benchmark（3 problems × 10 seeds × budgets {10, 20, 40}），报告 best_value vs budget 曲线。
- (b) 加 prompt-level ablation：把 BOCoDe problem metadata / action descriptions 注入 Laya system prompt，看决策是否变化。
- (c) 把 Laya 当作 Phase 9 RQ3 的 "System-1 reference"，重点测 calibration，不测 closed-loop 性能。

**评审需要回答**：
- Laya（mmBERT-style typed decision model）在 BO 任务上的 zero-shot 性能，文献 expectation 是什么？
- 是否应该对 Laya 做 BO-specific fine-tuning（小样本 LoRA），还是直接用 zero-shot？
- closed-loop benchmark 的标准 metric（simple regret / cumulative regret / best-so-far curve）哪个最合适？

---

### Q-T4. Router 当前只用了 confidence，还应当加什么信号？

**问题**：`ConfidenceRouter` 只用 `max(probabilities)` vs `tau_conf=0.5`。PROGRESS.md / PHASE9_DESIGN.md 都明确 `predict_score` 和 `predict_noul` 当前模型没 calibration target，因此不用。

**为什么卡住**：
- 只用 confidence 可能过于简单：相同 confidence 但 entropy 不同 → 决策含义不同。
- OOD signal（router/ood_aware.py 已实现）没和 confidence router 联动。
- 时间维度信号（前几轮与后几轮的 confidence 变化趋势）也未引入。

**候选方向**：
- (a) Router 加 entropy term：`route_score = confidence - λ·entropy`。
- (b) Router 加 OOD term：OOD score > τ 直接 escalate System-2。
- (c) Router 加 temporal term：连续 K 轮低 confidence 触发 System-2（避免 single-round 抖动）。

**评审需要回答**：
- selective reasoning literature 里 router 的 input 一般是什么？参考"model cascades", "FrugalGPT", "AutoMix"。
- 在 BO decision 这种 low-stakes per-step 决策里，单 round confidence 够吗？还是需要 temporal smoothing？
- OOD detection 在决策路由里是否必要？什么场景下 OOD 比 confidence 更早触发？

---

### Q-T5. HybridPolicy 当前的"high-confidence → System-1、low → System-2"是否应当换成 cascade？

**问题**：当前 HybridPolicy 是 hard routing（一步决策）。System-2 fallback 当前是 `FallbackSystem2`（委托给 RulePolicy），不是真 LLM。

**为什么卡住**：
- Hard routing 在 BO 这种长 horizon 场景可能不合理：某一轮 System-1 错了一步，影响后续 20 步。
- Cascade 模式（System-1 → 不确定 → System-2 → 不确定 → 第三层）可能更鲁棒但引入额外复杂度。
- 真 System-2（DeepSeek）当前只在 smoke 里，没进入常规 decision path。

**候选方向**：
- (a) 维持 hard routing，但把 System-2 换成真 DeepSeek，引入 token cost tracking。
- (b) 改成 cascade（System-1 → System-2 → third fallback），需要设计 third fallback。
- (c) 改成 soft routing（System-1 和 System-2 输出 weighted combination），需要把 System-2 输出也改成 typed decision。

**评审需要回答**：
- BO decision 这种 sequential 场景下，hard routing vs cascade vs soft routing 哪个更合适？
- System-2 输出也应是 typed decision（与 System-1 同 schema）还是 free-form（LLM 自由生成 action）？
- 长 horizon 决策里，cascade 的 fallback 链应当多深？

---

### Q-T6. Suspension（工程 transfer）的 state schema 是否要加 problem-specific block？

**问题**：`scripts/run_engineering_transfer.py` 存在但没跑通。当前 state 是 process-only，工程 transfer 时是否要加物理量？

**为什么卡住**：
- 选 A：维持 process-only state，工程 transfer 复用同一 schema。优点：迁移零成本；缺点：可能 trivial 通过 RQ2。
- 选 B：加 problem-specific feature block（sidecar），需要改 `OptimizationState` schema（破坏 Phase 2 已定稿的 5 块结构）。
- 选 C：把 problem-specific feature 作为 input embedding（不进 state frozen dataclass），只影响 Laya input。

**候选方向**：
- (a) 维持现状，先用 process-only state 跑 suspension transfer，看是否能 pass RQ2 trivial version。
- (b) 加 optional sidecar：`OptimizationState` 加 `Optional[ProblemFeatures]` 字段，向后兼容。
- (c) 推迟 suspension transfer，先完成 RQ1 / RQ3，再决定是否需要 sidecar。

**评审需要回答**：
- suspension 这种工程问题的 state representation 在 literature 里有没有标准？
- meta-BO 文献里 "task embedding" 是怎么处理的？参考 Wang et al. NeurIPS 2023。
- RQ2 的"迁移"必须 non-trivial 才有贡献吗？还是 trivial transfer 也算贡献？

---

### Q-T7. 当前 250 tests 是否足以支撑 Phase 9 框架的可靠性？

**问题**：测试覆盖率在 Phase 8.6B 后 151/151，Phase 9.0 后 250/250。但 Phase 9.0 大部分是 HybridPolicy / Router / Fallback 的 unit test，没有 closed-loop integration test 在 3 problems × 10 seeds 的规模下运行。

**为什么卡住**：
- Unit test 验证了 contract，但没验证"在真实 BOCoDe + Laya + HybridPolicy 端到端跑下不退化"。
- closed-loop integration test 需要 ~30 分钟跑一次（3 problems × 10 seeds × Laya inference），不能放 pytest 里。
- 现有 `scripts/phase9_baselines.py` 跑了 closed-loop 但只 holdout 2 seeds。

**候选方向**：
- (a) 加 nightly / weekly integration test（不是 pytest 默认跑，而是独立 job）。
- (b) 加 smoke-level integration test（1 seed × 1 problem，跑通即可），放 pytest。
- (c) 维持现状，所有 closed-loop 验证通过 scripts 驱动，不放 pytest。

**评审需要回答**：
- "framework skeleton" 这类工作，测试覆盖率的 industry standard 是什么？
- closed-loop integration test 应当多久跑一次？CI 触发还是 manual？
- 是否应当加 mutation testing / property-based testing？

---

### Q-T8. 当前 dataset pipeline（generator + replay + oracle）的可复现性是否足够？

**问题**：`DatasetGenerator` 已经 deterministic（`derive_run_seed` 用 SHA-256，不受 PYTHONHASHSEED 影响）。但 Phase 8.6B 的 `oracle_test_from_record` 依赖 backend version（BoTorch 版本绑定）。

**为什么卡住**：
- `backend_X / backend_objectives / backend_constraints` 写进 metadata，但 GP state 没写进（replay 时从空 history 重新 fit）。
- 这意味着 oracle_test 在 replay 时 GP 重建，**不会**和原 dataset 生成时的 GP 完全一致 → oracle test 有 GP-induced noise。
- 如果将来 BoTorch 升级（0.18 → 0.19），历史 dataset 可能跑出不同结果。

**候选方向**：
- (a) 把 GP state 也写进 metadata（`backend_gp_state`），让 oracle replay 完全 deterministic。
- (b) 把 BoTorch 版本写进 metadata，replay 时 assert version match；不一致则 raise。
- (c) 接受 GP-induced noise，在 oracle test 里加 bootstrap CI，明确报告 noise。

**评审需要回答**：
- dataset reproducibility 在 ML literature 里的 standard 是什么？是否需要 GP state？
- BoTorch 版本绑定对 research artifact 的长期可用性影响多大？
- 是否应当用 frozen docker / uv lock 把所有依赖固定？

---

## 4. 评审产物期望

后续单独评审时，建议每个问题提供：
1. **判断**：明确说当前方案是否合理 / 是否需要改。
2. **理由**：1–3 句理由（避免空泛）。
3. **替代方案**（如适用）：列出 1–2 个候选方向。
4. **优先级**：P0（影响 RQ1/RQ2/RQ3 能否实证）/ P1（影响工程实现质量）/ P2（nice-to-have）。

---

## 5. 相关文档清单

- `MASTER_PLAN.md` — 项目总纲
- `PHASE9_DESIGN.md` — Phase 9 框架设计（含 System-1/System-2/Router/HybridPolicy 接口）
- `PROGRESS.md` — 当前实证状态
- `runs/phase8_6b_h3_report.json` — Gate 2/3/4 实测结果
- `runs/phase9_baseline_smoke.json` — Phase 9 MLP baseline 对照
- 记忆文件（按主题索引）：
  - `project-opt-agent-v3.md` — 项目总览 + 3 个 RQ
  - `project-opt-agent-v3-phases.md` — Phase 0–14 + 严格 gating
  - `project-opt-agent-v3-research-core.md` — Research Core vs Engineering Infrastructure
  - `project-opt-agent-v3-phase8-6b.md` — Phase 8.6B machinery 详情
  - `project-opt-agent-v3-phase8-5.md` — Utility bias 修复历史
  - `project-opt-agent-v3-phase8-6a.md` — Scale wiring fix
