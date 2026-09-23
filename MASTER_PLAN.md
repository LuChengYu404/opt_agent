# Decision-Centric Adaptive Optimization Agent
## Research Background & Technical Implementation Plan

> 项目总纲。任何开发 Agent 在动手前必须先读本文档。
>
> 配套规则：见第 27 节"当前不要做的事情"与第 28 节"开发 Agent 的决策原则"。
> 当前唯一允许的 Milestone：见第 32 节。

---

## 1. 项目目标

本项目面向复杂工程系统的设计优化，研究一种 **Decision-Centric Adaptive Optimization Agent**。

复杂工程优化通常包含：

- Design Space Sampling
- Expensive Simulation / Objective Evaluation
- Surrogate Modeling
- Bayesian / Evolutionary / Multi-objective Optimization
- Result Analysis
- Strategy Adjustment

现有 Bayesian Optimization（BO）、Surrogate-Assisted Optimization 等方法主要解决：

> 在有限计算预算下，应该评估哪些设计点，从而尽可能高效地找到最优设计。

但真实工程优化过程并不是简单地反复执行固定优化算法。

工程人员通常需要持续观察：

- 当前样本是否覆盖充分；
- 是否存在边界区域缺样；
- surrogate 是否存在明显局部误差；
- surrogate uncertainty 是否过高；
- 当前搜索是否过度集中；
- Pareto front 是否仍在改善；
- optimization 是否陷入 stagnation；
- 剩余 simulation budget 是否足够；
- 当前 surrogate / optimizer 是否仍适合当前阶段。

随后执行更高层的策略调整：

```text
Observe
   ↓
Diagnose
   ↓
Decide
   ↓
Adjust Optimization Strategy
   ↓
Execute
   ↓
Observe New State
```

因此，本项目不把优化仅视为：

```text
x → f(x) → search x*
```

而进一步将整个优化过程建模为：

```text
s_t → a_t → optimization backend → s_{t+1}
```

其中：

- `s_t`：Optimization State
- `a_t`：Optimization Strategy Action
- backend：sampling / surrogate / optimizer / simulator
- `s_{t+1}`：执行策略后的新优化状态

研究重点因此从：

> "如何搜索设计变量？"

转变为：

> **"面对当前优化状态，下一步应该采取什么优化策略？"**

---

## 2. 核心研究假设

本研究建立在一个关键假设上：

虽然不同工程问题具有不同设计变量、物理模型和目标函数，但优化过程中会反复出现相似的**优化状态模式**。

例如：

```text
Case A
--------------------------------
coverage             low
boundary coverage    low
surrogate error      medium
uncertainty          high
stagnation           low
budget               high

Possible strategy:
→ exploration
→ uncertainty sampling
```

另一个问题可能具有完全不同的物理变量，但出现：

```text
Case B
--------------------------------
coverage             high
local error          low
uncertainty          low
stagnation           high
budget               medium

Possible strategy:
→ local exploitation
→ restart / modify optimizer
```

因此，希望学习：

\[
\pi(a|s)
\]

即：

> 给定 Optimization State，学习合理 Optimization Action 的概率分布。

最终希望获得具有跨问题迁移能力的：

```text
Optimization Decision Model
```

而不是针对某一个工程问题训练一个专用 optimizer。

---

## 3. 与普通 Optimization Agent 的区别

传统 LLM Optimization Agent 通常采用：

```text
Optimization History
        ↓
       LLM
        ↓
Reasoning
        ↓
Optimization Action
```

即每一步都重新进行语言推理。

本项目采用：

```text
             Optimization State
                     │
                     ▼
                 System-1
                  π(a|s)
                     │
             confidence / OOD
                     │
          ┌──────────┴──────────┐
          │                     │
      confident            uncertain
          │                     │
          ▼                     ▼
       Action               System-2
                               LLM
                                │
                                ▼
                              Action
```

目标是实现：

> Fast learned decision + selective deep reasoning.

常规状态由轻量决策模型直接处理；

复杂状态、OOD 状态和低置信度状态才调用生成式 LLM。

因此研究重点包括三个方面：

1. Optimization decision 是否具有可学习规律；
2. 如何构造可靠的 decision supervision；
3. 如何决定何时应该调用 expensive System-2 reasoning。

---

## 4. 核心技术路线

完整技术路线：

```text
Optimization Problems
        │
        ▼
Optimization Backend
        │
        ▼
State Extractor
        │
        ▼
Optimization State s_t
        │
        ├─────────────────────────┐
        │                         │
        ▼                         ▼
     System-1                System-2 LLM
     πθ(a|s)                 reasoning
        │                         │
        └──────── Router ─────────┘
                    │
                    ▼
          Optimization Action
                    │
                    ▼
          Optimization Backend
                    │
                    ▼
                  s_t+1
```

同时存在一条离线数据生成路径：

```text
Optimization State s_t
        │
        ▼
Environment Checkpoint
        │
 ┌──────┼──────┬──────┐
 ▼      ▼      ▼      ▼
 a1     a2     a3    ... ak
 │      │      │       │
 ▼      ▼      ▼       ▼
H-step Counterfactual Rollout
 │      │      │       │
 ▼      ▼      ▼       ▼
R1     R2     R3      Rk
 └──────┴──────┴───────┘
          │
          ▼
       Utility
          │
          ▼
       P*(a|s)
          │
          ▼
Optimization Decision Dataset
          │
          ▼
     Train System-1
```

这是整个项目最核心的技术闭环。

---

## 5. 第一层：Optimization Problem / Benchmark

第一版不要自己实现大量测试函数。

优先采用 BOCoDe。

BOCoDe 当前提供大量 synthetic、engineering、HPO optimization problems，并具有统一 problem registry，同时提供 BoTorch/Ax 接口。

第一阶段选择少量代表性问题即可。

建议：

### Synthetic

```text
Ackley
Rosenbrock
Rastrigin
Branin
Hartmann
```

### Multi-objective

```text
ZDT
DTLZ
```

### Engineering

后续从 BOCoDe 中选择：

```text
Truss
Pressure Vessel
Gear Train
Car design
Water resources
...
```

第一版原则：

> Problem 本身不是本项目创新点。

因此尽量直接调用 benchmark。

---

## 6. 第二层：Optimization Backend

Backend 负责真正执行 optimization primitive。

Agent 不应该直接依赖 BoTorch、pymoo 或某个具体 optimizer。

定义统一接口：

```python
class OptimizationBackend:

    def initialize(problem, config):
        ...

    def sample(strategy, n):
        ...

    def evaluate(X):
        ...

    def fit_surrogate(config):
        ...

    def optimize(config):
        ...

    def get_history():
        ...

    def checkpoint():
        ...

    def restore(checkpoint):
        ...
```

第一版优先：

```text
BOCoDe
   +
BoTorch
```

后续再增加：

```text
pymoo
PlugBO-style plugins
Engineering simulator
Suspension simulator
```

Backend 的设计原则是：

```text
Decision Layer
      ↓
Unified Backend API
      ↓
┌─────────┬─────────┬───────────┐
BoTorch   pymoo     Engineering
                    Simulator
```

Agent 不应知道具体 backend 的实现。

---

## 7. 第三层：Optimization State

这是本研究第一个真正核心的数据结构。

定义：

```python
@dataclass
class OptimizationState:
    sampling: SamplingState
    surrogate: SurrogateState
    optimization: OptimizationProgress
    resource: ResourceState
    history: HistoryState
```

### Sampling State

第一版：

```text
n_samples
coverage
boundary_coverage
min_pairwise_distance
sample_concentration
```

### Surrogate State

```text
validation_rmse
validation_mae
max_local_error
mean_uncertainty
max_uncertainty
calibration_error
```

### Optimization State

```text
best_value
recent_improvement
stagnation_steps
pareto_improvement
diversity
```

### Resource State

```text
used_budget
remaining_budget
remaining_budget_ratio
simulation_count
elapsed_cost
```

### History State

```text
last_action
recent_actions
recent_improvement
failure_count
```

核心设计原则：

**State 尽可能描述 optimization process，而不是具体物理变量。**

例如：

```text
BAD:

spring stiffness = 180000
damping = 12000

GOOD:

coverage = 0.42
local_error = 0.31
uncertainty = 0.47
stagnation = 5
budget_remaining = 0.36
```

这样才有可能实现：

```text
Numerical Benchmark
        ↓
Engineering Benchmark
        ↓
Suspension Optimization
```

之间的 decision transfer。

---

## 8. 第四层：Optimization Action

第一版采用有限结构化 Action Space。

不要让 LLM 自由生成任意 Python code。

建议第一版：

```python
class ActionType(Enum):

    GLOBAL_EXPLORATION
    LOCAL_EXPLOITATION

    UNCERTAINTY_SAMPLING
    ERROR_GUIDED_SAMPLING
    BOUNDARY_SAMPLING

    RETRAIN_SURROGATE

    CONTINUE_OPTIMIZATION
    RESTART_OPTIMIZATION

    STOP
```

Action 可以带参数：

```json
{
    "type": "UNCERTAINTY_SAMPLING",
    "params": {
        "n_samples": 10
    }
}
```

后续再逐渐扩展：

```text
SWITCH_SURROGATE
CHANGE_ACQUISITION
CHANGE_TRUST_REGION
CHANGE_OPTIMIZER
CHANGE_HYPERPARAMETERS
```

第一阶段控制 Action Space 的复杂度。

---

## 9. Counterfactual Rollout

这是整个研究最关键的方法模块之一。

对于一个 Optimization State：

\[
s_t
\]

保存 environment checkpoint：

```python
checkpoint = env.checkpoint()
```

然后分别测试：

```text
a1
a2
a3
...
ak
```

每个 action 从**完全相同的状态**开始：

```python
for action in candidate_actions:

    env.restore(checkpoint)

    env.step(action)

    rollout(H)

    result[action] = evaluate()
```

得到：

\[
R(s,a_1),R(s,a_2),...,R(s,a_K)
\]

从而回答：

> 如果当时采取另一个 optimization strategy，结果会怎么样？

这比让 LLM 或人工直接标注：

```text
state → best action
```

更加可验证。

---

## 10. Utility Function

Counterfactual outcome 需要统一转化成 decision quality。

第一版定义：

\[
U(a|s)
=
w_1 \Delta Q_{\text{optimization}}
+
w_2 \Delta Q_{\text{surrogate}}
+
w_3 \Delta Q_{\text{coverage}}
-
w_4 C(a)
\]

其中：

- optimization improvement；
- surrogate improvement；
- exploration / coverage improvement；
- simulation / computation cost。

然后：

\[
P^*(a|s)
=
\operatorname{softmax}
\left(
U(a|s)/T
\right)
\]

而不是只保存：

```text
best_action = A3
```

保存：

```json
{
    "global_exploration": 0.08,
    "local_exploitation": 0.05,
    "uncertainty_sampling": 0.46,
    "error_sampling": 0.31,
    "retrain": 0.07,
    "continue": 0.03
}
```

从而得到：

\[
P^*(a|s)
\]

---

## 11. Optimization Decision Dataset

最终形成：

```text
Optimization Decision Dataset
```

建议 JSONL：

```json
{
  "id": "...",

  "problem": {
    "family": "ackley",
    "dimension": 10,
    "instance": 3
  },

  "state": {
    "sampling": {},
    "surrogate": {},
    "optimization": {},
    "resource": {},
    "history": {}
  },

  "candidate_actions": [
    "...",
    "...",
    "..."
  ],

  "gold_probs": {
    "global_exploration": 0.08,
    "local_exploitation": 0.05,
    "uncertainty_sampling": 0.46,
    "error_sampling": 0.31,
    "retrain": 0.07,
    "continue": 0.03
  },

  "rollout_results": {},

  "metadata": {}
}
```

这里可以直接参考 NanoJev 的：

```text
state
question
candidate
gold_probs
train/dev/calibration/test/ood
```

设计。

NanoJev 已经支持 complete probability target 和 `gold_probs`，因此其数据设计尤其值得借鉴。

---

## 12. System-1 Decision Model

第一阶段不要急着训练。

先比较：

```text
Random
Rule-based
LLM
Laya zero-shot
```

确认任务本身成立。

随后建立：

```text
MLP baseline
```

再尝试：

```text
NanoJev fine-tuning
Laya fine-tuning
Jev-style decision model
```

目标：

\[
s \rightarrow P_\theta(a|s)
\]

训练目标：

\[
\mathcal L =
D_{KL}
(P^*(a|s)\|P_\theta(a|s))
\]

或者 distribution cross entropy。

Laya 当前公开实现本身就是 non-autoregressive System-1 decision engine，可对 `choice / score / noul` 类型问题输出概率，因此非常适合作为原型参考。

---

## 13. Calibration

System-1 的核心不只是：

```text
Accuracy
```

而是：

> 它是否知道自己什么时候不可靠？

评估：

```text
ECE
Brier Score
NLL
Reliability Diagram
Entropy
```

定义：

\[
C(s)=\max_aP(a|s)
\]

以及：

\[
H(s)=-\sum_aP(a|s)\log P(a|s)
\]

后续加入 OOD detector。

---

## 14. System-2

System-2 使用生成式 LLM。

第一版继续使用现有 Tau Harness。

输入：

```text
Optimization State
+
Recent History
+
Available Actions
+
Remaining Budget
```

输出：

```json
{
  "action": "...",
  "parameters": {},
  "reason": "...",
  "confidence": 0.0
}
```

System-2 第一阶段不负责直接编写整个 optimization program。

其职责是：

> High-level optimization strategy reasoning.

Code Agent、sampling implementation、simulation execution 均属于 execution layer。

---

## 15. System-1 / System-2 Router

第一版：

```python
if confidence >= tau:
    action = system1(state)
else:
    action = system2(state)
```

第二版：

```python
escalate = (
    confidence < tau_conf
    or entropy > tau_entropy
    or ood_score > tau_ood
)
```

最终研究：

\[
\text{Performance}
\leftrightarrow
\text{Reliability}
\leftrightarrow
\text{LLM Cost}
\]

这是论文核心实验之一。

---

## 16. Benchmark Protocol

Benchmark 分三层。

### Level 1 — Synthetic

目标：

验证基本 decision learning。

```text
Ackley
Rastrigin
Rosenbrock
Branin
Hartmann
ZDT
DTLZ
```

### Level 2 — Engineering Benchmark

采用 BOCoDe 中的 engineering problems。

目标：

测试：

```text
cross-problem generalization
cross-dimension generalization
constrained optimization
multi-objective optimization
mixed-variable optimization
```

BOCoDe 当前专门强调 engineering-centered BBO，并提供统一 API 和大量工程任务，因此非常适合作为这一层基础。

### Level 3 — Suspension Engineering Case

最终迁移到：

```text
Vehicle Suspension Optimization
```

作为真实工程案例。

用于验证：

> Numerical optimization decision knowledge 是否能够迁移到真实 engineering optimization workflow。

---

## 17. Evaluation Metrics

需要区分三类指标。

### Decision-level

```text
Top-1 Action Accuracy
Top-k Accuracy
KL(P* || Pθ)
NLL
Brier Score
ECE
OOD AUROC
```

### Optimization-level

```text
Best Objective
Hypervolume
Regret
Convergence Speed
Simulation Budget
Success Rate
```

### Agent-level

```text
LLM Call Rate
Token Cost
Decision Latency
Failure Rate
Retry Count
Total Optimization Cost
```

最终最重要的图之一：

```text
Optimization Performance
        ↑
        │                Full LLM
        │             ●
        │         ● Hybrid
        │      ●
        │   ● System-1
        │
        └────────────────────────→
               LLM Cost
```

研究：

> 是否可以用较少的 System-2 调用接近 full-LLM optimization performance。

---

## 18. Benchmark 的另一项重要思想

可以借鉴 Frontier-Eng 的评测哲学：

不要只评价：

```text
final answer correct?
```

而应该评价：

```text
fixed budget
    ↓
iterative interaction
    ↓
continuous feedback
    ↓
improvement trajectory
```

因此保存：

```text
round 0
round 1
round 2
...
round T
```

完整轨迹。

---

## 19. Logging / Trajectory

所有实验必须支持 reproducibility。

建议：

```text
runs/
└── session_id/
    ├── config.yaml
    ├── problem.json
    ├── trajectory.jsonl
    │
    ├── round_000/
    │   ├── state.json
    │   ├── decision.json
    │   ├── action.json
    │   ├── samples.csv
    │   ├── surrogate_metrics.json
    │   └── optimization_metrics.json
    │
    ├── round_001/
    └── ...
```

Trajectory 至少记录：

```text
state
action
action probability
router decision
LLM reasoning
tool calls
optimization outcome
budget
runtime
random seed
```

---

## 20. 软件架构

建议最终：

```text
opt_agent/

├── problems/
├── backend/
├── state/
├── actions/
├── environment/
├── rollout/
├── dataset/
├── system1/
├── system2/
├── router/
├── benchmark/
├── metrics/
├── logging/
├── configs/
└── runs/
```

但第一阶段不要一次全部创建。

坚持：

> Build only what the current experiment needs.

---

## 21. 第一阶段 MVP

第一阶段目标不是论文最终系统。

目标只有：

> 跑通完整 decision loop。

最小流程：

```text
BOCoDe Problem
      ↓
BoTorch Backend
      ↓
State Extractor
      ↓
OptimizationState
      ↓
Laya / Rule / LLM
      ↓
OptimizationAction
      ↓
Backend
      ↓
New State
```

第一版只支持：

```text
1–3 problems
1 surrogate
1 optimizer
4–6 actions
```

不要扩展。

---

## 22. 第二阶段：Decision Dataset

MVP 跑通以后：

```text
Optimization trajectory
        ↓
select checkpoints
        ↓
counterfactual actions
        ↓
H-step rollout
        ↓
utility
        ↓
P*(a|s)
        ↓
JSONL
```

目标：

```text
10k+
optimization decision states
```

但数量不作为第一目标。

首先验证：

```text
P*(a|s)
```

是否具有稳定、可解释的结构。

---

## 23. 第三阶段：System-1

依次：

```text
Rule
 ↓
MLP
 ↓
Laya zero-shot
 ↓
NanoJev-style fine-tuning
 ↓
Jev-style / Laya fine-tuning
```

回答：

> Optimization decision 是否可学习？

以及：

> 是否能够跨 optimization problem 泛化？

---

## 24. 第四阶段：Hybrid Agent

加入：

```text
Calibration
+
OOD
+
Router
+
Tau System-2
```

形成：

```text
state
 ↓
System-1
 ↓
confidence
 ├── high → action
 │
 └── low → System-2 → action
```

回答：

> selective reasoning 是否能够降低 LLM inference cost，同时保持 optimization quality？

---

## 25. 第五阶段：Engineering Transfer

最终：

```text
Numerical Benchmark
       ↓
Engineering Benchmark
       ↓
Suspension Optimization
```

验证：

```text
Decision transfer
Generalization
Engineering feasibility
```

---

## 26. 当前应该复用的项目

### BOCoDe

用途：

```text
Optimization Problems
Engineering Benchmark
Baseline Algorithms
```

不要重写问题库。

### BoTorch

用途：

```text
GP
Acquisition
BO
```

作为第一版 backend。

### Laya

用途：

```text
System-1 prototype
Typed decisions
Probability output
Calibration reference
```

### NanoJev

用途：

```text
Dataset schema
gold_probs
decision head
training pipeline
OOD split
```

### Tau

用途：

```text
Agent Harness
Tool Call
Trajectory
Session
Failure Recovery
System-2
```

### Frontier-Eng

用途：

```text
Benchmark philosophy
Fixed budget
Continuous reward
Improvement trajectory
Simulator/verifier
```

主要参考评测方式，而非作为核心 backend。

---

## 27. 当前不要做的事情

开发 Agent 应避免：

1. 从零实现 optimization benchmark；
2. 从零实现完整 Bayesian Optimization；
3. 第一阶段训练大型模型；
4. 第一阶段接 ADAMS；
5. 第一阶段做完整 suspension environment；
6. 第一阶段让 LLM 任意生成 optimization code；
7. 同时支持十几种 surrogate；
8. 同时支持几十种 actions；
9. 一开始设计复杂 multi-agent；
10. 为了软件工程完整性过度 abstraction。

当前目标始终是：

```text
最小系统
   ↓
验证科研假设
   ↓
发现问题
   ↓
逐步扩展
```

---

## 28. 开发 Agent 的决策原则

每次实现新功能前必须回答：

### Q1

这个模块是在验证哪个 research hypothesis？

### Q2

已有项目是否已经提供？

如果已有成熟实现：

> 优先 adapter / wrapper，而不是 rewrite。

### Q3

它属于：

```text
Research Core
or
Engineering Infrastructure
```

如果属于 infrastructure：

> 尽量复用。

如果属于：

```text
State
Action
Counterfactual Rollout
Decision Dataset
Calibration
Routing
Decision Transfer
```

则需要重点自行实现并保留清晰实验接口。

---

## 29. Research Core Boundary

本项目真正需要保护的核心是：

```text
Optimization State Representation
             ↓
Counterfactual Rollout
             ↓
Decision Supervision P*(a|s)
             ↓
Optimization Decision Dataset
             ↓
Calibrated System-1
             ↓
Uncertainty / OOD Routing
             ↓
Selective System-2 Reasoning
             ↓
Cross-problem Decision Transfer
```

而：

```text
BO algorithm
GP implementation
benchmark function
LLM API
Agent harness
simulation solver
```

不是主要创新点。

---

## 30. 最终科研问题

整个项目围绕三个 Research Questions 展开。

### RQ1 — Learnability

> Can optimization strategy selection be learned from optimization trajectories?

研究：

\[
s_t\rightarrow P(a_t|s_t)
\]

是否存在稳定规律。

---

### RQ2 — Generalization

> Can optimization decision policies generalize across optimization problems?

研究：

```text
train problems
      ↓
unseen numerical problems
      ↓
unseen engineering problems
```

的迁移。

---

### RQ3 — Selective Reasoning

> Can calibrated System-1/System-2 routing preserve optimization performance while reducing expensive LLM reasoning?

研究：

\[
\text{performance}
-
\text{reliability}
-
\text{reasoning cost}
\]

之间的 trade-off。

---

## 31. 开发顺序

严格按照：

```text
Phase 0
Run BOCoDe baseline
        ↓
Phase 1
Unified Backend
        ↓
Phase 2
OptimizationState
        ↓
Phase 3
OptimizationAction
        ↓
Phase 4
Basic Decision Loop
        ↓
Phase 5
Checkpoint / Restore
        ↓
Phase 6
Counterfactual Rollout
        ↓
Phase 7
Utility + P*(a|s)
        ↓
Phase 8
Decision Dataset
        ↓
Phase 9
System-1 Baselines
        ↓
Phase 10
Calibration / OOD
        ↓
Phase 11
Tau System-2
        ↓
Phase 12
Router
        ↓
Phase 13
Benchmark
        ↓
Phase 14
Engineering Transfer
```

任何阶段没有跑通：

> 不进入下一阶段。

### 2026-09-23 执行顺序调整

Phase 8.6B 的三步 rollout 标签在 Branin 上未通过完整预算的终局效果检验。
这个问题保留为后续研究方向，详见 `PROGRESS.md`。当前先完成双模型和路由的
工程闭环：Phase 9 可接入 Laya zero-shot 参考基线，随后接入 System-2 与
Router，并运行完整预算对照。基于这些暂定标签的 System-1 监督训练及性能
提升结论，仍需通过独立种子的终局评估后才能成立。

---

## 32. 当前第一个 Milestone

当前只完成：

```text
BOCoDe
   ↓
选 1 个简单 problem
   ↓
跑通 baseline BO
   ↓
包装 backend
   ↓
提取 state
   ↓
执行 action
   ↓
获得 new state
```

验收标准：

```python
state = env.reset()

action = policy(state)

next_state, result = env.step(action)
```

能够稳定执行。

暂时：

```text
NO training
NO RL
NO Jev fine-tuning
NO suspension
NO complex multi-agent
```

完成后再开始 Counterfactual Rollout。

---

## 33. 最终目标

最终系统不是：

> "LLM 自动运行一个优化算法"。

而是：

> **一个能够从优化经验中学习策略、知道自身决策何时可靠，并在必要时调用深度推理模型的自适应工程优化决策智能体。**

最终方法链：

\[
\boxed{
Optimization\ State
\rightarrow
Counterfactual\ Decision\ Supervision
\rightarrow
Calibrated\ System\text{-}1
\rightarrow
Selective\ System\text{-}2
\rightarrow
Adaptive\ Optimization
}
\]

这条链应始终作为整个开发项目的主线。
