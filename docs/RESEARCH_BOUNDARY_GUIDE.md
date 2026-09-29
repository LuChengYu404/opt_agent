# 研究边界与修改指导方略（Research Boundary & Modification Guide）

> Status: 2026-09-29. 本文档把围绕 LMABO（arXiv:2602.07904, ICLR 2026）的研读
> 与后续架构讨论收敛为规范：抽象参考架构、组件冻结/可调分类学、双层决策
> 结构与接口契约、修改操作规程、术语表。
>
> 文档关系：THEORY.md 是命题的形式化对侧（P1–P3, L1）；本文规定**架构与
> 修改规范**——即可修改与不可修改的组件范围，以及修改前须通过的检查。
> 文献细节见 `ref_papers/optimization_theory_survey.md`；实验映射见
> `docs/final_goal_experiment_protocol.md`。
>
> 规范性用语约定：**必须**（违反即缺陷）/ **禁止**（违反即越界，产出的
> 数据与结论不可用）/ **允许**（默认合法）/ **建议**（最佳实践）。
> 修改本文档时必须同步更新 §9 术语表与 §7 锚点。
>
> 修订记录：2026-09-29 round-1（codex 评审 `runs/codex_boundary_guide_review.log`，
> VERDICT: REVISE）后修复全部事实性/一致性错误；同日 round-2（分支
> exp/boundary-scheme-round2）落地方案类五项：§5.5 终止判据、§0.3/§5.2
> 监督限制同等地位、元层前置筛选实验（初判 POSITIVE，经同日严谨化重析
> 更正为 INCONCLUSIVE，见 §10.3）、§8 重排、注册表 + 版本戳强制机制；
> 同日完成全文语域规范化（隐喻性表述仅保留于明确标注的直觉注记）——
> 处置状态见 §10。
> 同日后续：E2 分析严谨化（判定 POSITIVE → INCONCLUSIVE 的方法更正）；
> 全文语域规范化（隐喻仅保留于标注的直觉注记）；剂量网格扩展检出
> 配置条件性（§5.2）。

---

## 0. 一页摘要

1. 系统是一个**优化域 agent harness**：状态摘要 → 决策 → 执行 → 反馈 → 学习，
   分两层（元层配置 / 内层动作），层间以 episode 边界隔离。
2. **episode 内模型规范不变**：模型族（核函数）与超参估计规则在
   episode 启动前确定，运行中不得变更。代理维护遵循惰性拟合生命
   周期：评估使拟合失效，依赖 GP 的动作按需重拟合，RETRAIN_
   SURROGATE 显式重拟合（`actions/executor.py` 实现现状）；重估
   **必须**为族内 MLE，频率（每轮 vs 每 k 轮）属 B3 可调项（§2.2）。
   运行中更换核函数、修改后验或替换代理属 C 类**禁止**行为，必须
   外置为元层的重启/配置动作。
3. 研究自由度 = **动作菜单扩张 × 每个动作位监督可测性的构造质量**。
   本项目的主贡献是第二项：用 counterfactual rollout 构造 per-decision
   监督（"creating the measurability"）。同等地位的实测限制：短视程
   首动作标签噪声高（R=5 后 flip 0.37）、对齐教师首动作终端优势 ≈0、
   首动作与延续效应不可加（8.6F）、微调策略闭环塌缩——**schedule/
   continuation 级标注 + 匹配闭环检验是扩大首动作标签投入前的下一
   决策点**（§5.2、§5.5）。
4. 新增任何动作/特征/模型修改，先过 §3.4 三项前置检查，再按 §6 操作规程执行
   （动作菜单由 `actions/boundary_registry.py` 注册表 + 同步测试强制，
   动作集变更未同步注册表与 §2.4 表格即导致测试失败）。
5. 跨问题经验有效累积的前提是**测量规范一致**：harness 按版本冻结
   （§4.5），数据带版本戳。

---

## 1. 抽象参考架构

### 1.1 五层结构（单/多目标统一）

```
┌─ 外生层（问题定义，非系统部件）
│    f: X→R 或 f: X→R^m、域 X、噪声/成本模型、预算 T
│    输出目标：argmin（单目标）/ Pareto 非支配集（多目标）
│
├─ 状态层  s_t = ( D_t 观测数据, M_t 代理诊断, B_t 预算/进度 )
│    M_t 例：lengthscales/outputscale 统计、最短采样距离、f_range、
│    停滞步数——本项目 15 维 RF 特征与 LMABO 状态摘要同属此类
│
├─ 策略层  π: s_t → a_t，动作空间分为两个子集：
│    A_pt（点选择动作：决定下一个评估点的位置——AF 选择/AF 参数/批量 q；
│         抽象槽位。当前实现 = 5 个采样模式 + surrogate_acquisition，
│         见 §2.4；"AF 选择/参数/批量"为预留的扩张方向，尚未实现）
│    A_flow（流程控制动作：控制优化流程的推进——stop/retrain/切换
│         采样模式。重启不属于内层动作集：它是元层动作（§1.2/§4.1），
│         内层仅能以终止型动作触发 episode 边界）
│
├─ 执行与维护层（标准 BO 迭代循环，episode 内逐轮执行）
│    a_t → x_t → y_t = f(x_t)+ε → D_{t+1} → 按需族内重拟合代理（MLE，惰性生命周期）
│
└─ 输出算子 O: D_T → 结果
     单目标：incumbent；多目标：非支配过滤 + 超体积（hypervolume）
```

**多目标扩展在抽象架构层面仅涉及两个组件的替换**：点选择 AF 族
（EI/JES → MO-JES 类，本地锚 `arxiv_2210.02905`）与输出算子
（incumbent → Pareto 过滤）。状态结构、策略分层、执行循环、双层
架构**不变**；实现面的完整改动清单见 §6.4（backend 向量值模型、
状态抽取、效用项、参考点、校验均需变更）。注意区分：可调的是
"何时输出"（STOP 是动作位），不可调的是"什么算最优"（支配关系
是数学定义）。

**A_flow 是人工流程控制决策的形式化对应物**：人工执行优化时的
停止判断、重训决定、策略切换直觉，均落在这层动作空间（重启例外：
属元层，§4.1）。"替代人工操作"即以可学习策略 π 覆盖
A_flow ∪ A_pt。

### 1.2 双层决策结构

```
问题层（task instance）：一个工程优化问题
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 元策略层（meta-level / configuration layer）
   状态：问题签名（维度、预算、已完成 episode 的结果摘要）
   动作：选配置（模型族/核/AF 菜单/初始设计/预算切分）→ 启动 episode；
        或终止当前 episode → 换配置重启
   反馈：outcome 级（最终 regret/超体积、耗时、可行性）
   轨迹：config₁ → outcome₁ → config₂ → ...（meta-trajectory）
   理论形态：bandit over configurations（有自己的 regret 理论，
            不依赖 BO 收敛定理）
━━━━━━ episode 边界 ━━━━━━
 运行层（inner level）
   固定配置下的标准 BO：GP + AF 菜单 + 采点循环
   轨迹：s₀ → a₀ → s₁ → ...（inner trajectory）
   反馈：per-decision 稠密监督（本项目 counterfactual rollout）
```

层级语义对应 options/SMDP 框架（Sutton, Precup & Singh, 1999）：一个
option = 一次时间上扩展的执行 = 一个 episode。元层的"换模型"因此不是
运行中变异，而是 option 切换。

### 1.3 与 agent harness 概念的对应

| Agent 术语 | 本项目对应物 |
|---|---|
| harness / scaffold（控制循环） | rollout 循环 |
| observation | 状态摘要（15 维特征 / LMABO 序列化文本） |
| tool use / typed actions | 8 个 ActionType + 参数 schema + 校验 |
| trajectory | 一次 rollout 记录 |
| episodic memory | 跨 episode 轨迹库 |
| policy | RF（当前）/ RLCD 微调 Laya（exp/rlcd-laya-finetune） |
| in-context 复用 | 检索历史轨迹入 prompt（Reflexion/Voyager 路线，弱监督） |
| 训练信号路线 | 轨迹 → U(a,s) 标签 → SFT/RL（本项目主线，强监督） |

**关键区分**：控制循环本身是通用基础设施；关键瓶颈在于将轨迹
转化为可靠训练信号的机制。本项目贡献在后者（per-decision 反事实
监督），而非循环本身。

---

## 2. 组件分类学

### 2.1 A 类：定义性/外生冻结（动了就不是同一个问题）

| 组件 | 冻结理由 |
|---|---|
| 目标函数 f、评估协议、单次评估成本 | 黑盒真值，问题定义而非系统部件 |
| 预算 T | 资源约束，外生 |
| 输出算子的**定义**（argmin / Pareto 支配 / 超体积的测度） | 数学定义，不随运行改变 |
| 策略的 adaptedness（可测性第一层） | 决策必须是观测历史的可测函数——随机化策略只要把自身随机性并入决策时信息即满足（§3.2），确定性/温度 0 是可复现性约定而非本不变量条件；违反则运行评估失效。**架构不变量**，任何设计不得触碰 |

### 2.2 B 类：episode 内允许调整（按证据强度分档）

**B1 · 有理论背书**
- STOP（Bayesian optimal stopping, Dai et al., ICML 2019；
  probabilistic-regret stopping 带正确性保证，本地锚
  `arxiv_2402.16811`）。
- 族内 MLE 超参重估：允许且是默认工程惯例。理论注记（引文须准确）：
  Bull 2011（JMLR，EI 收敛分析）在超参固定或经特定一致估计的假设下
  成立，并给出**序贯 MLE 超参估计不收敛的反例**与替代估计量——超参
  估计策略是理论敏感点：必须设计期选定、episode 内语义不变。重拟合
  频率（每轮 vs 每 k 轮）属 B3，**不是**定理强制的。

**B2 · 经验充分、理论真空（替换使用的风险有限：每步决策仍是标准
   采集函数计算）**
- AF 选择：GP-Hedge（Hoffman et al., 2011）→ No-PASt-BO
  （Vasconcelos et al., 2019）→ SETUP-BO（ESWA 188:115847, 2022）→
  ESP（Shahriari et al., 2014）→ LMABO（零样本 LLM 选择器）整条脉络。
- AF 内部参数：SAWEI（自调权重 λ，AutoML 2023 / PMLR v224）、
  PI-is-back（按数据质量切 PI/EI，`arxiv_2211.01455`）。
- 本项目 8 动作中的采样模式位（global_exploration、
  local_exploitation、uncertainty_sampling、error_guided_sampling、
  boundary_sampling）与 surrogate_acquisition 占此生态位。

**B3 · 可调但文献覆盖有限（工程常用，理论保证缺乏）**
- 批量 q、重拟合频率（每轮 vs 每 k 轮）、初始点数 2D+1。

### 2.3 C 类：episode 内禁止（外置为元层动作）

> C 类是**本项目协议**，不是数学禁令：带日志的运行中切换在原则上仍是
> 可定义、可分析的 adaptive algorithm，受控比较也可以做。禁止的理由是
> 保护本项目的估计量——归因、可比、以及与所引定理前提的一致性
> （§3.3）。违反者产出的数据在本项目测量体系内不可用。

| 动作 | 破坏什么 |
|---|---|
| 运行中更换核/模型族 | 破坏超参一致性前提（估计量序列不再是该族的 MLE 序列）；先前证据在新假设下的权重无定义；切换时点构成未记录的隐藏状态，破坏可比性 |
| 手改超参/钳制 σ/向 μ 注入非概率信息（修改后验） | 后验不再是贝叶斯更新规则的输出：EI 的"期望"仅相对于被修改后的非贝叶斯后验成立；σ 钳小将导致欠探索与线性遗憾，且不再受定理上界约束；监督信号无法归因 |
| 替换代理为无概率语义的预测器（换代理） | 在 GP 后验上陈述的 regret bound 不再适用；替换载体必须自行给出与其输出语义匹配的保证——校准的后验近似也**不自动继承** GP 型 bound（§3.3）。LLAMBO（`arxiv_2402.03921`）/LLMP 占此位，与 BO 理论断开 |

**相邻但不属于 C 类的设计（避免过度禁止）**：
- 摊销推断（PFN `arxiv_2112.10510`、meta-learned priors PMLR v205）
  绕开手工 GP 但输出**校准的后验近似**——判据不在"是不是 GP"，
  在"输出是不是定义良好的概率信念"（注意：校准后验不自动继承
  GP 型 regret bound，该路线需自证其保证）。
- 设计期核选择/组合核（通过核的和/积组合获得所需表达能力）、贝叶斯
  模型平均：自始至终保持单一一致的概率模型语义，**允许**。
- SMAC（Hutter et al., 2011）以序贯模型配置算法为工具；本项目与其实
  结构同构、对象互补——**配置对象是 BO 自身**。这是元层的文献支点。

### 2.4 本项目 8 动作菜单的归类（`src/opt_agent/actions/models.py`）

| ActionType | 归类 | 备注 |
|---|---|---|
| GLOBAL_EXPLORATION | B2 | 采样模式位 |
| LOCAL_EXPLOITATION | B2 | 同上 |
| UNCERTAINTY_SAMPLING | B2 | 同上 |
| ERROR_GUIDED_SAMPLING | B2 | 同上 |
| BOUNDARY_SAMPLING | B2 | 同上 |
| SURROGATE_ACQUISITION | B2 | 标准 AF 求点 |
| RETRAIN_SURROGATE | B1 | 惰性生命周期中的显式重拟合位（B1 邻接：重估语义受 B1 约束，频率属 B3）；**必须**限定为族内 MLE 重估；**禁止**将其扩展为换族/换核——那属于元层重启动作（§4.1） |
| STOP | B1 | 理论对应物：Bayesian optimal stopping / probabilistic-regret stopping；Gate-4 stop-gain 是其数据侧对应 |

---

## 3. 研究边界的形式化

### 3.1 冻结集 = 可测性的最小不变量

```
F = { 外生层, adaptedness, 输出算子定义, （若要 bound）探索存在性 }
```

这些不是设计偏好，是"运行可评估、结论可陈述、经验可积累"的最低
条件（在本项目协议意义上，§2.3）。理论敏感点举例：Bull 类 EI 分析
对超参估计方式敏感（序贯 MLE 有不收敛反例，Bull 2011）；GP-UCB 型
bound（Srinivas et al., 2010）要求 β_t 调度保证探索频率下界——
**混合/学习型策略若打破所引定理的前提，所失去的不是性能上界的
紧度，而是遗憾上界的存在性本身**（对该定理而言，regret 不再受任何
上界约束；更换模型族则须另行陈述并证明适用的定理，§3.3）。

### 3.2 可测性的两层含义（本项目术语，勿混用）

1. **数学层（adaptedness）**：决策是历史 σ-代数的可测函数；随机化
   策略同样可以 adapted（把策略自身的随机性并入决策时信息即可）。
   温度 0 不是 adaptedness 的条件，而是本项目的可复现性约定。
   LMABO 是 adapted 策略，只是无人陈述其定理。
2. **监督层（per-decision measurability）**：每个决策有无从数据可算
   的监督目标。x 空间层有（EI 值/后验方差/信息增益）；策略层动作
   （stop/refit/换采样）原生没有——闭式规则与逐决策监督双缺。
   本项目的 U(a,s)（THEORY.md §0）通过反事实 rollout **人造**此目标。

综述口径的锚句："LLM-era systems replace the decision-maker, not
create the measurability"——LMABO 零监督、靠 LLM 内化知识；本项目
造监督、靠学习。**贡献声明落在监督层。**

### 3.3 收敛性：C 类动作失去的究竟是什么

精确表述：C 类是**项目协议**而非数学禁令。运行中执行 C 类动作的
后果并非必然损害性能——即使变更恰好有利；带日志的切换规则在原则
上仍是可定义、可分析的 adaptive algorithm。失效的是三项：**定理前提**
（所引的固定模型定理不再覆盖该运行，除非为切换规则另行陈述并证明
新定理）、**可归因性**（性能变化无法区分来自数据还是模型变更）、
**可比较性**（运行间差异混入模型规范差异这一变量）。对产出包含
"哪个决策在什么状态下更优"这类知识的系统，后两项构成决定性缺陷
——这正是将 C 类定为禁止行为的全部理由。

> 直觉注记（非规范内容，仅为理解辅助）：可将 episode 内的模型规范
> 类比为一次测量实验中的仪器——运行中途更换仪器，单次读数未必变差，
> 但跨读数积累的知识失去了统一的标定基准。规范文本以上段的形式
> 表述为准。

运行内冻结与跨问题经验累积的关系（该关系易被颠倒）：**运行内冻结
是跨问题经验有效累积的前提，而非障碍**。经验层位于 episode 边界
之上；运行内部规范越稳定，边界之上的累积越可靠。若某 episode 中途
变更模型规范，其产出的标签与其他 episode 不属于同一测量规范，
数据集受到污染（P1 的仿射不变性正依赖于"每 episode 模型规范一致"
这一前提）。

### 3.4 边界扩张的成本模型（新动作/新自由度三项前置检查）

新增动作位 a_new 前必须依次回答：

1. **反事实可 rollout？** a_new 的后果能否进 U(a,s) 分支对比
   （短视程 rollout + 固定延续）？不能则监督造不出，动作位无学习依据。
2. **破坏了哪条不变性、如何归一化修复？** 对照 P1 scope 条款：每个
   非零的目标依赖项必须**整体仿射不变**——等变的差值还须除以等变的
   尺度（如 Δopt/σ_opt 的构造）；仅分子 α-等变会使该项整体乘 α，
   破坏不变性（w_sur≠0 时的 Δsur 即此例；THEORY.md P1 scope 措辞已
   同步修正）。Gate 5 的 2/17 JS breach 已示范：仅 GP 拟合数值不稳
   即可污染标签。
3. **A(s) 是否可能恒非正？** 对照 P3（前提是逐态、匹配延续的一步
   模型）与 P2（D 是固定 ρ 与 U 下的一步上界）：若该动作位对最优
   常动作的逐态优势非正，一步模型下任何路由/学习设计在该基准问题集上
   必输——动作位整体无价值时，先做 D(family) 筛选（P2）再谈学习。
   本关是**筛选性检验**（结论须带不确定度陈述），不是闭环定论；
   价值声明仍需整程闭环评估。

---

## 4. 接口契约（层级结构的成立条件）

1. **元动作只在 episode 边界触发。** 配置在 episode 启动前确定，
   运行中不可变更。"更换模型" = 终止当前 episode + 以新配置启动新
   episode（option 切换，非运行中变异）。
2. **观测数据可在 episode 间复用（warm start）。** 观测 {(x,y)} 是黑盒
   真值记录，不依附模型；重启时全量带入，新 episode 对全部数据
   重新拟合，内部自洽，且避免浪费昂贵评估。定理适用性注意：带入的
   数据是**自适应选择**的（非 i.i.d. 设计），预算记账需包含初始数据，
   且所引定理须对容许初始历史一致成立——"照常适用"要按这些条件
   逐条核对，不是自动继承。
3. **元层反馈是 outcome 级**，故元层是 bandit over configurations
   （严格说是有状态、跨臂共享观测、受剩余预算约束的非平稳 bandit，
   其估计量按实际的历史/预算模型陈述），分析不依赖 BO 收敛定理——
   这是 C 类动作外置的主要好处。
4. **P2/P3 层级不变。** 元层优势函数 A(s) = "换配置 vs 继续跑"的
   净优势，同样可能恒非正（问题签名区分度不足时）；筛选工具箱
   在两层原样复用。
5. **Harness 版本冻结。** 同一版本内，观测格式、动作菜单、标签管线
   一致；数据带版本戳。机制（round-2 落地）：`dataset/versioning.py`
   定义 HARNESS_VERSION / UTILITY_VERSION，generator 经 `stamp_metadata`
   为每条新记录盖双戳；`check_record_versions` 按 ok / legacy / mismatch
   分类（legacy = 旧值或缺戳，可读但**禁止**与当前版本记录静默混池；
   mismatch = 未知值，拒绝混池）。任何 A/S 类修改**必须**递增对应戳
   并在 `_KNOWN_PAST` 登记旧值；`tests/test_versioning.py` 强制 generator
   走戳。harness 演化发生在版本边界（与 episode 边界同理）。违反则
   轨迹库跨版本不可比，RF/微调吸收污染。

---

## 5. 期望实现的效果

### 5.1 能力目标：双层可学习的自主优化 agent

- 内层：π 覆盖 A_flow ∪ A_pt，在 D(family) 允许的域上超过最优常
  动作（P2 给上界与筛选）。
- 元层：配置策略随经验改进（问题签名 → 配置映射），episode 间
  warm start 复用数据。
- 终态：给定问题签名与预算，系统自主完成"配置选择 → episode 执行 →
  结果输出（解/Pareto 前沿）"全流程，人工介入点收敛到问题定义本身。

### 5.2 科学目标：在两层各造一次可测性

- 内层（基建完成，**升级为决策点管理**）：counterfactual rollout →
  U(a,s) → gold_probs → RF/Laya 监督。与其实测限制并读（round-2 起
  同等地位）：标签噪声主导（单次 flip 0.74，R=5 复制后仍 0.37）、
  Gate-4 对齐教师首动作终端优势 ≈0（Ackley/Hartmann CI 很宽，未排除
  全部首动作价值）、8.6F 不可加、微调策略闭环塌缩。**下一决策点**：
  schedule/continuation 级标注（把"价值在延续里"变成标注单位）+
  匹配的闭环检验——其结果出来前，不再扩大首动作标签投入。
- 元层（开放，研究空隙更大；**配置轴筛选已完成两阶段**）：配置级监督 =
  同一问题多 episode 结果差 + 跨问题签名匹配下的标签迁移。两阶段筛选
  （2026-09-29）：两档版（EXP-META-CONFIG-SCREEN）INCONCLUSIVE（交互
  p = 0.069，系检验力不足）；按预宣告升级路径执行的剂量网格版
  （EXP-META-CONFIG-GRID，n_init ∈ {5,8,12,18} × 60 种子）**检出条件性**
  ——斜率同质性 χ²₄ = 30.87、p < 10⁻⁴、I² = 87%；光滑族逐族
  显著负斜率、多峰高维 Ackley 显著正斜率；机理符号 5/5（p = 0.031）。
  更正：两档版的 Rastrigin 个体结论未能在新种子上复制（n = 240 深采样
  臂效应至多 d_z ≈ 0.11）。**当前许可级别：签名特征与 held-out 预测是
  下一步的正当投入对象；配置级监督管线仍不启动**（oracle 选择价值
  ≠ 签名可学性，LOFO 3/5 无分辨率）。文献现状：per-instance 配置 BO 的
  元学习比内层策略层更少人做（Rice 1976 框架 / racing / Hyperband
  在 HPO 域有，BO-self-configuration 域近乎空白）。

### 5.3 理论目标：层层干净 + 判据可迁移

每层内部理论自洽（内层标准 BO + 策略层经验比较；元层 bandit），
P1/P2/P3 作为层级不变的筛选与安全工具。定位锚：LMABO 是边界最保守
端的实证点（1 个动作位、零监督、最强 LLM、50 问题稳定增益）；
本项目在中间偏激进处（8 动作位 + 人造监督）。

### 5.4 可检验的成功判据（挂现有 Gate 体系）

- 决策级：标签拟合与校准（现有 RF/Gate 体系）。微调结果按臂与种子流
  陈述：BT 臂在两条独立种子流上均显著超过 RF（pooled +0.056 /
  +0.140 与 +0.188，5/5 folds CI 排零）；RLCD-proper 臂原始流 ≈ RF
  （CI 跨零）、复制流 +0.053 显著；两臂闭环均塌缩为常动作策略
  （PROGRESS_BRANCH §5/§9；RLCD_THEORY.md T4）。
- 闭环级：terminal value 对比的 CI 排零（Phase 8.6F 提示首动作与
  延续效应不可加，闭环必须整程评估）。
- 不变量：Gate 5 仿射不变性——标签侧 Spearman ρ=1 是**必要条件**
  （实测 17/17 桶中位 ρ=1，但预宣告联合 gate 因 2/17 JS breach
  FAIL，归因 GP 拟合数值，见 THEORY.md P1 与 affine 报告）。
- 停止位：Gate 4 stop-gain 为正的 family 上检验 STOP 学习——注意
  其估计量是"gold 首动作延续 vs 立即停止"的效用差，不等于"学到的
  STOP 策略有增益"，后者需另行闭环测量。
- 元层（筛选已执行，见 §5.2）：同问题多配置 episode 的 regret 差分布 +
  签名特征的可分性检验。

### 5.5 预宣告终止判据（kill criteria；round-2 新增，D3）

成功判据（§5.4）的配对物——**满足即停，不再扩投入**，触发时在
PROGRESS 记录条件与数据，方案回到 §10 重议：

1. **教师可靠性**：标注单位重设计（长视程 / schedule 级）后，可负担
   复制数（≤R=5）下标签 flip 率仍不低于现有水平（0.37）→ 冻结
   "构造可测性"的强声明，降格为"噪声标签仍可训练排序模型"。
2. **闭环增量**：任何新监督/策略在 ≥3 个 family 上对最优常动作或
   两段 schedule 的闭环 terminal-value 增量 CI 均含 0 → 该路线止步于
   决策级报告，不进主线。
3. **成本上限**：每条标签的期望 rollout 成本（CPU·s）超过闭环评估
   成本的 10× → 重新设计标注单位（更长视程、更少标签），而非加复制。
4. **元层优势**：某配置轴前置筛选 **FUTILE**（交互未检出 且 全族 CI
   半宽 < 0.25·sd 的精确零界）→ 该轴元层管线停建（§4.4）；**仅未检出
   但零不精确（INCONCLUSIVE）→ 暂缓决策**（加种子/加配置档位后重筛），
   不得按 FUTILE 处理（n_init 轴：剂量网格已检出条件性（§5.2），
   本条未被触发）。

---

## 6. 修改指导方略（操作规程）

### 6.1 新增动作位

流程：写动机（覆盖哪类人工操作）→ §3.4 三项前置检查 → ActionType +
参数 schema + 校验（`actions/models.py` 模式）→ 接入 U(a,s) 分支
（`rollout/utility.py` 权重与归一化，过 P1 等变检查）→ 小规模基准问题集上的
D(family) 筛选 → 进菜单。**禁止**跳过三项前置检查直接加位。执行机制
（round-2 起生效）：菜单的边界分类由 `actions/boundary_registry.py`
注册表承载，`tests/test_boundary_registry.py` 同步校验注册表 ↔
ActionType 菜单 ↔ 本文档 §2.4 表格——三处任一不同步即红。

### 6.2 新增/修改状态特征

流程：特征定义（须为观测数据的确定函数，保持 adaptedness）→
版本号递增 → 新旧版本标签 A/B（同数据）→ 文档记录。
**必须**保持 15 维特征向量与数据集版本戳同步演化（版本戳机制见
§4.5；A 类修改须递增 HARNESS_VERSION）。

### 6.3 模型族/核/代理的常见修改意向与规范处置

| 修改意向 | 规范处置 |
|---|---|
| 认为核函数表达能力不足 | 在设计期更换为组合核（和/积核），并开启新 harness 版本 |
| 认为运行中拟合质量不佳 | 执行族内 RETRAIN_SURROGATE（B1）；或经元层重启更换族 + warm start |
| 希望引入 LLM/专家先验 | 进入决策层（状态/prompt/动作），**禁止**进入后验；语义先验仅允许在元层配置选择或设计期核选择中使用 |
| 认为后验不确定度标定不当 | 检查核函数与 MLE 数值健康度（Gate 5 的教训），不得直接修改 σ |

### 6.4 多目标扩展

抽象架构上仅替换两个组件（点选择 AF 族 → MO-JES 类；输出算子 → 非支配
过滤 + 超体积，超体积需指定参考点）。实现面远不止这两个组件，**必须**
列出完整改动清单：backend 向量值代理（现为 SingleTaskGP + 标量
best_f）、状态抽取（现读目标第 0 列）、U(a,s) 的 Δopt 项重定义
（如超体积改进量）、输出算子、校验管线——逐项通过 §3.4 三项前置
检查（第 2 项：等变性/归一化按新效用项重新推导）。

### 6.5 LLM 引入方式（对照 LMABO 的边界纪律）

- LMABO 纪律值得沿用：状态字段逐个给语义注释；动作清单不给描述
  （靠模型内化知识，防偏置）；few-shot 示例用占位符（防小模型模仿）；
  显式反馈指令（"避免重复无效动作"——其消融中最大退化项，RP
  1.21→1.92）；多轮记忆替代历史重放；温度 0；解析失败兜底
  （其兜底率 0.11%）。
- 本项目对 LLM 的 P3 教训：RQ3 扫描中八个 LLM 调用臂无一胜 RF，
  always-LLM 三 family 显著输——一步模型下 A(s)≤0 的签名。rich-state
  实验（exp/lenz-state-jev-presentation）**已完成**，结论为否：更充分
  的状态未改变 gold-argmax 一致率（0.133 两态持平），token 成本
  2.1×（n=30 单模型 A/B，caveats 见 `runs/EXP_LENZ_JEV_REPORT.md`）。
  **LLM 进系统的前提是其逐态优势被证实非负**，否则按 P3
  （一步模型内）只能输。

### 6.6 监督定义组件（S 类，最小规程；完整规程属方案修订项）

以下组件改变标签但不改变任何动作/状态定义，此前未被分类学覆盖、
缺少修改规程（round-1 评审 D1）：utility 权重与尺度（`UtilityWeights`）、
反事实延续策略、rollout 视程、gold 温度 T、动作子集配额（quota
k<8）、rollout 复制数（teacher 噪声控制）。

规程（**必须**遵守）：

1. 任何 S 类修改 = `UTILITY_VERSION` 递增（`dataset/versioning.py`，
   旧值登记 `_KNOWN_PAST`；generator 经 `stamp_metadata` 自动盖戳）
   + 新旧版本**同数据** A/B + P1 等变复查（按 §3.4 第 2 项修正后
   的口径）。修改若同时触及观测格式/动作菜单，则同时递增
   `HARNESS_VERSION`。
2. S 类修改**禁止**与动作/特征修改混入同一版本（归因隔离）。
3. 标签信噪比是 S 类预算的一部分：新监督源**必须**报告可负担
   复制数下的 flip 率（参照 journal-round P1 实测链：单次 rollout
   标签 flip 0.74 → R=5 复制平均后 0.37——仍高，复制数与可接受
   flip 率须显式声明）；终止阈值见 §5.5 第 1 条。

完整规程（可靠性 / 不变性 / 结果对齐三类检查的形式化）：仍属
方案修订项（§10.5）。

---

## 7. 文献锚点对照（主张 → 来源映射；✓ = `docs/ref_papers/pdfs/` 有本地全文，✗ = 仅按引文格式引用）

| 主张（正文承载处） | 锚点 |
|---|---|
| AF 组合调度脉络（B2 层先例，§2.2） | GP-Hedge（Hoffman 2011，✗）→ No-PASt-BO（Vasconcelos 2019，✗）→ SETUP-BO（ESWA 188:115847, 2022，✗）→ ESP（Shahriari 2014，✗）→ LMABO（`arxiv_2602.07904`，✓） |
| AF 参数自调（§2.2 B2） | SAWEI（PMLR v224，✓）、PI-is-back（`arxiv_2211.01455`，✓） |
| 停止理论（B1，§2.2） | Dai et al. ICML 2019（`pmlr_v97_dai19a`，✓）；probabilistic-regret stopping（`arxiv_2402.16811`，✓） |
| EI 收敛分析对超参估计敏感、序贯 MLE 反例（§2.2 B1 注记、§3.1） | Bull 2011, JMLR（✗，按引文格式；特征表述经 JMLR 官方摘要核实，round-1 评审） |
| 探索频率下界型 regret bound（§3.1 β_t） | GP-UCB（Srinivas, Krause, Kakade & Seeger, ICML 2010，✗） |
| 摊销/元学习 BO——"换载体"的合法版（§2.3 合法邻近物） | PFN（`arxiv_2112.10510`，✓）、Rothfuss meta-priors（PMLR v205，✓） |
| C 类反面教材（§2.3） | LLAMBO（`arxiv_2402.03921`，✓，LLM 兼代理）、LLMP（Requeima 2024，✗） |
| 元层框架（§1.2/§4/§5.2） | Rice 1976（✗，算法选择）；F-Race/irace、Hyperband（✗，racing/预算分配）；options/SMDP（Sutton, Precup & Singh 1999，✗）；SMAC（Hutter 2011，✗，镜像问题） |
| agent harness 术语（§1.3） | Reflexion（Shinn 2023，✗）、Voyager（Wang 2023，✗）——episodic memory/skill library |
| 教科书锚（§1.1 "BO = 序贯决策"表述） | Garnett, *Bayesian Optimization*, 2023（`garnett_bayesoptbook`，✓） |
| LMABO 实测锚（引用数字时用） | 50 问题（30 COCO/BoTorch 合成 + 20 Bayesmark HPO）×10 种子；RP 1.21；对最好静态 AF/元策略/组合/LLM 基线 AUC 分别 −9.7%/−14.8%/−16.6%/−54.7%；温度 0；兜底率 0.11%；50 轮 ≈6000 tokens、每轮 ≈1s、Gemini 臂 50 轮平均 7.42 min（部分 benchmark 预算更长；GP-Hedge 109 min——组合法每轮需优化全部 12 个 AF）。数字已对本地 PDF 核实（round-1 评审） |

---

## 8. 开放问题（研究边界上的未决问题；round-2 按证据强度重排，E3）

1. **Locked transfer evaluation（最高优先）**：BT 臂跨 family 迁移
   （两条独立流 5/5 folds CI 排零，pooled +0.140/+0.188）是唯一两次
   复现的正面结果——在 backend allowlist 扩围（新增 BOCoDe family，
   冻结决定需单独评审）后做锁定族确认，优先于一切新方向。
2. **Schedule/continuation 级标注**（§5.2 决策点）：把"价值在延续里"
   变成标注单位（schedule 段级 U / TD 式目标），配匹配闭环检验——
   直接回应 8.6F 不可加与 Gate-4 首动作 ≈0。
3. **元层监督构造**：n_init 轴两阶段筛选完成——剂量网格已检出条件性
   （§5.2、`runs/EXP_META_CONFIG_GRID.md`）。**下一步为签名特征与
   held-out 预测**（跨族 LOFO + 维度/多峰代理特征；oracle 选择价值
   已确立，可学性未测）；通过后才进入配置级反事实监督的构造（同一
   问题、同预算切分下不同配置的 episode 对比、跨问题标签迁移的签名
   匹配准则；维度归一化特征 n_init/(2d+1) 优先于裸 n_init）。
4. **闭环可加性**：Phase 8.6F 已证首动作与延续效应不可加；闭环
   价值只能整程评估——反事实监督的短视程近似何时失真？
5. **重启动作建模**：把"换族重启 + warm start"建成元层动作位后，
   其三项前置检查（§3.4）在元层如何具体化？
6. **System-2 优势抬正**：rich-state A/B（exp/lenz-state-jev-presentation）
   已完成且为负结果（§6.5）。A(s)≤0 的诊断在哪些 state 族/
   基准问题集上反转？反转条件本身是否可学（元层特征）？
7. **多目标标签**：Δopt → 超体积改进后的 P1 等变条件形式。

---

## 9. 术语表

| 术语 | 英文 | 定义（本文档口径） |
|---|---|---|
| 采集函数 | acquisition function (AF) | 由代理后验构造的打分函数 α(x;D_t)，决定下一个评估点；经 argmax 转化为采样行动 |
| 信念 | (posterior) belief | 贝叶斯意义：对未知量（黑盒 f）的概率分布认识；GP 后验 = μ（猜测）+ σ（不确定度）。其核心在于量化并保留不确定性，而非绝对确信 |
| 可测性（两层） | measurability | ①数学层：决策是观测历史的可测（adapted）函数；②监督层：每个决策有可从数据算出的监督目标。本项目"create the measurability"指② |
| adaptedness | adaptedness | 同上①；随机化策略（随机性并入决策时信息）同样 adapted——温度 0 是可复现性约定，非本不变量的条件 |
| 遗憾 | (simple) regret | 最小化约定下 r_T = min_t f(x_t) − f(x*) ≥ 0（f* 为全局最优值）。本仓库 backend 内部按最大化实现（目标取负），文档口径统一用最小化约定。收敛性理论承诺其趋于零的速率，前提是决策规则固定且满足各定理条件 |
| episode | episode | 一次完整优化运行：初始设计 → [拟合→决策→采点→评估→更新]×T → 输出。模型构建在设计期，不属于 episode |
| 设计期 | design phase | episode 之前：选模型族/核/动作菜单/预算结构。冻结发生在这里 |
| 元层 | meta-level / configuration layer | episode 边界之上的配置决策层；SMDP 语义下选择 option；反馈为 outcome 级 |
| 内层 | inner level | episode 内部的策略层决策（8 动作）；反馈为 per-decision 级 |
| 元轨迹 | meta-trajectory | config₁ → outcome₁ → config₂ → … 的序列 |
| 暖启动 | warm start | 重启的新 episode 全量带入历史观测 {(x,y)}；数据是真值记录，不依附模型，故合法 |
| 选项/半马尔可夫 | options / SMDP | 分层 RL 框架（Sutton et al. 1999）：时间上扩展的动作；一个 option ≙ 一个 episode |
| 配置上的老虎机 | bandit over configurations | 元层的理论形态；有独立 regret 理论，不依赖 BO 收敛定理 |
| 改核 | kernel surgery（C 类） | 运行中更换核函数 = 更换先验假设体系；episode 内禁止 |
| 改后验 | posterior tampering（C 类） | 绕过数据更新通道直接篡改 μ/σ（手改超参、钳制 σ、注入非概率预测）；episode 内禁止 |
| 换代理 | surrogate replacement（C 类） | 以无概率语义的预测器（如 LLM 直预测）替代 GP；破坏理论可陈述性 |
| 摊销推断 | amortized inference | PFN/meta-priors 路线：预训练预测器输出校准的后验近似——"换载体"的合法版，判据是输出仍为良定义的概率信念 |
| 族内重估 | in-family MLE refit | 核不变、超参由边际似然重估；拟合遵循惰性生命周期（按需/显式重拟合），频率属 B3 可调 |
| episode 内模型规范冻结 | intra-episode model-specification freeze | 模型族（核函数）与超参估计规则在 episode 启动前确定、运行中不变；跨 episode 比较与经验累积以该不变性为前提（C 类与 §4.1 的形式基础） |
| 反事实 rollout | counterfactual rollout | 从状态 s 执行候选动作 a + 固定延续的短视程分支，构造 U(a,s) 监督 |
| 决策实用度 | U(a, s) | THEORY.md §0 定义的标签量，共**六**项：w_opt·Δopt/σ_opt + w_sur·Δsur + w_cov·Δcov − w_eval·n_new/σ_eval − w_fit·n_fit/σ_fit − w_stop_opportunity·剩余预算比·1[a=STOP]。代码默认 w_stop_opportunity=0.75（STOP 项默认激活，`rollout/utility.py`） |
| gold 概率 | gold_probs | softmax(U(·,s)/T)，在**实际 rollout 的候选集**上计算——该候选集可能为配额采样的动作子集（quota k<8 时，`dataset/generator.py`） |
| 逐态优势 | A(s) | P3 路由模型中 Q(s,a₂(s)) − Q(s,a₁(s))：System-2 对 System-1 的匹配延续净优势 |
| Gate 4 / Gate 5 | gates | 项目预宣告实验门槛：Gate 4 = 终端行为增益（含 stop-gain 子项）；Gate 5 = 标签仿射不变性。定义与通过状态见 PROGRESS 记录与 THEORY.md |
| 惰性拟合 | lazy fitting | 代理维护现状：评估使拟合失效，GP 依赖动作按需重拟合，RETRAIN 显式重拟合（`actions/executor.py`） |
| 监督定义组件 | S 类（supervision-defining） | 改变标签而不属于动作/状态的组件：utility 权重与尺度、延续策略、视程、gold 温度、动作子集配额、rollout 复制数。规程见 §6.6 |
| 终止判据 | kill criteria | §5.5 预宣告的投入终止条件（教师可靠性/闭环增量/成本/元层筛选）；触发即停并在 PROGRESS 记录 |
| meta-D | configuration-conditionality descriptor | 族标准化效用上的 E_f[max_a u] − max_a E_f[u]；总体意义 >0 ⟺ 各族优势 Δ_f 符号不全同。度量"已知族标签的 oracle 选择价值"，是**描述量而非检验统计量**，且与签名可学性无关 |
| HARNESS_VERSION / UTILITY_VERSION | version stamps | `dataset/versioning.py` 的双版本戳：前者冻结观测格式+动作菜单+标签管线整体，后者只追踪 S 类标签定义；修改对应组件必须递增并在 `_KNOWN_PAST` 登记旧值 |
| legacy / mismatch | stamp statuses | `check_record_versions` 的分类：legacy = 旧值或缺戳（可读、禁止与当前版本静默混池）；mismatch = 未知值（拒绝混池） |
| 动作菜单 | action menu | 类型化动作集（现 8 个 ActionType）；扩张受 §3.4 三项前置检查约束 |
| 点选择动作 | A_pt（point-selection） | 决定下一个评估点的位置（AF 选择/参数/批量；抽象槽位，当前实现见 §2.4） |
| 流程控制动作 | A_flow（process-control） | 控制优化流程的推进（stop/retrain/切换采样模式；重启属元层）；人工流程控制决策的形式化对应物 |
| 问题签名 | problem signature | 元层状态特征：维度、预算、历史 episode 结果摘要等 |
| 函数族 / 基准问题集 | family / panel | 数据集的函数族（Hartmann、SixHumpCamel…）及其汇集；P2 的 D(family) 按族计算。英文文献称 panel，中文行文统一用"基准问题集" |
| D(family) | one-step state-conditionality value | E_s[max_a U] − max_a E_s[U]；状态条件化策略对最优常动作的一步上界（P2） |
| harness | (agent) harness | 控制-观测-执行-反馈循环骨架；按版本冻结（§4.5） |
| 语义控制器 | semantic controller | LMABO 自我定位：LLM 只做调度决策，不碰数学骨干 |
| in-context 决策 | in-context decision-making | 不更新权重的决策方式（零样本 LLM 或检索增强） |
| 上下文内学习复用 | episodic memory (in-context reuse) | 检索历史轨迹作示例/反思材料；弱监督路线 |
| 超体积 | hypervolume | 多目标输出算子的质量度量；支配关系为定义性冻结 |
| warm-restarted episode | warm-restarted episode | 带全量历史数据重启的 episode |

---

## 10. Round-1 评审遗留与 round-2 处置（分支 exp/boundary-scheme-round2）

Round-1（`runs/codex_boundary_guide_review.log`，VERDICT: REVISE，
24 项：7 BLOCKER / 16 MAJOR / 1 MINOR）中的方案取舍类问题，
round-2 处置状态（2026-09-29）：

1. **失败/终止判据（D3）→ 已落地**：§5.5 预宣告四条 kill criteria
   （教师可靠性 / 闭环增量 / 成本上限 / 元层优势）。阈值属初版
   校准，接受人工评审调整。
2. **监督构造的地位（E1）→ 已落地**：§0.3 与 §5.2 将短视程监督的
   实测限制升至同等地位；schedule/continuation 级标注 + 匹配闭环
   检验定为下一决策点（§8.2）。
3. **元层前置筛选（E2）→ 已执行（两阶段）**：阶段一
   `scripts/exp_meta_config_screen.py`（n_init 5 vs 12，5×20×2 配对嵌套
   Sobol，预算 30，估计量/机理假设/判定规则均预宣告；主检验为 Δ_f
   同质性 Wald χ²，meta-D 降为描述量）→ **INCONCLUSIVE**（χ²₄ = 8.71、
   p = 0.069、I² = 0.54）。初判以 meta-D 边界 bootstrap 判 POSITIVE，
   属检验方法错误，同日更正（round-2 评审 D1）——分析全文见
   `runs/EXP_META_CONFIG_SCREEN.md`。阶段二（按阶段一预宣告的升级
   路径）`scripts/exp_meta_config_grid.py`：n_init ∈ {5,8,12,18} 剂量
   网格 × 60 种子 + Rastrigin 240 种子深采样臂 → **CONDITIONALITY_
   DETECTED**（斜率同质性 χ²₄ = 30.87、p < 10⁻⁴、I² = 87%）；并撤回
   阶段一的 Rastrigin 个体结论（种子区块间不复制，n = 240 效应至多
   d_z ≈ 0.11）。分析全文见 `runs/EXP_META_CONFIG_GRID.md`。
4. **开放问题排序（E3）→ 已落地**：§8 重排——locked transfer
   evaluation（唯一两次复现的正面结果）与 schedule/continuation
   标注列前，元层筛选已执行，state 富化靠后。
5. **强制机制（D2）→ 已落地（最小版）**：`actions/boundary_registry.py`
   注册表 + `tests/test_boundary_registry.py`（菜单 ↔ 注册表 ↔ §2.4
   表格三方同步）；`dataset/versioning.py` 双版本戳 +
   `stamp_metadata`/`check_record_versions` + `tests/test_versioning.py`。
   余项：§6.6 完整规程（可靠性/不变性/结果对齐检查）的形式化，
   消费端（训练管线）对 legacy/mismatch 的硬拒绝策略。
