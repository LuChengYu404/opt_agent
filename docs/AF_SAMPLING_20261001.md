# LMABO AF 采样移植（h2 执行器）— 设计记录 2026-10-01

**分支**: `exp/boundary-scheme-round2`
**动机**: h1 的点选择采样动作（Sobol 空间填充 / 最优点均匀扰动 /
Sobol 网格 std-argmax）无决策理论支撑。将 LMABO
（"Adaptive Acquisition Selection for Bayesian Optimization", ICLR 2026，
`arxiv_2602.07904`，本地 PDF `docs/ref_papers/pdfs/arxiv_2602.07904.pdf`）
的 12-AF 候选集移植为本项目的采样机制，落实 guide §1.1 A_pt 预留的
"AF 选择"扩张槽位。

**版本戳**: `HARNESS_VERSION` h1 → **h2 → h3**（同日两跳：h2 为 AF 移植，
h3 为菜单简化——删除 error_guided/boundary 两动作并去 seed 参数，见 §9；
guide §4.5 规则 5。h1/h2 均登记入 `_KNOWN_PAST`，旧记录读取为 LEGACY，
与 h3 数据混池需显式豁免 +
pooling_note）。动作菜单、观测格式、记录 schema 均未变；变的是
动作语义与拟合计数 → 轨迹与 U(a,s) 标签分布变化，属 harness 演化。

---

## 1. 移植内容（`src/opt_agent/backend/acquisitions.py`）

LMABO 附录 A 的 12-AF 候选集，逐项对本地 PDF 核实：

| 组 | AF | BoTorch 载体 | 最大化方式 |
|---|---|---|---|
| exploitative | PI, LogPI, EI, LogEI, PosMean | `acquisition.analytic.*` | 多起点 L-BFGS-B（`optimize_acqf`） |
| exploratory | UCB | `UpperConfidenceBound(beta=κ²)` | 同上 |
| exploratory | PosSTD | `PosteriorStandardDeviation` | 同上 |
| exploratory | TS | 后验 `rsample` | 候选集离散 argmax（论文附录 B.2） |
| exploratory | qKG, qMES, qJES | `qKnowledgeGradient` / `qMaxValueEntropy(candidate_set)` / `qJointEntropySearch` | 多起点 L-BFGS-B |
| exploratory | qPES | `qPredictiveEntropySearch(optimal_inputs)` | 候选集离散 argmax（论文附录 B.2） |

优化协议按论文附录 B.2：除 TS 与 PES 外全部多起点 L-BFGS-B；TS/PES
在有限随机候选集上离散最大化。qPES/qJES 所需的 optimal inputs/outputs
由后验 Thompson 式 argmax 采样生成（BoTorch ≥0.18 要求显式传入）。

**论文未给定、由本项目设计期定值的常量**（集中在模块顶部，改一处即改全部）：

| 常量 | 值 | 说明 |
|---|---|---|
| `UCB_KAPPA` | 2.0 | 论文固定 κ 但未给出数值；取 2.0（按论文自分组明显偏探索） |
| `DISCRETE_CANDIDATES` | 512 | TS/PES 离散候选集大小（论文只说"有限随机候选集"） |
| `NUM_OPTIMAL_SAMPLES` | 8 | qPES/qJES 的 optimal-input 采样数 |
| `NUM_RESTARTS`/`RAW_SAMPLES` | 4/16 | 沿用 h1 propose() 的优化器预算，LogEI 默认路径逐位一致 |

Branin 实测单次求点耗时：解析 AF ≤0.02s，qKG/qMES/qJES ≤0.26s。

## 2. 动作映射（动作 id / 参数 schema 兼容，仅内部语义变更）

| 动作 | h1 | h2 | 拟合计数变化 |
|---|---|---|---|
| `global_exploration` | Sobol n 点 | **TS** 贪心逐点（n_samples 个 LMABO 轮） | 0 → 每点 1 次惰性拟合 |
| `local_exploitation` | 最优点 ±step 均匀扰动 | **PI**（τ=incumbent），在 `[best ± step·span]∩域` 信任域内最大化；`step` 语义保留为信任域半径，局部性契约（\|x−best\| ≤ step·span）不变 | 0 → 1 次惰性拟合 |
| `uncertainty_sampling` | Sobol 网格 std-argmax | **PosSTD** 解析 AF 连续最大化 | 1 次（stale 时）→ 每点 1 次惰性拟合（n>1 时逐点重拟合） |
| `surrogate_acquisition` | 硬编码 LogEI q=1 | 默认 **LogEI**（不变）+ 新增可选 `af` 参数（12 名单，LMABO 的"菜单选 AF"机制移植到策略层） | 不变 |
| `error_guided_sampling` | 残差引导扰动 | **h3 已移除**（本分支不再支持启发式采样） | — |
| `boundary_sampling` | Sobol 近边界点 | **h3 已移除**（boundary_coverage 状态特征保留） | — |

多点评采样为序贯贪心：每点一次 fit→propose→evaluate（等价于固定该
AF 连跑 n 轮 LMABO）。随机性（候选集 scramble、后验采样）全部走全局
torch RNG 流，由 checkpoint 的 RNG 状态回放——反事实 rollout 的公平性
机制不变。采样动作的 `seed` 参数仍被 schema 接受（旧记录里的参数能通过
校验），但 AF 路径不再消费它；注意这只保证**参数兼容**，不等于旧记录
可重放——h1 记录的动作序列在 h2 执行器下语义不同（见 §4.1 版本戳与
`validation._assert_replayable_harness`）。

与 LMABO 的结构对应：论文中 LLM 每轮从菜单**选名字**（零样本、不调
参数）；本项目中该选择 agency 由策略层承担——通过动作菜单（固定映射
的采样模式位 + `surrogate_acquisition` 的 `af` 参数）表达。论文单点/
轮 ↔ 本实现贪心逐点。注意可达性分层：**后端 API**（`propose(af=...)`）
暴露全部 12 个 AF；但 factory 的规范候选 `surrogate_acquisition` 不带
`af`（默认 LogEI），策略经 factory 只会选到 LogEI——要用满菜单需显式
构造带 `af` 的动作（或后续给 factory/sample_params 加 AF 维度，预留）。

## 3. 变更文件

- `src/opt_agent/backend/acquisitions.py`（新增）：AF 注册表 + `propose_candidate`
- `src/opt_agent/backend/bocode_botorch.py`：`propose(n=1, af="LogEI", bounds=None)`（ABC 签名同步；`bounds` 供信任域覆写）
- `scripts/ioh_backend.py`：`propose` 同步支持 `af`/`bounds`（共享注册表；独立运行回退 LogEI）
- `src/opt_agent/actions/executor.py`：三动作 AF 化 + 贪心 helper + 动态 `n_model_fits`
- `src/opt_agent/actions/models.py`：`af` 参数校验（`_af_name`）
- `src/opt_agent/actions/boundary_registry.py`、`system1/_prompt.py`：guide_note/LLM 描述同步
- `docs/RESEARCH_BOUNDARY_GUIDE.md` §1.1（A_pt 槽位状态）、§2.4（备注列）
- `src/opt_agent/dataset/versioning.py`：h1→h2
- 测试：`tests/test_acquisitions.py`（新增）、`test_action_executor.py`、`test_counterfactual_rollout.py`、`test_attribution_suite.py`、`test_jev_policy.py`、`tests/conftest.py`（桩记录 af/bounds）

## 4. 研究协议影响（读数前必看）

1. **旧数据不可与 h2 混池**：h1 数据集 / 反事实标签 / RLCD 训练头对
   新执行器全部 off-policy。已有训练工件继续用 `version_policy=warn`
   读，但任何"旧数据 + 新轨迹"的合并需显式豁免（§4.5）。
2. **utility 标签面变化**：`rollout/utility.py` 的 `w_fit·n_model_fits`
   项现在作用于每个采样点（原 Sobol/扰动零拟合），U(a,s) 分布整体
   移动——跨 h1/h2 的 D 值/ Gate 指标不可直接比较。
3. **LLM System-1 提示词变化**：`_DESCRIPTIONS` 更新为 AF 语义，
   Laya/Jev/LLM-S1 各线需重新基线评测；RF 头特征结构不变、分布漂移。
4. **运行时成本**：global/local 每点一次 GP 拟合。数据生成与
   attribution suite 的 rollout 密集管线会变慢（Branin 量级仍为秒级/
   episode；qKG/qMES/qJES 仅显式 `af` 选择时使用）。
5. **遗留**：`sample_params`（Random/Uniform
   消融策略）未加 `af` 随机化——如需复现论文的 random-portfolio 基线，
   在 factory 里给 surrogate_acquisition 加 af 均匀采样即可（一行）。
   （`lenz_state` 的 acquisition 字段已在 round-2 同步为
   "LMABO portfolio (default LogEI)"，不再是遗留项。）
6. **未动的预留槽位**：AF 参数（κ 自调、SAWEI/PI-is-back 线）与批量 q
   仍为预留（guide §1.1），本记录不涉及。

## 5. codex round-1 处置（gpt-6.1-sol，runs/codex_af_sampling_review.log，REVISE → 全部处置）

| # | 级别 | 发现 | 处置 |
|---|---|---|---|
| A | MAJOR | `normalize_inputs=True` 时 qPES/qJES 的 EP 用已归一化 train_inputs 与物理 optimal_inputs 混拼、且经 posterior 二次变换；qMES 同病（train_inputs 默认取归一化值） | **ADDRESSED**：`acquisitions._entropy_workspace` 在变换模型上改用剥离变换的 deepcopy + 归一化边界 + 逆变换回物理坐标（已数值验证 posterior 一致、往返精确）；TS 不经此路径本就正确 |
| B | MAJOR | 贪心多点采样中途失败会留下已提交的点而无 ActionResult 报告，违反执行器"无部分变更"契约 | **ADDRESSED**：`_greedy_af_samples` 原子化——有 checkpoint 能力的后端在动作前快照、异常回滚（含 RNG）后重抛；无 checkpoint 能力的后端（IOHBackend）按旧语义执行并已在 docstring 注明 |
| D | MAJOR | `validation.py` 的 replay/oracle 入口不查 harness 戳，h1 记录会在 h2 执行器下被静默错误重放 | **ADDRESSED**：`_assert_replayable_harness` 守卫加入 `oracle_test_from_record` / `replay_with_config`；LEGACY 需显式 `allow_legacy_harness=True`，MISMATCH 永拒；三个冻结工件分析脚本（validate_phase8_6b / diagnose_qs5 / ablate_phase8_6d）显式豁免并注明 |
| F | MAJOR | executable criteria 仍称 global/local "不拟合代理"、seed 冒充可复现性控制 | **ADDRESSED**：`_prompt.py` 改为逐点惰性拟合描述 + seed 兼容性口径（backend RNG 流）+ af 参数展示 |
| D' | MINOR | versioning 戳定义未覆盖执行器语义 | **ADDRESSED**：docstring 已把"动作菜单的可执行语义"纳入 HARNESS_VERSION 冻结范围 |
| E | MINOR | 测试缺口 | **ADDRESSED（部分）**：新增原子性回归、IOH af 转发与独立回退、LogEI 默认路径等价、TS checkpoint 重放、归一化熵 AF、PI≠LogEI 行为区分、h1 分类/混池拒绝（见 tests/test_acquisitions.py、test_action_executor.py、test_versioning 补充） |
| F' | MINOR | AF_SAMPLING 表格 uncertainty 拟合计数口径、af 可达性分层；models.py 头部、lenz_state 固定 LogEI 描述过时 | **ADDRESSED**：本文档 §2 表格与可达性说明、models.py 头部、lenz_state 两处字符串均已更新 |

## 6. codex round-2 处置（gpt-6.1-sol，runs/codex_af_sampling_round2_review.log，REVISE → 全部处置）

Round-1 8 项中 6 项复核 ADDRESSED；A/B/E/F-minor 判 PARTIAL，衍生 2 MAJOR + 2 MINOR：

| # | 级别 | 发现 | 处置 |
|---|---|---|---|
| R2-A | MAJOR | `_entropy_workspace` 的 del-first 顺序在 restore 后的 GP 上错误：restore 重建的 GP 处于 train 态、train_inputs 为物理坐标，botorch 在 `eval()`（`_set_transformed_inputs`）时才归一化它们；先删属性再进 posterior 会跳过归一化 → EP 建在错误后验上（fit 后不经 predict 直接 propose 的路径同样暴露） | **ADDRESSED**：顺序改为先 `af_model.eval()`（train_inputs 归一化）再 `del input_transform`；已实证 del-first 在 restored 模型上不一致、eval-first 一致；新增 fresh-fit 与 restored 双路径后验等价回归 + restore 后熵 AF 物理坐标回归 |
| R2-B | MAJOR | IOHBackend 无 checkpoint/restore，多点贪心动作失败仍部分提交，违反原子性契约 | **ADDRESSED**：IOHBackend 实现与 Bocode 同构的 Checkpoint 协议（history + fitted GP state_dict + RNG；problem_id 编码 fid:dim:instance，BBOB 类假设已在 docstring 记录）；新增 checkpoint 往返与 IOH 多点动作原子性回归。顺带修复该文件 `get_history().__len__` 的 NameError（`X` → `self.X`，此前 execute_action 在 IOH 后端上必崩） |
| R2-C | MINOR | 回滚失败被静默吞掉 | **ADDRESSED**：回滚失败时以异常链抛出 RuntimeError（`from` 原异常），同时报告 original 与 rollback 两个异常 |
| R2-D | MINOR | 三个 legacy 回放脚本的产物未标注"当前执行器跑旧记录" | **ADDRESSED**：validate_phase8_6b / ablate 的输出行加 `replay_execution` 字段，diagnose_qs5 报告头加 Execution 行 |
| R2-E | MINOR | 熵 AF 覆盖只查边界不查后验正确性；回滚只测桩 | **ADDRESSED**：新增后验等价测试（fresh + restored）、真实后端（Bocode）原子回滚测试（含 RNG 恢复断言）、IOH 原子性测试 |
| R2-F | MINOR | 文档：seed"重放不受影响"表述误导、lenz 遗留项过时 | **ADDRESSED**：§2 改为"参数兼容 ≠ 可重放"并指向版本戳守卫；§4.5 lenz 项移出遗留 |

## 7. codex round-3 终审处置（gpt-6.1-sol，runs/codex_af_sampling_round3_review.log：两个 MAJOR 复核 ADDRESSED，剩 2 MINOR）

| # | 级别 | 发现 | 处置 |
|---|---|---|---|
| R3-A | MINOR | IOHBackend.restore 内部 GP/RNG 恢复失败被静默吞掉，绕过执行器的回滚失败上报 | **ADDRESSED**：restore 的 GP 重建与 RNG 恢复失败均直接抛出（executor 的原子回滚路径能看到失败并链式上报两个异常） |
| R3-B | MINOR | diagnose_qs5 独立保存的 JSON 产物缺执行来源标注（Markdown 有） | **ADDRESSED**：JSON metadata 增加 `replay_execution` 块（执行 harness 版本、记录侧 harness 戳集合、migration 注记） |

## 8. 评审闭环（round-4，runs/codex_af_sampling_round4_review.log）

round-4 复核两项 round-3 MINOR 均 ADDRESSED、新问题 PASS、**VERDICT: APPROVE**。
四轮轨迹：REVISE(4M+4m) → REVISE(2M+2m) → REVISE(2m，双 MAJOR 复核通过) → **APPROVE**。
全量测试 516 项通过（含 3 轮评审新增回归 16 项）。

## 9. h3 菜单简化（2026-10-01，同日于 h2；本分支不再支持旧采样方法）

用户决定：本分支只保留 AF 采样。变更：

- **动作菜单 8 → 6**：删除 `error_guided_sampling`、`boundary_sampling`
  （仅存的两条启发式采样；枚举、factory 规格、boundary_registry、
  guide §2.4、System-1 描述同步收缩；`state.extractor` 的
  `boundary_coverage` 特征保留，`BOUNDARY_TOL` 常量移至 extractor）。
- **参数接口收缩**：所有动作不再接受 `seed`（AF 采样的随机性本就走
  checkpoint 回放的 torch RNG 流）。参数全集变为
  `{n_samples, step, af}`；`candidate_actions_for_state(state)` 去掉
  seed 形参（25 处调用点同步）；`full_candidates()` 无参化；
  `sample_params` 只生成 n_samples/step；`LLMSystem2` 移除
  `candidate_seed`；ioh_backend 移除独立运行的内联 LogEI 回退；
  models.py 删除两个失去用途的参数校验器。
- **版本戳 h2 → h3**（h2 未产生过数据，属会话内迭代；h1/h2 均登记
  `_KNOWN_PAST`）。
- **归档数据注意**：`_shared_rq_features.ACTIONS` 收缩为 6 元组后，
  h1 时代的 attribution 归档工件（8 动作、SAMPLING 索引）不再能被
  当前分析脚本按索引解读——按声明弃用处理，需要时回到 h1 提交分析。
- 测试同步：删除两动作的专项测试与 seed/tol 参数测试，菜单类断言
  8→6；`affine_rescaling_test` 新增 `transforms` 参数（默认不变），
  tiny 冒烟改按平移对 + 噪声地板断言（h3 近平局密度升高，
  spearman≥0.75 / top1≥0.7 / JS≤0.02，生产门限 0.9/0.8 不变）。

## 9. h3 简化（2026-10-01，同日第二笔：菜单收敛 + 接口瘦身）

用户决策：本分支不需要支持旧采样方法。在 h2 的基础上进一步删减：

1. **动作菜单 8 → 6**：移除 `error_guided_sampling`（残差引导扰动）与
   `boundary_sampling`（Sobol 近边界）——两者均为启发式、无 portfolio
   对应物。现存菜单 = 纯 AF 点选择（global=TS、local=PI+step 信任域、
   uncertainty=PosSTD、surrogate=af 可选默认 LogEI）+ retrain + stop。
   `boundary_coverage` **状态特征保留**（extractor 自持 BOUNDARY_TOL，
   与动作解耦）。
2. **seed 参数全删**：无动作再消费 seed，随机性统一走 checkpoint 可回放
   的后端 RNG 流。连带删除：schema 中的 seed 校验、factory 的
   action_seed、`sample_params` 的 seed 生成、`candidate_actions_for_state`
   与 `full_candidates` 的 seed 形参（约 25 处调用点同步）、
   `LLMSystem2.candidate_seed`、System-2 的 seed 类型镜像项。
3. **其他瘦身**：IOHBackend 删除内联 LogEI 回退（直连共享注册表）；
   models.py 删除失去用途的 `_finite_float`/`_non_negative_int` 校验器。

版本戳 h2→h3（h1/h2 入 `_KNOWN_PAST`；h2 未产生过数据，属会话内迭代）。
测试对齐：删除 error/boundary/seed 专属用例（phase6 ×4、factory ×2、
system2 ×5、IOH 回退 ×1），改写以被删动作为样例的 jev/laya/
system2_fallback/decision_loop/counterfactual_rollout 用例；全量
pytest 通过（最终数字见提交时记录）。

遗留提示：归档 h1 数据的显式豁免回放（allow_legacy_harness=True）在
候选重建阶段会因未知动作类型/不支持参数而**响亮失败**——这是本分支
不支持旧采样的既定后果；需要分析旧数据时请在 h1 代码检出上跑。

## 10. 精简 pass（2026-10-01，用户决定：可维护性优先于防御性）

核心链路行数：bocode_botorch 566→274、acquisitions 263→159、executor 227→109、
models 162→97、ioh_backend 297→193（合计 −45%）。

**保留的正确性底线**（非防御仪式）：n_obj/n_constr 结构门（该栈处理不了
约束/多目标）、objective 仿射变换的有限性检查、propose 前必须 fit、
预算/陈旧态检查、贪心动作的 checkpoint 回滚（一条 try/except）、
熵类 AF 的归一化工作区、checkpoint 的 GP+RNG 精确还原。

**按用户决定移除的防御层**（相应测试一并删除；后续修改者自负其责）：
- 问题名 allowlist（任何单目标无约束 bocode 问题均可初始化）；
- `evaluate()` 的 rank/dim/finite 输入复验；
- checkpoint/restore 的 backend_id 匹配与 shape 复验、legacy
  normalize_inputs 戳推断、restore 失败的降级吞嗉（现一律向上抛）；
- `fit()` 失败时的半成品模型清理（失败即抛，模型对象遗留但
  `is_fitted=False` 挡住使用）；
- 执行器回滚失败的二级 RuntimeError 包装（回滚失败时直接让
  restore 的异常顶替原异常）；
- 参数校验双表（_ALLOWED_PARAMS/_PARAM_SCHEMAS）合并为单表
  `PARAM_VALIDATORS`。
