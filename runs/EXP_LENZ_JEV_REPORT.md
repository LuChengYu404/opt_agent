# EXP — lenz 式富状态 × Jev 式呈现：两项对照实验（2026-09-28）

Branch `exp/lenz-state-jev-presentation`。问题（survey §四）：**全量试验日志
（𝒟_t）里有没有 15 维压缩摘要之外的决策信号？** 两路独立检验——
可学性（RF 探针）与 LLM 行为（deepseek 对照）——在**未触碰的 s100–119
复现流**上执行，冻结 8-action 菜单与可执行判据两侧完全相同，只有状态呈现不同。

## E1：轨迹粒度可学性探针（`exp_lenz_gate2_probe.py`）

每家族 leave-one-seed-out 多输出 RF，三组特征对比（同折同记录）：

| Family | frozen15 top1 | +traj (ext) top1 | traj-only top1 |
|---|---:|---:|---:|
| ackley | 0.112 | 0.105 | 0.126 |
| branin | **0.496** | 0.445 | 0.244 |
| hartmann | 0.197 | 0.197 | 0.256 |
| rastrigin | 0.185 | 0.192 | 0.177 |
| sixhumpcamel | **0.655** | 0.648 | 0.566 |

（mean regret 同构：traj_ext 在 ackley/hartmann 上还略差。）

**读法：干净的阴性。** 手工轨迹特征（逐维覆盖统计、best-x 位置、成对距离
分位数、top-gap/改进曲线）在 5/5 家族上不优于 frozen15；traj-only 在两个
"有信号"家族（branin/sixhump）明显更差。**至少在这个特征粒度上，全量
𝒟_t 不含压缩摘要之外的可学信号**——与 P2 的 D 上限解读一致（信号薄在
问题里，不在压缩里），也实证支持 codex 对轨迹级状态的 NOT_RECOMMENDED。

## E2：双状态 LLM 对照（`exp_lenz_vs_frozen_llm.json`）

30 条分层记录（每家族 6 条、跨 phase）× 2 条件，deepseek-chat 温度 0：

| 指标 | frozen 叙事 | lenz 富叙事 |
|---|---:|---:|
| 条件间选择一致率 | — | **0.80**（6/30 改变） |
| gold-argmax 命中 | 0.133 | **0.133**（无差） |
| tokens/次 | 766 | **1,599（2.1×）** |
| 延迟 | 0.8s | 0.8s |

改变方向的细节：6 处改变中 5 处是 `global_exploration → surrogate_acquisition`
（集中在 mid/late phase——看到完整 trial log 后 LLM 略转保守/利用），1 处
`→ local_exploitation`（hartmann late，恰好命中 gold）。

**读法：富状态确实改变 LLM 行为（20% 的决策、方向系统性偏 exploit），
但不改变其与教师 argmax 的一致率，且 token 成本翻倍。** 结合 E1：
**两条路都指向"富状态在本面板上不产生决策价值，只产生成本"。**

## 合并结论与主线索引

两项实验为 codex 裁决 KEEP_FROZEN_QUALIFY 提供了实证补充（裁决当时只有
理论论证）：

1. 压缩状态没有丢掉可学的信号（E1，特征粒度）；
2. 富状态改变 LLM 的行为风格但不改善对齐，且成本 2×（E2）。

若期刊版引入新面板（HPO/工程，D 上限未知），再重跑本分支的两个实验作为
第一道诊断——脚本已就位且零改动可换数据集。

## 诚实限定

- E1 是**手工特征**的阴性——学习式序列编码器（transformer over trials）
  未测试；但 n≈130/家族下过拟合风险高，且 D 上限不因编码器改变。
- E2 n=30、单模型、温度 0、gold 命中本身受教师噪声影响（教师 argmax 跨
  re-roll 仅 29% 稳定）——"命中无差"的噪声容限较宽；但"2× 成本 + 行为
  系统性偏移"两个事实是硬的。
- E2 的 H_t 截断到 last_action（数据集记录不存完整动作序列）——富状态的
  记忆维度未被完全测试；完整 H_t 需活体运行（如 `run_lenz_jev_smoke.py`）。

## Artifacts

- `runs/exp_lenz_gate2_probe.json` + `scripts/exp_lenz_gate2_probe.py`
- `runs/exp_lenz_vs_frozen_llm.json` + `scripts/exp_lenz_vs_frozen_llm.py`
- `runs/lenz_jev_prompt_example.txt`（完整示例 prompt）
- 设计实现：`src/opt_agent/state/lenz_state.py`（LenzState + record 重建器）、
  `src/opt_agent/system1/lenz_prompt.py`（Jev 式呈现）、
  `LLMSystem1Policy(request_builder=...)` 注入点
