# EXP-META-CONFIG-SCREEN — 元层配置前置筛选（guide §10.3 / codex round-1 E2）

**Date**: 2026-09-29（分支 `exp/boundary-scheme-round2`）
**Question（预宣告于运行前）**: 在铺设元层监督管线之前，当前面板上是否存在
**可检出的配置条件性**？guide §4.4 警告元层优势同样可能恒非正；本实验是
round-1 评审 E2 要求的便宜配对筛选。

## 设计（脚本内预宣告）

- **配置轴**：初始设计规模 `n_init ∈ {5 ("small", harness 默认), 12 ("large")}`
  ——guide §1.2 的设计期配置（初始设计/预算切分），是现 backend 无需新模型
  代码即暴露的唯一元层旋钮。两臂总预算同为 30 次评估（同花费、不同切分）。
- **内层策略钉死** AlwaysSurrogate（隔离配置效应与策略效应）。
- **配对**：同 family + 同 seed → 同 scrambled Sobol 流（嵌套：draw(12) 的
  前 5 点 = draw(5)）。
- **端点**：terminal best objective（backend 最大化约定）。
- 5 families × 20 seeds × 2 configs = 200 条轨迹；`n_evals` 校验两臂均为 30。
- **预宣告判定规则**：POSITIVE ⇔（≥2 个 family CI 显著且方向相反）或
  （meta-D bootstrap CI 排零）；否则 NEGATIVE。

## 结果（`runs/exp_meta_config_screen.json`）

| family | Δ = mean(large − small) | 95% CI | 显著 |
|---|---:|---|---|
| Branin | −0.0052 | [−0.024, 0.019] | 否 |
| Hartmann | −0.3132 | [−0.694, 0.080] | 否（small 优，接近显著） |
| Ackley | +0.0383 | [−0.072, 0.168] | 否 |
| SixHumpCamel | −0.0408 | [−0.147, 0.052] | 否 |
| **Rastrigin** | **+8.3562** | **[1.278, 15.496]** | **是（large 优）** |

- 赢家身份分裂：small 胜 3 个 family（Branin/Hartmann/SixHumpCamel）、
  large 胜 2 个（Ackley/Rastrigin）。
- **meta-D = 0.124，95% CI [0.032, 0.231]，排零** → 判定规则触发。
- **VERDICT: POSITIVE**（经由 meta-D 分支；"≥2 显著且反向"分支未触发——
  显著的只有 Rastrigin 一个，单方向）。

## 解读（含诚实的 caveats）

1. **配置条件性存在，但主要由一个 family 驱动**：Rastrigin（强多峰）是唯一
   CI 显著的 family——更大的 Sobol 初始设计显著更好，与"多峰函数需要更多
   空间填充覆盖、LogEI 易陷局部盆地"的直觉一致；Hartmann 方向相反
   （small 优，CI 上沿 0.080）但未达显著。
2. **赢家身份 3:2 分裂是 D_meta > 0 的来源**：即"最优 n_init 因问题而异"
   在本面板成立 → 按问题签名选 n_init 的元层策略存在超越任何固定 n_init
   的空间（幅度以标准化效用计 ≈ 0.12）。
3. **标准化口径 caveat**：u = mean/s_f（族内两臂合并 std）是单位无量纲
   化，非标准效应量；D_meta 的幅度依赖该口径，但"赢家因族而异"的定性
   结论只依赖差的符号，对口径稳健。
4. **范围 caveats**：单一配置轴（n_init）× 单一替代值（12）；20 seeds；
   策略钉死 AlwaysSurrogate；n_init=12 占预算 40% 是较极端的切分。
5. **对 guide §10.3 的回答**：前置筛选**未否决**元层方向——在 n_init 轴上
   存在可检出条件性（尽管由单一 family 主导）。下一步若铺元层监督，应
   优先扩展配置轴（预算切分、更丰富的设计规模档位），并把"Rastrigin 型
   多峰签名"作为签名特征的可分性检验起点。

## 复现

```bash
uv run python scripts/exp_meta_config_screen.py            # 默认 5×20×2, ~3 min
```

脚本内含 torch 单线程初始化器（首次冒烟 85s/traj → 2.2s/4traj，超额订阅
防护，longhorizon 驱动先例）。
