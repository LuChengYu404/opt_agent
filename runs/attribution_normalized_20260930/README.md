# 归一化后续归因归档

协议：`docs/ATTRIBUTION_NORMALIZED_FOLLOWUP_20260930.md`。

生产后端增加 `BocodeBotorchBackend(normalize_inputs=True)`，拟合和快照恢复都保留输入变换；默认 False 用于保持旧实验行为。IOH 也有相同的显式选项。本实验 normalized 组的全部 GP 动作均启用，raw 组完全不启用。没有修改原 `runs/attribution_20260930` 归档。

统计复算可在当前项目根目录直接执行最后一条分析命令。模拟复算应在独立项目副本中先用 `source_snapshot/` 中的 11 个冻结文件覆盖对应路径，再按依赖顺序执行：

```bash
.venv/bin/python scripts/exp_attribution_normalized.py controls --workers 12
.venv/bin/python scripts/exp_attribution_normalized.py labels --workers 12
.venv/bin/python scripts/exp_attribution_normalized.py loops --workers 12
.venv/bin/python scripts/exp_attribution_normalized.py references --workers 20
.venv/bin/python scripts/exp_attribution_normalized.py positive
```

上述为主面板的实际执行版本。两组独立补充使用后续兼容性修补版本；在独立副本切换为 `source_snapshot_supplements/` 中的对应文件后继续：

```bash
.venv/bin/python scripts/exp_attribution_normalized_confirmation.py
.venv/bin/python scripts/exp_attribution_rastrigin_opportunity.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/audit_attribution_decision_metric.py
.venv/bin/python scripts/analyze_attribution_normalized.py
```

模型审计要求原数据集和本地 `runs/rlcd_laya_{bt,proper}_v1` checkpoints 已存在，模型权重哈希见审计产物；不请求外部模型。分析命令需要主面板、补充组及审计全部完成。统计复算不需要重跑模拟。

补充运行器的进程数已在源码冻结：归一化确认 8，机会确认开发 16/测试 20。

`controls` 完成后其它模拟阶段可并行。所有任务写入采用临时文件后原子替换；相同配置跳过已保存任务，不同来源哈希拒绝续跑。重新冻结配置时应在另一个目录保留原数据；不要覆盖本归档。`--smoke` 使用独立 smoke 子目录。

| 文件 | 数据 |
|---|---|
| `controls.json`、`selection.json` | 800 条开发轨迹，冻结各族最佳 raw/normalized 固定动作及 normalized schedule |
| `labels.json` | 25 个开发上下文、75 个相同状态，raw/normalized × 一步/五步 × 八次模拟 |
| `loops.json` | 1300 条新种子固定/schedule 闭环 |
| `references.json` | 100 条新种子特权参考，两个干预长度都每五点评估后重规划 |
| `positive_control.json` | 有解析收益 1/30 的单决策正向对照；选择与评价模拟分开 |
| `rastrigin_confirmation*.json` | 100 个独立种子、300 条闭环；GP 归一化稳健性 |
| `rastrigin_opportunity*.json` | 200 条开发 schedule + 120 条独立测试；反馈选择确认 |
| `decision_metric_audit.json` | 原数据切分下固定排序对照；两个模型各 20 个初始状态的 SDK/位置检查 |
| `analysis.json`、`REPORT.md` | 配对 95%/99% 区间、校正 p、完整结果和限制 |
| `verification.json` | 预算、共同前缀、来源、正向对照与验证记录 |
| `CAUSE_ASSESSMENT.md` | 结论修正、原因分级及下一步建议 |

全部真实闭环共 2820 条。独立补充协议分别为 `docs/ATTRIBUTION_RASTRIGIN_CONFIRMATION_20260930.md` 和 `docs/ATTRIBUTION_RASTRIGIN_OPPORTUNITY_20260930.md`；先冻结补充组参数与新种子，再完整运行。原探索组与确认组分开报告，未据确认结果追加样本。

参考阶段首次使用八个进程，之后在确认属于本次任务的独立进程组内停止并从原子检查点恢复为二十个进程，以利用空闲 CPU。配置、种子与任务没有改变；不丢弃慢任务或失败案例。`interim_baselines.json` 是基线阶段完成后的诊断快照，未用于调整剩余实验。

本轮先检查教师和特权选择器能否提供独立收益，再决定是否值得扩大标签和训练学生。没有把有限模拟参考当作闭环数学上限。

运行中发现并另行修复了两项兼容性问题：数据集元数据没有保留未拟合快照的归一化设置；跨问题或未初始化后端恢复时没有同步缓存边界。前者由生成/重放往返测试验证，后者的两项回归测试修复前失败、修复后通过，日志位于 runs/attribution_normalized_restore_bounds_*。本轮不经过数据集生成器/重放器，且始终在初始化后的同一问题内恢复，因此这些修补未改变主实验计算；运行中的进程继续使用已加载的冻结版本。`source_snapshot/manifest.json` 保留实际执行版本，分析程序据此验证来源。
