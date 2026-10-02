# 归因实验归档

先读 [CAUSE_ASSESSMENT.md](CAUSE_ASSESSMENT.md)，完整结果见 [REPORT.md](REPORT.md)。所有历史实验数据与生产实现未覆盖。

## 从现有归档重新计算统计与图表

在项目根目录执行：

```bash
.venv/bin/python scripts/analyze_attribution_suite.py
```

会读取完整原始面板、核对主实验与补充组来源、确认独立重复与训练目标，重写 `analysis.json`、`REPORT.md` 和两幅结果图；不重新执行优化实验。

## 完整实验的依赖顺序

主协议与两个补充协议保存在 `docs/ATTRIBUTION_*_20260930.md`，包含状态选取、种子、动作、预算、参考策略与局限。运行器固定输出到本目录；已有 stage 可从原子保存的任务记录续跑，配置不相同会拒绝续跑。`fit` 会重写模型，重建实验时应使用复制出的独立项目目录并保存本归档。

```bash
.venv/bin/python scripts/exp_attribution_suite.py labels --workers 12
.venv/bin/python scripts/exp_attribution_suite.py controls --workers 12
.venv/bin/python scripts/exp_attribution_suite.py fit
.venv/bin/python scripts/exp_attribution_suite.py loops --workers 12
.venv/bin/python scripts/exp_attribution_suite.py dim_controls --workers 12
.venv/bin/python scripts/exp_attribution_suite.py dim_labels --workers 12
.venv/bin/python scripts/exp_attribution_suite.py dim_loops --workers 12
.venv/bin/python scripts/exp_attribution_repeats.py --workers 12
.venv/bin/python scripts/exp_attribution_posterior.py --workers 8
.venv/bin/python scripts/analyze_attribution_suite.py
```

RF 固定训练随机种子，保存的 labels 行顺序和序列化模型共同确定本次结果。若从零重建，进程完成次序可能改变 labels 行顺序，RF bootstrap 抽样的对应训练行也会改变；精确复算本次闭环应使用归档的 labels 与 models。每个受控训练对照使用相同的保存行顺序。

## 文件

| 文件 | 内容 |
|---|---|
| `labels.json` | BOCoDe 280 个状态，四种标签处理，每种四次模拟 |
| `controls.json`、`controls_selection.json` | 1000 条开发控制轨迹与冻结选择 |
| `models.pkl`、`model_provenance.json` | RF 模型与训练来源 |
| `loops.json` | BOCoDe 2125 条新种子闭环 |
| `dim_controls.json`、`dim_controls_selection.json` | IOH 1350 条开发控制轨迹与冻结选择 |
| `dim_labels.json` | IOH 135 个上下文、405 个决策状态 |
| `dim_loops.json` | IOH 1350 条新实例闭环 |
| `repeat32.json` | 30 个开发初始状态的 32 次模拟，首四次保留主实验值 |
| `posterior_training_features.json`、`posterior_models.pkl`、`posterior_loops.json` | 现有 posterior 摘要可见性补充组，100 条闭环 |
| `analysis.json` | 配对/层次 bootstrap 区间、Holm p、产物哈希与敏感性分析 |
| `verification.json`、`serial_parallel_check.json` | 完整性、预算、来源、测试及串并行重放检查 |
| `attribution_effects.png`、`dimension_effects.png` | 可独立导出的统计图 |

模拟规划的目标调用不计入真实预算，单独记录在闭环行中的 `planning_evaluations`/`planning_fits`。参考策略不是可部署基线，也不是真正最优 oracle。
