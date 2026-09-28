# 参考文献库（面向期刊投稿）

> 建立日期：2026-09-27。用途：支撑期刊论文（升级自 workshop paper）的方法论定位、
> related work 写作与审稿应对。检索范围以 2024 年以后的 agentic BO / LLM-for-optimization
> 文献为主，保留少量无法替代的经典方法论文献。

## 文件索引

| 文件 | 内容 |
|---|---|
| [papers_2024plus.md](papers_2024plus.md) | 注释版文献清单：按"agentic BO 主线 / 路由与成本线 / 反事实监督 OPE 线 / 经典统计锚点"四条线组织，每条注明它支撑本项目哪个部分 |
| [journal_positioning.md](journal_positioning.md) | 与最近邻工作（Sara/lenz、LLAMBO、GPTOpt）的创新性差异定位、一句话卖点、期刊级完整性缺口（按优先级） |

## 使用警示（务必先读）

1. **arXiv 编号来自 web 搜索结果，引用前必须逐一下载 PDF 核对标题/作者/年份。**
   已知至少一处搜索结果指向错误论文（2608.00146 实为 DiffusionGemma；
   Agentic BO 论文的正确编号是 2608.00316，已通过全文抓取核实）。
2. 部分条目（标注 ⚠️ 的）只经过搜索摘要确认，尚未读全文；
   其表述（如"附录 E"细节）基于摘要/评论转述，引用前需核实。
3. 状态标记：✅ = 已读摘要/全文核实；⚠️ = 仅搜索结果确认，待核实。
| [state_action_design_survey.md](state_action_design_survey.md) | 状态/动作设计谱系调研（Sara/lenz、LLAMBO、GPTOpt、Volpp、TempoRL、FunBO vs 本项目），供 related work 与设计调整决策使用 |
| [model_interaction_survey.md](model_interaction_survey.md) | 大小模型交互 × 信息充分性四线调研（路由/级联、System1/2、何时求助、主动信息获取）：代表论文、叙述逻辑、本项目差异化定位 |
| [exploration_rounds.md](exploration_rounds.md) | 2026-09-28 六轮调研探索的轮次记录 + 分优先级后续优化方向（P0: GP-Hedge 基线/LMABO 划界/codex 复审；P1/P2 期刊版路线） |
