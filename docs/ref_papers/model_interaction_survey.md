# 大小模型交互 × 信息充分性：前沿调研（2026-09-28）

> 检索日期 2026-09-28，经 web 检索确认（多为摘要级；标注 ⚠️ 的条目引用前需
> 下载 PDF 核对标题/作者/venue）。组织方式：四条研究线，每条给出代表论文、
> 叙述逻辑（论文怎么讲故事）、与本项目的关系。

## 线 1：路由 / 级联（"同样准，更省钱"）

代表论文：
- **FrugalGPT**（Chen et al., Stanford, 2023）✅ 开山之作：小模型先答，
  打分不够好才升级到大模型，报告最高 98% 成本节省。
- **RouteLLM**（Yue et al., 2024）✅：训练一个路由器决定每个 query 送强
  模型还是弱模型；保留 ~95% 强模型质量、85% query 走便宜模型。⚠️ venue
  待核实（ICLR 2025 提交记录存在）。
- **C3PO**（Valkanas et al., OpenReview）⚠️：在概率成本约束下自监督优化
  级联。
- **Conformal Arbitrage**（NeurIPS）✅：用 conformal 风险控制平衡竞争
  LLM 的成本-准确率。
- **Dynamic Model Routing and Cascading**（2026 综述, arXiv）⚠️。

**叙述逻辑**：痛点（大模型贵）→ 机制（学出来的门/路由器/级联阈值）→
实验（cost-performance Pareto 曲线）→ 卖点（省 X% 钱不掉点）。
**共同隐含假设：切换/升级有收益**——整条线的图都是"新曲线包住旧曲线"。

## 线 2：快慢双系统 System 1 / System 2（"快管日常，慢管难事"）

代表论文：
- **SwiftSage**（Lin et al., NeurIPS 2023）✅ 本线锚点：小 seq2seq 模型当
  System 1（行为克隆），大 LM 当 System 2，**学一个"何时唤起慢系统"的
  策略**（ScienceWorld 任务）。
- **System-1.x: Learning to Balance Fast and Slow Planning**（2025,
  arXiv:2503.14601）⚠️：先生成快计划，再预测哪些步骤升级给慢推理。
- **From System 1 to System 2: A Survey of Reasoning LLMs**（2025,
  arXiv:2502.17419）✅ 综述：快系统 → 慢系统（RL 长思考、test-time
  scaling）的谱系。
- **Reasoning on a Spectrum**（2025）⚠️：对齐到 System 1/2 的精度-效率
  权衡。

**叙述逻辑**：认知科学双过程理论（Kahneman 快思慢想）开场 → 架构（小
模型 + 大模型 + 切换策略）→ 任务准确率对比 → 卖点（效率与能力兼得）。

## 线 3：自知之明 / 何时求助 / 何时停（"知道自己不知道"）

代表论文：
- **Robots that Ask for Help**（Varley et al., CoRL 2023）✅：LLM 规划器
  用不确定性对齐决定何时向人求助（本线早期锚点）。
- **HiL-Bench: Do Agents Know When to Ask for Help?**（Scale AI,
  2026 preprint）⚠️：人在环 benchmark，测 frontier agent 会不会意识到
  上下文不全并求助。
- **Agentic Abstention: Do Agents Know When to Stop**（2026 preprint）⚠️：
  多轮搜索/浏览/终端 agent 会不会在该停时停。
- **Knowing When to Ask for Help: Bayesian Self-Escalation**（preprint）⚠️。
- 校准/弃权线（NeurIPS/ICLR 多篇）：conformal prediction for LLMs、
  abstention benchmarks。

**叙述逻辑**：痛点（agent 过度自信、该问不问/该停不停）→ benchmark 或
  校准机制 → 发现"它们不知道自己不知道" → 方法提升求助/弃权质量。

## 线 4：主动信息获取（"要什么自己拿，拿够就停"）

代表论文：
- 经典（pre-LLM）：**Kapoor & Horvitz, Active Information Acquisition**
  (NeurIPS 2009)✅；**Jeong et al., Learning Q-network for AIA** (2019)✅；
  **Gong et al., Icebreaker** (2019)✅ —— 预算约束下"买哪个特征/做哪个
  测试"的形式化（metalevel control）。
- LLM 时代：**adaptive retrieval** 线（"When not to retrieve" /
  **Adaptive Retrieval Without Self-Knowledge?** Moskvoretskii et al.
  2025 ⚠️ / **Reinforced Internal-External Knowledge Synergistic Search**
  2025 ⚠️）：学"何时检索"，内部知识够就不检索。
- **Sara/lenz 的 probe 动作**（arXiv:2608.00316）✅：BO 语境下的主动查询
  （预测/诊断/历史/前沿），计算动作不耗评估预算。

**叙述逻辑**：信息获取本身是有成本的动作 → 学一个获取/停止策略 →
  实验（同样的任务成绩、更少的查询数）。

## 主流叙事的共同骨架（四条线通用）

1. 痛点：贵 / 慢 / 错 / 不自知（四选一或组合）；
2. 机制：一个学出来的门控（何时升级/切换/求助/检索/停）；
3. 实验：cost-performance 曲线，或求助/弃权的准确率 benchmark；
4. 卖点：省钱不掉点 / 该问才问 / 拿够就停。

**共同默认前提：门控的另一侧有收益可赚。** 阴性结果（门怎么设都没收益）
在这批文献里近乎缺席——这正是本项目 RQ3 的稀缺性所在。

## 本项目在四条线上的位置

| 线 | 本项目对应 | 差异化 |
|---|---|---|
| 路由/级联 | RQ3 成本前沿（450 轨迹） | 罕见的**阴性前沿** + P3 定理解释（若逐状态优势 ≤0，任何门控都无收益）——给整条线补对照证据 |
| System 1/2 | HybridPolicy 架构（RF System-1 + deepseek System-2 + router） | 别人学"何时切换"，我们给出"何时切换**不值得一学**"的可判据（A(s)≤0 的实证签名） |
| 自知之明 | E2 信息充分性 A/B（分支实验） | 归因反转：不是"agent 不知道缺信息"，是"信息本来不缺"（P2 上限）——把问题从 agent 移到任务 |
| 主动获取 | Sara probe 对比；本项目 probe=零成本动作的成本二分 | 我们把计算动作/评估动作的成本语义纳入 utility 与前沿度量（Sara 只报告不优化） |

写 related work 的建议：按线 1+2 合并为"大小模型交互"一节（本项目是其在
序贯决策/BO 领域的实例 + 阴性对照），线 3+4 合并为"信息充分性与获取"一节
（E2 + probe 成本语义作为贡献点）。

## 引用纪律

- ✅ 条目：FrugalGPT、RouteLLM、Conformal Arbitrage (NeurIPS)、SwiftSage
  (NeurIPS 2023)、System1→System2 综述、Robots that Ask for Help
  (CoRL 2023)、经典 AIA 三篇、Sara/lenz。
- ⚠️ 条目（2026 preprint 居多：HiL-Bench、Agentic Abstention、C3PO、
  System-1.x、adaptive retrieval 各篇）：引用前必须核对 PDF 与 venue。
