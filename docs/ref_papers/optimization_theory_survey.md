# 优化理论近5年（2022–2026）顶会顶刊调研 — Related Work 扩展

> 检索日期 2026-09-28/29。初版经 web 检索核实；2026-09-29 起逐篇下载 PDF
> 并用 pdftotext 核对标题/作者（**本地 PDF 库：`docs/ref_papers/pdfs/`，
> 26 个文件**）。核实标记：✅ = 标题/作者/venue 已对 PDF 或官方页面确认；
> ⚠️ = 待对 PDF。**本次核对纠正了初版的四处记录错误**（见各条 💢 标记与
> 文末"核对纠正记录"）。

## 线 A：采集函数理论 — 信息论族与改进量族的推进

| 论文 | venue | PDF | 核心内容 | 与本项目对接 |
|---|---|---|---|---|
| Hvarfner, Hutter, Nardi, *Joint Entropy Search for Maximally-Informed Bayesian Optimization* (arXiv:2206.04771) | NeurIPS 2022 | ✅ `arxiv_2206.04771.pdf` | 定价对象从 X\* 的熵升级为 (x\*, f\*) 的**联合熵**，MC 估计器实用化 | P2 划界句的当代延伸：JES 仍在"熵减关于最优解"泛函族内；D 是效用单位、关于状态标签 |
| Ament, Daulton, Eriksson, Balandat, Bakshy, *Unexpected Improvements to Expected Improvement for Bayesian Optimization* (LogEI/LogPI, arXiv:2310.20708) | NeurIPS 2023, arXiv:2310.20708 | ✅ `arxiv_2310.20708.pdf` | EI/PI 低噪声/近优区数值退化 → 对数变换修复；进 BoTorch | **必引直接依赖**：backend 的 acquisition 就是 `LogExpectedImprovement`；与 G5 教训同构（数值健康度足以制造/掩盖信号） |
| Tu, Gandy, Kantas, Shafei, *Joint Entropy Search for Multi-Objective Bayesian Optimization* (arXiv:2210.02905) | NeurIPS 2022 | ✅ `arxiv_2210.02905.pdf` | 同名不同文：多目标 JES（BoTorch 实现） | 消歧记录（引 JES 勿混两文） |

## 线 B：谁来定规则 — 权重与组合分配的自动化

> 💢 **初版记录纠正**：SAWEI 不在 AAAI；"WEI/WU AISTATS 2022 (PMLR
> v151)" 一条**删除**（未能核实，且被下方已核实的三段谱系取代）。

**已核实的谱系（PI-is-back → Towards-SAWEI → SAWEI）**：

| 论文 | venue | PDF | 核心内容 |
|---|---|---|---|
| Benjamins, Raponi, Doerr et al., *PI is back! Switching Acquisition Functions in Bayesian Optimization* (arXiv:2211.01455) | NeurIPS 2022 workshop (Gaussian Processes & Decision-making Systems) | ✅ `arxiv_2211.01455.pdf` | 重新为 PI 正名：按数据质量在 PI/EI 间**切换**优于任一固定采集 |
| Benjamins, Raponi, Jankovic, Thierens, Doerr, *Towards Self-Adjusting Weighted Expected Improvement* | GECCO 2023 Companion | ⚠️（HAL 有开放版，未下载） | 加权 EI 的 λ 在线自调（短版） |
| Benjamins, Raponi, Jankovic, Doerr, Lindauer, *Self-Adjusting Weighted Expected Improvement for Bayesian Optimization* | **AutoML 2023 (PMLR v224)** | ✅ `pmlr_v224_benjamins23a_sawei.pdf` | λ 由模型质量/平稳性驱动自调，无手动调参 |

**与本项目对接**：GP-Hedge→PI-is-back/SAWEI→LMABO 是"把规则的一个自由度
交给在线适应"的脉络；我们的封闭 8 动作空间是它的一般化，P3 给出退化判据。

| Vasconcelos, Souza, Virgolino, Mattos, Gomes, *Self-tuning portfolio-based Bayesian optimization* | **Expert Systems with Applications 188:115847, 2022**（无 arXiv 版；Elsevier 付费墙） | ❌ | GP-Hedge 后继：组合权重逐实例自调 | 同上脉络；引用时按期刊格式 |

> 💢 **初版记录纠正**：Vasconcelos 一条原记 "arXiv 2204.03846"——该编号
> 实为纽结理论论文（Jones 多项式），已删除错误文件。正确出处为 ESWA
> 期刊（上表），经 citing 文献与 Semantic Scholar 双重确认。

## 线 C：停止理论（本项目最应补引的一条线）

| 论文 | venue | PDF | 核心内容 | 与本项目对接 |
|---|---|---|---|---|
| Dai, Yu, Low, Jaillet, *Bayesian Optimization Meets Bayesian Optimal Stopping* (BO-BOS) | ICML 2019（窗口外锚点） | ✅ `pmlr_v97_dai19a.pdf` | GP-UCB × 贝叶斯最优停止，incumbent 上的最优停时 | STOP 的形式化起点 |
| Wilson, *Stopping Bayesian Optimization with Probabilistic Regret Bounds* (arXiv:2402.16811) | NeurIPS 2024 | ✅ `arxiv_2402.16811.pdf` | (ε,δ) 准则：以"找到的点以概率 ≥1−δ 落在最优 ε 邻域"替代跑满预算；证明正确终止 | **STOP-as-typed-action 最近理论对照**：我们的 stop = 封闭动作空间的成员动作 + utility 停机项 + Gate 4 stop-gain（5/5 家族、过 Bonferroni）——stop-gain 是这类准则在反事实数据集上的对应物 |
| Xie, Cai, Terenin, Frazier, Scully, *Cost-aware Stopping for Bayesian Optimization* (arXiv:2507.12453) | arXiv 2025-07 | ✅ `arxiv_2507.12453.pdf` | 停止规则 × 变评估成本：期望累计评估成本有界 | 成本×停止交叉点，RQ3 相关 |

## 线 D：GP-bandit 遗憾界 — x 层理论的密度

| 论文 | venue | PDF | 核心内容 |
|---|---|---|---|
| Iwazaki, Takeno, Tanabe, Irie, *Failure-Aware Gaussian Process Optimization with Regret Bounds* (F-GP-UCB) | NeurIPS 2023 | ✅ `fagpucb_neurips23.pdf` | 臂可失效设定：排除失效点邻域 + 首个遗憾上界 |
| Zi Wang, Dahl, Swersky, Lee, Nado, Gilmer, Mariet, …, Ghahramani, *Pre-trained Gaussian Processes for Bayesian Optimization* (HyperBO, arXiv:2109.08215) | **JMLR 25(212):1–83, 2024** | ✅ `jmlr_v25_212_hyperbo.pdf` | 跨任务预训练 GP（学核参数分布），无核调参的 BO |
| Iwazaki, *Improved Regret Bounds for Gaussian Process Upper Confidence Bound in Bayesian Optimization* (arXiv:2506.01393) | NeurIPS 2025 (**oral**) | ✅ `arxiv_2506.01393.pdf` | Matérn 光滑性下 GP-UCB 达 Õ(√T) 近最优遗憾 |
| Iwazaki, *Gaussian Process Upper Confidence Bound Achieves Nearly-Optimal Regret in Noise-Free Gaussian Process Bandits* | NeurIPS 2025 | ✅ `neurips25_iwazaki_noisefree_gpucb.pdf`（作者单位：LY Corporation） | 无噪声设定近极小极大率 |
| Takeno, Iwazaki, *Improved Regret Analysis for Parallel Gaussian Process Bandit Optimization* (arXiv:2608.16492) | arXiv 2026-08 | ✅ `arxiv_2608.16492.pdf` | 并行/批量设定遗憾界收紧 |

> 💢 **初版记录纠正**：HyperBO 一条原记 "NeurIPS 2024"——正式出处是
> JMLR 25(212) 2024（NeurIPS virtual 站上的是同名混淆项），已改。
> （注：virtual.neurips.cc 2024 poster/98319 指向的是另一篇，检索引擎混淆。）

**对接（定位句素材）**：x 层无遗憾理论近5年持续加密（极小极大率、失败感知、
并行、预训练先验），但策略层（stop/refit/采样策略切换）既无闭式规则也无逐
决策监督——本文补的是后者。§2 最有力的"层间划界"论据。

## 线 E：学习/摊销 BO 机制 — 学后验、学先验、学空间

| 论文 | venue | PDF | 核心内容 | 与本项目对接 |
|---|---|---|---|---|
| Müller, Hollmann, Pineda-Arango, Grabocka, Hutter, *Transformers Can Do Bayesian Inference* (PFN, arXiv:2112.10510) | **ICLR 2022** | ✅ `arxiv_2112.10510.pdf` | 先验数据拟合网络：单次前向摊销贝叶斯推断（TabPFN 引擎） | 分野：这条线学**推断机器**，本项目学**元决策**；两层正交可组合 |
| Rothfuss et al., *Meta-Learning Priors for Safe Bayesian Optimization* | ICML 2023 (PMLR v205) | ✅ `pmlr_v205_rothfuss23a.pdf` | 函数空间元学习先验加速安全 BO | RQ2 对照线：先验迁移 vs 决策标签迁移 |
| Shukuan Wang, Xue, Song, Huang, Qian, *Monte Carlo Tree Search based Space Transfer for Black-box Optimization* (arXiv:2412.07186) | NeurIPS 2024 | ✅ `arxiv_2412.07186.pdf` | MCTS 迁移搜索空间（最优解空间 warm start） | 空间迁移线（南大 LAMDA） |
| Jiarong Pan, Falkner, Berkenkamp, Vanschoren, *MALIBO: Meta-learning for Likelihood-free Bayesian Optimization* (arXiv:2307.03565) | arXiv 2023 | ✅ `arxiv_2307.03565.pdf` | 绕过代理直接学查询效用（meta-BO） | 与 utility 直学表面相似——引用前细读划界 |

锚点（库内已有，PDF 均已入库）：Volpp ICLR 2021、TempoRL/DAC ICML 2021、
LLAMBO（`arxiv_2402.03921.pdf`）、FunBO（`arxiv_2406.04824.pdf`）。

## 线 F：成本感知 BO — 成本对象的划界

| 论文 | venue | PDF | 核心内容 |
|---|---|---|---|
| Chen, Ament, Eriksson, Balandat, Low, Bakshy, Lin, *Efficiently Estimating Optimal Hyperparameter Scaling Laws through Power-Law Entropy Search* (PLES, arXiv:2609.01431) | arXiv 2026-09 | ✅ `arxiv_2609.01431.pdf` | 计算成本感知的多保真熵采集（缩放律估计；Meta + NUS） |
| 多保真 cost-aware BO | CMAME 403:115937, 2023 | ❌（付费墙） | 保真度×成本联合调度 |
| EI-per-cost 实践族（EIperSec 等） | （综述见 CAPS） | ⚠️ | 评估成本归一化采集 |

**对接（RQ3 划界句）**：这条线的成本对象是**评估成本**（保真度/秒）；本项目
的成本对象是**推理成本**（LLM 调用/token）进闭环非劣效前沿。同类思想、不同
成本项，一句话划清。

## 线 G：LLM-as-optimizer 顶会线（补齐两个锚）

| 论文 | venue | PDF | 核心内容 | 与本项目对接 |
|---|---|---|---|---|
| Yang, Wang, Lu, …, Chen, Liu, *Large Language Models as Optimizers* (OPRO, arXiv:2309.03409) | ICLR 2024 | ✅ `arxiv_2309.03409.pdf` | 轨迹全留自然语言，LLM 迭代生成候选；GSM8K 提示超越人工设计 | "委托"路线顶会锚：x/prompt 层、无过程监督 |
| Romera-Paredes et al., *Mathematical discoveries from program search with large language models* (FunSearch) | Nature（在线 2023-12，印刷 2024） | ✅ `funsearch_nature_2023.pdf`（开放获取） | LLM 提议程序 + 评估器进化：cap set 新下界、bin-packing 启发式 | 程序空间层 Nature 锚；与 FunBO 同线 |
| *GPTOpt: Towards Efficient LLM-Based Black-Box Optimization* (arXiv:2510.25404) | arXiv 2025（**v3 改名 SemanticOpt**） | ✅ `arxiv_2510.25404.pdf`（v3） | 微调 LLM 直接做连续黑盒优化，BBOB 基准 | LLM-as-optimizer 参照系；meta 层不同层 |

> 💢 **改名记录**：arXiv:2510.25404 v1 标题即 GPTOpt（已对 arXiv v1 页面
> 确认），v3 改名 *SemanticOpt: Towards LLM-Based Semantic Black-Box
> Optimization*（同一论文）。引用时写 "GPTOpt (renamed SemanticOpt in
> later versions)" 或引 v1，避免审稿人对不上号。

## 教科书锚点

- ✅ Garnett, *Bayesian Optimization*, Cambridge University Press, 2023
  （`garnett_bayesoptbook.pdf`，作者站免费 PDF）。记号与 "BO = 序贯决策"
  标准表述来源。

## 对 §2 Related Work 的整合建议（英文草稿，可直接改造）

在现有 "Decision-theoretic and classical-BO foundations" 段之后接：

> Recent theory has intensified on the *x-space* layer. Joint entropy search
> prices candidates by the entropy of the joint (x\*, f\*) (Hvarfner et al.,
> NeurIPS 2022); log-transformed improvement repairs EI's numerical
> degeneracy and is the acquisition our own backend runs (Ament et al.,
> NeurIPS 2023). Stopping — one of the eight typed actions — has
> been formalized as Bayesian optimal stopping (Dai et al., ICML 2019) and
> as probabilistic-regret termination with correctness guarantees (Wilson,
> NeurIPS 2024); our Gate-4 stop-gain is the dataset-side counterpart of
> these criteria, measuring STOP's terminal advantage directly on
> counterfactual labels. On the allocation side, the portfolio line has
> taught its free parameters to adapt — switching PI against EI by data
> quality (Benjamins et al., 2022) and self-adjusting weighted EI
> (Benjamins et al., AutoML 2023) — continuing from GP-Hedge to LMABO's
> zero-shot LLM selector. Meanwhile the GP-bandit regret program keeps
> tightening (failure-aware bounds, Iwazaki et al., NeurIPS 2023;
> near-minimax noise-free rates, Iwazaki, NeurIPS 2025), and a parallel
> line amortizes the inference machinery itself — prior-fitted
> transformers (Müller et al., ICLR 2022), meta-learned safe-BO priors
> (Rothfuss et al., ICML 2023), transferred search spaces (Wang et al.,
> NeurIPS 2024). All of this theory is about *which x to evaluate* or *how
> to infer the posterior*; the strategy layer above it — stop, refit,
> switch sampling strategy — has neither closed-form rules nor per-decision
> supervision. That is the layer this paper instruments, and none of the
> LLM-era systems (OPRO, ICLR 2024; FunSearch, Nature 2023; LLAMBO;
> GPTOpt; Sara/lenz; LMABO) change this: they replace the decision-maker,
> not create the measurability.

## 本地 PDF 库清单（`docs/ref_papers/pdfs/`，26 文件，均已 pdftotext 验证）

JES 2206.04771 · MO-JES 2210.02905 · LogEI 2310.20708 · PI-is-back 2211.01455 ·
SAWEI PMLR v224 · BO-BOS PMLR v97 · ProbRegretStopping 2402.16811 ·
CostAwareStopping 2507.12453 · F-GP-UCB (NeurIPS'23 proceedings.com) ·
HyperBO JMLR 25(212) · GP-UCB-improved 2506.01393 · noise-free GP-UCB
(NeurIPS'25) · parallel-GP-bandit 2608.16492 · PFN 2112.10510 ·
Rothfuss PMLR v205 · MCTS-transfer 2412.07186 · MALIBO 2307.03565 ·
PLES 2609.01431 · OPRO 2309.03409 · FunSearch (Nature OA) ·
GPTOpt/SemanticOpt 2510.25404 · LLAMBO 2402.03921 · FunBO 2406.04824 ·
Sara/lenz 2608.00316 · LMABO 2602.07904 · Garnett 教科书。

**仍无 PDF（付费墙，引用按期刊格式）**：Vasconcelos ESWA 188:115847 (2022)、
CMAME 403:115937 (2023)；**次要缺口**：GECCO'23 Towards-SAWEI 短版（HAL
开放，未下载——正式引用用 AutoML'23 长版即可）。

## 核对纠正记录（初版 → 现在）

1. Vasconcelos：~~arXiv:2204.03846~~（实为纽结理论文）→ ESWA 188:115847。
2. SAWEI：~~AAAI 2023~~ → AutoML 2023 (PMLR v224)；谱系前置 PI-is-back
   (arXiv:2211.01455, NeurIPS'22 workshop) 与 GECCO'23 短版。
3. "WEI/WU, AISTATS 2022, PMLR v151"：未能核实，整条删除（被谱系取代）。
4. HyperBO：~~NeurIPS 2024~~ → JMLR 25(212), 2024。
5. GPTOpt：v3 改名 SemanticOpt（v1 标题已核实为 GPTOpt）。

## 引用纪律（更新）

- ✅ 可直接进正文：上表全部 ✅ 行（26 篇 PDF 在库）。
- ⚠️ 引用前仍需处理：GECCO'23 短版（HAL）、EIperSec 族（综述二手）。
- ❌ 付费墙（按期刊格式引用，不必有 PDF）：Vasconcelos ESWA、CMAME。
- 作者名单全部以 PDF 第一页为准（本次已逐一提取）；venue 以官方
  proceedings 页面为准。
