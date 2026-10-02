"""Analyze the frozen attribution suite without choosing policies on test data."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from exp_attribution_suite import ACTIONS, FAMILIES, SAMPLING, OUT


def intervals(values, *, clusters=None, repetitions=6000):
    x = np.asarray(values, dtype=float)
    if not len(x):
        raise ValueError("empty uncertainty calculation")
    rng = np.random.default_rng(20260930)
    if clusters is None:
        samples = x[rng.integers(0, len(x), size=(repetitions, len(x)))].mean(axis=1)
    else:
        groups = {c: x[np.asarray(clusters) == c] for c in sorted(set(clusters))}
        keys = list(groups)
        samples = []
        for _ in range(repetitions):
            # Equal weight to instances; paired seeds are resampled within
            # each resampled instance. No row-wise pseudoreplication.
            chosen = rng.integers(0, len(keys), len(keys))
            samples.append(np.mean([rng.choice(groups[keys[i]], len(groups[keys[i]]), replace=True).mean() for i in chosen]))
        samples = np.asarray(samples)
    return {"mean": float(x.mean()), "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "ci99": np.quantile(samples, [0.005, 0.995]).tolist(), "n": len(x),
            "wins_ties_losses": [int((x > 1e-12).sum()), int((np.abs(x) <= 1e-12).sum()), int((x < -1e-12).sum())]}


def permutation_p(values):
    x = np.asarray(values, dtype=float)
    if not len(x) or np.all(np.abs(x) < 1e-12):
        return 1.0
    rng = np.random.default_rng(451)
    means = (rng.choice([-1, 1], size=(20000, len(x))) * x).mean(axis=1)
    return float((1 + (np.abs(means) >= abs(x.mean()) - 1e-14).sum()) / (1 + len(means)))


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    running, result = 0.0, {}
    for i, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[key]))
        result[key] = running
    return result


def seed_means(rows, values):
    grouped = defaultdict(list)
    for row, value in zip(rows, values):
        grouped[row["context"]["seed"]].append(value)
    return [float(np.mean(v)) for _, v in sorted(grouped.items())]


def vectors(row, profile, target, actions=SAMPLING):
    replicas = row["profiles"][profile]
    field = "composite" if target.endswith("composite") else "delta"
    horizon = "short" if target.startswith("short") else "terminal"
    return np.array([[rep[a][horizon][field] / (row["scale"] if field == "delta" else 1.0)
                      for a in actions] for rep in replicas], dtype=float)


def split_metrics(matrices):
    """One panel, two independent 2v2 replica splits. Ties reported separately."""
    array = np.stack(matrices)  # state, replica, action
    means = array.mean(axis=1)
    constant = int(means.mean(axis=0).argmax())
    descriptor = float(means.max(axis=1).mean() - means.mean(axis=0).max())
    gains, agreements, pair_flips, tie_shares = [], [], [], []
    for train_idx, test_idx in (((0, 1), (2, 3)), ((2, 3), (0, 1))):
        train = array[:, train_idx].mean(axis=1)
        test = array[:, test_idx].mean(axis=1)
        fixed = int(train.mean(axis=0).argmax())
        chosen = train.argmax(axis=1)
        gains.append(test[np.arange(len(test)), chosen] - test[:, fixed])
        agreements.append([float(test[i, chosen[i]] >= test[i].max() - 1e-12) for i in range(len(test))])
        flips, ties = [], []
        for i in range(len(array)):
            a, b = np.triu_indices(train.shape[1], 1)
            left, right = train[i, a] - train[i, b], test[i, a] - test[i, b]
            eligible = (np.abs(left) > 1e-12) & (np.abs(right) > 1e-12)
            flips.append(float((left[eligible] * right[eligible] < 0).mean()) if eligible.any() else None)
            ties.append(float(1.0 - eligible.mean()))
        pair_flips += [v for v in flips if v is not None]
        tie_shares += ties
    return {"D_descriptive": descriptor, "best_constant": SAMPLING[constant],
            "split_heldout_gains": np.mean(gains, axis=0).tolist(),
            "top_choice_test_optimal_share": float(np.mean(agreements)),
            "pair_flip_non_ties": float(np.mean(pair_flips)) if pair_flips else None,
            "pair_tie_share": float(np.mean(tie_shares))}


def label_analysis(rows, *, dimension=False):
    groups = defaultdict(list)
    for row in rows:
        c = row["context"]
        cell = f"f{c['fid']}/d{c['dim']}" if dimension else c["family"]
        groups[cell].append(row)
    output = {}
    for cell, group in sorted(groups.items()):
        summary = {"n_states": len(group), "n_seeds": len({r['context']['seed'] for r in group}),
                   "used_budget_counts": dict(Counter(r['state']['resource']['used_budget'] for r in group)),
                   "profiles": {}}
        for profile in group[0]["profiles"]:
            for target in ("short_composite", "short_objective", "terminal_objective"):
                if profile.startswith("legacy") and target.startswith("terminal"):
                    continue
                matrices = [vectors(r, profile, target) for r in group]
                metrics = split_metrics(matrices)
                gains = metrics.pop("split_heldout_gains")
                if dimension:
                    per_context = defaultdict(list)
                    for row, value in zip(group, gains):
                        c = row["context"]
                        per_context[(c["instance"], c["seed"])].append(value)
                    keys = sorted(per_context)
                    metrics["heldout_adaptive_gain"] = intervals([np.mean(per_context[k]) for k in keys], clusters=[k[0] for k in keys])
                else:
                    metrics["heldout_adaptive_gain"] = intervals(seed_means(group, gains))
                summary["profiles"][f"{profile}/{target}"] = metrics
        # On the same simulated paths, separate the reward definition from
        # lookahead length. Evaluate both choices on independent terminal reps.
        profile = "local/unit" if dimension else "rule/unit"
        change = {}
        for name, left_target, right_target in (
            ("horizon_terminal_minus_short", "terminal_objective", "short_objective"),
            ("score_short_objective_minus_composite", "short_objective", "short_composite"),
            ("joint_terminal_minus_short_composite", "terminal_objective", "short_composite"),
        ):
            values, agreement = [], []
            for row in group:
                left, right, terminal = (vectors(row, profile, t) for t in (left_target, right_target, "terminal_objective"))
                diffs = []
                for train_idx, test_idx in (((0, 1), (2, 3)), ((2, 3), (0, 1))):
                    a, b = int(left[list(train_idx)].mean(axis=0).argmax()), int(right[list(train_idx)].mean(axis=0).argmax())
                    test = terminal[list(test_idx)].mean(axis=0)
                    diffs.append(test[a] - test[b])
                    agreement.append(float(a == b))
                values.append(float(np.mean(diffs)))
            stats = intervals(seed_means(group, values)) if not dimension else {"mean": float(np.mean(values))}
            change[name] = {**stats, "train_choice_agreement": float(np.mean(agreement))}
        summary["target_changes"] = change
        if not dimension:
            agreement = []
            for row in group:
                local = vectors(row, "local/unit", "terminal_objective").mean(axis=0)
                rule = vectors(row, "rule/unit", "terminal_objective").mean(axis=0)
                agreement.append(float(local.argmax() == rule.argmax()))
            summary["local_rule_terminal_choice_agreement"] = float(np.mean(agreement))
        output[cell] = summary
    return output


def repeat_analysis(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["context"]["family"]].append(row)
    output = {}
    for family, group in sorted(groups.items()):
        output[family] = {}
        for profile in ("rule/unit", "local/unit"):
            array = np.stack([vectors(r, profile, "terminal_objective") for r in group])
            by_r = {}
            for n in (1, 2, 4, 8, 16):
                gains, agreement, flips = [], [], []
                for train_indices, eval_indices in ((list(range(n)), list(range(16, 32))),
                                                     (list(range(32-n, 32)), list(range(16)))):
                    train, test = array[:, train_indices].mean(axis=1), array[:, eval_indices].mean(axis=1)
                    selected, fixed = train.argmax(axis=1), int(train.mean(axis=0).argmax())
                    gains.append(test[np.arange(len(test)), selected] - test[:, fixed])
                    agreement.append([float(test[i, selected[i]] >= test[i].max() - 1e-12) for i in range(len(test))])
                left, right = array[:, :n].mean(axis=1), array[:, -n:].mean(axis=1)
                i, j = np.triu_indices(6, 1)
                for l, r in zip(left, right):
                    lg, rg = l[i] - l[j], r[i] - r[j]
                    eligible = (np.abs(lg) > 1e-12) & (np.abs(rg) > 1e-12)
                    if eligible.any():
                        flips.extend((lg[eligible] * rg[eligible] < 0).tolist())
                averaged = array[:, :n].mean(axis=1)
                by_r[str(n)] = {"heldout_gain": intervals(np.mean(gains, axis=0)),
                               "test_top_choice_agreement": float(np.mean(agreement)),
                               "pair_flip_non_ties": float(np.mean(flips)) if flips else None,
                               "D_selection_replicas_descriptive": float(averaged.max(axis=1).mean() - averaged.mean(axis=0).max())}
            output[family][profile] = by_r
    return output


def loop_analysis(rows, selections, *, dimension=False):
    lookup = {}
    for row in rows:
        c = row["context"]
        cell = f"f{c['fid']}/d{c['dim']}" if dimension else c["family"]
        k = (c.get("instance", 0), c["seed"])
        key = (cell, k, row["policy"])
        if key in lookup:
            raise ValueError("duplicate loop cell")
        lookup[key] = row
    groups = defaultdict(list)
    for row in rows:
        c = row["context"]
        cell = f"f{c['fid']}/d{c['dim']}" if dimension else c["family"]
        groups[cell].append(row)
    output = {}
    for cell, group in sorted(groups.items()):
        policies = sorted({r["policy"] for r in group})
        contrasts = {}
        pairs = [(p, selections[cell]["fixed"], f"{p}_minus_best_dev_fixed") for p in policies if p != selections[cell]["fixed"]]
        pairs += [(p, selections[cell]["schedule"], f"{p}_minus_frozen_schedule") for p in policies if p != selections[cell]["schedule"]]
        pairs += [("oracle_terminal_local", "oracle_short_local", "oracle_terminal_minus_short"),
                  ("normalized_surrogate", "surrogate_acquisition", "normalization_effect")]
        if not dimension:
            pairs.append(("rf_unit_terminal_posterior", "rf_unit_terminal_state", "posterior_visibility_effect"))
            for menu, mode in itertools.product(("original", "unit"), ("state", "time")):
                pairs.append((f"rf_{menu}_terminal_{mode}", f"rf_{menu}_short_{mode}", f"target_effect/{menu}/{mode}"))
            for menu, target in itertools.product(("original", "unit"), ("short", "terminal")):
                pairs.append((f"rf_{menu}_{target}_state", f"rf_{menu}_{target}_time", f"state_effect/{menu}/{target}"))
            for target, mode in itertools.product(("short", "terminal"), ("state", "time")):
                pairs.append((f"rf_unit_{target}_{mode}", f"rf_original_{target}_{mode}", f"menu_effect/{target}/{mode}"))
            for target in ("short", "terminal"):
                pairs.append((f"rf_original_{target}_state_noops", f"rf_original_{target}_state", f"zero_ops_mask/{target}"))
                pairs.append((f"rf_unit_{target}_state", f"rf_original_{target}_state_noops", f"batch_to_unit/{target}"))
        for left, right, name in pairs:
            lrows = [r for r in group if r["policy"] == left]
            if not lrows:
                continue
            diffs, ids = [], []
            for row in lrows:
                c = row["context"]
                k = (c.get("instance", 0), c["seed"])
                reference = lookup.get((cell, k, right))
                if reference is None:
                    raise ValueError(f"unpaired contrast {cell}/{left}/{right}")
                if row["initial_values"] != reference["initial_values"]:
                    raise ValueError("unequal initial evaluations")
                diffs.append(reference["normalized_regret"] - row["normalized_regret"] if dimension
                             else row["terminal_best"] - reference["terminal_best"])
                ids.append(c.get("instance", 0))
            result = intervals(diffs, clusters=ids if dimension else None)
            result.update(left=left, right=right, units="normalized regret reduction" if dimension else "raw objective gain")
            # Instance-level p-values only on the IOH panel; there are three
            # independent instances, so precision is deliberately limited.
            sign_values = [float(np.mean([v for v, i in zip(diffs, ids) if i == instance])) for instance in sorted(set(ids))] if dimension else diffs
            result["paired_sign_randomization_p"] = permutation_p(sign_values)
            contrasts[name] = result
        policy_summary = {}
        for policy in policies:
            q = [r for r in group if r["policy"] == policy]
            policy_summary[policy] = {
                "n": len(q), "mean_terminal_best": float(np.mean([r["terminal_best"] for r in q])),
                "mean_evaluations": float(np.mean([r["n_evaluations"] for r in q])),
                "mean_unused_budget": float(np.mean([r["unused_budget"] for r in q])),
                "mean_refits": float(np.mean([r["n_fits"] for r in q])),
                "mean_planning_evaluations": float(np.mean([r["planning_evaluations"] for r in q])),
                "action_counts": dict(Counter(a for r in q for a in r["actions"])),
            }
            if dimension:
                policy_summary[policy]["mean_normalized_regret"] = float(np.mean([r["normalized_regret"] for r in q]))
        output[cell] = {"controls": selections[cell], "contrasts": contrasts, "policies": policy_summary}
    # Correct the same predeclared contrast across families / dimension cells.
    names = {name for result in output.values() for name in result["contrasts"]}
    for name in names:
        entries = {cell: result["contrasts"][name]["paired_sign_randomization_p"] for cell, result in output.items() if name in result["contrasts"]}
        corrected = holm(entries)
        for cell, p in corrected.items():
            output[cell]["contrasts"][name]["holm_p"] = p
    return output


def pretty(stats):
    return f"{stats['mean']:+.4f} [{stats['ci95'][0]:+.4f}, {stats['ci95'][1]:+.4f}]"


def dimension_interactions(rows, controls):
    lookup = {(r['context']['fid'], r['context']['dim'], r['context']['instance'], r['context']['seed'], r['policy']): r for r in rows}
    output = {}
    for fid in (6, 8, 15):
        output[str(fid)] = {}
        keys = sorted({(r['context']['instance'], r['context']['seed']) for r in rows if r['context']['fid'] == fid})
        for name in ('reference_opportunity', 'normalization_effect'):
            for dim in (5, 10):
                differences = []
                for instance, seed in keys:
                    def effect(d):
                        left = 'oracle_terminal_local' if name == 'reference_opportunity' else 'normalized_surrogate'
                        right = controls[f'f{fid}/d{d}']['fixed'] if name == 'reference_opportunity' else 'surrogate_acquisition'
                        l = lookup[(fid, d, instance, seed, left)]
                        r = lookup[(fid, d, instance, seed, right)]
                        return r['normalized_regret'] - l['normalized_regret']
                    differences.append(effect(dim) - effect(2))
                output[str(fid)][f'{name}/d{dim}_minus_d2'] = intervals(differences, clusters=[i for i, _ in keys])
    return output


def make_report(result, folder):
    lines = ["# 归因实验报告 — 2026-09-30", "", "主实验配置预先冻结；重复次数和 posterior 补充组根据代码与开发集诊断另行冻结，未用于调选主实验测试策略。历史结果不覆盖。下面的区间默认是配对 95% bootstrap；分析 JSON 另含 99% 区间和跨族 Holm p 值。无显著收益不等于收益为零。", "",
             f"- 重算 BOCoDe 状态：{result['counts']['label_states']}，每状态四次模拟；训练种子 0–19。",
             f"- BOCoDe 开发控制：{result['counts']['controls']} 条；新种子 1000–1019 闭环：{result['counts']['loops']} 条。",
             f"- IOH 开发控制：{result['counts']['dim_controls']} 条；维度交叉闭环：{result['counts']['dim_loops']} 条；重算状态：{result['counts']['dim_label_states']}。", "",
             "参考策略知道测试函数的模拟结果，规划开销不计入 30 次真实评估，因此只用于诊断可利用机会，不能当作可部署方法或数学最优策略。rule 延续参考只测五个新种子，属于小样本辅助诊断。", "",
             "## 1. 标签：区分 horizon、效用与噪声", "",
             "这里只比较六个可采样动作；STOP/refit 不混入固定采样上界。D 是四次模拟均值上的描述量，不能充当闭环终局上界。held-out gain 用两个 replica 选动作，再用另两个评价，并反向交换平均。", "",
             "| 函数 | 旧 h3 D | rule h3 D | rule terminal D | terminal held-out gain | rule/local 首动作一致率 |", "|---|---:|---:|---:|---|---:|"]
    for f in FAMILIES:
        s = result["labels"][f]
        p = s["profiles"]
        lines.append(f"| {f} | {p['legacy/original/short_composite']['D_descriptive']:.4f} | {p['rule/unit/short_composite']['D_descriptive']:.4f} | {p['rule/unit/terminal_objective']['D_descriptive']:.4f} | {pretty(p['rule/unit/terminal_objective']['heldout_adaptive_gain'])} | {s['local_rule_terminal_choice_agreement']:.1%} |")
    lines += ["", "| 函数 | 延长 horizon 的 held-out terminal 改善 | 仅去成本项的改善 | terminal teacher 2v2 非 tie pair flip |", "|---|---|---|---:|"]
    for f in FAMILIES:
        s = result["labels"][f]
        lines.append(f"| {f} | {pretty(s['target_changes']['horizon_terminal_minus_short'])} | {pretty(s['target_changes']['score_short_objective_minus_composite'])} | {s['profiles']['rule/unit/terminal_objective']['pair_flip_non_ties']:.1%} |")
    lines += ["", "## 2. 闭环：模型、动作、状态信息", "",
              "控制动作和 schedule 仅在开发种子上选择，不在测试结果上挑最强对手。主表对照各族冻结的最佳固定采样动作。", "",
              "| 函数 | 开发最佳固定动作 | 旧 RF | unit terminal state RF | terminal local 参考 | normalized LogEI − raw LogEI |", "|---|---|---|---|---|---|"]
    for f in FAMILIES:
        s = result["loops"][f]
        c = s["contrasts"]
        lines.append(f"| {f} | {s['controls']['fixed']} | {pretty(c['rf_legacy_minus_best_dev_fixed'])} | {pretty(c['rf_unit_terminal_state_minus_best_dev_fixed'])} | {pretty(c['oracle_terminal_local_minus_best_dev_fixed'])} | {pretty(c['normalization_effect'])} |")
    for name, title in (("target_effect/unit/state", "terminal target − h3 composite target (unit, full state)"),
                        ("state_effect/unit/terminal", "full state − budget only (unit, terminal target)"),
                        ("zero_ops_mask/short", "去掉 refit/STOP，同一短期 RF 权重"),
                        ("batch_to_unit/terminal", "unit sampling − batch sampling (terminal full-state RF)"),
                        ("oracle_terminal_minus_short", "terminal reference − h3 composite reference (local continuation)")):
        lines += ["", f"### {title}", "", "| 函数 | 配对 terminal 改善 | Holm p |", "|---|---|---:|"]
        for f in FAMILIES:
            c = result["loops"][f]["contrasts"][name]
            lines.append(f"| {f} | {pretty(c)} | {c['holm_p']:.4g} |")
    lines += ["", "## 3. 同一函数内的维度交叉", "", "IOH 使用三个全新实例，每实例五个种子；区间先抽实例再抽种子。三实例的统计分辨率有限。正值表示比冻结开发最佳固定策略减少了 normalized regret。", "",
              "| IOH 函数/维度 | 开发最佳固定动作 | terminal local 参考 − 固定 | terminal − short 参考 | GP 输入归一化效果 |", "|---|---|---|---|---|"]
    for cell, s in result["dimension_loops"].items():
        c = s["contrasts"]
        lines.append(f"| {cell} | {s['controls']['fixed']} | {pretty(c['oracle_terminal_local_minus_best_dev_fixed'])} | {pretty(c['oracle_terminal_minus_short'])} | {pretty(c['normalization_effect'])} |")
    lines += ["", "以下是同实例/同种子的维度交互：d10 中的效果减去 d2 中的效果。", "",
              "| IOH 函数 | 参考选择机会 d10−d2 | 归一化修复效果 d10−d2 |", "|---|---|---|"]
    for fid, s in result['dimension_interactions'].items():
        lines.append(f"| f{fid} | {pretty(s['reference_opportunity/d10_minus_d2'])} | {pretty(s['normalization_effect/d10_minus_d2'])} |")
    lines += ["", "## 4. 补充归因：重复次数与 posterior 特征", "",
              "32-replica 组只使用预先选定的开发初始状态，每族六个种子。R 个 replica 选动作，固定的独立 16 个 replica 评价，再交换两半。主实验模型与测试策略未随结果调整。", "",
              "| 函数 | rule terminal pair flip R=1 | R=16 | R=16 held-out gain | posterior 可见 RF − 原 full-state RF |", "|---|---:|---:|---|---|"]
    for family in FAMILIES:
        reps = result["repeat32"][family]["rule/unit"]
        effect = result["loops"][family]["contrasts"]["posterior_visibility_effect"]
        lines.append(f"| {family} | {reps['1']['pair_flip_non_ties']:.1%} | {reps['16']['pair_flip_non_ties']:.1%} | {pretty(reps['16']['heldout_gain'])} | {pretty(effect)} |")
    lines += ["", "posterior 对照保持标签、RF 超参数、六个采样动作与原 raw-input GP 不变，在训练和部署决策前先拟合 GP，再读取原有 15 字段。它增加了拟合计算量，真实目标评估数仍为 30。只检验当前摘要字段的信息可见性。", ""]
    lines += ["", "## 5. 边界与复现", "",
              "- 两种 action menu 是完整处理包；桥接 mask 对照帮助拆分 zero-evaluation 操作与 batching。",
              "- 全状态 RF 只使用现有 15 个摘要特征；null contrast 不证明完整数据集/GP posterior 无信息。",
              "- 标签状态来自原行为轨迹，部分种子早停；报告中保留实际评估数分布，不能把三个目标检查点当作三个均匀阶段。",
              "- 大多数基础对照保持原 GP 实现。归一化是独立处理，未应用到 RF 或参考策略，以免同时改变多个因素。",
              "- interval crossing zero 不作为没有价值的证明；小样本 rule 参考与三实例 IOH 面板均需谨慎。", "",
              "- 冻结协议前曾用 SixHumpCamel/seed1000 的旧 runner 做一次耗时估算；未据它选策略或超参数。分析 JSON 另给出排除该初始种子的敏感性结果。", "",
              "配置：`docs/ATTRIBUTION_PROTOCOL_20260930.md`。运行器：`scripts/exp_attribution_suite.py`。分析：`scripts/analyze_attribution_suite.py`。",
              "所有原始结果、模型、来源哈希、逐步值与模拟成本保存在 `runs/attribution_20260930/`。"]
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n")


def figures(result, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    names = ("target_effect/unit/state", "state_effect/unit/terminal", "normalization_effect")
    titles = ("Terminal vs short target", "State vs budget-only features", "Normalized vs raw GP inputs")
    for ax, name, title in zip(axes, names, titles):
        values = [result["loops"][f]["contrasts"][name] for f in FAMILIES]
        scale = np.array([result["scales"][f] for f in FAMILIES])
        means = np.array([v['mean'] for v in values]) / scale
        lo = np.array([v['ci95'][0] for v in values]) / scale
        hi = np.array([v['ci95'][1] for v in values]) / scale
        ax.errorbar(means, np.arange(5), xerr=[means-lo, hi-means], fmt="o", capsize=3)
        ax.axvline(0, color="gray", linewidth=1)
        ax.set_yticks(np.arange(5), FAMILIES)
        ax.set_title(title)
        ax.set_xlabel("Paired terminal gain / dev family scale")
    fig.savefig(folder / "attribution_effects.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, fid in zip(axes, (6, 8, 15)):
        vals = [result["dimension_loops"][f"f{fid}/d{d}"]["contrasts"]["oracle_terminal_local_minus_best_dev_fixed"] for d in (2, 5, 10)]
        means = np.array([v['mean'] for v in vals])
        lo, hi = (np.array([v['ci95'][i] for v in vals]) for i in (0, 1))
        ax.errorbar([2, 5, 10], means, yerr=[means-lo, hi-means], fmt="o-", capsize=3)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_title(f"IOH f{fid}")
        ax.set_xticks([2, 5, 10])
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Privileged reference regret reduction")
    fig.savefig(folder / "dimension_effects.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=OUT)
    args = parser.parse_args()
    folder = args.folder
    stages = {name: json.loads((folder / f"{name}.json").read_text()) for name in
              ("labels", "loops", "controls", "dim_labels", "dim_loops", "dim_controls")}
    for name, data in stages.items():
        keys = [r["key"] for r in data["runs"]]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate task keys in {name}")
        for key in ("dataset_sha256", "script_sha256", "protocol_sha256", "packages"):
            if data["config"][key] != stages["labels"]["config"][key]:
                raise ValueError(f"mixed provenance for {key}")
    expected = {'labels': 280, 'controls': 1000, 'loops': 2125,
                'dim_controls': 1350, 'dim_labels': 135, 'dim_loops': 1350}
    for name, n in expected.items():
        if len(stages[name]['runs']) != n:
            raise ValueError(f'incomplete {name}: expected {n} tasks')
    labels = [r for task in stages["labels"]["runs"] for r in task["records"]]
    dim_labels = [r for task in stages["dim_labels"]["runs"] for r in task["records"]]
    selections = json.loads((folder / "controls_selection.json").read_text())
    dim_selections = json.loads((folder / "dim_controls_selection.json").read_text())
    repeats = json.loads((folder / "repeat32.json").read_text())
    posterior = json.loads((folder / "posterior_loops.json").read_text())
    labels_hash = hashlib.sha256((folder / "labels.json").read_bytes()).hexdigest()
    for name, data, label_key in (("repeat32", repeats, "source_labels_sha256"),
                                  ("posterior_loops", posterior, "labels_sha256")):
        keys = [r["key"] for r in data["runs"]]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate task keys in {name}")
        for key in ("dataset_sha256", "script_sha256", "protocol_sha256", "packages"):
            if data["config"][key] != stages["labels"]["config"][key]:
                raise ValueError(f"mixed supplementary provenance for {key}")
        if data["config"][label_key] != labels_hash:
            raise ValueError(f"changed source labels for {name}")
    if len(repeats["runs"]) != 30 or len(posterior["runs"]) != 100:
        raise ValueError("incomplete supplementary panel")
    features = json.loads((folder / "posterior_training_features.json").read_text())
    if {r['id'] for r in features} != {r['id'] for r in labels}:
        raise ValueError("posterior feature panel differs from primary training states")
    for row in features:
        if row['features'][11] != 1.0:
            raise ValueError("posterior features not current")
    label_by_id = {r['id']: r for r in labels}
    for row in features:
        source = label_by_id[row['id']]
        for action in SAMPLING:
            expected_y = np.mean([rep[action]['terminal']['delta']/source['scale'] for rep in source['profiles']['rule/unit']])
            if not math.isclose(row['targets'][action], expected_y, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('posterior intervention changed the supervision target')
    all_loop_rows = stages['loops']['runs'] + posterior['runs']
    result = {
        "counts": {name: len(data["runs"]) for name, data in stages.items()},
        "scales": stages["labels"]["config"]["scales"],
        "artifacts_sha256": {name: hashlib.sha256((folder / f"{name}.json").read_bytes()).hexdigest()
                             for name in (*stages, "repeat32", "posterior_loops", "posterior_training_features")},
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "labels": label_analysis(labels),
        "loops": loop_analysis(all_loop_rows, selections),
        "dimension_labels": label_analysis(dim_labels, dimension=True),
        "dimension_loops": loop_analysis(stages["dim_loops"]["runs"], dim_selections, dimension=True),
        "repeat32": repeat_analysis(repeats["runs"]),
        "dimension_interactions": dimension_interactions(stages['dim_loops']['runs'], dim_selections),
        "sensitivity_excluding_timing_pilot": loop_analysis([r for r in all_loop_rows if not (r['context']['family'] == 'sixhumpcamel' and r['context']['seed'] == 1000)], selections),
    }
    result["counts"].update(label_states=len(labels), dim_label_states=len(dim_labels))
    result["counts"].update(repeat32_states=len(repeats["runs"]), posterior_loops=len(posterior["runs"]))
    (folder / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    make_report(result, folder)
    figures(result, folder)
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
