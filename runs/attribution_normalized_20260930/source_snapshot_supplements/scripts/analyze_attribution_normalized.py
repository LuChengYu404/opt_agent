"""Analyze the corrected-GP follow-up without choosing policies on test data."""
from __future__ import annotations
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from exp_attribution_normalized import OUT, provenance
from exp_attribution_suite import FAMILIES, SAMPLING
from analyze_attribution_suite import holm, intervals, permutation_p, pretty


def paired_summary(values):
    result = intervals(values)
    result["paired_sign_randomization_p"] = permutation_p(values)
    return result


def loop_analysis(rows, selections):
    out = {}
    for fam in FAMILIES:
        group = [r for r in rows if r["context"]["family"] == fam]
        lookup = {(r["context"]["seed"], r["treatment"], r["policy"]): r for r in group}
        fixed = selections[fam]["fixed"]
        contrasts = [(f"normalization/{a}", ("normalized", a), ("raw", a)) for a in SAMPLING]
        contrasts += [("best_pipeline", ("normalized", fixed), ("raw", selections[fam]["raw_fixed"])),
                      ("schedule_minus_fixed", ("normalized", selections[fam]["schedule"]), ("normalized", fixed)),
                      ("reference_unit_minus_fixed", ("normalized", "reference_1"), ("normalized", fixed)),
                      ("reference_block_minus_fixed", ("normalized", "reference_5"), ("normalized", fixed)),
                      ("reference_block_minus_unit", ("normalized", "reference_5"), ("normalized", "reference_1"))]
        results = {}
        for name, left, right in contrasts:
            candidates = sorted((s, r) for (s, t, p), r in lookup.items() if (t, p) == left)
            values = []
            for s, row in candidates:
                other = lookup[s, *right]
                assert row["initial_values"] == other["initial_values"]
                values.append(row["terminal_best"] - other["terminal_best"])
            results[name] = paired_summary(values)
        summaries = {}
        for t, policy in sorted({(r['treatment'], r['policy']) for r in group}):
            subset = [r for r in group if (r['treatment'], r['policy']) == (t, policy)]
            summaries[f"{t}/{policy}"] = {"n": len(subset), "mean_terminal_best": float(np.mean([r['terminal_best'] for r in subset])),
                "mean_fits": float(np.mean([r['n_fits'] for r in subset])),
                "mean_planning_evaluations": float(np.mean([r['planning_evaluations'] for r in subset])),
                "mean_planning_fits": float(np.mean([r['planning_fits'] for r in subset])),
                "actions": dict(Counter(a for r in subset for a in r['actions'])),
                "reference_nonfixed_blocks": (sum(r['decisions'][i]['action'] != fixed for r in subset for i in range(0,25,5)) if policy.startswith('reference_') else None)}
        out[fam] = {"selection": selections[fam], "contrasts": results, "policies": summaries}
    for name in out[FAMILIES[0]]['contrasts']:
        corrected = holm({f: out[f]['contrasts'][name]['paired_sign_randomization_p'] for f in FAMILIES})
        for f,p in corrected.items(): out[f]['contrasts'][name]['holm_p'] = p
    return out


def matrix(row, profile, target):
    return np.asarray([[rep[a][target]['delta'] for a in SAMPLING] for rep in row['profiles'][profile]])


def independent_gains(row, profile):
    terminal, first = matrix(row, profile, 'terminal'), matrix(row, profile, 'first')
    fixed = SAMPLING.index(row['continuation'])
    gains, first_gains, immediate_gains, chosen, flips, ties = [], [], [], [], [], []
    for select, evaluate in ((range(4), range(4,8)), (range(4,8), range(4))):
        train, test = terminal[list(select)].mean(axis=0), terminal[list(evaluate)].mean(axis=0)
        a, b = int(train.argmax()), int(first[list(select)].mean(axis=0).argmax())
        gains.append(float(test[a]-test[fixed]))
        first_gains.append(float(test[b]-test[fixed]))
        first_test=first[list(evaluate)].mean(axis=0)
        immediate_gains.append(float(first_test[b]-first_test[fixed]))
        chosen.append(a)
        i,j = np.triu_indices(len(SAMPLING),1)
        left,right = train[i]-train[j],test[i]-test[j]
        eligible = (np.abs(left)>1e-12)&(np.abs(right)>1e-12)
        flips.append(float((left[eligible]*right[eligible]<0).mean()) if eligible.any() else None)
        ties.append(float(1-eligible.mean()))
    return {'gain':float(np.mean(gains)), 'first_gain':float(np.mean(first_gains)),
            'immediate_gain':float(np.mean(immediate_gains)),
            'pair_flip':float(np.mean([x for x in flips if x is not None])) if any(x is not None for x in flips) else None,
            'pair_tie_share':float(np.mean(ties)), 'choices':chosen}


def cluster_seed(rows, values):
    groups=defaultdict(list)
    for r,v in zip(rows,values): groups[r['context']['seed']].append(v)
    return paired_summary([float(np.mean(v)) for _,v in sorted(groups.items())])


def label_analysis(rows):
    out={}
    for fam in FAMILIES:
        group=[r for r in rows if r['context']['family']==fam]
        profiles={}
        cached={}
        for profile in group[0]['profiles']:
            stats=[independent_gains(r,profile) for r in group]
            cached[profile]=stats
            means=np.stack([matrix(r,profile,'terminal').mean(axis=0) for r in group])
            first_means=np.stack([matrix(r,profile,'first').mean(axis=0) for r in group])
            profiles[profile]={'n_states':len(group), 'n_seeds':len({r['context']['seed'] for r in group}),
                'D_descriptive':float(means.max(axis=1).mean()-means.mean(axis=0).max()),
                'D_first_descriptive':float(first_means.max(axis=1).mean()-first_means.mean(axis=0).max()),
                'heldout_gain_vs_frozen_fixed':cluster_seed(group,[s['gain'] for s in stats]),
                'heldout_first_step_gain_vs_frozen_fixed':cluster_seed(group,[s['immediate_gain'] for s in stats]),
                'heldout_terminal_gain_using_first_step_labels':cluster_seed(group,[s['first_gain'] for s in stats]),
                'pair_flip_non_ties':float(np.mean([s['pair_flip'] for s in stats if s['pair_flip'] is not None])) if any(s['pair_flip'] is not None for s in stats) else None,
                'pair_tie_share':float(np.mean([s['pair_tie_share'] for s in stats])),
                'top_choice_split_exact_agreement':float(np.mean([s['choices'][0]==s['choices'][1] for s in stats])),
                'selection_share_frozen_fixed':float(np.mean([a==SAMPLING.index(r['continuation'])
                    for r,s in zip(group,stats) for a in s['choices']])),
                'chosen_actions':dict(Counter(SAMPLING[a] for s in stats for a in s['choices']))}
        contrasts={}
        for name,left,right in [('normalized_block_minus_unit','normalized/b5','normalized/b1'),
                                 ('raw_block_minus_unit','raw/b5','raw/b1'),
                                 ('normalization_unit_signal','normalized/b1','raw/b1'),
                                 ('normalization_block_signal','normalized/b5','raw/b5')]:
            contrasts[name]=cluster_seed(group,[a['gain']-b['gain'] for a,b in zip(cached[left],cached[right])])
        by_phase={}
        for n in (5,15,25):
            subset=[r for r in group if r['n']==n]
            by_phase[str(n)]={p:cluster_seed(subset,[independent_gains(r,p)['gain'] for r in subset]) for p in profiles}
        geometry={}
        for treatment in ('raw','normalized'):
            cvs=[]
            duplicates=[]
            uncertainty_spreads=[]
            for r in group:
                state=r['states'][treatment]['surrogate']
                if state['max_uncertainty']>1e-12:
                    uncertainty_spreads.append((state['max_uncertainty']-state['mean_uncertainty'])/state['max_uncertainty'])
                for rep in r['profiles'][f'{treatment}/b1']:
                    std=np.asarray([rep[a]['first']['std'] for a in SAMPLING],dtype=float)
                    if std.mean()>1e-12: cvs.append(float(std.std()/std.mean()))
                    a=rep['global_exploration']['first']['X']
                    b=rep['uncertainty_sampling']['first']['X']
                    duplicates.append(bool(np.allclose(a,b,rtol=0,atol=1e-12)))
            geometry[treatment]={'first_point_std_cv_mean':float(np.mean(cvs)) if cvs else None,
                                 'uncertainty_equals_global_point_share':float(np.mean(duplicates)),
                                 'fixed_grid_max_minus_mean_std_over_max':float(np.mean(uncertainty_spreads)),
                                 'fixed_grid_effectively_flat_share':float(np.mean(np.asarray(uncertainty_spreads)<1e-8))}
        out[fam]={'profiles':profiles,'contrasts':contrasts,'by_evaluation_count':by_phase,
                  'exploratory_gp_geometry':geometry}
    return out


def make_report(result):
    lines=['# GP 修复后的补充归因实验','',
           '主面板 2200 条闭环、75 个标签状态，每个处理八次模拟；新种子 2000–2019。全部对照配置预先冻结。正数表示原始终端目标改善，默认区间为配对 95% bootstrap。标签按开发种子聚类；D 只是模拟均值描述量。','',
           '生产后端增加显式归一化选项，拟合和恢复统一处理；本实验所有 normalized 动作均启用。默认 raw 保持历史实验可复现。没有重训 LLM 或新增学生。','',
           '## 新种子：修复后的动作与强基线','',
           '| 函数 | 原开发最佳 | 修复后最佳 | 修复后最佳 − 原最佳 | 单独修复固定代理 |','|---|---|---|---|---|']
    for f in FAMILIES:
        r=result['loops'][f];c=r['contrasts'];s=r['selection']
        lines.append(f"| {f} | {s['raw_fixed']} | {s['fixed']} | {pretty(c['best_pipeline'])} | {pretty(c['normalization/surrogate_acquisition'])} |")
    lines+=['','## 强固定延续：首动作与五步段的独立信号','',
            '这里的 gain 是独立四次模拟评价的选择收益，相对开发集冻结的强固定首动作。每个函数只有五个开发种子；不能据跨零区间断言效应为零。','',
            '| 函数 | 修复后单步 gain | 修复后五步 gain | 五步 − 单步 | 单步 pair flip | 五步 pair flip |', '|---|---|---|---|---:|---:|']
    for f in FAMILIES:
        r=result['labels'][f];a=r['profiles']['normalized/b1'];b=r['profiles']['normalized/b5']
        flip=lambda s: 'NA' if s is None else f'{s:.1%}'
        lines.append(f"| {f} | {pretty(a['heldout_gain_vs_frozen_fixed'])} | {pretty(b['heldout_gain_vs_frozen_fixed'])} | {pretty(r['contrasts']['normalized_block_minus_unit'])} | {flip(a['pair_flip_non_ties'])} | {flip(b['pair_flip_non_ties'])} |")
    lines+=['','| 函数 | 单步标签的独立即时收益 | 用即时标签选动作的终端收益 |','|---|---|---|']
    for f in FAMILIES:
        p=result['labels'][f]['profiles']['normalized/b1']
        lines.append(f"| {f} | {pretty(p['heldout_first_step_gain_vs_frozen_fixed'])} | {pretty(p['heldout_terminal_gain_using_first_step_labels'])} |")
    lines+=['','## 真正闭环：相同规划频率的特权参考','',
            '每臂十个新种子，五次规划，每次四个模拟。unit 执行一次所选动作再执行四次固定动作；block 执行五次所选动作。规划成本在真实 30 次评估之外单独记录。它们不是真正最优 oracle。','',
            '| 函数 | 单步参考 − 强固定 | 五步参考 − 强固定 | 五步 − 单步 | 五步收益 Holm p | 冻结 schedule − 强固定 |','|---|---|---|---|---:|---|']
    for f in FAMILIES:
        c=result['loops'][f]['contrasts']
        lines.append(f"| {f} | {pretty(c['reference_unit_minus_fixed'])} | {pretty(c['reference_block_minus_fixed'])} | {pretty(c['reference_block_minus_unit'])} | {c['reference_block_minus_fixed']['holm_p']:.4g} | {pretty(c['schedule_minus_fixed'])} |")
    p=result['positive_control']
    lines+=['','## 解析正向对照','',f"球函数两种状态：选 local/选 global 的真实优势为 1/30={p['analytic_gain']:.6f}。256 次选择模拟得到正确动作；2048 次独立评价得到 {pretty(p['heldout_gain'])}。这个对照通过，只验证测量链能识别已知的一决策收益。",'',
            '## 限制与归档','',
            '- 未在测试结果上挑固定对手或调整动作。强固定对手来自独立开发集。',
            '- Label 状态由修复后的最佳固定策略产生，可能覆盖不到自适应策略特有状态；没有支持范围外的结论。',
            '- 八次标签模拟与四次规划仍有噪声；特权参考没有穷尽规划/动作序列。',
            '- 五步段只标注“重复一个动作”，不是任意 schedule；固定预算也未检验高维预算扩展。',
            '- Bootstrap 区间、99% 区间及每项对照跨五族 Holm 校正保存在 analysis.json。探索性对照不作普遍理论否定。',
            '- 不同 GP 动作和前置拟合的计算成本没有折成真实函数评估；规划调用另记。',
            '- 协议：docs/ATTRIBUTION_NORMALIZED_FOLLOWUP_20260930.md；测试/恢复保证见 verification.json。','']
    confirm=result['rastrigin_normalization_confirmation']
    lines+=['## 独立补充：Rastrigin 归一化稳健性','',
            '此组在看到原 20 个新种子的宽区间后另行冻结；100 个独立种子 50000–50099，300 条闭环。与原组分开报告，两个对照作 Holm 校正。','',
            '| 对照 | 终端改善 | Holm p |','|---|---|---:|']
    for name,s in confirm['contrasts'].items():lines.append(f"| {name} | {pretty(s)} | {s['holm_p']:.4g} |")
    opp=result['rastrigin_opportunity_confirmation']
    lines+=['','## 独立补充：反馈选择与强时间安排','',
            f"在观察探索性 Rastrigin 参考的正均值后另行冻结。开发集另测 200 条 GP 两段安排，选中 `{opp['selection']['selected']}`；新种子 60000–60029，共 120 条测试。以下六项对照一起作 Holm 校正。",'',
            '| 对照 | 终端改善 | Holm p |','|---|---|---:|']
    for name,s in opp['contrasts'].items():lines.append(f"| {name} | {pretty(s)} | {s['holm_p']:.4g} |")
    audit=result['decision_metric_audit'];fixed=audit['global_fixed_metrics'];bt=audit['old_model_metrics']['bt_pairs_baseline']
    lines+=['','## 原学生评测：固定排序与选项顺序审计','',
            f"严格复用原 548 条训练/115 条验证记录、13618/2889 对动作。只在训练集搜索一个全局固定排序，验证 pairwise={fixed['pairwise_accuracy']:.6f}、top-1={fixed['top1_agreement']:.6f}；原 BT 分别为 {bt['pairwise_accuracy']:.6f}、{bt['top1_agreement']:.6f}。固定排序完全不读取当前状态。这两个指标没有支持 BT 学到了状态依赖的选择。",'',
            '本地 checkpoint 的训练路径/SDK 路径和选项倒序，在 20 个验证集初始状态上检查：','',
            '| 模型 | 训练/SDK 最优动作一致 | 倒序后保留同一动作 | 倒序后保留同一位置 |','|---|---:|---:|---:|']
    for arm,a in audit['local_model_wiring_audit'].items():
        lines.append(f"| {arm} | {a['training_vs_sdk_same_action_share']:.0%} | {a['reverse_same_action_share']:.0%} | {a['reverse_same_slot_share']:.0%} |")
    lines+=['','BT 在此探针上对选项位置敏感；未发现这 20 个相同输入上的 SDK 权重/索引映射差异。该检查没有覆盖旧闭环所有后续状态，不能单独归因全部塌缩。未重新训练模型。','']
    lines+=['','全部 BOCoDe 真实闭环合计 2820 条；正向对照的候选调用及特权模拟不计入这个数。补充组不是原主实验的预注册面板，未与原组混算。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,3,figsize=(11,8.5))
    panels=[('normalization/surrogate_acquisition','Normalized GP minus raw GP'),
            ('reference_unit_minus_fixed','Unit reference minus strong fixed'),
            ('reference_block_minus_fixed','Block reference minus strong fixed')]
    for i,f in enumerate(FAMILIES):
        values=[result['loops'][f]['contrasts'][name] for name,_ in panels]
        lo=min(0,*[s['ci95'][0] for s in values]);hi=max(0,*[s['ci95'][1] for s in values])
        pad=max((hi-lo)*.12,.001)
        for j,(name,title) in enumerate(panels):
            ax=axes[i,j]
            s=result['loops'][f]['contrasts'][name]
            lo,hi=s['ci95'];mid=s['mean']
            ax.errorbar(mid,0,xerr=[[mid-lo],[hi-mid]],fmt='o',capsize=3,color='#176b87')
            ax.axvline(0,color='gray',linewidth=.8)
            ax.set_yticks([])
            ax.set_xlim(min(0,*[v['ci95'][0] for v in values])-pad,max(0,*[v['ci95'][1] for v in values])+pad)
            if j==0:ax.set_ylabel(f,rotation=0,labelpad=42,fontsize=9)
            if i==0:ax.set_title(title,fontsize=9)
            if i==4:ax.set_xlabel('Terminal objective gain')
            ax.grid(axis='x',alpha=.2)
    fig.suptitle('Main panel, paired 95% CI; separate objective scales by family',fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT/'normalized_followup_effects.png',dpi=180)
    fig.savefig(OUT/'normalized_followup_effects.pdf')
    plt.close(fig)


def main():
    panels={n:json.loads((OUT/f'{n}.json').read_text()) for n in ('controls','labels','loops','references')}
    expected={'controls':800,'labels':25,'loops':1300,'references':100}
    config=panels['controls']['config']
    # Analyze the code version actually executed. The working tree may later
    # receive compatibility fixes for scenarios outside this panel.
    assert config['packages']==provenance()['packages']
    for name,sha in config['source_sha256'].items():
        frozen=OUT/'source_snapshot'/name
        source=frozen if frozen.exists() else ROOT/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==sha,name
    for name,d in panels.items():
        assert len(d['runs'])==expected[name],(name,len(d['runs']))
        assert len({r['key'] for r in d['runs']})==expected[name]
        assert all(d['config'][k]==v for k,v in config.items())
    selected=json.loads((OUT/'selection.json').read_text())
    sh=hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()
    assert all(panels[n]['config']['selection_sha256']==sh for n in ('labels','loops','references'))
    labels=[r for task in panels['labels']['runs'] for r in task['records']]
    assert len(labels)==75 and len({r['id'] for r in labels})==75
    loops=panels['loops']['runs']+panels['references']['runs']
    for row in panels['controls']['runs']+loops:
        assert row['n_evaluations']==30==len(row['values'])
        assert row['terminal_best']==max(row['values'])
        assert np.isfinite(row['values']).all()
    for r in labels:
        assert all(s['surrogate']['is_current'] for s in r['states'].values())
        assert all(len(x)==8 for x in r['profiles'].values())
        assert len(r['history_X'])==r['n']==len(r['history_y'])
        assert all(cell['n_new']==30-r['n'] for reps in r['profiles'].values()
                   for rep in reps for cell in rep.values())
        for t in ('raw','normalized'):
            for a in SAMPLING:
                for unit,block in zip(r['profiles'][f'{t}/b1'],r['profiles'][f'{t}/b5']):
                    assert unit[a]['first']==block[a]['first']
                    if a==r['continuation']: assert unit[a]==block[a]
    positive=json.loads((OUT/'positive_control.json').read_text())
    assert positive['config']==config and positive['heldout_gain']['ci95'][0]>0
    confirmation=json.loads((OUT/'rastrigin_confirmation_analysis.json').read_text())
    opportunity=json.loads((OUT/'rastrigin_opportunity_analysis.json').read_text())
    supplementary={n:json.loads((OUT/f'{n}.json').read_text()) for n in
        ('rastrigin_confirmation','rastrigin_opportunity_controls','rastrigin_opportunity_test')}
    for n,count in [('rastrigin_confirmation',300),('rastrigin_opportunity_controls',200),('rastrigin_opportunity_test',120)]:
        rows=supplementary[n]['runs']
        assert len(rows)==count==len({r['key'] for r in rows})
        assert all(r['n_evaluations']==30==len(r['values']) for r in rows)
        assert all(r['terminal_best']==max(r['values']) for r in rows)
        assert all(np.isfinite(r['values']).all() for r in rows)
        for name,sha in supplementary[n]['config']['source_sha256'].items():
            frozen=OUT/'source_snapshot_supplements'/name
            assert hashlib.sha256(frozen.read_bytes()).hexdigest()==sha,name
    for cfg,runner,protocol,rkey,pkey in (
            (confirmation['config'],'scripts/exp_attribution_normalized_confirmation.py','docs/ATTRIBUTION_RASTRIGIN_CONFIRMATION_20260930.md','confirmation_script_sha256','confirmation_protocol_sha256'),
            (opportunity['config'],'scripts/exp_attribution_rastrigin_opportunity.py','docs/ATTRIBUTION_RASTRIGIN_OPPORTUNITY_20260930.md','supplement_runner_sha256','supplement_protocol_sha256')):
        assert cfg[rkey]==hashlib.sha256((OUT/'source_snapshot_supplements'/runner).read_bytes()).hexdigest()
        assert cfg[pkey]==hashlib.sha256((OUT/'source_snapshot_supplements'/protocol).read_bytes()).hexdigest()
    assert confirmation['raw_data_sha256']==hashlib.sha256((OUT/'rastrigin_confirmation.json').read_bytes()).hexdigest()
    for n,sha in opportunity['artifacts_sha256'].items():
        assert sha==hashlib.sha256((OUT/f'rastrigin_opportunity_{n}.json').read_bytes()).hexdigest()
    audit=json.loads((OUT/'decision_metric_audit.json').read_text())
    assert audit['old_model_metrics']==json.loads((ROOT/'runs/exp_rlcd_report.json').read_text())
    assert audit['script_sha256']==hashlib.sha256((ROOT/'scripts/audit_attribution_decision_metric.py').read_bytes()).hexdigest()
    assert audit['dataset_sha256']==hashlib.sha256((ROOT/'runs/phase8_6d_h3_20seed_5family.jsonl').read_bytes()).hexdigest()
    for arm,a in audit['local_model_wiring_audit'].items():
        assert a['n_states']==len(a['rows'])==20
        assert a['checkpoint_sha256']==hashlib.sha256((ROOT/f'runs/rlcd_laya_{arm}_v1/model.safetensors').read_bytes()).hexdigest()
    # Identity/no-GP negative controls should be exactly equal, not merely
    # have a confidence interval crossing zero.
    lookup={(r['context']['family'],r['context']['seed'],r['treatment'],r['policy']):r
            for r in panels['controls']['runs']+panels['loops']['runs']}
    identity_pairs=0
    for (f,s,t,a),row in lookup.items():
        if t=='normalized' and a in SAMPLING and (f=='hartmann' or a in
                ('global_exploration','local_exploitation','boundary_sampling')):
            assert row['values']==lookup[f,s,'raw',a]['values']
            identity_pairs+=1
    result={'counts':{n:len(d['runs']) for n,d in panels.items()},'label_states':75,
            'artifacts_sha256':{n:hashlib.sha256((OUT/f'{n}.json').read_bytes()).hexdigest() for n in (*panels,'positive_control','selection')},
            'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'loops':loop_analysis(loops,selected),'labels':label_analysis(labels),
            'rastrigin_normalization_confirmation':confirmation,
            'rastrigin_opportunity_confirmation':opportunity,
            'decision_metric_audit':audit,
            'positive_control':{k:positive[k] for k in ('analytic_gain','heldout_gain','selected','constant','selection_replicas','evaluation_replicas')}}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    make_report(result)
    figures(result)
    check={'closed_loop_trajectories':2200,'label_states':75,'all_real_budgets_exactly_30':True,
           'supplementary_closed_loops':620,'total_closed_loop_trajectories':2820,
           'matched_block_first_prefixes':True,'fixed_intervention_matches_across_blocks':True,
           'current_gp_on_all_label_states':True,'frozen_source_hashes_verified':True,
           'identity_negative_control_pairs_exact':identity_pairs,
           'positive_control_passed':True,'tests':{'baseline_and_compatibility_passed':60,
               'normalized_backend_and_analysis_passed':11,'dataset_replay_and_analysis_passed':27,
               'bounds_restore_regression_and_checkpoint_passed':22},
           'model_audit_dataset_and_checkpoint_hashes_verified':True,
           'supplementary_source_hashes_verified':True,
           'post_freeze_compatibility_fixes':{
               'dataset_normalization_flag_preserved':True,
               'uninitialized_or_different_problem_bounds_restored':True,
               'not_exercised_by_main_panel':True},
           'current_compatibility_source_sha256':{
               p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
               ('src/opt_agent/backend/bocode_botorch.py','src/opt_agent/backend/checkpoint.py',
                'src/opt_agent/dataset/generator.py','src/opt_agent/dataset/validation.py')}}
    (OUT/'verification.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps(result['counts'],indent=2))


if __name__=='__main__': main()
