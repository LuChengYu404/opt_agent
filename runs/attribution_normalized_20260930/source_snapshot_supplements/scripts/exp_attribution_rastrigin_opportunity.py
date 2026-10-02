"""Independent Rastrigin feedback-versus-time-schedule confirmation."""
import hashlib
import json
import statistics
from pathlib import Path

import torch
from exp_attribution_normalized import OUT, ROOT, backend, action_seed, perform, loop_task, provenance
from exp_attribution_suite import SAMPLING, run_tasks
from analyze_attribution_suite import intervals, permutation_p, holm


def switch_task(task):
    torch.set_num_threads(1)
    import warnings
    warnings.filterwarnings('ignore')
    if 'after_action' not in task:
        return loop_task(task)
    c=task['context'];be=backend(c,'normalized')
    initial=be.get_history().objectives[:,0].tolist()
    actions=[];fits=0
    while len(be.get_history())<30:
        n=len(be.get_history())
        a='surrogate_acquisition' if n<task['switch_n'] else task['after_action']
        result=perform(be,a,action_seed(c,n));fits+=result.n_model_fits;actions.append(a)
    values=be.get_history().objectives[:,0].tolist()
    return {'key':task['key'],'context':c,'treatment':'normalized','policy':task['policy'],
            'initial_values':initial,'values':values,'terminal_best':max(values),'n_evaluations':len(values),
            'n_fits':fits,'prefeature_fits':0,'planning_evaluations':0,'planning_fits':0,'actions':actions}


def main():
    torch.set_num_threads(1)
    config={**provenance(),
            'supplement_runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'supplement_protocol_sha256':hashlib.sha256((ROOT/'docs/ATTRIBUTION_RASTRIGIN_OPPORTUNITY_20260930.md').read_bytes()).hexdigest(),
            'source_controls_sha256':hashlib.sha256((OUT/'controls.json').read_bytes()).hexdigest()}
    tasks=[]
    for seed in range(20,30):
        c={'family':'rastrigin','seed':seed}
        for a in SAMPLING:
            if a=='surrogate_acquisition':continue
            for n in (10,15,20,25):
                policy=f'gp_then_{a}_{n}'
                tasks.append({'key':f'rastrigin/s{seed}/{policy}','context':c,'policy':policy,
                              'after_action':a,'switch_n':n})
    dev=run_tasks(tasks,switch_task,OUT/'rastrigin_opportunity_controls.json',16,config)
    assert len(dev)==200
    old=json.loads((OUT/'controls.json').read_text())['runs']
    gp=[r for r in old if r['context']['family']=='rastrigin' and r['treatment']=='normalized' and r['policy']=='surrogate_acquisition']
    assert len(gp)==10
    candidates={'surrogate_acquisition':{'mean':statistics.fmean(r['terminal_best'] for r in gp)}}
    for task in tasks:
        policy=task['policy']
        if policy not in candidates:
            candidates[policy]={'mean':statistics.fmean(r['terminal_best'] for r in dev if r['policy']==policy),
                                'after_action':task['after_action'],'switch_n':task['switch_n']}
    selected=max(candidates,key=lambda k:candidates[k]['mean'])
    selection={'selected':selected,'candidates':candidates}
    (OUT/'rastrigin_opportunity_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    config['schedule_selection_sha256']=hashlib.sha256((OUT/'rastrigin_opportunity_selection.json').read_bytes()).hexdigest()
    tasks=[]
    for seed in range(60000,60030):
        c={'family':'rastrigin','seed':seed}
        for policy in ('surrogate_acquisition','frozen_gp_schedule','reference_1','reference_5'):
            task={'key':f'rastrigin/s{seed}/{policy}','context':c,'treatment':'normalized','policy':policy,
                  'continuation':'surrogate_acquisition'}
            if policy=='frozen_gp_schedule':
                candidate=candidates[selected]
                task.update(after_action=candidate.get('after_action','surrogate_acquisition'),switch_n=candidate.get('switch_n',30))
            tasks.append(task)
    rows=run_tasks(tasks,switch_task,OUT/'rastrigin_opportunity_test.json',20,config)
    assert len(rows)==120 and all(r['n_evaluations']==30 for r in rows)
    lookup={(r['context']['seed'],r['policy']):r for r in rows}
    contrasts={}
    for name,left,right in [('schedule_minus_gp','frozen_gp_schedule','surrogate_acquisition'),
          ('unit_minus_gp','reference_1','surrogate_acquisition'),('block_minus_gp','reference_5','surrogate_acquisition'),
          ('unit_minus_schedule','reference_1','frozen_gp_schedule'),('block_minus_schedule','reference_5','frozen_gp_schedule'),
          ('block_minus_unit','reference_5','reference_1')]:
        values=[]
        for seed in range(60000,60030):
            a,b=lookup[seed,left],lookup[seed,right]
            assert a['initial_values']==b['initial_values']
            values.append(a['terminal_best']-b['terminal_best'])
        contrasts[name]={**intervals(values),'paired_sign_randomization_p':permutation_p(values)}
    corrected=holm({k:v['paired_sign_randomization_p'] for k,v in contrasts.items()})
    for k,p in corrected.items(): contrasts[k]['holm_p']=p
    result={'config':config,'n_test_seeds':30,'n_development_trajectories':200,'n_test_trajectories':120,
            'selection':selection,'contrasts':contrasts,
            'artifacts_sha256':{name:hashlib.sha256((OUT/f'rastrigin_opportunity_{name}.json').read_bytes()).hexdigest()
                               for name in ('controls','selection','test')}}
    (OUT/'rastrigin_opportunity_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'selected':selected,'contrasts':contrasts},indent=2))


if __name__=='__main__': main()
