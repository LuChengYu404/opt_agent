"""Independent, fixed-size Rastrigin GP normalization confirmation."""
import hashlib
import json
from pathlib import Path
import torch
from exp_attribution_normalized import OUT, ROOT, provenance, loop_task
from exp_attribution_suite import run_tasks
from analyze_attribution_suite import intervals, permutation_p, holm


def main():
    torch.set_num_threads(1)
    selection_path=OUT/'selection.json'
    selected=json.loads(selection_path.read_text())['rastrigin']
    assert selected['fixed']=='surrogate_acquisition'
    config={**provenance(),
            'confirmation_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'confirmation_protocol_sha256':hashlib.sha256((ROOT/'docs/ATTRIBUTION_RASTRIGIN_CONFIRMATION_20260930.md').read_bytes()).hexdigest(),
            'selection_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            'seeds':[50000,50100]}
    tasks=[]
    for seed in range(50000,50100):
        c={'family':'rastrigin','seed':seed}
        for t,a in [('raw','surrogate_acquisition'),('normalized','surrogate_acquisition'),('raw',selected['raw_fixed'])]:
            tasks.append({'key':f'rastrigin/s{seed}/{t}/{a}','context':c,'treatment':t,'policy':a})
    rows=run_tasks(tasks,loop_task,OUT/'rastrigin_confirmation.json',8,config)
    assert len(rows)==300 and all(r['n_evaluations']==30 for r in rows)
    lookup={(r['context']['seed'],r['treatment'],r['policy']):r for r in rows}
    contrasts={}
    for name,right in [('normalization','surrogate_acquisition'),('best_pipeline',selected['raw_fixed'])]:
        diffs=[]
        for seed in range(50000,50100):
            a,b=lookup[seed,'normalized','surrogate_acquisition'],lookup[seed,'raw',right]
            assert a['initial_values']==b['initial_values']
            diffs.append(a['terminal_best']-b['terminal_best'])
        contrasts[name]={**intervals(diffs),'paired_sign_randomization_p':permutation_p(diffs)}
    corrected=holm({k:v['paired_sign_randomization_p'] for k,v in contrasts.items()})
    for k,p in corrected.items(): contrasts[k]['holm_p']=p
    result={'config':config,'n_trajectories':300,'n_seeds':100,'contrasts':contrasts,
            'raw_data_sha256':hashlib.sha256((OUT/'rastrigin_confirmation.json').read_bytes()).hexdigest()}
    (OUT/'rastrigin_confirmation_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(contrasts,indent=2))


if __name__=='__main__': main()
