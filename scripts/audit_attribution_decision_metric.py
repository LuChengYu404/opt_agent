"""Does the old high pair accuracy require state-dependent decisions?"""
import hashlib
import itertools
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from exp_attribution_normalized import OUT
from _shared_rq_features import ACTIONS
from opt_agent.dataset import read_jsonl
from opt_agent.system1.rlcd import mine_pairs


def best_fixed_order(records):
    edges=np.zeros((8,8),dtype=int);idx={a:i for i,a in enumerate(ACTIONS)}
    for r in records:
        for p in mine_pairs(r):edges[idx[p.pos],idx[p.neg]]+=1
    orders=np.asarray(list(itertools.permutations(range(8))),dtype=int)
    scores=sum(edges[orders[:,i],orders[:,j]] for i in range(8) for j in range(i+1,8))
    return [ACTIONS[i] for i in orders[int(scores.argmax())]]


def fixed_metrics(records,order):
    total=correct=top=0
    for r in records:
        o=order[r.problem['family']] if isinstance(order,dict) else order
        rank={a:i for i,a in enumerate(o)}
        pairs=mine_pairs(r);total+=len(pairs)
        correct+=sum(rank[p.pos]<rank[p.neg] for p in pairs)
        predicted=min(r.gold_probs,key=rank.get)
        top+=predicted==max(r.gold_probs,key=r.gold_probs.get)
    return {'pairs':total,'pairwise_accuracy':correct/total,'top1_agreement':top/len(records),'records':len(records)}


def main():
    torch.set_num_threads(1)
    dataset=ROOT/'runs/phase8_6d_h3_20seed_5family.jsonl'
    records=read_jsonl(dataset,version_policy='warn')
    train=[r for r in records if r.problem['seed']<16];val=[r for r in records if r.problem['seed']>=16]
    global_order=best_fixed_order(train)
    family_order={f:best_fixed_order([r for r in train if r.problem['family']==f]) for f in sorted({r.problem['family'] for r in train})}
    assert len(train)==548 and len(val)==115
    assert sum(len(mine_pairs(r)) for r in train)==13618
    assert sum(len(mine_pairs(r)) for r in val)==2889
    old=json.loads((ROOT/'runs/exp_rlcd_report.json').read_text())
    result={'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'training_seeds':[0,16],'validation_seeds':[16,20],
            'global_fixed_order':global_order,'global_fixed_metrics':fixed_metrics(val,global_order),
            'family_fixed_order':family_order,'family_fixed_metrics':fixed_metrics(val,family_order),
            'always_local_top1_agreement':sum(max(r.gold_probs,key=r.gold_probs.get)=='local_exploitation' for r in val)/len(val),
            'train_teacher_top_actions':dict(Counter(max(r.gold_probs,key=r.gold_probs.get) for r in train)),
            'validation_teacher_top_actions':dict(Counter(max(r.gold_probs,key=r.gold_probs.get) for r in val)),
            'old_model_metrics':old}
    (OUT/'decision_metric_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('global_fixed_order','global_fixed_metrics','family_fixed_metrics','always_local_top1_agreement')},indent=2),flush=True)

    # Check only local checkpoints, with external model access disabled by
    # the command's environment. Same input through raw training and SDK paths.
    import laya
    from opt_agent.system1.rlcd import build_item,_forward,_record_choice_parts,RLCDConfig
    probes=[next(r for r in val if r.problem['family']==f and r.problem['seed']==s)
            for f in sorted({r.problem['family'] for r in val}) for s in range(16,20)]
    audits={}
    for arm in ('bt','proper'):
        model_dir=ROOT/f'runs/rlcd_laya_{arm}_v1'
        agent=laya.load(str(model_dir));agent.model.eval()
        cfg=RLCDConfig(device=str(agent.device));rows=[]
        for r in probes:
            item=build_item(r,agent.tok)
            with torch.no_grad():logits=_forward(agent.model,[item],agent.tok,agent.device,cfg)[0]
            raw_top=item['option_ids'][int(logits.argmax())]
            state,q=_record_choice_parts(r)
            question={'next_action':{'type':q['t'],'instructions':q['ins'],'criteria':q['crit']}}
            response=agent.predict(state,question)
            probs=response['answers']['next_action']['probabilities'];sdk_top=max(probs,key=probs.get)
            reversed_record=replace(r,candidate_actions=list(reversed(r.candidate_actions)))
            reversed_item=build_item(reversed_record,agent.tok)
            with torch.no_grad():reverse_logits=_forward(agent.model,[reversed_item],agent.tok,agent.device,cfg)[0]
            reverse_top=reversed_item['option_ids'][int(reverse_logits.argmax())]
            rows.append({'id':r.id,'raw_top':raw_top,'sdk_top':sdk_top,'reversed_top':reverse_top,
                         'same_slot_after_reverse':int(logits.argmax())==int(reverse_logits.argmax())})
        audits[arm]={'n_states':len(rows),'training_vs_sdk_same_action_share':sum(r['raw_top']==r['sdk_top'] for r in rows)/len(rows),
                     'raw_top_actions':dict(Counter(r['raw_top'] for r in rows)),
                     'reverse_same_action_share':sum(r['raw_top']==r['reversed_top'] for r in rows)/len(rows),
                     'reverse_same_slot_share':sum(r['same_slot_after_reverse'] for r in rows)/len(rows),
                     'checkpoint_sha256':hashlib.sha256((model_dir/'model.safetensors').read_bytes()).hexdigest(),
                     'rows':rows}
        print(arm,json.dumps({k:v for k,v in audits[arm].items() if k!='rows'}),flush=True)
        del agent
        import gc
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    result['local_model_wiring_audit']=audits
    (OUT/'decision_metric_audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
