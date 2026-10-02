import numpy as np
from scripts.analyze_attribution_normalized import independent_gains, cluster_seed
from scripts.exp_attribution_suite import SAMPLING


def row(values):
    return {'continuation':SAMPLING[0], 'profiles':{'normalized/b1':[
        {a:{'first':{'delta':float(y)},'terminal':{'delta':float(y)}} for a,y in zip(SAMPLING,x)} for x in values]}}


def test_independent_gain_does_not_score_lucky_selection_on_same_replicas():
    values=np.zeros((8,6))
    values[:4,1]=4
    values[4:,2]=4
    result=independent_gains(row(values),'normalized/b1')
    assert result['choices']==[1,2]
    assert result['gain']==0


def test_independent_gain_recovers_real_advantage_and_clusters_checkpoints():
    values=np.zeros((8,6));values[:,1]=2
    result=independent_gains(row(values),'normalized/b1')
    assert result['gain']==2 and result['choices']==[1,1]
    rows=[{'context':{'seed':s}} for s in (1,1,1,2,2,2)]
    assert cluster_seed(rows,[2,2,2,2,2,2])['n']==2
