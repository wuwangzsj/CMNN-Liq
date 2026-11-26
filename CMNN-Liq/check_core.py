"""Non-training algebra and diagnostic checks; synthetic inputs, not case histories."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import numpy as np
import pandas as pd
from features import inputs,physical_features
from path_metrics import self_check
from inference import activation

def main():
    d=pd.DataFrame(dict(pga=[.1,.2,.4],mw=[7.5]*3,depth=[5.]*3,
                        stress_ratio=[1.5]*3,qt1n=[60.]*3,ic=[2.]*3))
    x=inputs(d); f=physical_features(d)
    assert np.allclose(x[:,5],x[0,5])
    assert np.allclose(np.log(f['fs_current']),x[:,5]-x[:,6])
    assert np.allclose(f['fs_current']*d.pga,(f['fs_current']*d.pga).iloc[0])
    assert self_check()
    h=np.repeat(np.linspace(-6,6,101)[:,None],32,axis=1)
    assert np.diff(activation(h),axis=0).min()>=-1e-12
    import cmnn,training,comparators,factorial,factorial_training,support_query,reference_mw,evaluation
    print('PASS: feature identity, mixed activation, support-gap-aware metrics and core imports.')
    print('No model fitting and no manuscript data used.')

if __name__=='__main__': main()
