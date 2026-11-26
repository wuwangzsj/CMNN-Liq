"""Optional dual-reference extension. activate_adapter changes module classes;
use a separate process for extension training and base-model training.
"""
import numpy as np
import torch
from scipy.special import expit
import cmnn
import training as repair
from features import juang,validate

def new_inputs(d):
    validate(d)
    ref=d.copy()
    ref['pga']=0.1
    ref['mw']=7.5
    m=np.log(1.0/(d.mw.to_numpy()/7.5)**(-2.56))
    return np.column_stack([m,d[['depth','stress_ratio','qt1n','ic']].to_numpy(float),
                            np.log(juang(ref)),np.log(d.pga.to_numpy()/0.1)])

class DualMonoLayer(cmnn.MonoLayer):
    def effective(self):
        if self.first:
            return torch.cat([self.weight[:,:1].abs(),self.weight[:,1:-1],self.weight[:,-1:].abs()],1)
        return self.weight.abs()

class ReferenceMwCMNN(cmnn.CMNN):
    def inputs(self,d):
        return new_inputs(d)

def activate_adapter():
    cmnn.MonoLayer=DualMonoLayer
    repair.CMNN=ReferenceMwCMNN

def numpy_predict(w,d):
    x=(new_inputs(d)-w['mean'])/w['scale']
    h=x@w['w1'].T+w['b1']
    n=h.shape[1];a=round(0.4*n);b=round(0.4*n)
    def elu(v):
        return np.where(v>0,v,np.expm1(np.minimum(v,0)))
    z=h[:,a+b:]
    h=np.concatenate([elu(h[:,:a]),-elu(-h[:,a:a+b]),
        np.where(z<=0,elu(z+1)-1,-elu(1-z)+1)],axis=1)
    return expit((h@w['w2'].T+w['b2']).ravel())
