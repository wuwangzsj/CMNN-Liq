"""Saved effective-weight inference for reference-state M/U networks."""
import numpy as np
from features import inputs

def elu(v):
    return np.maximum(v,0)+np.expm1(np.minimum(v,0))

def activation(v, derivative=False):
    h=v.shape[-1]; a=round(.4*h); b=round(.4*h)
    if derivative:
        f=lambda z:np.exp(np.minimum(z,0))
        sat=np.exp(np.minimum(1-np.abs(v[...,a+b:]),0))
        return np.concatenate([f(v[...,:a]),f(-v[...,a:a+b]),sat],axis=-1)
    z=v[...,a+b:]
    return np.concatenate([elu(v[...,:a]),-elu(-v[...,a:a+b]),np.where(z<=0,elu(z+1)-1,-elu(1-z)+1)],axis=-1)

def evaluate(d, weights):
    x=(inputs(d)-weights['mean'])/weights['scale']
    h=x@weights['w1'].T+weights['b1']
    logit=(activation(h)@weights['w2'].T+weights['b2']).ravel()
    p=np.exp(-np.logaddexp(0,-logit))
    dz_du=(activation(h,True)*(weights['w1'][:,-1]*weights['w2'].ravel())).sum(1)/weights['scale'][-1]
    return p,logit,dz_du
