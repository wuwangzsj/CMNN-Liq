"""Original training-derived support and per-network numerical root routines.
Intervals are in u=ln(a/0.1); query roots are in g. No averaging across seeds here.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit,logit
from scipy.optimize import brentq
from inference import activation
RHO=[.1,.5,.9]
LOW,HIGH=np.log(.01/.1),np.log(1.5/.1)

class Support:
 def __init__(self,v,k,q,m):
  self.mu=v.mean(0);self.sd=v.std(0);self.sd[self.sd<1e-12]=1;self.lo=v.min(0);self.hi=v.max(0);self.z=(v-self.mu)/self.sd;self.tree=cKDTree(self.z);self.r=float(np.quantile(self.tree.query(self.z,k=k+1)[0][:,-1],q));self.m=m
 def contains(self,v):return ((v>=self.lo-1e-12)&(v<=self.hi+1e-12)).all(1)&(self.tree.query((v-self.mu)/self.sd,k=self.m)[0][:,-1]<=self.r+1e-12)
 def intervals(self,x):
  if not ((x>=self.lo[:-1])&(x<=self.hi[:-1])).all():return []
  q=(x-self.mu[:-1])/self.sd[:-1];d2=((self.z[:,:-1]-q)**2).sum(1);ok=d2<=self.r**2
  h=np.sqrt(np.maximum(0,self.r**2-d2[ok]));cen=self.z[ok,-1]
  lo=np.maximum((cen-h)*self.sd[-1]+self.mu[-1],self.lo[-1]);hi=np.minimum((cen+h)*self.sd[-1]+self.mu[-1],self.hi[-1]);ok=hi>lo
  if ok.sum()<self.m:return []
  starts=lo[ok];ends=hi[ok];pts=np.unique(np.r_[starts,ends]);mid=(pts[:-1]+pts[1:])/2
  count=((mid[:,None]>=starts)&(mid[:,None]<=ends)).sum(1);ans=[]
  for j in np.where(count>=self.m)[0]:
   a,b=pts[j],pts[j+1]
   if ans and abs(a-ans[-1][1])<1e-12:ans[-1][1]=b
   else:ans.append([a,b])
  return [[max(a,LOW),min(b,HIGH)] for a,b in ans if min(b,HIGH)-max(a,LOW)>1e-10]

class Curve:
 def __init__(self,w,x,slope=None):
  self.q=w['w1'][:,:6]@((x-w['mean'][:6])/w['scale'][:6])+w['b1']-w['w1'][:,-1]*w['mean'][-1]/w['scale'][-1]
  self.r=w['w1'][:,-1]/w['scale'][-1] if slope is None else slope
  self.v=w['w2'].ravel();self.out=float(w['b2'].ravel()[0]);self.b=2*round(.4*len(self.v))
 def z(self,u,der=False):
  t=self.q[None,:]+np.atleast_1d(u)[:,None]*self.r[None,:]
  return (activation(t,der)*self.r)@self.v if der else activation(t)@self.v+self.out
 def p(self,u):return expit(self.z(u))
 def roots(self,segs):
  found={r:[] for r in RHO};flat={r:0 for r in RHO}
  for si,(lo,hi) in enumerate(segs):
   transitions=[]
   for j,(q,r) in enumerate(zip(self.q,self.r)):
    if abs(r)>1e-14:transitions.extend([(t-q)/r for t in ([-1,0,1] if j>=self.b else [0]) if lo<(t-q)/r<hi])
   grid=np.unique(np.r_[np.linspace(lo,hi,257),transitions]);der=self.z(grid,True)
   ext=[brentq(lambda u:float(self.z(u,True)[0]),grid[k],grid[k+1],xtol=1e-12) for k in np.where(der[:-1]*der[1:]<0)[0]]
   grid=np.unique(np.r_[grid,ext]);z=self.z(grid)
   for rho in RHO:
    f=z-logit(rho);flat[rho]+=int(((abs(f[:-1])<=1e-9)&(abs(f[1:])<=1e-9)&(abs(self.z((grid[:-1]+grid[1:])/2,True))<=1e-8)).sum())
    roots=list(grid[abs(f)<=1e-10])+[brentq(lambda u:float(self.z(u)[0]-logit(rho)),grid[k],grid[k+1],xtol=1e-12) for k in np.where(f[:-1]*f[1:]<0)[0]]
    ded=[]
    for u in sorted(roots):
     if not ded or abs(u-ded[-1])>1e-7:ded.append(u)
    for u in ded:
     err=float(abs(self.p(u)[0]-rho));assert err<=1e-8
     found[rho].append(dict(u=float(u),pga=float(.1*np.exp(u)),segment=si,residual=err))
  return found,flat
