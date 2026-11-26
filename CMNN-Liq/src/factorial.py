"""Projected effective-coefficient geometry; G1=M0, G3=A0; R/C unchanged."""
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

def tensor(x):return torch.tensor(x,dtype=torch.float64)

def mixed_activation(x):
 a=round(.4*x.shape[-1]);b=a;z=x[...,a+b:]
 return torch.cat((nn.functional.elu(x[...,:a]),-nn.functional.elu(-x[...,a:a+b]),torch.where(z<=0,nn.functional.elu(z+1)-1,-nn.functional.elu(1-z)+1)),dim=-1)

class MonoLayer(nn.Module):
 def __init__(self,ni,no,first=False,activation=True):
  super().__init__();self.weight=nn.Parameter(torch.empty(no,ni,dtype=torch.float64));nn.init.xavier_uniform_(self.weight);self.bias=nn.Parameter(torch.zeros(no,dtype=torch.float64))

def project_pair(pair,rho,cone):
 # Euclidean projection in the SAME standardized reference coordinates for every group.
 x=pair[...,0];y=pair[...,1]
 if cone=='M':return torch.stack((x,y.clamp_min(0)),dim=-1)
 if cone=='R':return torch.stack((x.clamp_max(0),y.clamp_min(0)),dim=-1)
 feasible=((y>=0) if cone=='C' else (x<=0))&(y+rho*x>=0)
 ray0=torch.stack((x.clamp_min(0),torch.zeros_like(y)),dim=-1) if cone=='C' else torch.stack((torch.zeros_like(x),y.clamp_min(0)),dim=-1)
 t=((-x+rho*y)/(1+rho*rho)).clamp_min(0)
 ray1=torch.stack((-t,rho*t),dim=-1)
 choose=((ray0-pair)**2).sum(-1)<=((ray1-pair)**2).sum(-1)
 out=torch.where(choose[...,None],ray0,ray1)
 return torch.where(feasible[...,None],pair,out)

class Net(nn.Module):
 def __init__(self,group,seed,stats,H=32):
  super().__init__();self.group=group;self.seed=int(seed);self.native=group=='G6';self.cone=group if group in ['R','C'] else ('M' if group in ['G1','G2'] else 'A')
  self.register_buffer('mu',tensor(stats['mu']));self.register_buffer('scale',tensor(stats['scale']))
  self.register_buffer('muv',tensor(stats['muv']));self.register_buffer('sv',tensor(stats['sv']))
  self.rho=float(self.scale[6]/self.scale[5])
  with torch.random.fork_rng():
   torch.manual_seed(self.seed);hidden=MonoLayer(7,H,True);output=MonoLayer(H,1,activation=False)
  orig=hidden.weight.detach().clone();b=hidden.bias.detach().clone()
  # Initial A effective weights are recreated from seed, never fitted historical weights.
  wa=torch.cat((orig[:,:5],-orig[:,5:6].abs(),orig[:,6:7].abs()),1)
  wc,bc=self.from_native(wa,b)
  if group=='G5':
   wc=torch.cat((orig[:,:6],orig[:,6:7].abs()),1)
   wc=torch.cat((wc[:,:5],project_pair(wc[:,5:7],self.rho,'A')),1);bc=b
  if self.native:w,b0=self.to_native(wc,bc)
  else:w,b0=wc,bc
  self.W=nn.Parameter(w.clone());self.b=nn.Parameter(b0.clone())
  self.V=nn.Parameter(output.weight.detach().abs().clone());self.bo=nn.Parameter(output.bias.detach().clone())
  self.c=1.
  with torch.no_grad():
   if group=='G7':self.c=float(self.ra()/self.rm());assert self.c>0 and np.isfinite(self.c)
 def from_native(self,w,b):
  p=w[:,5:6];q=w[:,6:7]
  ws=p*self.scale[5]/self.sv;wu=q-p*self.scale[6]/self.sv
  delta=(self.mu[5]-self.mu[6]-self.muv)/self.sv
  return torch.cat((w[:,:5],ws,wu),1),b+p.ravel()*delta
 def to_native(self,w,b):
  p=w[:,5:6]*self.sv/self.scale[5];q=w[:,6:7]+w[:,5:6]*self.scale[6]/self.scale[5]
  delta=(self.mu[5]-self.mu[6]-self.muv)/self.sv
  return torch.cat((w[:,:5],p,q),1),b-p.ravel()*delta
 def canonical(self):return self.from_native(self.W,self.b) if self.native else (self.W,self.b)
 def forward(self,a):
  x=(a-self.mu)/self.scale
  if self.native:x=torch.cat((x[:,:5],((a[:,5]-a[:,6]-self.muv)/self.sv)[:,None],x[:,6:7]),1)
  return nn.functional.linear(mixed_activation(nn.functional.linear(x,self.W,self.b)),self.V,self.bo).ravel()
 def rm(self):
  w,b=self.canonical();return .5*(w.square().sum()+b.square().sum()+self.V.square().sum()+self.bo.square().sum())
 def ra(self):
  w,b=self.canonical();a,ab=self.to_native(w,b)
  return .5*(a.square().sum()+ab.square().sum()+self.V.square().sum()+self.bo.square().sum())
 def penalty(self):return self.c*self.rm() if self.group in ['G1','G3','G7','R','C'] else self.ra()
 def constraints(self):
  w,b=self.canonical();return dict(min_u=float(w[:,6].detach().min()),max_s=float(w[:,5].detach().max()),min_A_slack=float((w[:,6]+self.rho*w[:,5]).detach().min()),min_output=float(self.V.detach().min()))
 def project(self):
  with torch.no_grad():
   w,b=self.canonical();pair=project_pair(w[:,5:7],self.rho,self.cone)
   nw=torch.cat((w[:,:5],pair),1);v=self.V.clamp_min(0)
   norm=float(torch.sqrt((nw-w).square().sum()+(v-self.V).square().sum()))
   nactive=int(((pair-w[:,5:7]).abs().max(1).values>1e-14).sum())
   if self.native:nw,b=self.to_native(nw,b)
   self.W.copy_(nw);self.b.copy_(b);self.V.copy_(v)
  return norm,nactive
 def assert_feasible(self):
  c=self.constraints();assert c['min_u']>=-1e-12 and c['min_output']>=-1e-12,c
  if self.cone in ['A','R']:assert c['max_s']<=1e-12,c
  if self.cone in ['A','C']:assert c['min_A_slack']>=-1e-12,c
