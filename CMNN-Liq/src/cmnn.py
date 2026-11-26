"""Original CMNN layers. Use training.Repaired for the paper training objective.
CMNN.fit is the legacy initializer used with epochs=0 by that wrapper.
"""
import time
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from features import SOIL, physical_reference, validate
torch.set_num_threads(1)

def mixed_activation(x):
 n=x.shape[-1];a=round(.4*n);b=round(.4*n)
 convex=nn.functional.elu(x[...,:a]);concave=-nn.functional.elu(-x[...,a:a+b]);z=x[...,a+b:]
 saturated=torch.where(z<=0,nn.functional.elu(z+1)-1,-nn.functional.elu(1-z)+1)
 return torch.cat([convex,concave,saturated],dim=-1)

class MonoLayer(nn.Module):
 def __init__(self,ni,no,first=False,activation=True):
  super().__init__();self.weight=nn.Parameter(torch.empty(no,ni,dtype=torch.float64));nn.init.xavier_uniform_(self.weight);self.bias=nn.Parameter(torch.zeros(no,dtype=torch.float64));self.first=first;self.activation=activation
 def effective(self):return torch.cat([self.weight[:,:-1],self.weight[:,-1:].abs()],1) if self.first else self.weight.abs()
 def forward(self,x):
  z=nn.functional.linear(x,self.effective(),self.bias);return mixed_activation(z) if self.activation else z

class CMNN:
 def __init__(self,config,seed):self.config=config;self.seed=seed
 def inputs(self,d):
  validate(d);return np.column_stack([d[SOIL].to_numpy(float),np.log(physical_reference(d)),np.log(d.pga.to_numpy()/.1)])
 def fit(self,d):
  start=time.perf_counter();torch.manual_seed(self.seed);torch.use_deterministic_algorithms(True)
  self.scaler=StandardScaler().fit(self.inputs(d));x=torch.tensor(self.scaler.transform(self.inputs(d)),dtype=torch.float64);y=torch.tensor(d.y.to_numpy(),dtype=torch.float64)
  cfg=self.config;layers=[MonoLayer(7,cfg['width'],True)]
  for i in range(cfg['depth']-1):layers.append(MonoLayer(cfg['width'],cfg['width']))
  layers.append(MonoLayer(cfg['width'],1,activation=False));self.net=nn.Sequential(*layers);opt=torch.optim.Adam(self.net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);self.loss_trace=[]
  for epoch in range(cfg['epochs']):
   opt.zero_grad();z=self.net(x).ravel();loss=nn.functional.binary_cross_entropy_with_logits(z,y);loss.backward();opt.step()
   if epoch%25==0 or epoch==cfg['epochs']-1:self.loss_trace.append([epoch,float(loss.detach())])
  self.train_seconds=time.perf_counter()-start;self.pga_range=(float(d.pga.min()),float(d.pga.max()));return self
 def predict(self,d):
  with torch.no_grad():return torch.sigmoid(self.net(torch.tensor(self.scaler.transform(self.inputs(d)),dtype=torch.float64))).numpy().ravel()
 def audit(self,d):
  start=time.perf_counter();finite=all(torch.isfinite(p).all().item() for p in self.net.parameters());mins=[float(layer.effective().detach()[:,-1:].min()) if layer.first else float(layer.effective().detach().min()) for layer in self.net]
  x=torch.tensor(self.scaler.transform(self.inputs(d)),dtype=torch.float64,requires_grad=True);grad=torch.autograd.grad(self.net(x).sum(),x)[0][:,-1].detach().numpy()
  return dict(structural_monotonicity=finite and min(mins)>=0,min_constrained_weight=min(mins),min_logit_derivative_scaled=float(grad.min()),certification_seconds=time.perf_counter()-start)
 def thresholds(self,d,rho=.5):
  lo=np.full(len(d),np.log(self.pga_range[0]));hi=np.full(len(d),np.log(self.pga_range[1]));cf=d.copy();cf.pga=np.exp(lo);pl=self.predict(cf);cf.pga=np.exp(hi);ph=self.predict(cf);valid=(pl<=rho)&(ph>=rho)&(ph>pl)
  for _ in range(50):
   mid=(lo+hi)/2;cf.pga=np.exp(mid);p=self.predict(cf);lo=np.where(p<rho,mid,lo);hi=np.where(p>=rho,mid,hi)
  root=(lo+hi)/2;cf.pga=np.exp(root);err=abs(self.predict(cf)-rho)
  return np.where(valid,root,np.nan),valid,err
