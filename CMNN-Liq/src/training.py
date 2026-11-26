"""CMNN branch of the original full-batch Adam / explicit L2 trainer."""
import time, copy
import numpy as np
import torch
from torch import nn
from cmnn import CMNN
DT=torch.float64

class Repaired:
 def __init__(self,name,config,seed):
  if name != "CMNN": raise ValueError("This release retains only the CMNN branch")
  self.name=name;self.config=config;self.seed=seed
 def tensors(self,d):
  x=torch.tensor(self.base.scaler.transform(self.base.inputs(d)),dtype=DT)
  return (x,)
 def forward(self,t):
  return self.base.net(t[0]).ravel(),None
 def fit(self,tr,va=None,epochs=None):
  start=time.perf_counter();cfg=self.config
  init=dict(width=cfg['width'],epochs=0,lr=cfg['lr'],depth=1,weight_decay=0.,bmin=.05,curvature=cfg['curvature'],regularization=0.)
  self.base=CMNN(init,self.seed).fit(tr)
  torch.use_deterministic_algorithms(True);net=self.base.net;opt=torch.optim.Adam(net.parameters(),lr=cfg['lr'],weight_decay=0.)
  t=self.tensors(tr);y=torch.tensor(tr.y.to_numpy(),dtype=DT)
  g=None
  if va is not None:
   vt=self.tensors(va);vy=torch.tensor(va.y.to_numpy(),dtype=DT);_,counts=np.unique(va.region,return_counts=True);cw=va.region.map(va.region.value_counts());vw=torch.tensor(1/cw.to_numpy()/va.region.nunique(),dtype=DT)
  self.trace=[];best=float('inf');bestepoch=0;state=None;cap=epochs if epochs is not None else cfg['max_epochs']
  for ep in range(1,cap+1):
   opt.zero_grad();z,c=self.forward(t);bce=nn.functional.binary_cross_entropy_with_logits(z,y);reg=.5*sum(p.square().sum() for p in net.parameters());cur=((c@g)*c).sum(1).mean() if g is not None else torch.tensor(0.,dtype=DT);loss=bce+cfg['lambda']*reg+cfg['curvature']*cur
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
   loss.backward();grad=float(torch.sqrt(sum(p.grad.square().sum() for p in net.parameters())));opt.step()
   if ep%cfg['check_every']==0 or ep==cap:
    rec=dict(epoch=ep,objective=float(loss.detach()),bce=float(bce.detach()),regularizer=float(reg.detach()),curvature=float(cur.detach()),gradient=grad)
    if va is not None:
     with torch.no_grad():score=float((((torch.sigmoid(self.forward(vt)[0])-vy)**2)*vw).sum())
     rec['validation_macro_brier']=score
     if score<best-cfg['min_delta']:best=score;bestepoch=ep;state=copy.deepcopy(net.state_dict())
    self.trace.append(rec)
    if va is not None and ep-bestepoch>=cfg['patience']:break
  self.last_epoch=ep;self.best_epoch=bestepoch if va is not None else cap
  if va is not None:net.load_state_dict(state)
  self.n_parameters=sum(p.numel() for p in net.parameters());self.train_seconds=time.perf_counter()-start
  return self
 def predict(self,d):return self.base.predict(d)

def candidates(name,cfg):
 return [dict(cfg,width=w,**{'lambda':lam},curvature=0.) for w in cfg['widths'] for lam in cfg['lambdas']]
