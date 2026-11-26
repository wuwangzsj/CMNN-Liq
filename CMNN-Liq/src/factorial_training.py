"""Original matched training kernel, without file-cache and Git wrappers."""
import time,copy
import numpy as np
from features import inputs
from factorial import Net,tensor,torch,nn

def train_fixed(tr,model,seed,H,lam,epochs=None,va=None):
 x=inputs(tr);mu=x.mean(0);sc=x.std(0);sc=np.where(sc==0,1.,sc);t=x[:,5]-x[:,6]
 stats=dict(mu=mu,scale=sc,muv=t.mean(),sv=t.std() or 1.)
 net=Net({'M0':'G1','A0':'G3'}.get(model,model),seed,stats,H=H);net.assert_feasible()
 xx=tensor(x);yy=tensor(tr.y.to_numpy(float));opt=torch.optim.Adam(net.parameters(),lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=0.)
 if va is not None:
  vx=tensor(inputs(va));vy=tensor(va.y.to_numpy(float));vw=tensor(1/va.region.map(va.region.value_counts()).to_numpy()/va.region.nunique())
 best=np.inf;bestep=0;state=None;trace=[];start=time.perf_counter()
 for ep in range(1,(1000 if epochs is None else int(epochs))+1):
  opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(net(xx),yy)+lam*net.penalty()
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
  loss.backward();opt.step();net.project();net.assert_feasible()
  if ep%10==0 or ep==epochs:
   row=dict(epoch=ep,loss=float(loss.detach()))
   if va is not None:
    with torch.no_grad():val=float(((torch.sigmoid(net(vx))-vy).square()*vw).sum())
    row['valid_macro_brier']=val
    if val<best-1e-5:best=val;bestep=ep;state=copy.deepcopy(net.state_dict())
   trace.append(row)
   if va is not None and ep-bestep>=100:break
 if va is not None:net.load_state_dict(state)
 w,b=net.canonical();out=dict(mean=mu,scale=sc,w1=w.detach().numpy(),b1=b.detach().numpy(),w2=net.V.detach().numpy(),b2=net.bo.detach().numpy())
 meta=dict(H=H,lam=lam,seed=seed,model=model,epochs=bestep if va is not None else ep,actual_epochs=ep,seconds=time.perf_counter()-start,train_ids=tr.row_id.tolist(),valid_ids=[] if va is None else va.row_id.tolist(),best_score=best if va is not None else None,parameter_norm=float(np.sqrt(sum(p.detach().square().sum() for p in net.parameters()))),effective_units=int((net.V.detach().numpy()>1e-10).sum()),trace=trace)
 return out,meta
