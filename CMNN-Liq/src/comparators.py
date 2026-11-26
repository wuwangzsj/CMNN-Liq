"""Current-state A/N, M-noFS, and B0/B1 original model routines."""
import copy,time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.special import expit
from cmnn import MonoLayer,mixed_activation
from features import inputs
DT=torch.float64
def tensor(x): return torch.tensor(x,dtype=DT)
core=SimpleNamespace(inputs=inputs)
NN_FAMILIES=["M-noFS","N","A"]

def features(d,family):
    raw=core.inputs(d)
    if family=='M-noFS':return raw[:,[0,1,2,3,4,6]]
    if family=='B0':return (raw[:,-2]-raw[:,-1])[:,None]
    if family in ('N','A'):
        raw[:,-2]-=raw[:,-1]
    return raw

class Network(nn.Module):
    def __init__(self,family,H,seed,mu,scale):
        super().__init__();self.family=family;self.H=H
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.hidden=MonoLayer(7,H,True);self.out=MonoLayer(H,1,activation=False)
        self.original_initial={k:v.clone() for k,v in self.state_dict().items()}
        if family=='M-noFS':
            self.hidden.weight=nn.Parameter(self.hidden.weight[:,[0,1,2,3,4,6]].clone())
        self.register_buffer('mu',tensor(mu));self.register_buffer('scale',tensor(scale))
    def effective(self):
        w=self.hidden.weight
        if self.family=='A':return torch.cat((w[:,:5],-w[:,5:6].abs(),w[:,6:].abs()),1)
        return torch.cat((w[:,:-1],w[:,-1:].abs()),1)
    def preactivation(self,raw):return nn.functional.linear((raw-self.mu)/self.scale,self.effective(),self.hidden.bias)
    def forward(self,raw):
        h=self.preactivation(raw)
        return nn.functional.linear(mixed_activation(h),self.out.weight.abs(),self.out.bias).ravel()
    def path_du(self,raw):
        h=self.preactivation(raw);a=round(.4*self.H);b=2*a;z=h[:,b:]
        deriv=torch.cat((torch.where(h[:,:a]>0,1.,torch.exp(h[:,:a])),
            torch.where(h[:,a:b]<0,1.,torch.exp(-h[:,a:b])),
            torch.where(z<=0,torch.where(z+1>0,1.,torch.exp(z+1)),
                torch.where(1-z>0,1.,torch.exp(1-z)))),1)
        w=self.effective();slope=w[:,-1]/self.scale[-1]
        if self.family in ('N','A'):slope=slope-w[:,-2]/self.scale[-2]
        return (self.out.weight.abs()*deriv*slope).sum(1)

def fit(family,H,lam,seed,tr,va=None,epochs=None):
    st=time.perf_counter();cpu_start=time.process_time();f=features(tr,family);sc=StandardScaler().fit(f)
    x=(f-sc.mean_)/sc.scale_;y=tr.y.to_numpy(float)
    common=dict(family=family,H=H,lam=lam,seed=seed,mu=sc.mean_,scale=sc.scale_,
        train_ids=tr.row_id.tolist(),valid_ids=[] if va is None else va.row_id.tolist(),
        input_order=(['ln_FS_current'] if family=='B0' else
            ['mw','depth','stress_ratio','qt1n','ic']+([] if family=='M-noFS' else
                ['ln_FS_current' if family in ('N','A') else 'ln_FS_ref'])+['ln_PGA_ratio']))
    if family in ['B0','B1']:
        if family=='B0':x=-x
        X=np.column_stack((np.ones(len(x)),x))
        def fun(theta):
            z=X@theta;p=expit(z)
            return np.mean(np.logaddexp(0,z)-y*z)+lam/2*(theta@theta),X.T@(p-y)/len(y)+lam*theta
        bounds=[(None,None)]*X.shape[1];bounds[-1]=(0,None)
        opt=minimize(fun,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',bounds=bounds,
            options=dict(ftol=1e-12,gtol=1e-8,maxiter=10000,maxls=50))
        loss,grad=fun(opt.x);pg=grad.copy()
        if opt.x[-1]<=0 and grad[-1]>0:pg[-1]=0
        audit=dict(success=bool(opt.success),status=int(opt.status),message=str(opt.message),
            iterations=int(opt.nit),evaluations=int(opt.nfev),objective=float(loss),
            projected_gradient=float(np.max(np.abs(pg))),gradient=grad,coefficients=opt.x)
        if not np.isfinite(opt.x).all() or not np.isfinite(loss) or opt.nit>=10000 or np.max(np.abs(pg))>1e-6:
            raise RuntimeError('LOGISTIC_NUMERICAL_FAIL '+str(audit))
        return dict(**common,coef=opt.x,initial=np.zeros(X.shape[1]),solver=audit,
            parameters=len(opt.x),seconds=time.perf_counter()-st,cpu_seconds=time.process_time()-cpu_start,best_epochs=0,actual_epochs=int(opt.nit),
            trace=[],best_score=None)
    net=Network(family,H,seed,sc.mean_,sc.scale_);initial=copy.deepcopy(net.state_dict())
    xx=tensor(f);yy=tensor(y);opt=torch.optim.Adam(net.parameters(),lr=.003,weight_decay=0.)
    if va is not None:
        vx=tensor(features(va,family));vy=tensor(va.y.to_numpy())
        vw=tensor(1/va.region.map(va.region.value_counts()).to_numpy()/va.region.nunique())
    best=np.inf;best_epoch=0;state=None;trace=[]
    for epoch in range(1,(1000 if epochs is None else epochs)+1):
        opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(net(xx),yy)
        loss=loss+lam/2*sum(p.square().sum() for p in net.parameters())
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();opt.step()
        if epoch%10==0 or epoch==epochs:
            row=dict(epoch=epoch,objective=float(loss.detach()))
            if va is not None:
                with torch.no_grad():v=float(((torch.sigmoid(net(vx))-vy).square()*vw).sum())
                if not np.isfinite(v):raise RuntimeError('Nonfinite validation')
                row['validation_macro_brier']=v
                if v<best-1e-5:best=v;best_epoch=epoch;state=copy.deepcopy(net.state_dict())
            trace.append(row)
            if va is not None and epoch-best_epoch>=100:break
    if va is not None:
        if state is None:raise RuntimeError('No selected checkpoint')
        net.load_state_dict(state)
    net.eval()
    return dict(**common,state=net.state_dict(),initial=initial,original_initial=net.original_initial,
        initial_effective=Network(family,H,seed,sc.mean_,sc.scale_).effective().detach(),
        parameters=sum(p.numel() for p in net.parameters()),seconds=time.perf_counter()-st,cpu_seconds=time.process_time()-cpu_start,
        best_epochs=best_epoch if va is not None else epochs,actual_epochs=epoch,trace=trace,
        best_score=best if va is not None else None)

def restore(obj):
    net=Network(obj['family'],obj['H'],obj['seed'],obj['mu'],obj['scale'])
    net.load_state_dict(obj['state']);net.eval();return net

def predict(obj,d):
    raw=features(d,obj['family'])
    if obj['family'] in ['B0','B1']:
        x=(raw-obj['mu'])/obj['scale']
        if obj['family']=='B0':x=-x
        return expit(obj['coef'][0]+x@obj['coef'][1:])
    with torch.no_grad():return torch.sigmoid(restore(obj)(tensor(raw))).numpy()
