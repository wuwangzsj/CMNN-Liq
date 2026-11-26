"""Region-equal and record-equal scores from supplied held-out predictions."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss

def scores(pred):
 rows=[]
 for (m,s),g in pred.groupby(['model','seed']):
  loss=(g.p-g.y)**2
  rows.append(dict(model=m,seed=s,n=len(g),regions=g.region.nunique(),macro_brier=loss.groupby(g.region).mean().mean(),record_brier=loss.mean(),logloss=log_loss(g.y,g.p,labels=[0,1]),pooled_auc=roc_auc_score(g.y,g.p) if g.y.nunique()==2 else np.nan))
 return pd.DataFrame(rows)
