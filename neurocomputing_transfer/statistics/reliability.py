"""Established regression baselines and development-only configuration selection."""
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from .core import route,queue_metrics


def development_split(source_ids,flow,seed,fit_fraction=.7):
    sources=np.asarray(source_ids,dtype=str)
    if sources.ndim!=1 or len(sources)<2 or not 0<fit_fraction<1:raise ValueError('Invalid development split')
    rng=np.random.default_rng(seed)
    fit=np.zeros(len(sources),dtype=bool)
    if flow=='record_random':
        n=max(1,min(len(sources)-1,int(np.floor(fit_fraction*len(sources)))))
        fit[rng.permutation(len(sources))[:n]]=True
    elif flow=='source_disjoint':
        unique=np.unique(sources)
        if len(unique)<2:raise ValueError('At least two development sources required')
        n=max(1,min(len(unique)-1,int(np.floor(fit_fraction*len(unique)))))
        fit=np.isin(sources,rng.permutation(unique)[:n])
    else:raise ValueError('Unknown split flow')
    return np.flatnonzero(fit),np.flatnonzero(~fit)


@dataclass
class Reliability:
    scaler: object
    regressor: object
    constant: float|None
    feature_count: int

    def predict(self,features):
        features=np.asarray(features,dtype=float)
        if features.ndim!=2 or features.shape[1]!=self.feature_count or not np.isfinite(features).all():
            raise ValueError('Invalid prediction features')
        if self.constant is not None:return np.full(len(features),self.constant)
        return np.clip(self.regressor.predict(self.scaler.transform(features)),0,1)


def fit_reliability(fit_features,fit_scores,kind,parameter=None):
    """Callers supply F only; scaling never uses validation or holdout inputs."""
    X=np.asarray(fit_features,dtype=float);y=np.asarray(fit_scores,dtype=float)
    if X.ndim!=2 or y.ndim!=1 or len(X)!=len(y) or not len(y):raise ValueError('Invalid fit arrays')
    if not np.isfinite(X).all() or not np.isfinite(y).all() or np.any((y<0)|(y>1)):
        raise ValueError('Missing/invalid features or official scores')
    if kind=='constant':return Reliability(None,None,float(y.mean()),X.shape[1])
    scaler=StandardScaler().fit(X);transformed=scaler.transform(X)
    if kind=='ridge':
        if parameter is None or parameter<=0:raise ValueError('Positive ridge alpha required')
        model=Ridge(alpha=float(parameter),solver='lsqr',tol=1e-8)
    elif kind=='knn':
        if parameter is None or int(parameter)!=parameter or not 1<=parameter<=len(y):
            raise ValueError('Invalid k, or too few F records')
        model=KNeighborsRegressor(n_neighbors=int(parameter),weights='uniform',n_jobs=1)
    else:raise ValueError('Unknown reliability baseline')
    model.fit(transformed,y)
    return Reliability(scaler,model,None,X.shape[1])


def select_on_validation(predicted_scores,official_losses,record_ids,coverage=.7,seed=0):
    """Select by V accepted loss. No holdout argument exists in this interface."""
    if not predicted_scores or set(predicted_scores)!=set(official_losses):
        raise ValueError('Candidate panel mismatch')
    comparisons=[]
    for candidate in sorted(predicted_scores):
        if len(official_losses[candidate])!=len(record_ids):raise ValueError('Incomplete candidate panel')
        mask=route(predicted_scores[candidate],record_ids,coverage,seed,candidate)
        metrics=queue_metrics(official_losses[candidate],mask)
        comparisons.append({'candidate_id':candidate,**metrics})
    chosen=min(comparisons,key=lambda r:(r['accepted_loss_per_total_record'],r['candidate_id']))
    return {'selected_candidate_id':chosen['candidate_id'],'validation_comparisons':comparisons,
            'tie_rule':'lexicographic candidate_id','selection_target':'accepted_loss_per_total_record'}
