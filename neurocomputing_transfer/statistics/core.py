"""Standard estimators with explicit targets and paired resampling.

These are established statistical components, not claimed as new methods.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
from scipy.stats import t


def seed_for(*parts):
    payload=json.dumps(parts,sort_keys=True,separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],'big')


def checked(values, sources):
    values=np.asarray(values,dtype=float)
    sources=np.asarray(sources,dtype=str)
    if values.ndim!=1 or sources.ndim!=1 or len(values)!=len(sources) or not len(values):
        raise ValueError('Nonempty aligned one-dimensional vectors required')
    if not np.isfinite(values).all():raise ValueError('Non-finite observations')
    if any(not x for x in sources):raise ValueError('Empty source ID')
    return values,sources


def cluster_totals(values,sources):
    values,sources=checked(values,sources)
    unique,inverse=np.unique(sources,return_inverse=True)
    size=np.bincount(inverse).astype(float)
    sums=np.bincount(inverse,weights=values)
    return unique,size,sums


def point_estimate(values,sources,target='record_workload'):
    _,size,sums=cluster_totals(values,sources)
    if target=='record_workload':return float(sums.sum()/size.sum())
    if target=='source_equal':return float(np.mean(sums/size))
    raise ValueError('Unknown target')


def analytic_interval(values,sources,target='record_workload',alpha=.05):
    _,size,sums=cluster_totals(values,sources)
    count=len(size)
    if count<2:raise ValueError('At least two sources required for an interval')
    if not 0<alpha<1:raise ValueError('Invalid alpha')
    if target=='record_workload':
        estimate=sums.sum()/size.sum()
        residual=sums-estimate*size
        variance=count/(count-1)*np.sum(residual**2)/size.sum()**2
    elif target=='source_equal':
        source_means=sums/size
        estimate=source_means.mean()
        variance=np.var(source_means,ddof=1)/count
    else:raise ValueError('Unknown target')
    se=np.sqrt(variance);critical=t.ppf(1-alpha/2,count-1)
    return {'estimate':float(estimate),'se':float(se),'lower':float(estimate-critical*se),
            'upper':float(estimate+critical*se),'sources':count,'records':int(size.sum()),
            'target':target,'alpha':alpha,'df':count-1,'clipped':False}


def bootstrap_draws(values,sources,target='record_workload',unit='source',
                    replicates=4999,seed=0,batch_size=128):
    values,sources=checked(values,sources)
    _,size,sums=cluster_totals(values,sources)
    rng=np.random.default_rng(seed)
    if replicates<1 or batch_size<1:raise ValueError('Positive replicate/batch count required')
    if unit not in ['source','record']:raise ValueError('Unknown sampling unit')
    if unit=='record' and target!='record_workload':
        raise ValueError('IID-record mean does not estimate the source-equal target')
    if target not in ['record_workload','source_equal']:raise ValueError('Unknown target')
    outputs=[]
    for offset in range(0,replicates,batch_size):
        batch=min(batch_size,replicates-offset)
        if unit=='record':
            indices=rng.integers(0,len(values),size=(batch,len(values)))
            draws=values[indices].mean(axis=1)
        else:
            weights=rng.multinomial(len(size),np.full(len(size),1/len(size)),size=batch)
            if target=='record_workload':draws=(weights@sums)/(weights@size)
            else:draws=(weights@(sums/size))/len(size)
        outputs.append(draws)
    return np.concatenate(outputs)


def contrast_draws(a,b,sources,target='record_workload',unit='source',paired=True,
                   replicates=4999,seed=0):
    a,sources=checked(a,sources);b,_=checked(b,sources)
    if paired:
        return bootstrap_draws(a-b,sources,target,unit,replicates,seed)
    # Unpaired inference is a diagnostic when observations truly share sources.
    return (bootstrap_draws(a,sources,target,unit,replicates,seed_for(seed,'A'))
            -bootstrap_draws(b,sources,target,unit,replicates,seed_for(seed,'B')))


def percentile_interval(draws,alpha=.05):
    draws=np.asarray(draws,dtype=float)
    if draws.ndim!=1 or not len(draws) or not np.isfinite(draws).all():
        raise ValueError('Finite one-dimensional bootstrap draws required')
    if not 0<alpha<1:raise ValueError('Invalid alpha')
    lower,upper=np.quantile(draws,[alpha/2,1-alpha/2],method='linear')
    return float(lower),float(upper)


def route(predicted_score,record_ids,coverage=.7,seed=0,candidate_id='policy'):
    """Label-free exact-capacity selection; outcome is intentionally not an argument."""
    scores=np.asarray(predicted_score,dtype=float)
    ids=list(map(str,record_ids))
    if scores.ndim!=1 or len(scores)!=len(ids) or not len(ids):raise ValueError('Invalid routing inputs')
    if not np.isfinite(scores).all() or len(set(ids))!=len(ids):raise ValueError('Nonfinite scores or duplicate occurrence IDs')
    if not 0<=coverage<=1:raise ValueError('Invalid coverage')
    tie=np.array([seed_for(seed,candidate_id,r) for r in ids],dtype=np.uint64)
    order=np.lexsort((tie,-scores))
    mask=np.zeros(len(ids),dtype=bool)
    mask[order[:int(np.floor(coverage*len(ids)))]]=True
    return mask


def queue_metrics(loss,mask):
    loss=np.asarray(loss,dtype=float);mask=np.asarray(mask)
    if loss.ndim!=1 or mask.shape!=loss.shape or mask.dtype!=bool or not len(loss):
        raise ValueError('Aligned loss and boolean mask required')
    if not np.isfinite(loss).all() or np.any((loss<0)|(loss>1)):raise ValueError('Loss outside [0,1]')
    accepted=int(mask.sum());total=float(loss[mask].sum())
    return {'accepted_loss_per_total_record':total/len(loss),
            'selective_risk':total/accepted if accepted else None,
            'coverage':accepted/len(loss),'accepted':accepted,'records':len(loss)}
