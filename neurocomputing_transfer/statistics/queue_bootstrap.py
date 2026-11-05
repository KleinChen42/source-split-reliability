"""Conditional-on-fitted-policy cluster bootstrap with capacity recomputed per queue."""
import numpy as np
from .core import seed_for


def ranked_order(scores,record_ids,seed=0,candidate_id='policy'):
    scores=np.asarray(scores,dtype=float);ids=list(map(str,record_ids))
    if scores.ndim!=1 or not len(scores) or len(ids)!=len(scores) or len(set(ids))!=len(ids):
        raise ValueError('Unique aligned record IDs required')
    if not np.isfinite(scores).all():raise ValueError('Nonfinite reliability')
    tie=np.array([seed_for(seed,candidate_id,r) for r in ids],dtype=np.uint64)
    return np.lexsort((tie,-scores))


def weighted_queue_loss(loss,order,record_multiplicity,coverage=.7):
    """Each repeated record retains its tie key; copies have identical loss.

    The capacity floor(q*N*) is rebuilt for every bootstrap queue, including
    partial acceptance of identical copies at the boundary. This is not the
    bootstrap of a fixed acceptance mask.
    """
    loss=np.asarray(loss,dtype=float);order=np.asarray(order)
    weights=np.asarray(record_multiplicity)
    if weights.ndim==1:weights=weights[None,:]
    if loss.ndim!=1 or weights.ndim!=2 or weights.shape[1]!=len(loss):raise ValueError('Unaligned queue')
    if not np.array_equal(np.sort(order),np.arange(len(loss))):raise ValueError('Invalid order')
    if np.any(weights<0) or not np.isfinite(weights).all() or np.any(weights!=np.floor(weights)):
        raise ValueError('Nonnegative integer multiplicities required')
    if not np.isfinite(loss).all() or np.any((loss<0)|(loss>1)) or not 0<=coverage<=1:
        raise ValueError('Invalid loss or coverage')
    w=weights[:,order];totals=w.sum(axis=1)
    if np.any(totals==0):raise ValueError('Empty bootstrap queue')
    before=np.cumsum(w,axis=1)-w
    capacities=np.floor(coverage*totals)[:,None]
    accepted=np.minimum(w,np.maximum(capacities-before,0))
    return accepted@loss[order]/totals


def paired_queue_draws(policies,sources,record_ids,replicates=4999,seed=0,coverage=.7,batch_size=16):
    """Policies: sequence of paired dictionaries A/B with score, loss, tie_seed/id.

    A-minus-B is averaged across split seeds using the SAME source bootstrap
    weights. Split seeds are never treated as independent holdout samples.
    Policies must already be fixed without H labels; this function only infers.
    """
    sources=np.asarray(sources,dtype=str)
    if sources.ndim!=1 or len(sources)!=len(record_ids) or not policies:raise ValueError('Invalid panel')
    unique,inverse=np.unique(sources,return_inverse=True)
    if len(unique)<2 or replicates<1 or batch_size<1:raise ValueError('Insufficient bootstrap units')
    prepared=[]
    for pair in policies:
        if set(pair)!={'A','B'}:raise ValueError('Paired A/B policies required')
        sides=[]
        for side in ['A','B']:
            p=pair[side]
            order=ranked_order(p['scores'],record_ids,p.get('tie_seed',0),p.get('candidate_id','policy'))
            if len(order)!=len(sources):raise ValueError('Incomplete policy')
            sides.append((p['loss'],order))
        prepared.append(sides)
    rng=np.random.default_rng(seed);draws=[]
    for offset in range(0,replicates,batch_size):
        n=min(batch_size,replicates-offset)
        source_weights=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=n)
        w=source_weights[:,inverse];delta=np.zeros(n)
        for sides in prepared:
            delta+=weighted_queue_loss(*sides[0],w,coverage)-weighted_queue_loss(*sides[1],w,coverage)
        draws.extend(delta/len(prepared))
    return np.asarray(draws)
