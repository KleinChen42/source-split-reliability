"""Outcome-free split stressors with measured overlap and fixed fit records."""
import numpy as np

def controlled_overlap_split(sources,seed,overlap_fraction,fit_source_fraction=.7,within_source_fit_fraction=.7,validation_fraction=.2):
    sources=np.asarray(sources,dtype=str)
    if sources.ndim!=1 or len(sources)<4 or not 0<=overlap_fraction<=1:raise ValueError('Invalid overlap stress inputs')
    unique=np.unique(sources)
    if len(unique)<3:raise ValueError('At least three sources needed')
    rng=np.random.default_rng(seed);shuffled=rng.permutation(unique)
    n=max(1,min(len(unique)-1,int(np.floor(len(unique)*fit_source_fraction))));fit_sources=set(shuffled[:n])
    F=[];leaky=[];unseen=[]
    for source in unique:
        indices=rng.permutation(np.flatnonzero(sources==source))
        if source in fit_sources:
            count=max(1,int(np.floor(len(indices)*within_source_fit_fraction)));F.extend(indices[:count]);leaky.extend(indices[count:])
        else:unseen.extend(indices)
    leaky=rng.permutation(leaky);unseen=rng.permutation(unseen)
    size=min(len(leaky),len(unseen),max(1,int(np.floor(len(sources)*validation_fraction))))
    if not size:raise ValueError('No common validation size across overlap endpoints')
    n_leaky=int(np.floor(size*overlap_fraction+.5));V=np.concatenate([leaky[:n_leaky],unseen[:size-n_leaky]]).astype(int)
    F=np.asarray(F,dtype=int)
    if set(F)&set(V):raise AssertionError('Record leakage in stress construction')
    overlap=np.isin(sources[V],sources[F])
    return F,V,{'requested_overlap_fraction':overlap_fraction,'actual_validation_record_overlap':float(overlap.mean()),
                'fit_records':len(F),'validation_records':len(V),'unused_development_records':len(sources)-len(F)-len(V),
                'design':'F fixed across alpha; fixed V record count; mix held-back records of F sources with entirely unseen sources'}

def fragment_source_ids(sources,seed,fraction):
    """Fragment a fixed fraction of source IDs; retain true IDs for H assessment."""
    sources=np.asarray(sources,dtype=str)
    if not 0<=fraction<=1:raise ValueError('Invalid corruption fraction')
    unique=np.unique(sources);rng=np.random.default_rng(seed)
    selected=set(rng.permutation(unique)[:int(np.floor(len(unique)*fraction+.5))]);observed=[]
    counts={}
    for source in sources:
        ordinal=counts.get(source,0);counts[source]=ordinal+1
        observed.append(f'{source}::fragment{ordinal%2}' if source in selected else source)
    return np.array(observed),{'selected_sources':len(selected),'total_sources':len(unique),'requested_source_fraction':fraction,
                              'sources_actually_split':sum(counts[s]>1 for s in selected)}


def merge_source_ids(sources,seed,fraction):
    """Pair randomly chosen source IDs; an odd unpaired source stays intact."""
    sources=np.asarray(sources,dtype=str)
    if not 0<=fraction<=1:raise ValueError('Invalid corruption fraction')
    unique=np.unique(sources);rng=np.random.default_rng(seed)
    selected=rng.permutation(unique)[:int(np.floor(len(unique)*fraction+.5))]
    mapping={s:s for s in unique}
    for i in range(0,len(selected)-1,2):
        a,b=selected[i:i+2];mapping[a]=mapping[b]=f'corrupt_merged_pair::{i//2}'
    return np.array([mapping[s] for s in sources]),{'requested_source_fraction':fraction,
        'selected_sources':len(selected),'actually_merged_sources':2*(len(selected)//2),'total_sources':len(unique)}
