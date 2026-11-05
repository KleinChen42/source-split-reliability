"""Descriptive dependence and continuous-score calibration; no coverage claims."""
import numpy as np
from .core import checked,cluster_totals

def source_structure(values,sources):
    values,sources=checked(values,sources)
    unique,size,sums=cluster_totals(values,sources)
    # cluster_totals returns the unique labels, counts, and totals.
    S=len(size);N=len(values);means=sums/size;grand=values.mean()
    within=sum(float(np.sum((values[sources==g]-means[i])**2)) for i,g in enumerate(unique))
    between=float(np.sum(size*(means-grand)**2));rho=None
    if S>1 and N>S:
        msb=between/(S-1);msw=within/(N-S);n0=(N-float(size@size)/N)/(S-1)
        denominator=msb+(n0-1)*msw
        if denominator>0:rho=(msb-msw)/denominator
    return {'sources':S,'records':N,'size_min':int(size.min()),'size_max':int(size.max()),
            'size_quartiles':np.quantile(size,[.25,.5,.75]).tolist(),
            'size_mean':float(size.mean()),'source_weight_max':float(size.max()/N),
            'source_weight_squared_sum':float(np.sum((size/N)**2)),
            'record_mean':float(grand),'source_equal_mean':float(means.mean()),
            'record_minus_source_mean':float(grand-means.mean()),'oneway_anova_icc_descriptive':rho,
            'size_score_covariance_population_denominator':float(np.mean((size-size.mean())*(means-means.mean()))),
            'icc_caveat':'one-way random-intercept moment summary; not proof of exchangeability or causal dependence'}

def continuous_calibration(predicted,observed,edges=None):
    p=np.asarray(predicted,dtype=float);y=np.asarray(observed,dtype=float)
    if p.ndim!=1 or y.shape!=p.shape or not len(p) or not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Finite aligned arrays required')
    if np.any((p<0)|(p>1)|(y<0)|(y>1)):raise ValueError('Scores must be in [0,1]')
    edges=np.linspace(0,1,11) if edges is None else np.asarray(edges,dtype=float)
    if edges.ndim!=1 or len(edges)<2 or edges[0]!=0 or edges[-1]!=1 or np.any(np.diff(edges)<=0):raise ValueError('Bins must partition [0,1]')
    ids=np.minimum(np.searchsorted(edges,p,side='right')-1,len(edges)-2);bins=[];ece=0.
    for b in range(len(edges)-1):
        mask=ids==b;n=int(mask.sum())
        pm=float(p[mask].mean()) if n else None;ym=float(y[mask].mean()) if n else None
        if n:ece+=n/len(p)*abs(pm-ym)
        bins.append({'lower':float(edges[b]),'upper':float(edges[b+1]),'records':n,'predicted_mean':pm,'observed_mean':ym})
    return {'score_prediction_mse':float(np.mean((p-y)**2)),'binned_absolute_calibration_gap':ece,
            'bins':bins,'interpretation':'MSE includes irreducible error; binned gap is descriptive and bin-dependent'}
