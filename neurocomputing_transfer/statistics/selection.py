"""F/V fitting and label-free H routing with no H-outcome argument."""
import numpy as np
from .reliability import development_split,fit_reliability,select_on_validation
from .core import route

def fit_select_policy(XD,YD,XH,source_ids,record_ids_D,record_ids_H,flow,seed,coverage=.7,fit_fraction=.7,
                      native_D=None,native_H=None,ridge_alphas=(.1,1,10),knn_k=(5,20,50),include_constant=True,split_indices=None):
    candidates=sorted(XD)
    if not candidates or set(candidates)!=set(YD) or set(candidates)!=set(XH):raise ValueError('Candidate panel mismatch')
    if (native_D is None)!=(native_H is None):raise ValueError('Native confidence must be available on D and H')
    if native_D is not None and (set(native_D)!=set(candidates) or set(native_H)!=set(candidates)):raise ValueError('Incomplete native panel')
    F,V=development_split(source_ids,flow,seed,fit_fraction) if split_indices is None else split_indices
    F=np.asarray(F,dtype=int);V=np.asarray(V,dtype=int)
    if not len(F) or not len(V) or len(set(F))!=len(F) or len(set(V))!=len(V) or set(F)&set(V):raise ValueError('Invalid F/V indices')
    if np.any(F<0) or np.any(V<0) or np.any(F>=len(record_ids_D)) or np.any(V>=len(record_ids_D)):raise ValueError('F/V outside D')
    fitted={};validation={};losses={};base={}
    for candidate in candidates:
        xd=np.asarray(XD[candidate]);yd=np.asarray(YD[candidate]);xh=np.asarray(XH[candidate])
        if len(xd)!=len(record_ids_D) or len(yd)!=len(xd) or len(xh)!=len(record_ids_H):raise ValueError('Incomplete record panel')
        for kind,param in ([('constant',None)] if include_constant else [])+[('ridge',x) for x in ridge_alphas]+[('knn',x) for x in knn_k]:
            key=f'{candidate}::{kind}:{param}';fitted[key]=fit_reliability(xd[F],yd[F],kind,param)
            validation[key]=fitted[key].predict(xd[V]);losses[key]=1-yd[V];base[key]=candidate
        if native_D is not None:
            key=f'{candidate}::native';validation[key]=np.asarray(native_D[candidate])[V]
            losses[key]=1-yd[V];base[key]=candidate;fitted[key]=None
    selection=select_on_validation(validation,losses,np.asarray(record_ids_D)[V],coverage,seed)
    key=selection['selected_candidate_id'];candidate=base[key];model=fitted[key]
    h_score=model.predict(XH[candidate]) if model is not None else np.asarray(native_H[candidate],dtype=float)
    h_mask=route(h_score,record_ids_H,coverage,seed,key)
    return {'selected_config':key,'selected_candidate':candidate,'model':model,'H_reliability':h_score,'H_mask':h_mask,
            'F_indices':F,'V_indices':V,'validation':selection['validation_comparisons'],
            'F_V_source_overlap':len(set(np.asarray(source_ids)[F])&set(np.asarray(source_ids)[V]))}
