"""Known-truth mechanisms for the v2 protocol. Smoke results are not confirmatory."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import beta as beta_function, hyp1f1
from scipy.stats import t, norm
from .core import seed_for, percentile_interval


def beta_parameters(mu,rho):
    if not 0<mu<1 or not 0<rho<1:raise ValueError('Interior mu/rho required')
    return mu*(1/rho-1),(1-mu)*(1/rho-1)


def weighted_p(mu,rho,kappa):
    if rho==0:return mu
    a,b=beta_parameters(mu,rho)
    return float(mu*hyp1f1(a+1,a+b+1,kappa)/hyp1f1(a,a+b,kappa))


def primary_truth(p,kappa_magnitude=4):
    mu=p['mu'];m=p['mean_group_size'];rho=p['rho'];regime=p['size_regime']
    if regime in ['balanced','unequal_noninformative']:return mu,mu
    kappa=kappa_magnitude*(1 if regime=='informative_positive' else -1)
    return (mu+(m-1)*weighted_p(mu,rho,kappa))/m,mu


def primary_sample(p,rng,kappa_magnitude=4):
    S=p['sources'];mu=p['mu'];rho=p['rho'];m=p['mean_group_size']
    latent=np.full(S,mu) if rho==0 else rng.beta(*beta_parameters(mu,rho),size=S)
    regime=p['size_regime']
    if regime=='balanced':size=np.full(S,m,dtype=int)
    elif regime=='unequal_noninformative':size=1+rng.poisson(m-1,size=S)
    else:
        kappa=kappa_magnitude*(1 if regime=='informative_positive' else -1)
        if rho==0:normalizer=1.
        else:
            a,b=beta_parameters(mu,rho)
            normalizer=np.exp(-kappa*mu)*hyp1f1(a,a+b,kappa)
        size=1+rng.poisson((m-1)*np.exp(kappa*(latent-mu))/normalizer)
    success=rng.binomial(size,latent)
    return size,success


def paired_probabilities(p,rng,eta=.5):
    S=p['sources'];rho=p['rho'];delta=p['marginal_effect'];c=p['within_record_coupling']
    if not 0<=eta<=1 or not 0<=c<=1 or not 0<=delta<1:raise ValueError('Invalid coupling/effect')
    latent=rng.beta(*beta_parameters(.5,rho),size=S)
    other=rng.beta(*beta_parameters(.5,rho),size=S)
    other=np.where(rng.random(S)<eta,latent,other)
    pa=delta+(1-delta)*latent;pb=(1-delta)*other
    both=(1-c)*pa*pb+c*np.minimum(pa,pb)
    probs=np.stack([both,pa-both,pb-both,1-pa-pb+both],axis=1)
    if probs.min() < -1e-12:raise ValueError('Invalid joint probabilities')
    probs=np.maximum(probs,0);probs/=probs.sum(axis=1,keepdims=True)
    return pa,pb,probs


def primary_intervals(size,success,B,rng):
    S=len(size);N=size.sum();estimate=success.sum()/N;means=success/size
    weights=rng.multinomial(S,np.full(S,1/S),size=B)
    iid=rng.binomial(N,estimate,size=B)/N
    record=(weights@success)/(weights@size)
    equal=(weights@means)/S
    ratio_se=np.sqrt(S/(S-1)*np.sum((success-estimate*size)**2)/N**2)
    equal_se=np.std(means,ddof=1)/np.sqrt(S)
    tc=t.ppf(.975,S-1);zc=norm.ppf(.975)
    return {
        'record_iid_bootstrap':percentile_interval(iid),
        'record_source_bootstrap':percentile_interval(record),
        'record_source_sandwich_t':(estimate-tc*ratio_se,estimate+tc*ratio_se),
        'source_equal_bootstrap':percentile_interval(equal),
        'source_equal_t':(means.mean()-tc*equal_se,means.mean()+tc*equal_se),
        'source_equal_z':(means.mean()-zc*equal_se,means.mean()+zc*equal_se),
    }


def paired_intervals(joint,m,B,rng):
    S=len(joint);N=S*m
    a=joint[:,0]+joint[:,1];b=joint[:,0]+joint[:,2]
    empirical=joint.sum(axis=0)/N
    resampled_joint=rng.multinomial(N,empirical,size=B)
    record_paired=(resampled_joint[:,1]-resampled_joint[:,2])/N
    record_unpaired=(rng.binomial(N,a.sum()/N,size=B)-rng.binomial(N,b.sum()/N,size=B))/N
    weights=rng.multinomial(S,np.full(S,1/S),size=B)
    independent=rng.multinomial(S,np.full(S,1/S),size=B)
    source_paired=(weights@(a-b))/N
    source_unpaired=(weights@a-independent@b)/N
    return {name:percentile_interval(draws) for name,draws in [
        ('record_paired',record_paired),('record_unpaired',record_unpaired),
        ('source_paired',source_paired),('source_unpaired',source_unpaired)]}


def reference_checks():
    results=[]
    for rho in [.1,.3,.5]:
        a,b=beta_parameters(.5,rho)
        for kappa in [-4,4]:
            factor=beta_function(a,b)
            z=quad(lambda x:np.exp(kappa*x)/factor,0,1,weight='alg',wvar=(a-1,b-1),epsabs=1e-10)[0]
            first=quad(lambda x:x*np.exp(kappa*x)/factor,0,1,weight='alg',wvar=(a-1,b-1),epsabs=1e-10)[0]
            error=abs(weighted_p(.5,rho,kappa)-first/z)
            assert error<1e-9
            results.append({'rho':rho,'kappa':kappa,'independent_quadrature_error':error})
    return results


def run_cell(cell,plan,mode):
    p=json.loads(cell['parameters_json']);family=cell['family']
    settings=plan['simulation'][mode];R=settings['mc_replicates'];B=settings['bootstrap_replicates']
    methods=plan['simulation'][family]['methods']
    coverage={m:[] for m in methods};width={m:[] for m in methods};rejections={m:[] for m in methods}
    truth_r,truth_s=primary_truth(p,plan['simulation']['informative_size_kappa']) if family=='primary' else (p['marginal_effect'],p['marginal_effect'])
    started=time.monotonic()
    for replicate in range(R):
        data_rng=np.random.default_rng(seed_for(plan['rng']['master_seed'],plan['plan_version'],cell['cell_id'],mode,replicate,'data'))
        boot_rng=np.random.default_rng(seed_for(plan['rng']['master_seed'],plan['plan_version'],cell['cell_id'],mode,replicate,'bootstrap'))
        if family=='primary':
            size,success=primary_sample(p,data_rng,plan['simulation']['informative_size_kappa'])
            intervals=primary_intervals(size,success,B,boot_rng)
        else:
            _,_,probs=paired_probabilities(p,data_rng,plan['simulation']['paired_source_shared_probability'])
            joint=data_rng.multinomial(p['group_size'],probs)
            intervals=paired_intervals(joint,p['group_size'],B,boot_rng)
        for method,(low,high) in intervals.items():
            truth=truth_s if method.startswith('source_equal') else truth_r
            coverage[method].append(low<=truth<=high);width[method].append(high-low)
            rejections[method].append(not low<=0<=high)
    rows=[]
    for method in methods:
        cp=float(np.mean(coverage[method]));reject=float(np.mean(rejections[method]))
        rows.append({'cell_id':cell['cell_id'],'family':family,'method':method,'mode':mode,
                     'parameters_json':cell['parameters_json'],'replicates':R,'bootstrap_replicates':B,
                     'target':('source_equal_success' if method.startswith('source_equal') else 'record_success') if family=='primary' else 'paired_success_difference',
                     'truth':truth_s if method.startswith('source_equal') else truth_r,
                     'coverage':cp,'coverage_mcse':float(np.sqrt(cp*(1-cp)/R)),
                     'mean_width':float(np.mean(width[method])),
                     'reject_zero':reject if family=='paired' else None,
                     'reject_zero_mcse':float(np.sqrt(reject*(1-reject)/R)) if family=='paired' else None,
                     'zero_test_role':('type_I' if truth_r==0 else 'raw_power_not_size_adjusted') if family=='paired' else None,
                     'cell_elapsed_seconds':time.monotonic()-started})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--grid',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['smoke','formal'],default='smoke')
    parser.add_argument('--cell-id');parser.add_argument('--protocol',type=Path)
    args=parser.parse_args();plan=json.loads(args.config.read_text())
    if args.mode=='formal':
        if args.protocol is None:raise ValueError('Formal simulation requires frozen protocol')
        from neurocomputing_transfer.protocol import load_frozen,bound_path
        root=Path(__file__).resolve().parents[2]
        protocol,frozen_plan,_=load_frozen(root,args.protocol)
        if args.config.resolve()!=bound_path(root,protocol['bindings']['plan']) or plan!=frozen_plan:raise ValueError('Frozen config mismatch')
        if args.grid.resolve()!=bound_path(root,protocol['bindings']['simulation_grid']):raise ValueError('Frozen simulation grid mismatch')
    reference=reference_checks()
    with args.grid.open(encoding='utf-8-sig',newline='') as f:cells=list(csv.DictReader(f))
    if args.cell_id:cells=[c for c in cells if c['cell_id']==args.cell_id]
    if not cells:raise ValueError('No cells selected')
    args.output.mkdir(parents=True,exist_ok=True)
    for cell in cells:
        target=args.output/(cell['cell_id']+'.json')
        if target.exists():raise ValueError('Refusing to overwrite an existing simulation cell')
        result=run_cell(cell,plan,args.mode)
        target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'cell':cell['cell_id'],'mode':args.mode,'method_rows':len(result)}),flush=True)
    (args.output/'reference_checks.json').write_text(json.dumps(reference,indent=2)+'\n')


if __name__=='__main__':main()
