"""Aggregate the complete extension matrix and infer with paired source queues.

Default: report preparation status only. --gpu-ready analyzes actual completed
fits after resource handoff. --self-test uses synthetic arrays exclusively.
Portable numerical inputs permit later CPU replay without media or model runs.
"""
from pathlib import Path
from collections import Counter,defaultdict
from functools import lru_cache
import argparse
import csv
import gzip
import json
import sys
import tempfile
import numpy as np

from extension_layout import DATASETS,COHORTS,FLOWS,SEEDS,cells,units,unit_path,plan
from prepare_extension import original_seed

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CAPACITIES=np.array([.1,.3,.5,.7,.9])
sys.path.insert(0,str(ROOT))


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+".part")
    partial.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    partial.replace(path)


def save_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+".part")
    with partial.open("wb") as f:np.savez_compressed(f,**arrays)
    partial.replace(path)


def write_csv(path,rows):
    iterator=iter(rows)
    first=next(iterator,None)
    if first is None:raise ValueError("Empty analysis table")
    path.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if path.suffix==".gz" else open
    with opener(path,"wt",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(first))
        writer.writeheader();writer.writerow(first);writer.writerows(iterator)


@lru_cache(maxsize=1)
def identity_index():
    with (HERE/"prepared/all_record_source_indices.csv").open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def source_design(dataset,cohort,record_ids,source_ids):
    expected={r["record_id"]:r for r in identity_index() if r["dataset"]==dataset and r["partition"]==cohort}
    actual=dict(zip(map(str,record_ids),map(str,source_ids)))
    if len(actual)!=len(record_ids) or actual!={rid:r["source_id"] for rid,r in expected.items()}:
        raise ValueError(f"{dataset}/{cohort}: evaluation IDs differ from the recovered cohort")
    unique,inverse=np.unique(source_ids,return_inverse=True)
    strata={}
    for rid,sid in actual.items():
        label=expected[rid]["official_split"] if (dataset,cohort)==("activitynetqa","H2") else "all"
        if sid in strata and strata[sid]!=label:raise ValueError("A video source spans different bootstrap strata")
        strata[sid]=label
    return unique,inverse,np.array([strata[str(s)] for s in unique])


def geometry_key(rep):
    return {"weighted_TVn_0.00":"TVn","weighted_TVn_1.00":"block_TVn"}.get(rep,rep)


def load_fits(results,dataset,cohort):
    descriptors=cells(dataset)
    index={(r["task"],r["representation"],r["panel"],r["fixed_candidate"]):i for i,r in enumerate(descriptors)}
    count=len(descriptors)
    assignment=np.full((SEEDS,2,count),-1,dtype=np.int32)
    configs=np.empty((SEEDS,2,count),dtype="U160");candidates=np.empty((SEEDS,2,count),dtype="U100")
    orders=[];losses=[];semantic=defaultdict(list);policy_keys=[];routing={}
    reference_ids=None;reference_sources=None
    for u in (x for x in units() if x["dataset"]==dataset):
        folder=unit_path(results,u)
        complete=json.loads((folder/"complete.json").read_text())
        if complete.get("status")!="UNIT_COMPLETE" or any(complete[k]!=u[k] for k in u):
            raise ValueError(f"Unexpected completed-unit metadata: {folder}")
        selected=json.loads((folder/"D_selected_policies.json").read_text())
        rows=json.loads((folder/"results.json").read_text())
        point_lookup={(r["cohort"],r["panel"],r["fixed_candidate"]):r["J"] for r in rows}
        with np.load(folder/"evaluation_arrays.npz",allow_pickle=False) as z:
            ids=z[cohort+"_record_ids"];sources=z[cohort+"_source_ids"]
            if reference_ids is None:
                source_design(dataset,cohort,ids,sources)
                reference_ids=ids.copy();reference_sources=sources.copy()
            elif not np.array_equal(ids,reference_ids) or not np.array_equal(sources,reference_sources):
                raise ValueError(f"Unaligned paired evaluation records in {folder}")
            n=len(ids)
            for i,p in enumerate(selected):
                if any(p[k]!=u[k] for k in ["dataset","seed","flow","representation"]):raise ValueError("Selection/unit metadata mismatch")
                ci=index[u["task"],u["representation"],p["panel"],p["fixed_candidate"]]
                fi=FLOWS.index(u["flow"]);s=u["seed"]
                if assignment[s,fi,ci]!=-1:raise ValueError("Duplicate planned cell")
                if len(p["config"])>160 or len(p["candidate"])>100:raise ValueError("Unexpected policy identifier length")
                if s in routing and routing[s]!=p["routing_seed"]:raise ValueError("Paired flows use different routing seeds")
                routing[s]=p["routing_seed"]
                order=z[f"{cohort}_p{i:03d}_order"].astype(np.int32,copy=False)
                loss=z[f"{cohort}_p{i:03d}_loss"].astype(np.float64,copy=False)
                raw=p["config"].endswith(("::native","::constant:None"))
                key=(p["routing_seed"],p["config"],"raw" if raw else geometry_key(p["representation"]),None if raw else p["flow"])
                # Repeated panel selections and weight endpoints often have the
                # same policy. Reuse only exactly equal orders AND losses;
                # otherwise retain the distinct numerical policy. No hashing.
                uid=next((j for j in semantic[key] if np.array_equal(order,orders[j]) and np.array_equal(loss,losses[j])),None)
                if uid is None:
                    if loss.shape!=(n,) or order.shape!=(n,) or not np.array_equal(np.sort(order),np.arange(n)):
                        raise ValueError("Invalid fixed queue order")
                    if not np.isfinite(loss).all() or np.any((loss<0)|(loss>1)):raise ValueError("Invalid benchmark loss")
                    uid=len(orders);semantic[key].append(uid);orders.append(order);losses.append(loss);policy_keys.append(list(key))
                j=float(loss[order[:int(.7*n)]].sum()/n)
                if not np.isclose(j,point_lookup[cohort,p["panel"],p["fixed_candidate"]],atol=1e-12,rtol=0):
                    raise ValueError("Stored J and fixed queue arrays disagree")
                assignment[s,fi,ci]=uid;configs[s,fi,ci]=p["config"];candidates[s,fi,ci]=p["candidate"]
    if np.any(assignment<0):raise ValueError("The complete planned cell/seed/flow matrix is not present")
    unique,_,strata=source_design(dataset,cohort,reference_ids,reference_sources)
    return dict(dataset=dataset,cohort=cohort,cells=descriptors,policy_keys=policy_keys,
                orders=np.stack(orders),losses=np.stack(losses),assignment=assignment,selected_configs=configs,selected_candidates=candidates,
                routing_seeds=np.array([routing[s] for s in range(SEEDS)],dtype=np.uint64),record_ids=reference_ids,source_ids=reference_sources,
                unique_source_ids=unique,source_strata=strata)


ARRAY_KEYS=("orders","losses","assignment","selected_configs","selected_candidates","routing_seeds","record_ids","source_ids","unique_source_ids","source_strata")


def save_portable(folder,p):
    save_npz(folder/"queue_inputs.npz",**{k:p[k] for k in ARRAY_KEYS})
    save_json(folder/"problem.json",{k:p[k] for k in ["dataset","cohort","cells","policy_keys"]})


def read_portable(folder):
    p=json.loads((folder/"problem.json").read_text())
    with np.load(folder/"queue_inputs.npz",allow_pickle=False) as z:p.update({k:z[k] for k in ARRAY_KEYS})
    if p["cells"]!=cells(p["dataset"]) or p["assignment"].shape!=(SEEDS,2,len(p["cells"])):
        raise ValueError("Portable inputs do not cover the declared thirty-seed matrix")
    return p


def capacity_points(p):
    n=p["orders"].shape[1]
    cumulative=np.cumsum(np.take_along_axis(p["losses"],p["orders"],axis=1),axis=1)
    return np.stack([cumulative[:,int(q*n)-1]/n if int(q*n) else np.zeros(len(cumulative)) for q in CAPACITIES],axis=1)


def point_tables(folder,p):
    values=capacity_points(p)[p["assignment"]]
    primary=values[:,:,:,3]
    def seed_rows():
        for c,desc in enumerate(p["cells"]):
            deltas=primary[:,0,c]-primary[:,1,c]
            loo=(deltas.sum()-deltas)/(len(deltas)-1)
            for s in range(len(deltas)):
                yield dict(dataset=p["dataset"],cohort=p["cohort"],**desc,seed=s,J_R=primary[s,0,c],J_S=primary[s,1,c],delta=deltas[s],
                           delta_per_1000=1000*deltas[s],leave_this_seed_out=loo[s],R_config=p["selected_configs"][s,0,c],S_config=p["selected_configs"][s,1,c])
    def summaries():
        for c,desc in enumerate(p["cells"]):
            d=primary[:,0,c]-primary[:,1,c];n=len(d);loo=(d.sum()-d)/(n-1)
            yield dict(dataset=p["dataset"],cohort=p["cohort"],**desc,seeds=n,J_R=primary[:,0,c].mean(),J_S=primary[:,1,c].mean(),delta=d.mean(),
                       delta_per_1000=1000*d.mean(),seed_min=d.min(),seed_max=d.max(),leave_one_out_min=loo.min(),leave_one_out_max=loo.max(),
                       positive_seeds=int((d>1e-12).sum()),negative_seeds=int((d< -1e-12).sum()),
                       native_R=int(np.char.endswith(p["selected_configs"][:,0,c],"::native").sum()),native_S=int(np.char.endswith(p["selected_configs"][:,1,c],"::native").sum()),
                       same_selected_candidate=int((p["selected_candidates"][:,0,c]==p["selected_candidates"][:,1,c]).sum()))
    def capacities():
        n=p["orders"].shape[1]
        for c,desc in enumerate(p["cells"]):
            for qi,q in enumerate(CAPACITIES):
                r,s=values[:,0,c,qi].mean(),values[:,1,c,qi].mean();fraction=int(q*n)/n
                yield dict(dataset=p["dataset"],cohort=p["cohort"],**desc,capacity=q,incoming_records=n,accepted_records=int(q*n),
                           J_R=r,J_S=s,delta=r-s,delta_per_1000=1000*(r-s),risk_R=r/fraction if fraction else None,risk_S=s/fraction if fraction else None,
                           selection_capacity=.7,role="DESCRIPTIVE_FIXED_POLICY_CAPACITY_SENSITIVITY")
    write_csv(folder/"seed_details.csv",seed_rows());write_csv(folder/"cell_points.csv",summaries());write_csv(folder/"capacity_points.csv",capacities())
    return primary.mean(axis=0)


def sampled_source_weights(rng,strata,n):
    weights=np.zeros((n,len(strata)),dtype=np.int64)
    for label in sorted(set(strata)):
        ix=np.flatnonzero(strata==label);k=len(ix)
        weights[:,ix]=rng.multinomial(k,np.full(k,1/k),size=n)
    return weights


def queue_cpu(loss,order,weights,q=.7):
    ordered=weights[:,order];totals=ordered.sum(axis=1)
    before=np.cumsum(ordered,axis=1)-ordered
    accepted=np.minimum(ordered,np.maximum(np.floor(q*totals)[:,None]-before,0))
    return accepted@loss[order]/totals


def memberships(p):
    linked=defaultdict(Counter)
    for s in range(p["assignment"].shape[0]):
        for f in range(2):
            for c,uid in enumerate(p["assignment"][s,f]):linked[int(uid)][f,c]+=1
    return linked


def cuda_inputs(p):
    import torch
    if not torch.cuda.is_available():raise RuntimeError("CUDA queue inference was requested but no CUDA device is available")
    order=torch.as_tensor(p["orders"],device="cuda",dtype=torch.int64)
    loss=torch.as_tensor(p["losses"],device="cuda",dtype=torch.float64)
    return order,torch.gather(loss,1,order)


def bootstrap_batch(p,weights,device="cpu",linked=None,gpu_inputs=None):
    if linked is None:linked=memberships(p)
    nseeds=p["assignment"].shape[0]
    result=np.zeros((len(weights),2,len(p["cells"])),dtype=np.float64)
    if device=="cuda":
        import torch
        if gpu_inputs is None:gpu_inputs=cuda_inputs(p)
        w=torch.as_tensor(weights,device="cuda",dtype=torch.int64)
        totals=w.sum(1);capacity=torch.as_tensor(np.floor(.7*weights.sum(1)),device="cuda",dtype=torch.int64)[:,None]
    for uid,targets in linked.items():
        if device=="cpu":values=queue_cpu(p["losses"][uid],p["orders"][uid],weights)
        else:
            order=gpu_inputs[0][uid];loss=gpu_inputs[1][uid]
            ordered=w[:,order];before=ordered.cumsum(1)-ordered
            accepted=torch.minimum(ordered,torch.clamp(capacity-before,min=0))
            values=((accepted.to(torch.float64)@loss)/totals).cpu().numpy()
        for (f,c),count in targets.items():result[:,f,c]+=values*(count/nseeds)
    return result


def bootstrap_problem(folder,p,replicates,batch_size,device,stop_after_batches=None):
    unique,inverse=np.unique(p["source_ids"],return_inverse=True)
    if not np.array_equal(unique,p["unique_source_ids"]):raise ValueError("Source strata/order mismatch")
    seed=original_seed("POST_REVIEW_EXTENSION_QUEUE",p["dataset"],p["cohort"])
    config=dict(dataset=p["dataset"],cohort=p["cohort"],replicates=replicates,batch_size=batch_size,device=device,capacity=.7,
                development_seeds=p["assignment"].shape[0],cells=len(p["cells"]),unique_policies=len(p["orders"]),records=len(inverse),seed=seed)
    done=folder/"bootstrap_complete.json"
    if done.is_file():
        if json.loads(done.read_text())["config"]!=config:raise ValueError("Analysis settings changed; use a fresh output directory")
        with np.load(folder/"flow_means_bootstrap.npz",allow_pickle=False) as z:return z["draws"]
    progress=folder/"bootstrap_progress.json";buffer=folder/"partial_flow_means.npy"
    rng=np.random.default_rng(seed);offset=0;folder.mkdir(parents=True,exist_ok=True)
    if progress.is_file():
        saved=json.loads(progress.read_text())
        if saved["config"]!=config:raise ValueError("Analysis settings changed; use a fresh output directory")
        offset=saved["next_replicate"];rng.bit_generator.state=saved["rng_state"]
        out=np.lib.format.open_memmap(buffer,mode="r+")
    else:out=np.lib.format.open_memmap(buffer,mode="w+",dtype=np.float64,shape=(replicates,2,len(p["cells"])))
    if out.shape!=(replicates,2,len(p["cells"])):raise ValueError("Unexpected resumable draw buffer shape")
    batches=0;linked=memberships(p);gpu_inputs=cuda_inputs(p) if device=="cuda" else None
    while offset<replicates:
        n=min(batch_size,replicates-offset)
        weights=sampled_source_weights(rng,p["source_strata"],n)[:,inverse]
        out[offset:offset+n]=bootstrap_batch(p,weights,device,linked,gpu_inputs);out.flush();offset+=n;batches+=1
        save_json(progress,dict(config=config,next_replicate=offset,rng_state=rng.bit_generator.state))
        if stop_after_batches is not None and batches>=stop_after_batches:return None
        if batches%10==0 or offset==replicates:print(json.dumps(dict(stage="source_queues",dataset=p["dataset"],cohort=p["cohort"],replicates_complete=offset,total=replicates)),flush=True)
    draws=np.array(out);del out
    save_npz(folder/"flow_means_bootstrap.npz",draws=draws)
    save_json(done,dict(status="CONDITIONAL_PAIRED_SOURCE_QUEUE_BOOTSTRAP_COMPLETE",config=config,
                       source_strata=dict(Counter(p["source_strata"].tolist())),seeds_treated_as_independent_samples=False))
    # The completed compressed array supersedes this explicitly bounded local
    # work buffer. Leave the small progress record for interrupted-run context.
    if buffer.resolve().parent!=folder.resolve():raise ValueError("Unexpected temporary draw-buffer path")
    buffer.unlink()
    return draws


def family_blocks(point,draws):
    theta=np.vstack([point,point[0]-point[1]])
    sampled=np.concatenate([draws,draws[:,0:1]-draws[:,1:2]],axis=1)
    for e,endpoint in enumerate(["J_R","J_S","delta"]):
        yield endpoint,None,np.arange(theta.shape[1]),theta[e],sampled[:,e]
        for left in range(theta.shape[1]-1):
            right=np.arange(left+1,theta.shape[1])
            yield endpoint,left,right,theta[e,left]-theta[e,right],sampled[:,e,left,None]-sampled[:,e,right]


def standardized(block,estimate):
    se=block.std(axis=0,ddof=1);active=se>1e-12
    residual=block-estimate
    if np.any(np.max(np.abs(residual[:,~active]),axis=0)>1e-10):raise ValueError("Degenerate bootstrap endpoint is not equal to its point estimate")
    z=np.zeros_like(block);z[:,active]=np.abs(residual[:,active]/se[active])
    return se,z


def joint_inference(output,problems,replicates):
    maxima=np.zeros(replicates);total=0
    for p,point,draws in problems:
        for _,_,right,estimate,block in family_blocks(point,draws):
            _,z=standardized(block,estimate);maxima=np.maximum(maxima,z.max(axis=1));total+=len(right)
    critical=float(np.quantile(maxima,.95,method="higher"))
    def rows(pairs):
        for p,point,draws in problems:
            for endpoint,left,right,estimate,block in family_blocks(point,draws):
                if (left is not None)!=pairs:continue
                se,_=standardized(block,estimate);nominal=np.quantile(block,[.025,.975],axis=0)
                for i,r in enumerate(right):
                    yield dict(dataset=p["dataset"],cohort=p["cohort"],endpoint=endpoint,
                               left=p["cells"][left if pairs else r]["cell_id"],right=p["cells"][r]["cell_id"] if pairs else "zero",
                               estimate=estimate[i],bootstrap_se=se[i],nominal95_low=nominal[0,i],nominal95_high=nominal[1,i],
                               simultaneous95_low=estimate[i]-critical*se[i],simultaneous95_high=estimate[i]+critical*se[i],
                               estimate_per_1000=1000*estimate[i],simultaneous95_low_per_1000=1000*(estimate[i]-critical*se[i]),
                               simultaneous95_high_per_1000=1000*(estimate[i]+critical*se[i]))
    write_csv(output/"cell_intervals.csv",rows(False));write_csv(output/"pairwise_intervals.csv.gz",rows(True))
    save_npz(output/"family_maxima.npz",maxima=maxima)
    save_json(output/"multiplicity_method.json",dict(status="EXPLORATORY_COMPLETE_FAMILY_INFERENCE",family_endpoints=total,replicates=replicates,
              critical95=critical,method="Maximum absolute standardized paired-source-bootstrap deviation; one critical value for all cells and within-dataset/cohort pairs, all three endpoints",
              cohort_streams="Independent deterministic dataset/cohort streams paired by draw number to approximate their product law",
              inference_scope="Conditional on observed D and the finite fitted seed collection; inspected H/H2; not new independent confirmation"))
    return critical,total,maxima


def self_test(device="cpu"):
    from neurocomputing_transfer.statistics.queue_bootstrap import weighted_queue_loss
    loss=np.array([0.,.2,1.,.6,.4,1.,0.]);order=np.array([3,5,0,6,2,4,1])
    w=np.array([[1,2,0,3,0,1,1],[0,0,0,0,0,1,0],[2,0,1,1,3,0,1]],dtype=np.int64)
    reference=[]
    for weights in w:
        expanded=np.repeat(order,weights[order]);reference.append(loss[expanded[:int(.7*len(expanded))]].sum()/len(expanded))
    np.testing.assert_allclose(queue_cpu(loss,order,w),reference,atol=1e-14,rtol=0)
    np.testing.assert_allclose(queue_cpu(loss,order,w),weighted_queue_loss(loss,order,w),atol=1e-14,rtol=0)
    rng=np.random.default_rng(11426);strata=np.array(["train","train","val"])
    sw=sampled_source_weights(rng,strata,13)
    assert np.all(sw[:,:2].sum(1)==2) and np.all(sw[:,2]==1)
    p=dict(dataset="synthetic",cohort="H2",cells=[dict(cell_id="a"),dict(cell_id="b")],
           orders=np.stack([order,order[::-1]]),losses=np.stack([loss,loss[::-1]]),
           assignment=np.array([[[0,1],[1,0]],[[0,0],[1,1]],[[1,0],[0,1]]]),
           source_ids=np.array(["a","a","b","b","b","c","c"]),unique_source_ids=np.array(["a","b","c"]),source_strata=strata)
    weights=sw[:,np.array([0,0,1,1,1,2,2])]
    batch=bootstrap_batch(p,weights)
    brute=np.zeros_like(batch)
    for s in range(3):
        for f in range(2):
            for c in range(2):
                u=p["assignment"][s,f,c]
                brute[:,f,c]+=weighted_queue_loss(p["losses"][u],p["orders"][u],weights)/3
    np.testing.assert_allclose(batch,brute,atol=1e-14,rtol=0)
    if device=="cuda":np.testing.assert_allclose(bootstrap_batch(p,weights,"cuda"),brute,atol=1e-12,rtol=0)
    with tempfile.TemporaryDirectory(prefix="neucom-queue-test-",dir=HERE) as temporary:
        folder=Path(temporary).resolve()
        if not folder.is_relative_to(HERE.resolve()):raise ValueError("Synthetic temporary directory leaves recovery workspace")
        assert bootstrap_problem(folder/"resume",p,11,4,"cpu",stop_after_batches=1) is None
        resumed=bootstrap_problem(folder/"resume",p,11,4,"cpu")
        fresh=bootstrap_problem(folder/"fresh",p,11,4,"cpu")
        np.testing.assert_array_equal(resumed,fresh)
        problem_rows=[];dense=[];estimates=[]
        for c in [3,2]:
            point=rng.uniform(.1,.6,size=(2,c));draws=point+rng.normal(0,.01,size=(19,2,c))
            if c==3:
                # Include one exactly duplicated cell and nonzero pairwise
                # contrasts, across two independently generated problems.
                point[:,1]=point[:,0];draws[:,:,1]=draws[:,:,0]
            descriptor=dict(dataset="synthetic",cohort=str(c),cells=[dict(cell_id=str(i)) for i in range(c)])
            problem_rows.append((descriptor,point,draws))
            for e in range(3):
                theta=point[e] if e<2 else point[0]-point[1]
                sampled=draws[:,e] if e<2 else draws[:,0]-draws[:,1]
                for i in range(c):estimates.append(theta[i]);dense.append(sampled[:,i])
                for i in range(c):
                    for j in range(i+1,c):estimates.append(theta[i]-theta[j]);dense.append(sampled[:,i]-sampled[:,j])
        matrix=np.column_stack(dense);theta=np.array(estimates);se=matrix.std(axis=0,ddof=1);active=se>1e-12
        expected=np.abs((matrix[:,active]-theta[active])/se[active]).max(axis=1)
        critical,total,maximum=joint_inference(folder/"joint",problem_rows,19)
        assert total==27
        np.testing.assert_allclose(maximum,expected,atol=1e-14,rtol=0)
        assert critical==np.quantile(expected,.95,method="higher")
    status=dict(status="SYNTHETIC_ANALYSIS_SOFTWARE_CHECKS_COMPLETED",checks=["Complete-queue expansion including partial boundary copies agrees with the existing kernel",
            "Fixed train/val source strata and paired seed means","Interrupted bootstrap resumes to identical arrays","Streamed complete-family maximum matches a dense calculation, including duplicate endpoints"],
            dataset_experiments_run=0,actual_cuda_kernel_checked=device=="cuda")
    save_json(HERE/"extension_analysis_software_checks.json",status)
    print(json.dumps(status,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--gpu-ready",action="store_true");parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--results",type=Path,default=HERE/"extension_results");parser.add_argument("--output",type=Path)
    parser.add_argument("--replay-portable",type=Path,help="Replay completed portable numerical inputs; no models or fitting")
    parser.add_argument("--replicates",type=int,default=1999);parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu");a=parser.parse_args()
    if a.threads<1:raise SystemExit("--threads must be positive")
    try:
        from threadpoolctl import threadpool_limits
        thread_controller=threadpool_limits(limits=a.threads)
    except ImportError:thread_controller=None
    if a.self_test:self_test(a.device);return
    prepared=plan(a.results)
    if not a.gpu_ready and a.replay_portable is None:
        print(json.dumps(prepared,indent=2));return
    if a.replicates<2 or a.batch_size<1:raise SystemExit("At least two replicates and a positive batch size are required")
    if a.replay_portable is None and prepared["units_missing"]:
        raise SystemExit(f"WAITING_FOR_COMPLETE_FITS: {prepared['units_missing']} of {prepared['units_expected']} planned units are missing; no empirical inference performed")
    output=a.output or a.results/"analysis"
    output.mkdir(parents=True,exist_ok=True)
    problems=[]
    for dataset in DATASETS:
        for cohort in COHORTS:
            folder=output/dataset/cohort
            portable=(a.replay_portable/dataset/cohort) if a.replay_portable is not None else folder
            if (portable/"problem.json").is_file():p=read_portable(portable)
            else:
                p=load_fits(a.results,dataset,cohort);save_portable(folder,p)
            if (p["dataset"],p["cohort"])!=(dataset,cohort):raise ValueError("Portable inputs are assigned to the wrong dataset/cohort")
            point=point_tables(folder,p)
            draws=bootstrap_problem(folder,p,a.replicates,a.batch_size,a.device)
            problems.append(({k:p[k] for k in ["dataset","cohort","cells"]},point,draws))
            del p
    critical,total,_=joint_inference(output,problems,a.replicates)
    if total!=prepared["family_endpoints"]:raise ValueError("Computed family differs from the complete declared matrix")
    save_json(output/"analysis_complete.json",dict(status="COMPLETE_EXPLORATORY_EXTENSION_ANALYSIS",family_endpoints=total,critical95=critical,
              datasets=list(DATASETS),cohorts=list(COHORTS),seeds=SEEDS,replicates=a.replicates,device=a.device,
              portable_input_root=str(a.replay_portable or output),
              numerical_replay="queue_inputs.npz and problem.json per dataset/cohort; keep input policies unchanged when resuming, and use a fresh output directory if they change",
              next_stage="Create evidence-backed figures; update manuscript/responses; publish the final code and numerical release"))
    print(json.dumps(dict(status="COMPLETE_EXPLORATORY_EXTENSION_ANALYSIS",directory=str(output),family_endpoints=total,critical95=critical),indent=2))


if __name__=="__main__":main()
