"""Prepare original-seed indices and a bounded experiment plan without fitting."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/"output/neurocomputing_execution_20260910/formal_results_c9b7cb93c34b/research_provenance"


def original_seed(*parts):
    # This is the original experiment's random-stream derivation from strings.
    # It does not hash files or perform a provenance/checksum audit.
    payload=json.dumps(parts,sort_keys=True,separators=(",",":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],"big")


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def main():
    plan=json.loads((OLD/"artifacts/NEURO_R1/protocol/ready_plan_v4.json").read_text())
    splits=HERE/"splits";splits.mkdir(exist_ok=True)
    summary=[]
    for ds in ["vqav2","activitynetqa"]:
        records=[json.loads(x) for x in (HERE/f"prepared/{ds}/D.records.jsonl").read_text(encoding="utf-8").splitlines()]
        ids=np.array([r["record_id"] for r in records]);sources=np.array([r["source_id"] for r in records])
        assert list(ids)==sorted(ids.tolist())
        u=np.unique(sources);n=len(ids);seeds=[];masks={f:[] for f in ["record_random","source_disjoint"]}
        for s in range(30):
            seed=original_seed(plan["rng"]["master_seed"],plan["plan_version"],ds,"development_split",s)
            seeds.append(seed)
            for flow in masks:
                rng=np.random.default_rng(seed);mask=np.zeros(n,dtype=bool)
                if flow=="record_random":mask[rng.permutation(n)[:int(.7*n)]]=True
                else:mask=np.isin(sources,rng.permutation(u)[:int(.7*len(u))])
                masks[flow].append(mask)
                fs=set(sources[mask]);vs=set(sources[~mask])
                summary.append(dict(dataset=ds,seed=s,flow=flow,fit_records=int(mask.sum()),validation_records=int((~mask).sum()),
                                    fitting_sources=len(fs),validation_sources=len(vs),overlapping_sources=len(fs&vs)))
        np.savez_compressed(splits/f"{ds}_30_seeds.npz",record_ids=ids,source_ids=sources,
                            routing_seeds=np.array(seeds,dtype=np.uint64),
                            record_random_fit=np.stack(masks["record_random"]),source_disjoint_fit=np.stack(masks["source_disjoint"]))
    old_counts={}
    with (HERE.parent/"analysis/development_record_counts.csv").open(encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):old_counts[r["dataset"],int(r["seed"]),r["flow"]]=(int(r["fit_records"]),int(r["validation_records"]))
    for r in summary:
        if r["seed"]<5:assert (r["fit_records"],r["validation_records"])==old_counts[r["dataset"],r["seed"],r["flow"]],"Recovered F/V counts differ from saved metadata"
    with (splits/"development_counts_30_seeds.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    extension={
        "status":"PREPARED_AWAITING_USER_GPU_NO_NEW_FITS",
        "authorization":"Continue restoring original development inputs now; execute remaining experiments and final public release after the user supplies GPU resources.",
        "datasets":["vqav2","activitynetqa"],"seeds":list(range(30)),"flows":["record_random","source_disjoint"],
        "cohorts":["D","H","H2"],"q":.7,"fit_fraction":.7,"selection_on":"V only; no full-D refitting",
        "historical_scope":"The original five-seed H2 family remains historical. Post-review extra seeds/representations/weights on inspected H/H2 are exploratory robustness extensions, not fresh independent confirmation.",
        "recovery_priority":["Reuse compatible original D/H/H2 banks or raw cached predictions/features on a supplied shared disk if available", "Otherwise regenerate missing outputs using recovered exact record/source memberships and the pinned generator recipe"],
        "seed_extension":{"representations":["T","TV","Tn","TVn","block_TVn","PCA32_TVn","V"],
                          "original_family":"constant, ridge alpha in [0.1,1,10], uniform kNN in [5,20,50], raw-native comparator", 
                          "controls_panels":["original_mix","ridge_original","ridge_expanded","knn","native_only"],
                          "reuse":"Do not repeat historical Monte Carlo, scoring sensitivities or unrelated checks"},
        "question_conditioned":{"encoder":"qwen3_vl_4b at the pinned original revision", "prompt":"direct_short",
                                "feature":"Final decoder hidden vector at the last prompt position before the first answer token; captured on first LM-head pre-hook call during original greedy generation",
                                "scope":"Common question/media-conditioned encoder across candidate generators; no generated answer or reference label in this vector",
                                "representations":["block_TQn","PCA32_TQn"],"comparators":["Tn","block_TVn","PCA32_TVn","native_only"],
                                "F_only":"Coordinate standardization, effective dimensions and PCA fit on F only"},
        "block_weight":{"dataset":"activitynetqa","candidate_scope":"All four fixed model/prompt candidates, plus separately labelled pooled selection",
                        "lambda":[0,.25,.5,.75,1],"weights":"For standardized blocks b in [T,V,n], w_b(lambda)=d_b**(-lambda/2); d_b counts nonconstant F coordinates, with minimum 1",
                        "endpoints":"lambda=0 recovers the original standardized geometry; lambda=1 recovers the block-normalized geometry",
                        "ridge_alpha":[.1,1,10],"hold_fixed":"Within each candidate/seed/flow, same F/V indices, model, prompt and ridge candidate grid; report each alpha as well as V-selected alpha",
                        "interpretation":"A controlled path through relative block penalties; it does not isolate a universal causal effect of visual information"},
        "inference":{"scope":"Conditional on observed D and fitted finite seed collection; paired source weights rebuild complete queues", 
                     "multiplicity":"Report every planned implementation cell; retain a defined joint family for all published cell/within-cohort pairwise endpoints. Do not choose tests based on favorable observed signs.",
                     "units":"Accepted errors per 1,000 arrivals for binary video loss; benchmark-loss units per 1,000 for VQA"},
        "gpu_smoke":"Check the first prompt's question-conditioned vector, generation/native-confidence fields and expected dimensions on a minimal record before starting the full queue",
        "publication":{"owner":"KleinChen42","status":"WAIT_FOR_EXPERIMENTS_AND_FINAL_PACKAGE", "target":"A public code repository with a versioned numerical release and an observed permanent repository/release URL; DOI optional", 
                       "exclude":["review letters","raw questions/references/media","model weights","credentials","private connection settings"],
                       "existing_repository":"No matching research repository was returned by the connected account; do not repurpose its unrelated private repository"}
    }
    save(HERE/"extension_plan.json",extension)
    save(HERE/"original_runtime_profiles.json",dict(plan_version=plan["plan_version"],models=plan["models"],prompts=plan["prompts"],panels=plan["panels"],generation=plan["generation"],runtime_environments=plan["runtime_environments"]))
    controls=json.loads((ROOT/"revision/r3/protocol/controls_manifest_v2.json").read_text())
    strong=json.loads((ROOT/"revision/r3/protocol/controls_strong_manifest.json").read_text())
    save(HERE/"fit_parameters.json",dict(ridge_alphas=controls["ridge_alphas"],knn_k=controls["knn_k"],hgb_parameters=strong["model_parameters"]))
    repositories={"qwen3_vl_4b":"Qwen/Qwen3-VL-4B-Instruct","internvl35_8b":"OpenGVLab/InternVL3_5-8B","gemma3_12b":"google/gemma-3-12b-it"}
    model_inputs=[]
    for item in plan["models"]:
        old=json.loads((OLD/f"artifacts/NEURO_R1/manifests/{item['id']}_download.json").read_text())
        model_inputs.append(dict(model=item["id"],registry="ModelScope",repository=repositories[item["id"]],revision=item["revision"],
                                 files=[dict(path=f["file"],bytes=f["bytes"]) for f in old["files"]]))
    save(HERE/"pinned_model_downloads.json",model_inputs)
    jobs=[]
    for task in ["seed","question","strong","block_weight"]:
        for ds in (["activitynetqa"] if task=="block_weight" else ["vqav2","activitynetqa"]):
            for seed in range(30):
                for flow in ["record_random","source_disjoint"]:
                    jobs.append(dict(task=task,dataset=ds,seed=seed,flow=flow,status="PENDING_USER_GPU"))
    (HERE/"fit_job_queue.jsonl").write_text("".join(json.dumps(j)+"\n" for j in jobs),encoding="utf-8")
    save(HERE/"preparation_status.json",dict(status="PREPARED_AWAITING_USER_GPU",D_record_counts={ds:sum(1 for _ in (HERE/f"prepared/{ds}/D.records.jsonl").open(encoding="utf-8")) for ds in ["vqav2","activitynetqa"]},
                                            planned_development_splits=len(summary),first_five_count_pairs_match_saved_metadata=True,
                                            models_fitted=0,question_conditioned_vectors_generated=0,block_weight_experiments_run=0))
    print(json.dumps(dict(status="PREPARED_AWAITING_USER_GPU",development_splits=len(summary),first_five_FV_counts_match=True),indent=2))


if __name__=="__main__":main()
