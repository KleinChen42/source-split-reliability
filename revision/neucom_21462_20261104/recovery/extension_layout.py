"""The complete post-review extension matrix, shared by fitting and inference."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
DATASETS=("vqav2","activitynetqa")
COHORTS=("H","H2")
FLOWS=("record_random","source_disjoint")
TASKS=("seed","question","strong","block_weight")
WEIGHTS=(0.,.25,.5,.75,1.)
SEEDS=30


def candidate_ids(dataset):
    models=["qwen3_vl_4b","internvl35_8b"]+(["gemma3_12b"] if dataset=="vqav2" else [])
    return sorted(m+"__"+p for m in models for p in ["direct_short","task_constrained_short"])


def representations(task):
    return {"seed":["T","TV","Tn","TVn","block_TVn","PCA32_TVn","V"],
            "question":["block_TQn","PCA32_TQn"],"strong":["Tn","PCA32_TVn"],
            "block_weight":[f"weighted_TVn_{w:.2f}" for w in WEIGHTS]}[task]


def panels(task,rep):
    if task=="strong":return ["strong_mix","hgb_only","native_only","constant_only"]
    if task=="block_weight":return ["ridge_original","ridge_only","native_only","alpha_0.1","alpha_1","alpha_10"]
    result=["original_mix","native_only"]
    if task=="seed" and rep in ["Tn","TVn","block_TVn","PCA32_TVn"]:
        result += ["ridge_original","ridge_expanded","knn"]
    if rep=="TV":result += ["constant_only","common_only"]
    return result


def cells(dataset):
    result=[]
    for task in TASKS:
        if task=="block_weight" and dataset!="activitynetqa":continue
        for rep in representations(task):
            for panel in panels(task,rep):
                for scope in ["ALL"]+candidate_ids(dataset):
                    result.append(dict(task=task,representation=rep,panel=panel,fixed_candidate=scope,
                                       selection_scope="pooled_candidate" if scope=="ALL" else "fixed_candidate",
                                       cell_id="/".join([task,rep,panel,scope])))
    return result


def jobs():
    return [dict(task=t,dataset=d,seed=s,flow=f) for t in TASKS for d in DATASETS
            if t!="block_weight" or d=="activitynetqa" for s in range(SEEDS) for f in FLOWS]


def units():
    return [dict(**job,representation=rep) for job in jobs() for rep in representations(job["task"])]


def unit_path(root,unit):
    return Path(root)/unit["task"]/unit["dataset"]/f"{unit['flow']}_{unit['seed']:02d}"/unit["representation"]


def plan(root):
    expected=units()
    missing=[str(unit_path(root,u)) for u in expected if not (unit_path(root,u)/"complete.json").is_file()]
    counts={d:len(cells(d)) for d in DATASETS}
    return dict(status="POST_REVIEW_EXPLORATORY_ANALYSIS_PLAN",jobs_expected=len(jobs()),units_expected=len(expected),
                units_present=len(expected)-len(missing),units_missing=len(missing),missing_examples=missing[:4],
                cells_per_dataset=counts,cohorts=list(COHORTS),development_seeds=SEEDS,
                family_endpoints=sum(3*c*(c+1)//2 for c in counts.values())*len(COHORTS),
                family="J_R, J_S, and R-minus-S for every cell; each endpoint for every unordered cell pair within dataset/cohort; one joint critical value across all four dataset/cohort problems",
                resampling="Paired complete-source queues; preserve video H2 train/val source counts; split seeds are fixed and averaged, not independently resampled",
                capacity=.7,capacity_sensitivity=[.1,.3,.5,.7,.9],bootstrap_replicates=1999,
                evidence_role="Exploratory extension on inspected H/H2; no new independent confirmation")


if __name__=="__main__":
    print(json.dumps(plan(HERE/"extension_results"),indent=2))
