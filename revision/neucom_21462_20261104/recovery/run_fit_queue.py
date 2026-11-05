"""Run prepared fitting jobs after resources arrive; unit outputs resume safely."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import subprocess
import sys
from extension_layout import DATASETS, jobs as planned_jobs, representations, unit_path

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument("--gpu-ready",action="store_true");p.add_argument("--python",default=sys.executable)
    p.add_argument("--bank-dir",type=Path,default=HERE/"rebuilt/banks");p.add_argument("--output",type=Path,default=HERE/"extension_results")
    p.add_argument("--task",action="append");p.add_argument("--max-jobs",type=int);p.add_argument("--threads",type=int,default=4)
    p.add_argument("--dataset",action="append",choices=DATASETS)
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--select-only",action="store_true",help="Save F/V selections for later holdout evaluation")
    p.add_argument("--skip-analysis",action="store_true");p.add_argument("--analysis-device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--replicates",type=int,default=1999);p.add_argument("--bootstrap-batch-size",type=int,default=16);a=p.parse_args()
    if a.max_jobs is not None and a.max_jobs<1:raise SystemExit("--max-jobs must be positive")
    if a.workers < 1 or a.threads < 1:
        raise SystemExit("--workers and --threads must be positive")
    jobs=[json.loads(x) for x in (HERE/"fit_job_queue.jsonl").read_text().splitlines()]
    jobs=[j for j in jobs if not a.task or j["task"] in a.task]
    jobs=[j for j in jobs if not a.dataset or j["dataset"] in a.dataset]
    if a.max_jobs:jobs=jobs[:a.max_jobs]
    keys=lambda rows:{(j["task"],j["dataset"],j["seed"],j["flow"]) for j in rows}
    expected=keys(planned_jobs());selected=keys(jobs)
    if not selected or selected-expected or len(selected)!=len(jobs):raise SystemExit("Unknown, empty or duplicated fitting job selection")
    full=selected==expected
    if not a.gpu_ready:
        print(json.dumps(dict(status="QUEUE_PREPARED_WAITING_USER_GPU",jobs=len(jobs),full_planned_queue=full,experiments_started=0,
                              automatic_analysis_after_full_queue=not (a.skip_analysis or a.select_only)),indent=2));return
    a.output.mkdir(parents=True,exist_ok=True)
    pending = []
    for i, job in enumerate(jobs, 1):
        paths = [unit_path(a.output, {**job, "representation": rep}) for rep in representations(job["task"])]
        if not all((path / "complete.json").is_file() or
                   (a.select_only and (path / "D_fitted_selected.joblib").is_file()) for path in paths):
            pending.append((i, job))

    def run_job(item):
        i, j = item
        command=[a.python,str(HERE/"run_extension_fits.py"),"--gpu-ready","--task",j["task"],"--dataset",j["dataset"],"--seed",str(j["seed"]),
                 "--flow",j["flow"],"--bank-dir",str(a.bank_dir),"--output",str(a.output),"--threads",str(a.threads)]
        if a.select_only:
            command.append("--select-only")
        print(json.dumps(dict(job=i,total=len(jobs),**{**j,"status":"RUNNING_PREPARED_JOB"})),flush=True)
        return subprocess.run(command).returncode

    print(json.dumps(dict(planned=len(jobs), pending=len(pending), reused=len(jobs)-len(pending), workers=a.workers)),flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(run_job, item) for item in pending]
        try:
            for future in as_completed(futures):
                return_code = future.result()
                if return_code:
                    raise SystemExit(return_code)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    if a.select_only:
        scope = "_".join(sorted(set(a.dataset or DATASETS)))
        name = f"fit_queue_development_{scope}.json"
        status = "DEVELOPMENT_SELECTION_SLICE_COMPLETE" if a.max_jobs or a.task else "DEVELOPMENT_SELECTION_QUEUE_COMPLETE"
        receipt = dict(status=status, jobs=len(jobs),
                       planned_jobs=len(expected), whole_queue_selected=full,
                       next_stage="Apply the persisted choices to H and H2")
        (a.output / name).write_text(json.dumps(receipt, indent=2) + "\n")
        return
    receipt=dict(status="FITTING_QUEUE_COMPLETE" if full else "SELECTED_JOB_SLICE_COMPLETE",jobs=len(jobs),full_planned_jobs=len(expected),
                 complete_declared_queue=full,analysis_complete=False,
                 next_stage="Paired source inference, figures, manuscript/response revision and the final public code/numerical release" if full else
                            "Complete the remaining prepared fitting jobs; then run paired source inference and finish the revision")
    name="fit_queue_complete.json" if full else "fit_queue_slice_complete.json"
    if not full and a.dataset:
        name = "fit_queue_slice_" + "_".join(sorted(set(a.dataset))) + "_complete.json"
    (a.output/name).write_text(json.dumps(receipt,indent=2)+"\n")
    if full and not a.skip_analysis:
        command=[a.python,str(HERE/"analyze_extensions.py"),"--gpu-ready","--results",str(a.output),"--device",a.analysis_device,
                 "--replicates",str(a.replicates),"--batch-size",str(a.bootstrap_batch_size),"--threads",str(a.threads)]
        result=subprocess.run(command)
        if result.returncode:raise SystemExit(result.returncode)
        receipt["analysis_complete"]=True
        receipt["next_stage"]="Evidence-backed figures, manuscript/response revision and the final public code/numerical release"
        (a.output/name).write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":main()
