"""Reuse original trusted project banks when a supplied shared disk has them.

No GPU inference or provenance hashing. Record/source joins establish which
recovered cohort is loaded. Conversion waits for the user's resource handoff.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import joblib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument("--from-root",type=Path,required=True);p.add_argument("--gpu-ready",action="store_true")
    p.add_argument("--output",type=Path,default=HERE/"rebuilt/banks");a=p.parse_args()
    found=[(ds,part,a.from_root/f"revision/banks/{ds}_{part}.joblib") for ds in ["vqav2","activitynetqa"] for part in ["D","H","H2"]]
    if not a.gpu_ready:
        print(json.dumps(dict(status="SHARED_BANK_LOOKUP_ONLY",available=[str(path) for _,_,path in found if path.is_file()],banks_converted=0),indent=2));return
    a.output.mkdir(parents=True,exist_ok=True)
    for ds,part,path in found:
        if not path.is_file():continue
        target=a.output/f"{ds}_{part}.npz"
        if target.exists():continue
        bank=joblib.load(path)
        current=[json.loads(x) for x in (HERE/f"prepared/{ds}/{part}.records.jsonl").read_text(encoding="utf-8").splitlines()]
        original={r["record_id"]:(i,r["source_id"]) for i,r in enumerate(bank["records"])}
        assert set(original)=={r["record_id"] for r in current},"This bank belongs to a different record cohort"
        assert all(original[r["record_id"]][1]==r["source_id"] for r in current)
        order=np.array([original[r["record_id"]][0] for r in current])
        candidates=sorted(bank["X"])
        payload=dict(record_ids=np.array([r["record_id"] for r in current]),source_ids=np.array([r["source_id"] for r in current]),candidates=np.array(candidates),
                     text=np.stack([bank["X"][c][order,:13] for c in candidates]),visual=np.asarray(bank["X"][candidates[0]])[order,13:],
                     native=np.stack([bank["native"][c][order] for c in candidates]),utility=np.stack([bank["scores"]["original"][c][order] for c in candidates]),
                     cache_origin=np.array("ORIGINAL_PROJECT_CACHE_RECOVERED"))
        for key in ["text","visual","native","utility"]:assert np.isfinite(payload[key]).all()
        np.savez_compressed(target,**payload)
        report=dict(status="ORIGINAL_BANK_CONVERTED",source=str(path),dataset=ds,partition=part,records=len(current),candidates=len(candidates),
                    scope="Original numeric features/outputs retained; IDs joined to the recovered cohort. New question-conditioned vectors remain to be generated.")
        target.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report),flush=True)


if __name__=="__main__":main()
