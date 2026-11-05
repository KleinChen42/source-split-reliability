"""Prepare or resume the original ModelScope model snapshots, with no inference.

Defaults to showing the download plan. Downloads wait until --gpu-ready so
they can go directly to the supplied machine rather than a speculative copy.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import json
import urllib.parse
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def fetch(job,root):
    model,entry=job
    folder=root/(model["model"]+"_"+model["revision"][:12])
    target=(folder/entry["path"]).resolve()
    if not target.is_relative_to(folder.resolve()):raise ValueError("Model path leaves target directory")
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.stat().st_size==entry["bytes"]:return dict(model=model["model"],file=entry["path"],status="CACHED")
    query=urllib.parse.urlencode(dict(Revision=model["revision"],FilePath=entry["path"]))
    url=f"https://modelscope.cn/api/v1/models/{model['repository']}/repo?{query}"
    partial=target.with_name(target.name+".part")
    for attempt in range(3):
        try:
            start=partial.stat().st_size if partial.exists() else 0
            headers={"User-Agent":"NEUCOM-input-recovery/1.0"}
            if start:headers["Range"]=f"bytes={start}-"
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90) as response:
                append=start>0 and response.status==206
                with partial.open("ab" if append else "wb") as f:
                    for block in iter(lambda:response.read(4*1024**2),b""):f.write(block)
            if partial.stat().st_size!=entry["bytes"]:raise ValueError("Incomplete model file length")
            partial.replace(target)
            return dict(model=model["model"],file=entry["path"],status="DOWNLOADED",bytes=entry["bytes"])
        except Exception:
            if attempt==2:raise


def main():
    p=argparse.ArgumentParser();p.add_argument("--gpu-ready",action="store_true");p.add_argument("--model",action="append")
    p.add_argument("--models-dir",type=Path,default=ROOT/"models");p.add_argument("--workers",type=int,default=3);a=p.parse_args()
    models=json.loads((HERE/"pinned_model_downloads.json").read_text())
    models=[m for m in models if not a.model or m["model"] in a.model]
    if not a.gpu_ready:
        print(json.dumps(dict(status="MODEL_DOWNLOAD_PLAN_ONLY",models=[dict(model=m["model"],registry=m["registry"],repository=m["repository"],revision=m["revision"],files=len(m["files"]),bytes=sum(f["bytes"] for f in m["files"])) for m in models],
                              scope="Original revisions are ModelScope revision IDs. No inference, weight download, file hashing or cross-registry revision comparison performed."),indent=2));return
    jobs=[(m,f) for m in models for f in m["files"]]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for fut in as_completed([pool.submit(fetch,j,a.models_dir) for j in jobs]):print(json.dumps(fut.result()),flush=True)


if __name__=="__main__":main()
