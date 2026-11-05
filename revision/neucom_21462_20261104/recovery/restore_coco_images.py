"""Restore only the selected original COCO images, without model inference."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import json
import urllib.request
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def acquire(relative):
    p=ROOT/relative
    if p.is_file():return dict(media_ref=relative,status="CACHED",bytes=p.stat().st_size)
    p.parent.mkdir(parents=True,exist_ok=True)
    split=p.name.split("_")[1]
    error=None
    for scheme in ["https","http"]:
        url=f"{scheme}://images.cocodataset.org/{split}/{p.name}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"NEUCOM-input-recovery/1.0"}),timeout=25) as response:
                data=response.read(20*1024**2+1)
            if len(data)>20*1024**2:raise ValueError("Unexpected oversized COCO image")
            partial=p.with_suffix(".jpg.part");partial.write_bytes(data)
            with Image.open(partial) as image:image.verify()
            partial.replace(p)
            return dict(media_ref=relative,status="RESTORED",bytes=len(data),source_url=url)
        except Exception as exc:error=str(exc)
    return dict(media_ref=relative,status="MISSING",error=error)


def main():
    p=argparse.ArgumentParser();p.add_argument("--limit",type=int);p.add_argument("--workers",type=int,default=8);a=p.parse_args()
    names=set()
    for part in ["D","H","H2"]:
        for line in (HERE/f"prepared/vqav2/{part}.records.jsonl").read_text(encoding="utf-8").splitlines():names.add(json.loads(line)["media_ref"])
    names=sorted(names)
    if a.limit:names=names[:a.limit]
    # One small real download determines whether the endpoint is usable before
    # dispatching the remaining selected inputs.
    first=acquire(names[0]);print(json.dumps(first),flush=True)
    results=[first]
    if first["status"]=="MISSING":
        (HERE/"image_restore_status.json").write_text(json.dumps(dict(status="ENDPOINT_UNAVAILABLE",first=first),indent=2),encoding="utf-8")
        raise SystemExit(2)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        todo=[pool.submit(acquire,n) for n in names[1:]]
        for f in as_completed(todo):
            result=f.result();results.append(result)
            if len(results)%100==0:print(json.dumps(dict(images=len(results),failed=sum(r["status"]=="MISSING" for r in results))),flush=True)
    result=dict(status="SELECTED_IMAGES_RESTORED" if not a.limit and all(r["status"]!="MISSING" for r in results) else "PARTIAL_IMAGE_RESTORATION",
                planned=len(names),available=sum(r["status"]!="MISSING" for r in results),total_bytes=sum(r.get("bytes",0) for r in results),
                failures=[r for r in results if r["status"]=="MISSING"],model_inference_started=False)
    (HERE/"image_restore_status.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
