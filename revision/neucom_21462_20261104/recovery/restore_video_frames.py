"""Restore the exact selected video inputs as eight-frame CPU caches.

Uses the pinned archive byte ranges, discards each temporary MP4 after successful
decoding, and performs no model inference, fitting or file-checksum audit.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import Lock
import argparse
import json
import struct
import sys
import zlib

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"runtime"))
from index_selected_videos import HTTPVolumes
from rebuild_gpu_inputs import frames,safe


def main():
    p=argparse.ArgumentParser();p.add_argument("--limit",type=int);p.add_argument("--partition",action="append",choices=["D","H","H2"])
    p.add_argument("--workers",type=int,default=2);p.add_argument("--output",type=Path,default=HERE/"rebuilt");a=p.parse_args()
    spec=json.loads((HERE/"selected_video_ranges.json").read_text())
    jobs=[x for x in spec["entries"] if not a.partition or x["partition"] in a.partition]
    # Develop­ment inputs first. Within each stage use smaller files first.
    priority={"D":0,"H":1,"H2":2}
    jobs.sort(key=lambda j:(priority[j["partition"]],j["entry"]["bytes"]))
    if a.limit:jobs=jobs[:a.limit]
    volumes={}
    for j in jobs:
        key=tuple(j["urls"])
        if key not in volumes:volumes[key]=HTTPVolumes(j["urls"],j["sizes"])
    temporary=HERE/"video_temporary";temporary.mkdir(exist_ok=True)
    progress=HERE/"video_frame_progress.jsonl";lock=Lock()
    def one(j):
        sid=j["source_id"];folder=a.output/"frames"/safe(sid)
        result=dict(source_id=sid,partition=j["partition"])
        if (folder/"timing.json").exists() and all((folder/f"{i:02d}.png").exists() for i in range(8)):
            return dict(result,status="CACHED")
        path=(temporary/(safe(sid)+".mp4")).resolve()
        if not path.is_relative_to(temporary.resolve()):raise ValueError("Temporary path leaves recovery directory")
        try:
            e=j["entry"];vol=volumes[tuple(j["urls"])]
            if not path.exists():
                header=struct.unpack("<4s5H3I2H",vol.read(e["disk"],e["offset"],30))
                if header[0]!=b"PK\x03\x04" or header[2]&1:raise ValueError("Invalid/encrypted video ZIP member")
                start=e["offset"]+30+header[-2]+header[-1]
                data=vol.read(e["disk"],start,e["compressed_bytes"])
                if e["compress_type"]==8:data=zlib.decompress(data,-15)
                elif e["compress_type"]!=0:raise ValueError("Unsupported video member compression")
                if len(data)!=e["bytes"]:raise ValueError("Incomplete video payload")
                partial=path.with_suffix(".mp4.part");partial.write_bytes(data);partial.replace(path)
                del data
            timing=frames(path,folder)
            path.unlink()
            result.update(status="FRAMES_RESTORED",frames=8,duration=timing["duration"],decoder=timing["decoder"],source_video_bytes=e["bytes"])
        except Exception as exc:result.update(status="INPUT_RECOVERY_FAILED",error=str(exc))
        with lock:
            with progress.open("a",encoding="utf-8") as f:f.write(json.dumps(result)+"\n")
        return result
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        todo=[pool.submit(one,j) for j in jobs]
        for f in as_completed(todo):
            r=f.result();results.append(r)
            if r["status"]!="CACHED" and (len(results)<=2 or len(results)%10==0):print(json.dumps(dict(processed=len(results),planned=len(jobs),last=r["source_id"],failures=sum(x["status"]=="INPUT_RECOVERY_FAILED" for x in results))),flush=True)
    result=dict(status="SELECTED_VIDEO_FRAME_INPUTS_RESTORED" if not a.limit and not a.partition and all(r["status"]!="INPUT_RECOVERY_FAILED" for r in results) else "PARTIAL_VIDEO_FRAME_RECOVERY",
                planned=len(jobs),available=sum(r["status"]!="INPUT_RECOVERY_FAILED" for r in results),failures=[r for r in results if r["status"]=="INPUT_RECOVERY_FAILED"],
                inference_started=False,scope="Eight-frame inputs and actual timing recovered by CPU decoding; original raw MP4 files are not retained")
    (HERE/"video_frame_restore_status.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
