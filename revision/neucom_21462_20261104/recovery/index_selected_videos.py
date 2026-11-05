"""Resolve only the original selected video IDs in pinned ZIP central directories."""
from pathlib import Path
import json
import sys
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from neurocomputing_transfer.data_sources.multivolume_zip import RemoteVolumes


class HTTPVolumes(RemoteVolumes):
    def __init__(self,urls,sizes):
        super().__init__(urls,sizes);self.resolved=list(urls);self.network_bytes=0
    def read(self,disk,offset,length):
        if disk<0 or offset<0 or length<0:raise ValueError("Invalid range")
        chunks=[]
        while length:
            if offset>=self.sizes[disk]:offset-=self.sizes[disk];disk+=1;continue
            size=min(length,self.sizes[disk]-offset,8*1024**2)
            headers={"User-Agent":"NEUCOM-input-recovery/1.0","Range":f"bytes={offset}-{offset+size-1}"}
            try:response=urllib.request.urlopen(urllib.request.Request(self.resolved[disk],headers=headers),timeout=60)
            except Exception:
                self.resolved[disk]=self.urls[disk]
                response=urllib.request.urlopen(urllib.request.Request(self.urls[disk],headers=headers),timeout=60)
            with response:
                if response.status!=206:raise ValueError("Provider did not honor bounded byte range")
                if not response.headers.get("Content-Range","").startswith(f"bytes {offset}-{offset+size-1}/"):
                    raise ValueError("Provider returned a different byte range")
                self.resolved[disk]=response.geturl();data=response.read(size+1)
            if len(data)!=size:raise ValueError("Truncated archive range")
            chunks.append(data);self.network_bytes+=size;offset+=size;length-=size
        return b"".join(chunks)


def main():
    spec=json.loads((HERE/"video_archive_locations.json").read_text())
    sizes={f["path"]:f["bytes"] for f in spec["files"]}
    base=f"https://huggingface.co/datasets/{spec['repository']}/resolve/{spec['revision']}/"
    desired={}
    for part in ["D","H","H2"]:
        for line in (HERE/f"prepared/activitynetqa/{part}.records.jsonl").read_text(encoding="utf-8").splitlines():
            r=json.loads(line);desired[r["source_id"]]=dict(partition=part,official_split=r["official_split"])
    output=[];network_bytes=0
    for split,names in [("train",["train_videos_merge.z01","train_videos_merge.z02","train_videos_merge.zip"]),("val",["validation_videos.zip"])]:
        urls=[base+n for n in names];vol=HTTPVolumes(urls,[sizes[n] for n in names])
        entries=vol.index();network_bytes+=vol.network_bytes
        matches={}
        for e in entries:
            if not e["name"].lower().endswith(".mp4"):continue
            sid="video:"+Path(e["name"]).stem.removeprefix("v_")
            if sid in desired and desired[sid]["official_split"]==split:
                matches[sid]={k:v for k,v in e.items() if k!="crc32"}
        output.extend(dict(source_id=sid,**desired[sid],urls=urls,sizes=vol.sizes,entry=e) for sid,e in matches.items())
        print(json.dumps(dict(split=split,selected_videos_resolved=len(matches),index_network_bytes=vol.network_bytes)),flush=True)
    missing=sorted(set(desired)-{r["source_id"] for r in output})
    result=dict(status="SELECTED_VIDEO_RANGES_LOCATED" if not missing else "SOME_ORIGINAL_VIDEO_IDS_UNRESOLVED",repository=spec["repository"],revision=spec["revision"],
                selected_sources=len(output),missing_source_ids=missing,selected_video_bytes=sum(r["entry"]["bytes"] for r in output),
                index_network_bytes=network_bytes,videos_downloaded=0,entries=output)
    (HERE/"selected_video_ranges.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="entries"},indent=2))


if __name__=="__main__":main()
