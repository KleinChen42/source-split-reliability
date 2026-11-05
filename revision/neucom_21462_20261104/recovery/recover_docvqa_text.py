"""Recover selected DocVQA QA columns with HTTP range reads, excluding images."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import io
import json
import urllib.request
import csv
import numpy as np
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/"output/neurocomputing_execution_20260910/formal_results_c9b7cb93c34b/research_provenance"


def source_spec():
    public=HERE/"docvqa_source_files.json"
    return json.loads((public if public.is_file() else OLD/"artifacts/NEURO_R1/manifests/docvqa_preparation.json").read_text())


class RemoteRanges(io.RawIOBase):
    def __init__(self,url,size):
        super().__init__();self.url=url;self.original_url=url;self.size=size;self.pos=0;self.transferred=0;self.cache={}
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError("Negative file position")
        return self.pos
    def read(self,n=-1):
        if self.pos>=self.size:return b""
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if n==0:return b""
        if n>16*1024**2:raise ValueError("Refusing an unexpected >16 MiB range; image columns must stay excluded")
        key=(self.pos,n)
        if key in self.cache:
            self.pos+=n;return self.cache[key]
        req=urllib.request.Request(self.url,headers={"User-Agent":"NEUCOM-input-recovery/1.0","Range":f"bytes={self.pos}-{self.pos+n-1}"})
        with urllib.request.urlopen(req,timeout=60) as response:
            if response.status!=206:raise ValueError("Provider did not honor the byte range; refusing full image-shard download")
            self.url=response.geturl()
            data=response.read(n+1)
        if len(data)!=n:raise ValueError(f"Unexpected range length {len(data)} != {n}")
        self.cache[key]=data
        self.pos+=n;self.transferred+=n;return data
    def readinto(self,b):
        data=self.read(len(b));b[:len(data)]=data;return len(data)


def read_shard(spec,revision):
    cache=HERE/"downloads/docvqa_text"/(Path(spec["file"]).stem+".jsonl")
    if cache.exists():return [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines()],0
    url=f"https://huggingface.co/datasets/lmms-lab/DocVQA/resolve/{revision}/{spec['file']}"
    stream=RemoteRanges(url,spec["bytes"])
    columns=["questionId","ucsf_document_id","ucsf_document_page_no","question","answers","data_split"]
    table=pq.read_table(stream,columns=columns,use_threads=False,pre_buffer=False)
    rows=table.to_pylist();cache.parent.mkdir(parents=True,exist_ok=True)
    partial=cache.with_suffix(".jsonl.part")
    partial.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    partial.replace(cache)
    return rows,stream.transferred


def main():
    p=argparse.ArgumentParser();p.add_argument("--probe",action="store_true");a=p.parse_args()
    prep=source_spec()
    files=prep["files"][:1] if a.probe else prep["files"]
    rows=[];bytes_read=0
    with ThreadPoolExecutor(max_workers=3) as pool:
        todo={pool.submit(read_shard,f,prep["revision"]):f["file"] for f in files}
        for fut in as_completed(todo):
            batch,n=fut.result();rows.extend(batch);bytes_read+=n
            print(json.dumps(dict(shard=todo[fut],rows=len(batch),range_bytes=n)),flush=True)
    if a.probe:
        print(json.dumps(dict(status="TEXT_COLUMN_RANGE_PROBE_COMPLETED",rows=len(rows),download_bytes=bytes_read,image_columns_downloaded=False)));return
    alloc={}
    with (HERE/"prepared/docvqa_parent_sources.csv").open(encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):alloc[r["source_id"]]=r["partition"]
    result={"D":([],[]),"H":([],[])}
    for row in rows:
        sid="ucsf:"+row["ucsf_document_id"].strip()
        if sid not in alloc:continue
        part=alloc[sid];parent=sid.removeprefix("ucsf:");page=str(row["ucsf_document_page_no"]);rid="docvqa:"+str(row["questionId"])
        result[part][0].append(dict(dataset_id="docvqa",dataset_version=prep["revision"],record_id=rid,source_id=sid,page_id=parent+"_"+page,
                                   partition=part,question=row["question"],official_split=row["data_split"],media_ref=f"data/raw/docvqa/selected_images/{parent}_{page}.png"))
        result[part][1].append(dict(record_id=rid,source_id=sid,partition=part,answers=row["answers"]))
    summary=[];group_order_check=None;expected=[]
    public_index=HERE/"prepared/all_record_source_indices.csv"
    if public_index.is_file():
        with public_index.open(encoding="utf-8",newline="") as f:
            expected=[r for r in csv.DictReader(f) if r["dataset"]=="docvqa"]
    for part,(records,labels) in result.items():
        records.sort(key=lambda r:r["record_id"]);labels.sort(key=lambda r:r["record_id"])
        assert len(records)=={"D":6316,"H":15695}[part]
        assert len({r["record_id"] for r in records})==len(records)
        if expected:assert {r["record_id"]:r["source_id"] for r in records}=={r["record_id"]:r["source_id"] for r in expected if r["partition"]==part},"Rejoined documents differ from released record/source IDs"
        if part=="H" and (ROOT/"output/neurocomputing_execution_20260910/formal_results_c9b7cb93c34b/release/docvqa_primary.npz").is_file():
            original=ROOT/"output/neurocomputing_execution_20260910/formal_results_c9b7cb93c34b/release/docvqa_primary.npz"
            with np.load(original,allow_pickle=False) as z:
                _,inverse=np.unique([r["source_id"] for r in records],return_inverse=True)
                assert np.array_equal(inverse,z["source_index"]),"Recovered record/source ordering differs from original numerical cohort"
                group_order_check=True
        folder=HERE/"prepared/docvqa";folder.mkdir(exist_ok=True)
        for suffix,items in [("records",records),("labels",labels)]:
            (folder/f"{part}.{suffix}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in items),encoding="utf-8")
        summary.append(dict(partition=part,records=len(records),sources=len({r["source_id"] for r in records})))
    status=dict(status="DOCVQA_QA_AND_SOURCE_IDENTITIES_RECOVERED",cohorts=summary,download_bytes_this_run=bytes_read,
                image_columns_downloaded=False,original_H_group_order_matches=group_order_check,released_record_source_index_matches=True if expected else None,remaining="Original document image bytes and model output/feature caches")
    (HERE/"docvqa_text_recovery_status.json").write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(status,indent=2))


if __name__=="__main__":main()
