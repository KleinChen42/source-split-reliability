"""Recover exact source membership and public QA inputs; no GPU or hash audit.

Source membership comes from the archived, post-repair formal feature indices.
H2 record/source pairs come from the existing numerical arrays. Public questions
and reference answers are rejoined by their original IDs. Cached predictions
and embeddings are not represented as recovered by this script.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from functools import lru_cache
import argparse
import csv
import json
import time
import urllib.request
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/"output/neurocomputing_execution_20260910/formal_results_c9b7cb93c34b/research_provenance"
RAW=HERE/"downloads"
OUT=HERE/"prepared"


@lru_cache(maxsize=1)
def released_index():
    path=OUT/"all_record_source_indices.csv"
    if not path.is_file():return []
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def jsonlines(path,items):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in items),encoding="utf-8")


def download(name,url):
    path=RAW/name
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size>0:
        return dict(file=name,url=url,status="CACHED",bytes=path.stat().st_size)
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"NEUCOM-input-recovery/1.0"})
            with urllib.request.urlopen(req,timeout=45) as response,path.with_suffix(path.suffix+".part").open("wb") as out:
                total=0
                while chunk:=response.read(1024*1024):
                    total+=len(chunk)
                    if total>250*1024**2:raise ValueError("Unexpected annotation size exceeds 250 MiB")
                    out.write(chunk)
            path.with_suffix(path.suffix+".part").replace(path)
            return dict(file=name,url=url,status="DOWNLOADED",bytes=path.stat().st_size)
        except Exception as exc:
            if attempt==2:raise RuntimeError(f"Cannot acquire {name}: {exc}") from exc
            time.sleep(1+attempt)


def source_membership(ds,part):
    public=[r for r in released_index() if r["dataset"]==ds and r["partition"]==part]
    if public:return sorted({r["source_id"] for r in public})
    if part=="H2":
        with np.load(ROOT/f"revision/numeric_release/E2_confirmation/{ds}/joint_queue_arrays.npz",allow_pickle=False) as z:
            ids=z["source_ids"].astype(str)
        return sorted(set(ids))
    p=OLD/f"features/{ds}_{part}.index.jsonl"
    names=[Path(json.loads(x)["media_ref"]).name for x in p.read_text().splitlines() if x]
    if ds=="vqav2":return sorted("coco:"+Path(x).stem.rsplit("_",1)[1] for x in names)
    if ds=="docvqa":return sorted(set("ucsf:"+Path(x).stem.rsplit("_",1)[0] for x in names))
    return sorted("video:"+Path(x).stem for x in names)


def recover_document_source_indices():
    pages=[];parents=[]
    for part in ["D","H"]:
        source_ids=source_membership("docvqa",part)
        assert len(source_ids)==(1000 if part=="D" else 938)
        parents.extend(dict(dataset="docvqa",partition=part,source_id=s) for s in source_ids)
        public=[r for r in released_index() if r["dataset"]=="docvqa" and r["partition"]==part]
        names=sorted({r["page_id"]+".png" for r in public}) if public else [Path(json.loads(line)["media_ref"]).name for line in (OLD/f"features/docvqa_{part}.index.jsonl").read_text().splitlines()]
        for name in names:
            parent,page=Path(name).stem.rsplit("_",1)
            pages.append(dict(partition=part,source_id="ucsf:"+parent,page_id=parent+"_"+page,
                              media_ref="data/raw/docvqa/selected_images/"+name))
    with (OUT/"docvqa_page_source_index.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(pages[0]));w.writeheader();w.writerows(pages)
    with (OUT/"docvqa_parent_sources.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(parents[0]));w.writeheader();w.writerows(parents)
    return dict(status="ORIGINAL_PARENT_AND_PAGE_IDENTIFIERS_RECOVERED",D_parents=1000,H_parents=938,pages=len(pages),
                scope="Source/page identities from original feature paths; document question/answer rows and image bytes are not recovered by this export.")


def read_zip(path,key):
    with zipfile.ZipFile(path) as z:
        names=[x for x in z.namelist() if x.endswith(".json")]
        assert len(names)==1
        return json.loads(z.read(names[0]))[key]


def build_vqa():
    qrows,arows={},{}
    for split in ["Train","Val"]:
        for q in read_zip(RAW/f"v2_Questions_{split}_mscoco.zip","questions"):
            qrows[q["question_id"]]=(q,split.lower()+"2014")
        for a in read_zip(RAW/f"v2_Annotations_{split}_mscoco.zip","annotations"):
            arows[a["question_id"]]=a
    assert qrows.keys()==arows.keys()
    grouped=defaultdict(list)
    for qid,(q,split) in qrows.items():
        a=arows[qid]
        assert q["image_id"]==a["image_id"] and len(a["answers"])==10
        grouped[f"coco:{q['image_id']:012d}"].append((q,a,split))
    result={}
    for part in ["D","H","H2"]:
        records=[];labels=[]
        for sid in source_membership("vqav2",part):
            assert sid in grouped
            for q,a,split in grouped[sid]:
                rid="vqa:"+str(q["question_id"])
                records.append(dict(dataset_id="vqav2",dataset_version="official-v2-2017",record_id=rid,source_id=sid,
                                    partition=part,question=q["question"],official_split=split,
                                    media_ref=f"data/raw/vqav2/images/COCO_{split}_{q['image_id']:012d}.jpg"))
                labels.append(dict(record_id=rid,source_id=sid,partition=part,answers=[x["answer"] for x in a["answers"]]))
        result[part]=(records,labels)
    return result


def build_video():
    grouped=defaultdict(list)
    for split in ["train","val"]:
        questions=json.loads((RAW/f"activitynetqa_{split}_q.json").read_text(encoding="utf-8"))
        answers=json.loads((RAW/f"activitynetqa_{split}_a.json").read_text(encoding="utf-8"))
        amap={str(a["question_id"]):a for a in answers}
        assert len(amap)==len(answers)==len(questions)
        for q in questions:
            sid="video:"+Path(q["video_name"]).stem.removeprefix("v_")
            grouped[sid].append((q,amap[str(q["question_id"])],split))
    result={}
    for part in ["D","H","H2"]:
        records=[];labels=[]
        for sid in source_membership("activitynetqa",part):
            assert sid in grouped and len(grouped[sid])==10
            for q,a,split in grouped[sid]:
                rid="activitynetqa:"+str(q["question_id"])
                records.append(dict(dataset_id="activitynetqa",dataset_version="original-public-QA-rejoined",record_id=rid,
                                    source_id=sid,partition=part,question=q["question"],official_split=split,
                                    media_ref="data/raw/activitynetqa/videos/"+sid.removeprefix("video:")+".mp4"))
                labels.append(dict(record_id=rid,source_id=sid,partition=part,answers=[a["answer"]]))
        result[part]=(records,labels)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--indices-only",action="store_true");args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    document_status=recover_document_source_indices()
    dump(HERE/"document_source_recovery.json",document_status)
    all_sources=[]
    for ds in ["vqav2","activitynetqa"]:
        for part in ["D","H","H2"]:
            members=source_membership(ds,part)
            assert len(members)==(995 if (ds,part)==("activitynetqa","H2") else 1000)
            all_sources.extend(dict(dataset=ds,partition=part,source_id=s) for s in members)
    with (OUT/"source_membership.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["dataset","partition","source_id"]);w.writeheader();w.writerows(all_sources)
    if args.indices_only:
        print(json.dumps(dict(status="SOURCE_MEMBERSHIP_RECOVERED",sources=len(all_sources)),indent=2));return
    jobs=[]
    for split in ["Train","Val"]:
        for kind in ["Questions","Annotations"]:
            name=f"v2_{kind}_{split}_mscoco.zip"
            jobs.append((name,"https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/"+name))
    for split in ["train","val"]:
        for kind in ["q","a"]:
            jobs.append((f"activitynetqa_{split}_{kind}.json",f"https://raw.githubusercontent.com/MILVLG/activitynet-qa/master/dataset/{split}_{kind}.json"))
    fetched=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        todo=[pool.submit(download,*x) for x in jobs]
        for fut in as_completed(todo):
            r=fut.result();fetched.append(r);print(json.dumps(r),flush=True)
    dump(HERE/"annotation_acquisition.json",fetched)
    expected={"vqav2":{"D":5286,"H":5369,"H2":5059},"activitynetqa":{"D":10000,"H":10000,"H2":9950}}
    summary=[]
    for ds,build in [("vqav2",build_vqa),("activitynetqa",build_video)]:
        result=build()
        source_sets={p:{r["source_id"] for r in rs} for p,(rs,_) in result.items()}
        assert not(source_sets["D"]&source_sets["H"] or source_sets["D"]&source_sets["H2"] or source_sets["H"]&source_sets["H2"])
        for part,(rs,ls) in result.items():
            rs=sorted(rs,key=lambda r:r["record_id"]);ls=sorted(ls,key=lambda r:r["record_id"])
            assert len(rs)==expected[ds][part] and len({r["record_id"] for r in rs})==len(rs)
            public={r["record_id"]:r["source_id"] for r in released_index() if r["dataset"]==ds and r["partition"]==part}
            if public:
                assert {r["record_id"]:r["source_id"] for r in rs}==public,"Rejoined QA records differ from the released record/source index"
            if part=="H2" and (ROOT/f"revision/numeric_release/E2_confirmation/{ds}/joint_queue_arrays.npz").is_file():
                with np.load(ROOT/f"revision/numeric_release/E2_confirmation/{ds}/joint_queue_arrays.npz",allow_pickle=False) as z:
                    actual={r["record_id"]:r["source_id"] for r in rs}
                    saved=dict(zip(z["record_ids"].astype(str),z["source_ids"].astype(str)))
                assert actual==saved,"Recovered H2 record/source join differs from numerical input"
            jsonlines(OUT/ds/f"{part}.records.jsonl",rs)
            jsonlines(OUT/ds/f"{part}.labels.jsonl",ls)
            counts=Counter(r["official_split"] for r in rs)
            summary.append(dict(dataset=ds,partition=part,sources=len(source_sets[part]),records=len(rs),official_split_record_counts=dict(counts)))
    dump(HERE/"recovery_status.json",dict(status="QA_TABLES_AND_SOURCE_MEMBERSHIP_RECOVERED",cohorts=summary,
             source_membership="Archived final D/H feature indices; existing H2 numerical IDs. No resampling or new cohort selection.",
             checks="Expected cohort counts, disjoint source sets, unique record IDs and exact H2 record/source joins.",
             remaining=["Raw images/videos at the selected media paths", "Frozen candidate output/native-confidence cache", "ResNet50 feature vectors", "Question-conditioned representation vectors"],
             not_claimed="This does not recover generated answers, embeddings, fitting banks or unavailable original media bytes."))
    print(json.dumps(dict(status="QA_TABLES_AND_SOURCE_MEMBERSHIP_RECOVERED",cohorts=summary),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
