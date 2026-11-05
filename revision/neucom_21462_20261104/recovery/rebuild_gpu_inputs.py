"""Resume missing input generation after user-supplied GPU resources arrive.

`probe` is read-only and uses no GPU. Other stages require --gpu-ready.
No generated prediction/feature cache is claimed to be an original recovered
cache: new outputs are marked REGENERATED. Existing project adapters are reused.
"""
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import argparse
import ast
import json
import math
import re
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    partial.replace(path)


def records(ds,part):
    return [json.loads(x) for x in (HERE/f"prepared/{ds}/{part}.records.jsonl").read_text(encoding="utf-8").splitlines()]


def safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]","_",value)


def frames(path,folder):
    from filelock import FileLock
    folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/"decode.lock"),timeout=300):
        receipt=folder/"timing.json"
        if receipt.exists() and all((folder/f"{i:02d}.png").exists() for i in range(8)):
            return json.loads(receipt.read_text())
        import av
        with av.open(str(path)) as container:
            stream=container.streams.video[0];stream.thread_type="NONE";stream.codec_context.thread_count=1
            start=stream.start_time or 0;tb=float(stream.time_base)
            duration=float(stream.duration*stream.time_base) if stream.duration else float(container.duration/av.time_base)
            fps=float(stream.average_rate) if stream.average_rate else None
            if not fps or not math.isfinite(duration) or duration<=0:raise ValueError("Invalid video timing")
            requested=[duration*(i+.5)/8 for i in range(8)];timestamps=[]
            for i,target in enumerate(requested):
                container.seek(start+int(target/tb),stream=stream,backward=True,any_frame=False)
                selected=None
                for f in container.decode(stream):
                    if f.pts is not None and (f.pts-start)*tb+1e-9>=target:selected=f;break
                if selected is None or selected.is_corrupt:raise ValueError("Missing/corrupt midpoint frame")
                selected.to_image().convert("RGB").save(folder/f"{i:02d}.png")
                timestamps.append((selected.pts-start)*tb)
            result=dict(decoder="pyav-"+av.__version__,duration=duration,fps=fps,total_num_frames=stream.frames or None,
                        frame_count=8,requested_timestamps=requested,timestamps=timestamps)
        save(receipt,result);return result


def load_media(row,media_root,out,frame_root=None):
    from PIL import Image
    path=media_root/row["media_ref"]
    if row["dataset_id"]!="activitynetqa":
        if not path.is_file():raise FileNotFoundError(path)
        return [Image.open(path).convert("RGB")],None
    folder=(frame_root if frame_root is not None else out/"frames")/safe(row["source_id"])
    if (folder/"timing.json").exists() and all((folder/f"{i:02d}.png").exists() for i in range(8)):
        timing=json.loads((folder/"timing.json").read_text())
    else:
        if not path.is_file():raise FileNotFoundError(path)
        timing=frames(path,folder)
    fps=timing["fps"]
    meta=dict(total_num_frames=timing["total_num_frames"] or round(timing["duration"]*fps),fps=fps,
              duration=timing["duration"],frames_indices=[t*fps for t in timing["timestamps"]])
    return [Image.open(folder/f"{i:02d}.png").convert("RGB") for i in range(8)],meta


class PromptVector:
    """Capture the first LM-head input, before any generated answer token."""
    def __init__(self,model):
        self.vector=None
        head=model.get_output_embeddings();self.expected_width=getattr(head,"in_features",None)
        self.handle=head.register_forward_pre_hook(self.capture)
    def capture(self,module,args):
        if self.vector is None:
            h=args[0]
            if h.ndim!=3 or h.shape[0]!=1:raise ValueError("Expected one prompt's decoder states")
            if self.expected_width is not None and h.shape[-1]!=self.expected_width:raise ValueError("Unexpected decoder hidden width")
            self.vector=h[0,-1].detach().float().cpu().numpy().copy()
    def close(self):self.handle.remove()


@lru_cache(maxsize=1)
def vqa_normalizer():
    """Reuse supplied official normalization code without a file-hash audit."""
    path=ROOT/"neurocomputing_transfer/vendor_sources/vqaEval.py"
    text=re.sub(r"(?m)^(\s*)print (.+)$",r"\1print(\2)",path.read_text(encoding="utf-8"))
    ns={};exec(compile(ast.parse(text),str(path),"exec"),ns)
    class Empty:
        def getQuesIds(self):return []
    return ns["VQAEval"](Empty(),None)


def utility(ds,prediction,answers):
    if ds=="activitynetqa":return float(prediction in answers)
    if ds!="vqav2" or len(answers)!=10:raise ValueError("Invalid benchmark utility input")
    obj=vqa_normalizer()
    def normalize(x):
        x=str(x).replace("\n"," ").replace("\t"," ").strip()
        return obj.processDigitArticle(obj.processPunctuation(x))
    pred=normalize(prediction);refs=[normalize(a) for a in answers]
    return sum(min(1.,sum(a==pred for j,a in enumerate(refs) if i!=j)/3) for i in range(10))/10


def probe(a):
    status=[]
    for ds in ["vqav2","docvqa","activitynetqa"]:
        for part in (["D","H"] if ds=="docvqa" else ["D","H","H2"]):
            rs=records(ds,part);media={r["media_ref"]:r["source_id"] for r in rs}
            raw_present=sum((a.media_root/p).is_file() for p in media)
            frame_present={p for p,sid in media.items() if ds=="activitynetqa" and
                           (a.frame_root/safe(sid)/"timing.json").is_file() and
                           all((a.frame_root/safe(sid)/f"{i:02d}.png").is_file() for i in range(8))}
            missing=[p for p in media if not(a.media_root/p).is_file() and p not in frame_present]
            status.append(dict(dataset=ds,partition=part,records=len(rs),media=len(media),
                               present_raw_media=raw_present,usable_cached_frame_sources=len(frame_present),
                               present_media=len(media)-len(missing),missing_media=len(missing),missing_media_examples=sorted(missing)[:2]))
    prior=[]
    for shared in a.search_root:
        for ds in ["vqav2","activitynetqa"]:
            for part in ["D","H","H2"]:
                p=shared/f"revision/banks/{ds}_{part}.joblib"
                if p.is_file():prior.append(str(p))
    result=dict(status="INPUT_PROBE_ONLY",cohorts=status,original_bank_candidates=prior,
                next_step="Reuse compatible original caches if present; otherwise stage media/models and regenerate on the supplied GPU.",gpu_jobs_started=0)
    save(a.output/"input_probe.json",result);print(json.dumps(result,indent=2))


def visual(a):
    import torch
    from torchvision.models import resnet50,ResNet50_Weights
    if not torch.cuda.is_available():raise RuntimeError("User-supplied CUDA GPU is not available")
    weights=ResNet50_Weights.IMAGENET1K_V2
    device=torch.device(a.encoder_device);torch.set_num_threads(4)
    model=resnet50(weights=weights).eval().to(device);model.fc=torch.nn.Identity();transform=weights.transforms()
    rs=records(a.dataset,a.partition)
    by_source={r["source_id"]:r for r in rs}
    selected=list(sorted(by_source))[:a.limit] if a.limit else sorted(by_source)
    for i,sid in enumerate(selected,1):
        target=a.output/"visual"/a.dataset/(safe(sid)+".npy");target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():continue
        images,_=load_media(by_source[sid],a.media_root,a.output,a.frame_root)
        with torch.inference_mode():vector=model(torch.stack([transform(im) for im in images]).to(device)).mean(0).float().cpu().numpy()
        if vector.shape!=(2048,) or not np.isfinite(vector).all():raise ValueError("Invalid ResNet vector")
        np.save(target,vector,allow_pickle=False)
        if i%25==0:print(json.dumps(dict(stage="visual",dataset=a.dataset,partition=a.partition,processed=i)),flush=True)
    save(a.output/f"visual_{a.dataset}_{a.partition}.json",dict(status="SMOKE_OUTPUT" if a.limit else "REGENERATED",sources=len(selected),encoder="ResNet50_IMAGENET1K_V2",device=a.encoder_device,runtime=dict(torch=torch.__version__)))


def read_predictions(cache):
    """Resume complete JSONL records after an interrupted append."""
    if not cache.exists():
        return {}
    old = {}
    with cache.open("r+b") as stream:
        while line := stream.readline():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b"\n"):
                    raise
                stream.truncate(stream.tell() - len(line))
                break
            if not line.endswith(b"\n"):
                stream.write(b"\n")
            old[row["record_id"], row["prompt_id"]] = row
    return old


def save_vector(path, vector):
    partial = path.with_suffix(".npy.part")
    with partial.open("wb") as stream:
        np.save(stream, vector, allow_pickle=False)
    partial.replace(path)


def predictions(a, selected_records=None):
    if not a.model_path:raise ValueError("--model-path must identify the pinned local model")
    import torch
    if not torch.cuda.is_available():raise RuntimeError("User-supplied CUDA GPU is not available")
    if a.model=="gemma3_12b" and a.dataset=="activitynetqa":raise ValueError("Gemma is not in the video panel")
    from neurocomputing_transfer.models.qwen import Qwen
    from neurocomputing_transfer.models.internvl import InternVL
    from neurocomputing_transfer.models.gemma import Gemma
    model={"qwen3_vl_4b":Qwen,"internvl35_8b":InternVL,"gemma3_12b":Gemma}[a.model](a.model_path)
    memo = None
    if getattr(a, "reuse_frozen_vision", False):
        from frozen_vision_memo import FrozenVisionMemo
        memo = FrozenVisionMemo(model, a.model)
    profile=json.loads((HERE/"original_runtime_profiles.json").read_text())
    rs=records(a.dataset,a.partition) if selected_records is None else list(selected_records)
    if a.limit:rs=rs[:a.limit]
    if memo:
        rs.sort(key=lambda row: (row["source_id"], row["media_ref"], row["record_id"]))
    assert len({safe(r["record_id"]) for r in rs})==len(rs)
    folder=a.output/"predictions"/a.dataset/a.partition;folder.mkdir(parents=True,exist_ok=True)
    cache=folder/(a.model+".jsonl")
    old = read_predictions(cache)
    qfolder=a.output/"question_vectors"/a.dataset/a.partition;qfolder.mkdir(parents=True,exist_ok=True)
    last_media = None
    for i,row in enumerate(rs,1):
        current_media = (row["source_id"], row["media_ref"])
        if memo is None or current_media != last_media:
            images = meta = None
            if memo:
                memo.clear()
        last_media = current_media
        for prompt in profile["prompts"]:
            key=row["record_id"],prompt["id"]
            qpath=qfolder/(safe(row["record_id"])+".npy")
            need_vector=a.model=="qwen3_vl_4b" and prompt["id"]=="direct_short" and not qpath.exists()
            if key in old and not need_vector:continue
            if images is None:images,meta=load_media(row,a.media_root,a.output,a.frame_root)
            text=prompt["text"].format(question=row["question"],medium="video" if meta else "image")
            capture=PromptVector(model.model) if need_vector else None
            try:
                metadata = meta
                if memo and meta is not None:
                    metadata = {**meta, "frames_indices": list(meta["frames_indices"])}
                r=model.generate(images,text,video=meta is not None,metadata=metadata,max_new_tokens=64)
            finally:
                if capture:capture.close()
            if capture:
                vector=capture.vector
                if vector is None or vector.ndim!=1 or not np.isfinite(vector).all():raise ValueError("Prompt-vector hook did not capture valid pre-answer state")
                save_vector(qpath, vector)
            item=dict(dataset=a.dataset,partition=a.partition,record_id=row["record_id"],source_id=row["source_id"],prompt_id=prompt["id"],model=a.model,
                      prediction=r["prediction"],mean_token_logprob=float(r["mean_token_logprob"]),generated_tokens=r["generated_tokens"],status="ok",cache_origin="REGENERATED")
            if key not in old:
                with cache.open("a",encoding="utf-8") as f:f.write(json.dumps(item,ensure_ascii=False)+"\n")
                old[key]=item
            elif old[key]["prediction"]!=item["prediction"]:
                with (folder/"generation_differences.jsonl").open("a",encoding="utf-8") as f:
                    f.write(json.dumps(dict(record_id=row["record_id"],prompt_id=prompt["id"],old=old[key]["prediction"],new=item["prediction"]))+"\n")
        if i%25==0:print(json.dumps(dict(stage="predictions",model=a.model,dataset=a.dataset,partition=a.partition,processed=i)),flush=True)
    save(folder/(a.model+"_status.json"),dict(status="SMOKE_OUTPUT" if a.limit else "REGENERATED",records=len(rs),prompts=2,
                                             question_vector="First pre-answer LM-head input for canonical direct_short prompt" if a.model=="qwen3_vl_4b" else None,
                                             torch=torch.__version__,
                                             frozen_vision_cache_hits=memo.hits if memo else 0,
                                             frozen_vision_encodes=memo.encodes if memo else None))


def question(a):
    """Generate Q features without regenerating an existing answer cache."""
    if not a.model_path:raise ValueError("--model-path must identify the pinned Qwen model")
    import torch
    if not torch.cuda.is_available():raise RuntimeError("User-supplied CUDA GPU is not available")
    from neurocomputing_transfer.models.qwen import Qwen
    model=Qwen(a.model_path)
    profile=json.loads((HERE/"original_runtime_profiles.json").read_text())
    prompt=next(p for p in profile["prompts"] if p["id"]=="direct_short")
    rs=records(a.dataset,a.partition)
    if a.limit:rs=rs[:a.limit]
    folder=a.output/f"question_vectors/{a.dataset}/{a.partition}";folder.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rs,1):
        path=folder/(safe(row["record_id"])+".npy")
        if path.exists():continue
        images,metadata=load_media(row,a.media_root,a.output,a.frame_root)
        text=prompt["text"].format(question=row["question"],medium="video" if metadata else "image")
        capture=PromptVector(model.model)
        try:model.generate(images,text,video=metadata is not None,metadata=metadata,max_new_tokens=1)
        finally:capture.close()
        v=capture.vector
        if v is None or v.ndim!=1 or not np.isfinite(v).all():raise ValueError("Missing pre-answer vector")
        save_vector(path, v)
        if i%25==0:print(json.dumps(dict(stage="question",dataset=a.dataset,partition=a.partition,processed=i)),flush=True)
    save(folder/"status.json",dict(status="SMOKE_OUTPUT" if a.limit else "QUESTION_FEATURES_GENERATED",records=len(rs),
                                   feature="First LM-head input at last prompt token; the single generated token is discarded, and native confidence caches are not replaced"))


def attach_question(a):
    path=a.output/f"banks/{a.dataset}_{a.partition}.npz"
    with np.load(path,allow_pickle=False) as z:bank={k:z[k] for k in z.files}
    folder=a.output/f"question_vectors/{a.dataset}/{a.partition}"
    bank["question_hidden"]=np.stack([np.load(folder/(safe(str(rid))+".npy"),allow_pickle=False) for rid in bank["record_ids"]])
    bank["question_hidden_origin"]=np.array("NEW_FROZEN_ENCODER_PRE_ANSWER_FEATURE")
    tmp=path.with_name(path.name+".part")
    with tmp.open("wb") as f:np.savez_compressed(f,**bank)
    tmp.replace(path)
    print(json.dumps(dict(status="QUESTION_VECTORS_ATTACHED",dataset=a.dataset,partition=a.partition,records=len(bank["record_ids"]),
                          original_prediction_and_visual_arrays_retained=True)))


def assemble(a):
    from filelock import FileLock

    folder = a.output / "banks"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{a.dataset}_{a.partition}.npz"
    receipt = target.with_suffix(".json")
    with FileLock(str(target.with_suffix(".npz.lock"))):
        assemble_bank(a, target, receipt)


def assemble_bank(a, target, receipt):
    from neurocomputing_transfer.statistics.features import text_features
    rs=records(a.dataset,a.partition)
    labels={r["record_id"]:r["answers"] for r in [json.loads(x) for x in (HERE/f"prepared/{a.dataset}/{a.partition}.labels.jsonl").read_text(encoding="utf-8").splitlines()]}
    models=["qwen3_vl_4b","internvl35_8b"]+(["gemma3_12b"] if a.dataset=="vqav2" else [])
    outputs={}
    for m in models:
        for line in (a.output/f"predictions/{a.dataset}/{a.partition}/{m}.jsonl").read_text(encoding="utf-8").splitlines():
            p=json.loads(line);outputs[m,p["record_id"],p["prompt_id"]]=p
    candidates=sorted(m+"__"+p for m in models for p in ["direct_short","task_constrained_short"])
    visual=np.stack([np.load(a.output/f"visual/{a.dataset}"/(safe(r["source_id"])+".npy"),allow_pickle=False) for r in rs])
    qhidden=np.stack([np.load(a.output/f"question_vectors/{a.dataset}/{a.partition}"/(safe(r["record_id"])+".npy"),allow_pickle=False) for r in rs])
    text=[];native=[];scores=[]
    for c in candidates:
        m,p=c.split("__",1);pred=[outputs[m,r["record_id"],p] for r in rs]
        text.append(np.stack([text_features(r["question"],x["prediction"]) for r,x in zip(rs,pred)]))
        native.append([x["mean_token_logprob"] for x in pred])
        scores.append([utility(a.dataset,x["prediction"],labels[r["record_id"]]) for r,x in zip(rs,pred)])
    bank=dict(record_ids=np.array([r["record_id"] for r in rs]),source_ids=np.array([r["source_id"] for r in rs]),candidates=np.array(candidates),
              text=np.array(text,dtype=np.float64),native=np.array(native,dtype=np.float64),utility=np.array(scores,dtype=np.float64),
              visual=visual,question_hidden=qhidden,cache_origin=np.array("REGENERATED_FROM_RECOVERED_SOURCE_MEMBERSHIP"))
    if not all(np.isfinite(bank[k]).all() for k in ["text","native","utility","visual","question_hidden"]):raise ValueError("Incomplete/nonfinite bank")
    partial = target.with_suffix(".npz.part")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **bank)
    partial.replace(target)
    result=dict(status="REGENERATED_BANK_COMPLETE",dataset=a.dataset,partition=a.partition,records=len(rs),candidates=len(candidates),visual_dim=visual.shape[1],question_dim=qhidden.shape[1],
                scope="Newly regenerated caches, not a claim of byte-identical restoration of original outputs")
    save(receipt,result);print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["probe","visual","predictions","question","attach_question","assemble"],default="probe")
    p.add_argument("--gpu-ready",action="store_true");p.add_argument("--dataset",choices=["vqav2","activitynetqa"])
    p.add_argument("--partition",choices=["D","H","H2"]);p.add_argument("--model",choices=["qwen3_vl_4b","internvl35_8b","gemma3_12b"])
    p.add_argument("--model-path",type=Path);p.add_argument("--media-root",type=Path,default=ROOT);p.add_argument("--search-root",type=Path,action="append",default=[])
    p.add_argument("--encoder-device",choices=["cpu","cuda"],default="cpu",help="Original ResNet recipe used CPU; CUDA must be explicitly recorded if selected")
    p.add_argument("--output",type=Path,default=HERE/"rebuilt");p.add_argument("--frame-root",type=Path,help="Shared recovered video-frame cache; also used by isolated smoke outputs")
    p.add_argument("--limit",type=int)
    p.add_argument("--reuse-frozen-vision",action="store_true")
    a=p.parse_args()
    if a.frame_root is None:a.frame_root=a.output/"frames"
    if a.stage!="probe":
        if not a.gpu_ready:raise SystemExit("WAITING_USER_GPU: non-probe stages are deferred until the user provides GPU resources.")
        if not a.dataset or not a.partition:raise SystemExit("--dataset and --partition are required")
        if a.stage=="predictions" and not a.model:raise SystemExit("--model is required")
        if a.limit:a.output=a.output/"smoke"
    globals()[a.stage](a)


if __name__=="__main__":main()
