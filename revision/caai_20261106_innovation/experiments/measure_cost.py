"""Synchronized measurements on outcome-independent, real video questions."""
from pathlib import Path
import argparse
import copy
import csv
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
RECOVERY = ROOT / "revision/neucom_21462_20261104/recovery"
sys.path[:0] = [str(ROOT), str(RECOVERY)]
from rebuild_gpu_inputs import records, load_media, PromptVector


def timed(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return time.perf_counter() - start, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "internvl"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    by_source = {}
    for row in records("activitynetqa", "D"):
        by_source.setdefault(row["source_id"], []).append(row)
    sources = np.array(sorted(by_source))
    sources = sources[np.random.default_rng(2026110601).permutation(len(sources))[:67]]
    sample = [sorted(by_source[source], key=lambda x: x["record_id"])[0] for source in sources]
    if args.model == "qwen":
        from neurocomputing_transfer.models.qwen import Qwen
        model_path = ROOT / "models/qwen3_vl_4b_0196dc7bb23f"
        factory = Qwen
    else:
        from neurocomputing_transfer.models.internvl import InternVL
        model_path = ROOT / "models/internvl35_8b_ae389705594f"
        factory = InternVL
    start = time.perf_counter()
    adapter = factory(model_path)
    load_seconds = time.perf_counter() - start
    profile = json.loads((RECOVERY / "original_runtime_profiles.json").read_text())
    prompts = {p["id"]: p["text"] for p in profile["prompts"]}
    result_rows, mismatches = [], []
    for index, row in enumerate(sample):
        images, metadata = load_media(row, ROOT / "data/raw", RECOVERY / "rebuilt",
                                      RECOVERY / "rebuilt/frames")
        def generate(tokens=64, capture=False, prompt_id="direct_short"):
            text = prompts[prompt_id].format(question=row["question"], medium="video")
            hook = PromptVector(adapter.model) if capture else None
            try:
                result = adapter.generate(images, text, video=True, metadata=copy.deepcopy(metadata),
                                          max_new_tokens=tokens)
            finally:
                if hook:
                    hook.close()
            if hook:
                if hook.vector is None or hook.vector.shape != (2560,):
                    raise ValueError("Pre-answer state was not captured")
            return result

        if index < 3:
            generate()
            generate(prompt_id="task_constrained_short")
            if args.model == "qwen":
                generate(capture=True)
                generate(tokens=1, capture=True)
            continue
        modes = [("answer_native", dict(prompt_id="direct_short"))]
        if args.model == "qwen":
            modes.append(("answer_native_and_Q", dict(capture=True, prompt_id="direct_short")))
            if index % 2:
                modes.reverse()
            modes.append(("separate_Q_pass", dict(tokens=1, capture=True, prompt_id="direct_short")))
        modes.append(("answer_native", dict(prompt_id="task_constrained_short")))
        outputs = {}
        for mode, parameters in modes:
            seconds, result = timed(lambda: generate(**parameters))
            outputs[mode, parameters["prompt_id"]] = result
            result_rows.append(dict(model=args.model, mode=mode, prompt=parameters["prompt_id"], sample_index=index-3,
                record_id=row["record_id"], source_id=row["source_id"], seconds=seconds,
                generated_tokens=result["generated_tokens"], frames=8))
        if args.model == "qwen":
            if outputs["answer_native", "direct_short"]["prediction"] != outputs["answer_native_and_Q", "direct_short"]["prediction"]:
                mismatches.append(row["record_id"])
        if (index - 2) % 8 == 0:
            print(json.dumps(dict(model=args.model, measured=index-2, target=64)), flush=True)
    if args.model == "internvl":
        # Measure the common visual feature on the same 64 sources, after the
        # generator timing. I/O and frame decoding remain outside this boundary.
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        encoder = resnet50(weights=weights).eval().cuda()
        encoder.fc = torch.nn.Identity()
        transform = weights.transforms()
        for index, row in enumerate(sample):
            images, _ = load_media(row, ROOT / "data/raw", RECOVERY / "rebuilt",
                                   RECOVERY / "rebuilt/frames")
            def extract():
                with torch.inference_mode():
                    return encoder(torch.stack([transform(im) for im in images]).cuda()).mean(0).cpu().numpy()
            seconds, vector = timed(extract)
            if vector.shape != (2048,):
                raise ValueError("Wrong visual feature width")
            if index >= 3:
                result_rows.append(dict(model="resnet50", mode="visual_per_source", prompt="none", sample_index=index-3,
                    record_id=row["record_id"], source_id=row["source_id"], seconds=seconds,
                    generated_tokens=0, frames=8))
    fields = list(result_rows[0])
    with (args.output / f"latency_{args.model}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader(); writer.writerows(result_rows)
    summary = dict(model=args.model, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        measured_questions=64, warmup_questions=3, model_load_seconds=load_seconds,
        timing="Synchronized wall time: preprocessing, transfer, model execution and score/state extraction; excludes disk I/O, frame decoding, loading and warm-up",
        source_features="ResNet is computed once per source and can be shared by its ten questions",
        separate_Q="One Qwen generation step; its answer token is discarded",
        answer_changes_with_capture=mismatches, statistics={})
    for mode, prompt_id in sorted({(r["mode"],r["prompt"]) for r in result_rows}):
        values = np.array([r["seconds"] for r in result_rows if r["mode"] == mode and r["prompt"] == prompt_id])
        summary["statistics"][mode + "__" + prompt_id] = dict(n=len(values), mean_seconds=float(values.mean()),
            median_seconds=float(np.median(values)), p95_seconds=float(np.quantile(values,.95)))
    (args.output / f"latency_{args.model}.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
