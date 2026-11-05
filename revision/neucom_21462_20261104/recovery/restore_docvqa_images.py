"""Recover the selected original DocVQA pages with bounded HTTP range reads.

Uses the pinned public Parquet files and already recovered row identities. No
model runs, replacement sampling, or historical file-checksum audit is involved.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Lock
import argparse
import io
import json
import time
import urllib.parse
import urllib.request

import pyarrow.parquet as pq
from PIL import Image

from recover_docvqa_text import HERE, ROOT, source_spec

PLAN = HERE / "docvqa_image_ranges.json"
DEST = ROOT / "data/raw/docvqa/selected_images"


class ImageRanges(io.RawIOBase):
    def __init__(self, url, size):
        super().__init__()
        self.original_url = url
        self.url = url
        self.size = size
        self.pos = 0
        self.transferred = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0: raise ValueError("Negative file position")
        return self.pos

    def read(self, n=-1):
        if self.pos >= self.size: return b""
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        chunks = []
        while n:
            size = min(n, 8 * 1024**2)
            start = self.pos
            end = start + size - 1
            for attempt in range(4):
                # A fresh canonical redirect also avoids a stale signed URL.
                url = self.url if attempt == 0 else self.original_url + "?" + urllib.parse.urlencode({"recovery_range": f"{start}-{end}"})
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "NEUCOM-input-recovery/1.0", "Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=60) as response:
                        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                            raise ValueError("Provider did not honor the requested bounded range")
                        data = response.read(size + 1)
                        self.url = response.geturl()
                    if len(data) != size: raise ValueError("Incomplete Parquet range")
                    break
                except Exception:
                    if attempt == 3: raise
                    time.sleep(attempt + 1)
            chunks.append(data)
            self.pos += size
            self.transferred += size
            n -= size
        return b"".join(chunks)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def selected_pages():
    selected = set()
    for part in ["D", "H"]:
        for line in (HERE / f"prepared/docvqa/{part}.records.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            selected.add(row["page_id"])
    for page in selected:
        parent, number = page.rsplit("_", 1)
        if not parent.isalnum() or not number.isdigit(): raise ValueError("Unsafe document page ID")
    return selected


def shard_groups(spec, revision, desired):
    cache = HERE / "downloads/docvqa_text" / (Path(spec["file"]).stem + ".jsonl")
    rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    url = f"https://huggingface.co/datasets/lmms-lab/DocVQA/resolve/{revision}/{spec['file']}"
    stream = ImageRanges(url, spec["bytes"])
    parquet = pq.ParquetFile(stream)
    if parquet.metadata.num_rows != len(rows): raise ValueError("Text cache and pinned Parquet row counts differ")
    groups = []
    offset = 0
    for i in range(parquet.num_row_groups):
        meta = parquet.metadata.row_group(i)
        pages = [r["ucsf_document_id"].strip() + "_" + str(r["ucsf_document_page_no"]) for r in rows[offset:offset + meta.num_rows]]
        needed = sorted(set(pages) & desired)
        if needed:
            image_bytes = sum(meta.column(j).total_compressed_size for j in range(meta.num_columns) if meta.column(j).path_in_schema == "image.bytes")
            groups.append(dict(file=spec["file"],file_bytes=spec["bytes"],group=i,row_offset=offset,rows=meta.num_rows,image_column_bytes=image_bytes,selected_pages=needed))
        offset += meta.num_rows
    return groups, stream.transferred


def make_plan():
    desired = selected_pages()
    prep = source_spec()
    groups = []
    network = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(shard_groups, s, prep["revision"], desired) for s in prep["files"]]
        for future in as_completed(futures):
            batch, n = future.result()
            groups.extend(batch)
            network += n
    # A page can appear in many QA rows and shards. Choose a small cover of
    # compressed image columns rather than downloading every duplicate image.
    remaining = set(desired)
    chosen = []
    while remaining:
        useful = [(g, set(g["selected_pages"]) & remaining) for g in groups]
        useful = [(g, pages) for g, pages in useful if pages]
        if not useful: raise ValueError(f"Selected pages absent from pinned shards: {sorted(remaining)[:3]}")
        g, pages = min(useful, key=lambda pair: (pair[0]["image_column_bytes"] / len(pair[1]), pair[0]["file"], pair[0]["group"]))
        chosen.append(g)
        remaining -= pages
    result = dict(status="SELECTED_DOCUMENT_IMAGE_RANGES_LOCATED",revision=prep["revision"],selected_pages=len(desired),
                  row_groups=len(chosen),image_column_bytes=sum(g["image_column_bytes"] for g in chosen),footer_network_bytes=network,groups=chosen)
    PLAN.write_text(json.dumps(result,indent=2) + "\n",encoding="utf-8")
    return result


def restore(plan, workers, limit):
    DEST.mkdir(parents=True,exist_ok=True)
    desired = selected_pages()
    jobs = plan["groups"][:limit] if limit else plan["groups"]
    lock = Lock()
    progress = HERE / "docvqa_image_progress.jsonl"

    def one(g):
        needed = {p for p in g["selected_pages"] if not (DEST / (p + ".png")).is_file()}
        result = dict(file=g["file"],group=g["group"],selected_pages=len(g["selected_pages"]))
        if not needed: return dict(result,status="CACHED",new_pages=0,network_bytes=0)
        try:
            text = HERE / "downloads/docvqa_text" / (Path(g["file"]).stem + ".jsonl")
            rows = [json.loads(line) for line in text.read_text(encoding="utf-8").splitlines()][g["row_offset"]:g["row_offset"] + g["rows"]]
            url = f"https://huggingface.co/datasets/lmms-lab/DocVQA/resolve/{plan['revision']}/{g['file']}"
            stream = ImageRanges(url,g["file_bytes"])
            table = pq.ParquetFile(stream).read_row_group(g["group"],columns=["image.bytes"],use_threads=False)
            images = table.to_pylist()
            if len(images) != len(rows): raise ValueError("Image and identity row counts differ")
            new = 0
            for row, image_row in zip(rows,images):
                page = row["ucsf_document_id"].strip() + "_" + str(row["ucsf_document_page_no"])
                if page not in needed: continue
                payload = image_row["image"]["bytes"]
                with Image.open(io.BytesIO(payload)) as image: image.verify()
                path = DEST / (page + ".png")
                # Overlapping row groups may be in flight. Serialize only the
                # atomic commit, so an already restored page is left intact.
                with lock:
                    if not path.is_file():
                        partial = path.with_suffix(".png.part")
                        partial.write_bytes(payload)
                        partial.replace(path)
                        new += 1
                needed.remove(page)
            if needed: raise ValueError(f"Missing image payloads for {len(needed)} selected pages")
            result.update(status="IMAGES_RESTORED",new_pages=new,network_bytes=stream.transferred)
        except Exception as exc:
            result.update(status="IMAGE_RECOVERY_FAILED",error=str(exc))
        with lock:
            with progress.open("a",encoding="utf-8") as f: f.write(json.dumps(result) + "\n")
        return result

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one,g) for g in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) <= 2 or len(results) % 10 == 0:
                print(json.dumps(dict(groups_processed=len(results),groups_planned=len(jobs),new_pages=sum(r.get("new_pages",0) for r in results),failures=sum(r["status"]=="IMAGE_RECOVERY_FAILED" for r in results))),flush=True)
    available = sum((DEST / (page + ".png")).is_file() for page in desired)
    status = dict(snapshot_time_utc=datetime.now(timezone.utc).isoformat(),status="SELECTED_DOCUMENT_IMAGES_RESTORED" if available==len(desired) else "PARTIAL_DOCUMENT_IMAGE_RECOVERY",
                  selected_pages=len(desired),available=available,missing=len(desired)-available,total_bytes=sum((DEST / (page + ".png")).stat().st_size for page in desired if (DEST / (page + ".png")).is_file()),
                  failed_groups=[r for r in results if r["status"]=="IMAGE_RECOVERY_FAILED"],model_inference_started=False)
    (HERE / "docvqa_image_restore_status.json").write_text(json.dumps(status,indent=2) + "\n",encoding="utf-8")
    print(json.dumps(status,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore",action="store_true")
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--limit-groups",type=int)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text()) if PLAN.exists() else make_plan()
    print(json.dumps({k:v for k,v in plan.items() if k!="groups"},indent=2),flush=True)
    if args.restore: restore(plan,args.workers,args.limit_groups)


if __name__=="__main__": main()
