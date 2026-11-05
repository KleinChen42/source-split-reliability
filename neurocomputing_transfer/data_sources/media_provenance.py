"""Recover expected media bytes from acquisition receipts, including repairs."""
import json
from pathlib import Path

def expected_media_hashes(root):
    root=Path(root);hashes={}
    for dataset in ['vqav2','docvqa','activitynetqa']:
        info=json.loads((root/f'artifacts/NEURO_R1/manifests/{dataset}_preparation.json').read_text())
        if dataset=='vqav2':found={x['file']:x['sha256'] for x in info['assets'] if x['status']=='AVAILABLE'}
        elif dataset=='docvqa':found={str(root/'data/raw/docvqa/selected_images'/name):sha for name,sha in info['media_hashes'].items()}
        else:
            found={str(root/'data/raw/activitynetqa/videos'/(x['source_id']+'.mp4')):x['sha256'] for x in info['assets'] if 'sha256' in x}
            repairs=json.loads((root/'data/prepared/activitynetqa/source_repairs.json').read_text())
            for item in repairs['replacements']:
                row=item['replacement_validation'];found[row['file']]=row['sha256']
        selected={json.loads(x)['media_ref'] for x in (root/f'data/prepared/{dataset}/records.base.jsonl').read_text().splitlines()}
        if not selected<=set(found):raise ValueError('Missing acquisition hash')
        hashes.update({path:found[path] for path in selected})
    return hashes
