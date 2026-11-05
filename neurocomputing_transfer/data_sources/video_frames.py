"""PyAV decode with actual presentation timestamps, including variable-rate video."""
from pathlib import Path
import hashlib,json,math

def extract_frames(path,cache,count=8):
    import fcntl
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    with (cache/'decode.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _extract_frames(path,cache,count)

def _extract_frames(path,cache,count):
    import av
    path=Path(path);cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(path.read_bytes()).hexdigest();receipt=cache/'timestamps.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved.get('decoder')=='pyav-15.1.0' and saved.get('media_sha256')==digest and saved.get('frame_count')==count:
            if all((cache/f'{i:02d}.png').exists() for i in range(count)):return saved
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_type='NONE';stream.codec_context.thread_count=1
        start=stream.start_time or 0;time_base=float(stream.time_base)
        duration=float(stream.duration*stream.time_base) if stream.duration else float(container.duration/av.time_base)
        fps=float(stream.average_rate) if stream.average_rate else None
        if duration<=0 or not math.isfinite(duration) or not fps or fps<=0:raise ValueError('Invalid video timing')
        requested=[duration*(i+.5)/count for i in range(count)];timestamps=[]
        for i,target in enumerate(requested):
            container.seek(start+int(target/time_base),stream=stream,backward=True,any_frame=False)
            selected=None
            for frame in container.decode(stream):
                if frame.pts is None:continue
                actual=(frame.pts-start)*time_base
                if actual+1e-9>=target:selected=frame;break
            if selected is None:raise ValueError('No decoded frame at requested midpoint')
            if selected.is_corrupt:raise ValueError('Selected video frame is corrupt')
            selected.to_image().convert('RGB').save(cache/f'{i:02d}.png')
            timestamps.append((selected.pts-start)*time_base)
        result={'decoder':'pyav-15.1.0','media_sha256':digest,'duration':duration,'fps':fps,
                'total_num_frames':stream.frames or None,'frame_count':count,'requested_timestamps':requested,
                'timestamps':timestamps,'frame_sha256':[hashlib.sha256((cache/f'{i:02d}.png').read_bytes()).hexdigest() for i in range(count)]}
    receipt.write_text(json.dumps(result,indent=2)+'\n');return result
