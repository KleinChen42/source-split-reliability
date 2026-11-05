"""Fixed, label-free text features; learned transformations belong inside F."""
import re
import numpy as np

PREFIXES=('what','who','where','when','why','how','is','are','does','do','other')

def text_features(question,prediction):
    if not isinstance(question,str) or prediction is not None and not isinstance(prediction,str):raise ValueError('Text expected')
    words=re.findall(r'[a-z]+',question.lower());prefix=words[0] if words and words[0] in PREFIXES[:-1] else 'other'
    return np.array([np.log1p(len(question)),np.log1p(len(prediction or ''))]+[float(prefix==p) for p in PREFIXES])

def assemble(question,prediction,visual=None,native_logprob=None,kind='common'):
    if kind=='native_only':
        if native_logprob is None or not np.isfinite(native_logprob):raise ValueError('Native confidence unavailable')
        return np.array([native_logprob])
    text=text_features(question,prediction)
    if kind=='text_only':return text
    visual=np.asarray(visual,dtype=float)
    if visual.ndim!=1 or len(visual)!=2048 or not np.isfinite(visual).all():raise ValueError('Invalid frozen visual representation')
    if kind=='visual_only':return visual
    common=np.concatenate([text,visual])
    if kind=='common':return common
    if kind=='common_plus_native':return np.concatenate([common,assemble(question,prediction,native_logprob=native_logprob,kind='native_only')])
    raise ValueError('Unknown feature specification')
