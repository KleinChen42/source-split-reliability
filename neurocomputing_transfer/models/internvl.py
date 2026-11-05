"""InternVL public chat API with lossless generation-score capture.

Preprocessing follows the pinned OpenGVLab model card. Weight/source files are
never edited. The public chat method still receives a token-sequence return.
"""
import math
from PIL import Image


def tiles(image,max_tiles=12,size=448):
    width,height=image.size;aspect=width/height
    ratios={(w,h) for w in range(1,max_tiles+1) for h in range(1,max_tiles+1) if w*h<=max_tiles}
    ratios=sorted(ratios,key=lambda pair:pair[0]*pair[1])
    best=(1,1);distance=float('inf')
    for w,h in ratios:
        current=abs(aspect-w/h)
        if current<distance or (current==distance and width*height>.5*size*size*w*h):
            best=(w,h);distance=current
    w,h=best;resized=image.resize((w*size,h*size))
    result=[resized.crop((x*size,y*size,(x+1)*size,(y+1)*size)) for y in range(h) for x in range(w)]
    if len(result)>1:result.append(image.resize((size,size)))
    return result


class InternVL:
    def __init__(self,path):
        import torch
        from transformers import AutoModel,AutoTokenizer
        from torchvision import transforms as T
        self.torch=torch
        self.model=AutoModel.from_pretrained(str(path),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,
                         use_flash_attn=False,trust_remote_code=True,local_files_only=True).eval().cuda()
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),trust_remote_code=True,use_fast=False,local_files_only=True)
        self.transform=T.Compose([T.Resize((448,448),interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),
                                  T.Normalize((.485,.456,.406),(.229,.224,.225))])

    def generate(self,images,text,video=False,max_new_tokens=64,check_native=False,metadata=None):
        torch=self.torch;patches=[];counts=[]
        for image in images:
            group=tiles(image.convert('RGB'),max_tiles=1 if video else 12)
            counts.append(len(group));patches.extend(self.transform(x) for x in group)
        pixels=torch.stack(patches).to(device=self.model.device,dtype=torch.bfloat16)
        if video:question=''.join(f'Frame{i+1}: <image>\n' for i in range(len(images)))+text
        else:
            if len(images)!=1:raise ValueError('Single image required')
            question='<image>\n'+text
        original=self.model.generate;captured={}
        def capture(*args,**kwargs):
            kwargs.update(return_dict_in_generate=True,output_scores=True)
            result=original(*args,**kwargs);captured['result']=result
            return result.sequences
        generation={'max_new_tokens':max_new_tokens,'do_sample':False}
        with torch.inference_mode():
            self.model.generate=capture
            try:answer=self.model.chat(self.tokenizer,pixels,question,dict(generation),num_patches_list=counts)
            finally:self.model.generate=original
            generated=captured['result']
            scores=self.model.language_model.compute_transition_scores(generated.sequences,generated.scores,normalize_logits=True)
            mean=float(scores.float().mean())
            if check_native:
                reference=self.model.chat(self.tokenizer,pixels,question,dict(generation),num_patches_list=counts)
                if answer!=reference:raise RuntimeError('Score capture changed native decoded answer')
        if not math.isfinite(mean):raise RuntimeError('Nonfinite generation confidence')
        return {'prediction':answer,'mean_token_logprob':mean,'generated_tokens':len(generated.scores),
                'native_api_equivalence_checked':check_native,'patch_counts':counts}
