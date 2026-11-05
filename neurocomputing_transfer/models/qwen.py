"""Native Qwen3-VL adapter for images and pre-sampled timestamped videos."""
import math

class Qwen:
    def __init__(self,path):
        import torch
        from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
        self.torch=torch
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(str(path),torch_dtype=torch.bfloat16,
                        device_map={'':0},attn_implementation='sdpa',local_files_only=True).eval()
        self.processor=AutoProcessor.from_pretrained(str(path),local_files_only=True)

    def generate(self,images,text,video=False,max_new_tokens=64,metadata=None,**kwargs):
        import numpy as np
        if video:
            if metadata is None:raise ValueError('Video timestamps required')
            messages=[{'role':'user','content':[{'type':'video'},{'type':'text','text':text}]}]
            rendered=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=self.processor(text=[rendered],videos=[np.stack([np.asarray(x) for x in images])],
                      video_metadata=[metadata],do_sample_frames=False,return_tensors='pt').to(self.model.device)
            if int(inputs['video_grid_thw'][0,0])!=len(images)//2:raise RuntimeError('Unexpected temporal frame count')
        else:
            messages=[{'role':'user','content':[{'type':'image','image':images[0]},{'type':'text','text':text}]}]
            inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                      return_dict=True,return_tensors='pt').to(self.model.device)
        with self.torch.inference_mode():
            generated=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,
                                           return_dict_in_generate=True,output_scores=True)
            transitions=self.model.compute_transition_scores(generated.sequences,generated.scores,normalize_logits=True)
        tokens=generated.sequences[:,inputs['input_ids'].shape[1]:]
        prediction=self.processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
        confidence=float(transitions.float().mean())
        if not math.isfinite(confidence):raise RuntimeError('Nonfinite confidence')
        return {'prediction':prediction,'mean_token_logprob':confidence,'generated_tokens':int(tokens.shape[1]),
                'input_tokens':int(inputs['input_ids'].shape[1])}
