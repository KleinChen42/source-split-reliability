"""Native Gemma-3 image adapter; no video-capability claim."""
import math

class Gemma:
    def __init__(self,path):
        import torch
        from transformers import Gemma3ForConditionalGeneration,AutoProcessor
        self.torch=torch
        self.model=Gemma3ForConditionalGeneration.from_pretrained(str(path),torch_dtype=torch.bfloat16,
                      device_map={'':0},attn_implementation='sdpa',local_files_only=True).eval()
        self.processor=AutoProcessor.from_pretrained(str(path),local_files_only=True,use_fast=False)

    def generate(self,images,text,video=False,max_new_tokens=64,**kwargs):
        if video or len(images)!=1:raise ValueError('Gemma panel is single-image only')
        messages=[{'role':'user','content':[{'type':'image','image':images[0]},{'type':'text','text':text}]}]
        inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                       return_dict=True,return_tensors='pt').to(self.model.device)
        with self.torch.inference_mode():
            generated=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,
                                          return_dict_in_generate=True,output_scores=True)
            # Transformers 4.50's generic transition helper assumes a flat
            # config.vocab_size; Gemma3 stores it under text_config. Greedy
            # decoding has no beam ancestry, so gather the same normalized
            # token log probabilities directly from the generation scores.
            steps=len(generated.scores)
            emitted=generated.sequences[:,-steps:]
            transitions=self.torch.stack([self.torch.log_softmax(score.float(),dim=-1).gather(1,emitted[:,i:i+1]).squeeze(1)
                                          for i,score in enumerate(generated.scores)],dim=1)
        tokens=generated.sequences[:,inputs['input_ids'].shape[1]:]
        # This pinned tokenizer does not mark every generation EOS ID as a
        # tokenizer special token. Remove terminal control IDs by config, not
        # by answer-dependent string cleanup. Confidence still includes EOS.
        eos=self.model.generation_config.eos_token_id
        eos={eos} if isinstance(eos,int) else set(eos or [])
        answer_tokens=tokens[0].tolist()
        while answer_tokens and answer_tokens[-1] in eos:answer_tokens.pop()
        prediction=self.processor.tokenizer.decode(answer_tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
        confidence=float(transitions.float().mean())
        if not math.isfinite(confidence):raise RuntimeError('Nonfinite confidence')
        return {'prediction':prediction,'mean_token_logprob':confidence,'generated_tokens':int(tokens.shape[1])}
