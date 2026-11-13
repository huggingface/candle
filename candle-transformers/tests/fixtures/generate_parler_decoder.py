"""Regenerate the CPU Parler decoder/cache oracle with the official package.

Install parler-tts at d108732cd57788ec86bc857d99a6cabd66663d68 and
its pinned transformers==4.46.1 dependency in a separate QA environment.
Run from the Candle workspace root. No pretrained weights are required.
"""
import json,hashlib,inspect
from pathlib import Path
from parler_tts import ParlerTTSForCausalLM
from parler_tts.configuration_parler_tts import ParlerTTSDecoderConfig
import torch,transformers,tokenizers
cfg=ParlerTTSDecoderConfig(hidden_size=8,ffn_dim=16,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,num_cross_attention_key_value_heads=2,num_codebooks=3,vocab_size=6,bos_token_id=5,eos_token_id=4,pad_token_id=4,max_position_embeddings=32,rope_embeddings=False,dropout=0.,attention_dropout=0.,activation_dropout=0.)
cfg._attn_implementation='eager'
m=ParlerTTSForCausalLM(cfg).eval()
with torch.no_grad():
 for i,(name,p) in enumerate(m.named_parameters()):
  if name.endswith('embed_positions.weights'):continue
  value=torch.sin(torch.arange(p.numel()).reshape(p.shape)*.17+i)*.2
  if name.endswith('layer_norm.weight'):value=value+1.
  p.copy_(value)
 ids=torch.tensor([[5,0,1,2],[5,1,2,3],[5,2,3,0]])
 prompt=torch.cos(torch.arange(16).reshape(1,2,8)*.23)
 context=torch.sin(torch.arange(24).reshape(1,3,8)*.31)
 full=m(ids,prompt_hidden_states=prompt,encoder_hidden_states=context,use_cache=False).logits
 cache=None;parts=[]
 for i in range(4):
  result=m(ids[:,i:i+1],prompt_hidden_states=prompt if i==0 else None,encoder_hidden_states=context,past_key_values=cache,use_cache=True)
  parts.append(result.logits);cache=result.past_key_values
 inc=torch.cat(parts,dim=1)
 print('shape',full.shape,'cache diff',(full-inc).abs().max().item())
 fixture={'source':'huggingface/parler-tts','revision':'d108732cd57788ec86bc857d99a6cabd66663d68','transformers':transformers.__version__,'torch':torch.__version__,'tokenizers':tokenizers.__version__,'source_sha256':hashlib.sha256(inspect.getsource(inspect.getmodule(m)).encode()).hexdigest(),'config':cfg.to_dict(),'weights':{k:{'shape':list(v.shape),'values':v.flatten().tolist()} for k,v in m.state_dict().items()},'ids':ids.flatten().tolist(),'prompt':prompt.flatten().tolist(),'context':context.flatten().tolist(),'expected':full.flatten().tolist(),'expected_incremental':inc.flatten().tolist()}
 Path('candle-transformers/tests/fixtures/parler_decoder.json').write_text(json.dumps(fixture,indent=2)+'\n')
