"""Regenerate the independent MusicGen decoder fixture from the repository root.

Requires torch==2.15.0 and transformers==5.22.2. This is fixture preparation;
Candle native inference does not invoke Python or Transformers.
"""
import hashlib,inspect,json
from pathlib import Path
import torch,transformers
from transformers import MusicgenDecoderConfig
from transformers.models.musicgen.modeling_musicgen import MusicgenForCausalLM
config=MusicgenDecoderConfig(vocab_size=8,hidden_size=8,num_hidden_layers=2,ffn_dim=16,num_attention_heads=2,num_codebooks=4,max_position_embeddings=64,pad_token_id=8,bos_token_id=8,dropout=0.,attention_dropout=0.,activation_dropout=0.)
config._attn_implementation='eager'
m=MusicgenForCausalLM(config).eval()
with torch.no_grad():
 for i,(name,p) in enumerate(m.named_parameters()):
  values=torch.sin(torch.arange(p.numel()).reshape(p.shape)*.17+i)*.2
  if name.endswith('layer_norm.weight'): values=values+1.
  p.copy_(values)
 ids=torch.tensor([[8,0,1,2],[8,1,2,3],[8,2,3,4],[8,3,4,5]])
 context=torch.sin(torch.arange(24).reshape(1,3,8)*.23)
 logits=m(ids,encoder_hidden_states=context,use_cache=False).logits
 unconditional=m(ids,encoder_hidden_states=torch.zeros_like(context),use_cache=False).logits
 delay=[]
 for channels,books in [(1,4),(2,8)]:
  mc=MusicgenDecoderConfig(vocab_size=8,hidden_size=8,num_hidden_layers=1,ffn_dim=16,num_attention_heads=2,num_codebooks=books,audio_channels=channels,pad_token_id=8,bos_token_id=8)
  dm=MusicgenForCausalLM(mc).eval()
  _,mask=dm.build_delay_pattern_mask(torch.full((books,1),8),8,max_length=8)
  delay.append({'channels':channels,'books':books,'length':8,'frames':4,'mask':mask.flatten().tolist()})
 fixture={'source':'transformers.models.musicgen.modeling_musicgen','version':transformers.__version__,'torch':torch.__version__,'source_sha256':hashlib.sha256(inspect.getsource(inspect.getmodule(m)).encode()).hexdigest(),'config':config.to_dict(),'weights':{k:{'shape':list(v.shape),'values':v.flatten().tolist()} for k,v in m.state_dict().items()},'input_ids':ids.flatten().tolist(),'context':context.flatten().tolist(),'expected':logits.flatten().tolist(),'expected_unconditional':unconditional.flatten().tolist(),'delay':delay}
 Path('candle-transformers/tests/fixtures/musicgen_decoder.json').write_text(json.dumps(fixture,indent=2)+'\n')
 print(transformers.__version__,logits.shape)

# Preserve known reference codes even where a delayed codebook extends beyond
# the first codebook's prefill. Compare actual HF generate, not a custom loop.
with torch.no_grad():
 prefix=torch.tensor([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
 prompt=torch.cat([torch.full((4,1),8),prefix],dim=1)
 _,mask=m.build_delay_pattern_mask(prompt,8,max_length=11)
 generated=m.generate(input_ids=prompt,encoder_hidden_states=context,max_length=11,do_sample=False,guidance_scale=1.,pad_token_id=8,bos_token_id=8,decoder_start_token_id=8,eos_token_id=None,num_return_sequences=1,num_beams=1,use_cache=False)
 long_prefix=torch.stack([(torch.arange(33)+book)%8 for book in range(4)])
 long_prompt=torch.cat([torch.full((4,1),8),long_prefix],dim=1)
 long_generated=m.generate(input_ids=long_prompt,encoder_hidden_states=context,max_length=41,do_sample=False,guidance_scale=1.,pad_token_id=8,bos_token_id=8,decoder_start_token_id=8,eos_token_id=None,num_return_sequences=1,num_beams=1,use_cache=False)
 continuation={'prefix':prefix.flatten().tolist(),'prompt_frames':3,'new_frames':4,'mask':mask.flatten().tolist(),'expected_codes':generated.flatten().tolist(),'reference_use_cache':False,'long_prefix':long_prefix.flatten().tolist(),'long_expected_codes':long_generated.flatten().tolist()}
 # Stereo reference delays have paired rows. Only the pattern needs the stereo
 # model: its actual decoder numerical path is covered by the mono fixture.
 stereo_prefix=torch.stack([prefix[0],prefix[0],prefix[1],prefix[1],prefix[2],prefix[2],prefix[3],prefix[3]])
 stereo_prompt=torch.cat([torch.full((8,1),8),stereo_prefix],dim=1)
 _,stereo_mask=dm.build_delay_pattern_mask(stereo_prompt,8,max_length=11)
 continuation['stereo_prefix']=stereo_prefix.flatten().tolist()
 continuation['stereo_mask']=stereo_mask.flatten().tolist()
 path=Path('candle-transformers/tests/fixtures/musicgen_decoder.json')
 fixture=json.loads(path.read_text());fixture['continuation']=continuation;path.write_text(json.dumps(fixture,indent=2)+'\n')
 print('continuation',generated.shape)
# The adapter crate owns its golden fixture so packaged tests do not depend on
# the workspace layout of a separate crate.
adapter_fixture={key:fixture[key] for key in ['source','version','torch','source_sha256','config','weights','context','continuation']}
Path('candle-audio/tests/fixtures/musicgen-continuation.json').write_text(json.dumps(adapter_fixture,indent=2)+'\n')
