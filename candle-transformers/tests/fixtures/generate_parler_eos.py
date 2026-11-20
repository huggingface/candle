"""Regenerate from the repository root with torch==2.15.0 and transformers==5.22.2.
Only the pinned processor source is fetched; no checkpoints are loaded. CPU
isin_mps_friendly is exactly torch.isin, injected to avoid unrelated MPS imports.
"""
import ast,hashlib,json,math
from pathlib import Path
import torch
from transformers import LogitsProcessor
from urllib.request import urlopen
revision='d108732cd57788ec86bc857d99a6cabd66663d68'
source_url=f'https://raw.githubusercontent.com/huggingface/parler-tts/{revision}/parler_tts/logits_processors.py'
source=urlopen(source_url).read().decode()
assert hashlib.sha256(source.encode()).hexdigest()=='29e4bbf190f6586cd367ce0281cc7030066376a075cf15aac0ba5ea0d16ec2d8'
cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='ParlerTTSLogitsProcessor')
namespace={'torch':torch,'math':math,'LogitsProcessor':LogitsProcessor,'isin_mps_friendly':torch.isin}
exec(compile(ast.Module(body=[cls],type_ignores=[]),'official-parler-eos','exec'),namespace)
processor=namespace['ParlerTTSLogitsProcessor'](4,3,1,'cpu')
histories=[[[5],[5],[5]],[[5,4],[5,0],[5,1]],[[5,4,4],[5,0,4],[5,1,2]],[[5,4,4,4],[5,0,4,4],[5,1,2,4]]]
scores=torch.tensor([[.1,.2,.3,.4,9.,.5],[.5,.4,.3,.2,8.,.1],[.7,.2,.1,.9,7.,.6]])
steps=[]
for history in histories:
 ids=torch.tensor(history);actual=processor(ids,scores.clone())
 steps.append({'finished':(ids==4).any(1).tolist(),'scores':scores.tolist(),'expected':[[v if math.isfinite(v) else None for v in row] for row in actual.tolist()]})
fixture={'source':f'https://github.com/huggingface/parler-tts/blob/{revision}/parler_tts/logits_processors.py','source_sha256':hashlib.sha256(source.encode()).hexdigest(),'torch':torch.__version__,'eos':4,'codebooks':3,'vocab':6,'steps':steps}
fixture['revision']=revision
Path('candle-transformers/tests/fixtures/parler_eos.json').write_text(json.dumps(fixture,indent=2)+'\n')
