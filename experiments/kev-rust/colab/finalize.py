import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

ROOT=Path('/content/candle/experiments/kev-rust')
STATE=Path('/content/kev-cpu-final-status.json')
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'','RAYON_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}
ENV['PATH']='/root/.cargo/bin:'+ENV['PATH']
START=time.time()


def run(cmd,out=None):
    print('+',' '.join(map(str,cmd)),flush=True)
    if out:
        with out.open('w') as f: subprocess.run(list(map(str,cmd)),cwd=ROOT,env=ENV,stdout=f,check=True)
    else: subprocess.run(list(map(str,cmd)),cwd=ROOT,env=ENV,check=True)


try:
    STATE.write_text(json.dumps({'status':'running','pid':os.getpid(),'started':START}))
    run(['cargo','fmt']);run(['cargo','test','--release']);run(['cargo','build','--release'])
    py=ROOT/'.venv/bin/python'; result=ROOT/'results-v2'
    run([ROOT/'target/release/kev-rust','--input','fixtures/parity-v2.jsonl'],result/'default-parity.jsonl')
    run([py,'scripts/compare.py',result/'python-parity.jsonl',result/'default-parity.jsonl','--out',result/'default-parity.json'])
    rows=[json.loads(s) for s in (result/'default-parity.jsonl').read_text().splitlines()]
    assert all(r['batched'] and r['mode']=='prefix' for r in rows)
    report=json.loads((result/'report.json').read_text())
    report['default_mode']='prefix-batched'
    report['default_parity']=json.loads((result/'default-parity.json').read_text())
    report['final_source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'Cargo.toml',*sorted((ROOT/'src').glob('*.rs'))]}
    (result/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    run(['tar','-czf','/content/kev-cpu-final-results.tar.gz','results','results-v2','src','Cargo.toml','Cargo.lock'])
    STATE.write_text(json.dumps({'status':'complete','pid':os.getpid(),'started':START,'ended':time.time()}))
except BaseException as e:
    STATE.write_text(json.dumps({'status':'failed','pid':os.getpid(),'started':START,'error':str(e)}))
    traceback.print_exc();raise
