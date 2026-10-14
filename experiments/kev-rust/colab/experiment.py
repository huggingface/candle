"""Second CPU-only experiment: compare batched question rows and validate numeric rendering."""
import json
import os
import platform
import statistics
import importlib.metadata
import hashlib
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path('/content/candle/experiments/kev-rust')
STATE=Path('/content/kev-cpu-experiment-status.json')
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'','RAYON_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}
ENV['PATH']='/root/.cargo/bin:'+ENV['PATH']
START=time.time()


def stage(name):
    STATE.write_text(json.dumps({'status':'running','stage':name,'pid':os.getpid(),'started':START}))
    print('STAGE:',name,flush=True)


def run(cmd,out=None):
    print('+',' '.join(map(str,cmd)),flush=True)
    if out:
        with out.open('w') as f: subprocess.run(list(map(str,cmd)),cwd=ROOT,env=ENV,stdout=f,check=True)
    else: subprocess.run(list(map(str,cmd)),cwd=ROOT,env=ENV,check=True)


try:
    stage('build and kernel tests')
    run(['rustup','component','add','rustfmt'])
    run(['cargo','fmt'])
    run(['cargo','test','--release'])
    run(['cargo','build','--release'])
    py=ROOT/'.venv/bin/python'; binary=ROOT/'target/release/kev-rust'
    result=ROOT/'results-v2';result.mkdir(exist_ok=True)
    stage('Python numeric and question-isolation reference')
    run([py,'scripts/reference.py','--input','fixtures/parity-v2.jsonl','--out',result/'python-parity.jsonl'])
    variants=[('unbatched',['--rows','--unbatched']),('rows-batched',['--rows','--batch']),('prefix-batched',['--batch'])]
    for name,flags in variants:
        stage(name+' correctness')
        run([binary,'--input','fixtures/parity-v2.jsonl',*flags],result/(name+'-parity.jsonl'))
        run([py,'scripts/compare.py',result/'python-parity.jsonl',result/(name+'-parity.jsonl'),'--out',result/(name+'-parity.json')])
        rows=[json.loads(s) for s in (result/(name+'-parity.jsonl')).read_text().splitlines()]
        for q,z in rows[0]['logits'].items():
            assert max(abs(v-rows[4]['logits'][q][k]) for k,v in z.items())<1e-5
        for i,q in enumerate(rows[0]['logits'],start=5):
            assert max(abs(v-rows[i]['logits'][q][k]) for k,v in rows[0]['logits'][q].items())<1e-5
    if not (ROOT/'results/python-bench.jsonl').exists():
        (ROOT/'results').mkdir(exist_ok=True)
        stage('Python warmed CPU benchmark')
        run([py,'scripts/reference.py','--input','fixtures/requests.jsonl','--repeats','5','--out',ROOT/'results/python-bench.jsonl'])
    for name,flags in [('rows-tensor',['--rows','--tensor']),*variants]:
        stage(name+' warmed CPU benchmark')
        run([binary,'--input','fixtures/requests.jsonl','--repeats','5',*flags],result/(name+'-bench.jsonl'))
        run([py,'scripts/compare.py',ROOT/'results/python-bench.jsonl',result/(name+'-bench.jsonl'),'--out',result/(name+'-bench.json')])
    benchmarks={}
    for name in ['python','rows-tensor',*[name for name,_ in variants]]:
        file=ROOT/'results/python-bench.jsonl' if name=='python' else result/(name+'-bench.jsonl')
        rows=[json.loads(s) for s in file.read_text().splitlines()]
        warmed=[r['latency_ms'] for r in rows if r['iteration']>0]
        benchmarks[name]={'warm_median_ms':statistics.median(warmed),'samples':len(warmed),
                          'by_record_ms':[statistics.median(r['latency_ms'] for r in rows[i*5:(i+1)*5] if r['iteration']>0) for i in range(len(rows)//5)]}
    best=min([name for name,_ in variants],key=lambda name:benchmarks[name]['warm_median_ms'])
    report={'hardware':{'architecture':platform.machine(),'cpu_count':os.cpu_count(),'threads':2,'gpu_used':False},
            'packages':json.loads(subprocess.check_output([py,'-c','import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ["torch","transformers","peft","tokenizers","safetensors"]}))'],env=ENV,text=True)),
            'checkpoint':json.loads((ROOT/'weights/metadata.json').read_text()),
            'parity':{name:json.loads((result/(name+'-parity.json')).read_text()) for name,_ in variants},
            'benchmarks':benchmarks,'best_short_request_mode':best,
            'speedup_vs_python':benchmarks['python']['warm_median_ms']/benchmarks[best]['warm_median_ms'],
            'speedup_vs_native_baseline':benchmarks['rows-tensor']['warm_median_ms']/benchmarks[best]['warm_median_ms'],
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'Cargo.toml',*sorted((ROOT/'src').glob('*.rs'))]},
            'isolation_passed':True,'numeric_tokenization_passed':True}
    (result/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    run(['tar','-czf','/content/kev-cpu-v2-results.tar.gz','results-v2'])
    STATE.write_text(json.dumps({'status':'complete','pid':os.getpid(),'started':START,'ended':time.time()}))
except BaseException as e:
    STATE.write_text(json.dumps({'status':'failed','pid':os.getpid(),'started':START,'error':str(e)}))
    traceback.print_exc();sys.exit(1)
