"""CPU-only Colab job, launched through the official Colab CLI."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path('/content/candle')
STATE = Path('/content/kev-cpu-status.json')
ENV = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2',
       'RAYON_NUM_THREADS': '2', 'HF_HUB_DISABLE_TELEMETRY': '1'}
ENV['PATH'] = '/root/.cargo/bin:/root/.local/bin:' + ENV['PATH']
START = time.time()


def stage(name):
    STATE.write_text(json.dumps({'status':'running','stage':name,'pid':os.getpid(),'started':START,'updated':time.time()}))
    print(f'\nSTAGE: {name}', flush=True)


def run(cmd, cwd=None):
    print('+ ' + ' '.join(map(str,cmd)), flush=True)
    subprocess.run(list(map(str,cmd)), cwd=cwd, env=ENV, check=True)


def repo(url, commit, path, sparse=None):
    path.mkdir(parents=True, exist_ok=True)
    run(['git','init',path])
    run(['git','remote','add','origin',url],path)
    run(['git','fetch','--depth','1','--filter=blob:none','origin',commit],path)
    if sparse:
        run(['git','sparse-checkout','init','--cone'],path)
        run(['git','sparse-checkout','set',sparse],path)
    run(['git','checkout','--detach','FETCH_HEAD'],path)


try:
    stage('inspect CPU runtime')
    run(['lscpu'])
    run(['free','-h'])
    stage('fetch pinned source and unpack Rust experiment')
    repo('https://github.com/huggingface/candle.git','c68b24997319b8d76e5a4e775dab197c7f26322a',WORK)
    repo('https://github.com/jaredpalmer/kev.git','0a6d3fafeee7f3f76a9a4a522bf8b9d5b50bd7e5',WORK/'experiments/kev',sparse='kev')
    root=WORK/'experiments/kev-rust'
    root.mkdir(parents=True,exist_ok=True)
    run(['tar','-xzf','/content/kev-rust-colab.tar.gz','-C',root])
    stage('install Rust and CPU Python dependencies')
    run(['bash','-c','curl -LsSf https://astral.sh/uv/install.sh | sh'])
    run(['bash','-c','curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'])
    run(['uv','venv','--python','3.13','.venv'],root)
    py=root/'.venv/bin/python'
    run(['uv','pip','install','--python',py,'torch==2.8.0','--index-url','https://download.pytorch.org/whl/cpu'],root)
    run(['uv','pip','install','--python',py,'-r','requirements.txt'],root)
    stage('export pinned Kev-0.8B weights')
    run([py,'scripts/prepare.py'],root)
    stage('build, verify Python parity, and benchmark CPU inference')
    if os.environ.get('KEV_COLAB_EXPERIMENT') == '1':
        run([py,root/'colab/experiment.py'],root)
        (root/'results').mkdir(exist_ok=True)
        (root/'results/report.json').write_bytes((root/'results-v2/report.json').read_bytes())
    else:
        run([py,'scripts/validate.py','--repeats','5'],root)
    results=root/'results'
    with (results/'hardware.txt').open('w') as f:
        subprocess.run(['lscpu'],stdout=f,check=True)
    with (results/'rustc.txt').open('w') as f:
        subprocess.run(['rustc','--version'],stdout=f,env=ENV,check=True)
    dirs=['results']+(['results-v2'] if (root/'results-v2').exists() else [])
    run(['tar','-czf','/content/kev-cpu-results.tar.gz','-C',root,*dirs])
    STATE.write_text(json.dumps({'status':'complete','pid':os.getpid(),'started':START,'ended':time.time(),
                                'report':str(results/'report.json'),'artifact':'/content/kev-cpu-results.tar.gz'}))
    print('CPU JOB COMPLETE',flush=True)
except BaseException as e:
    STATE.write_text(json.dumps({'status':'failed','pid':os.getpid(),'started':START,'ended':time.time(),'error':str(e)}))
    traceback.print_exc()
    sys.exit(1)
