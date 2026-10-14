"""Build, validate and package the CUDA executable in an existing Colab GPU VM.

Uses only its own experiment directory; never restarts or stops the shared VM.
"""
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path('/content/candle/experiments/kev-rust-gpu')
STATE = Path('/content/kev-gpu-status.json')
ENV = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'CUDA_COMPUTE_CAP': '75',
       'RAYON_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2', 'CARGO_BUILD_JOBS': '2',
       'CARGO_TARGET_DIR': '/content/candle/target',
       'HF_HUB_DISABLE_TELEMETRY': '1'}
ENV['PATH'] = '/root/.cargo/bin:/root/.local/bin:/usr/local/cuda/bin:' + ENV['PATH']
ENV['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + ENV.get('LD_LIBRARY_PATH', '')
START = time.time()
timings = {}


def stage(name):
    STATE.write_text(json.dumps({'status': 'running', 'stage': name,
                                'pid': os.getpid(), 'started': START}))
    print('STAGE:', name, flush=True)


def run(cmd, output=None, cwd=ROOT):
    print('+', ' '.join(map(str, cmd)), flush=True)
    start = time.monotonic()
    if output:
        with output.open('w') as f:
            subprocess.run(list(map(str, cmd)), cwd=cwd, env=ENV, stdout=f, check=True)
    else:
        subprocess.run(list(map(str, cmd)), cwd=cwd, env=ENV, check=True)
    timings[' '.join(map(str, cmd))] = time.monotonic() - start


try:
    stage('check pinned Candle source and build tools')
    candle = ROOT.parents[1]
    pin = 'c68b24997319b8d76e5a4e775dab197c7f26322a'
    if not (candle / '.git').exists():
        candle.mkdir(parents=True, exist_ok=True)
        run(['git', 'init', candle], cwd=Path('/content'))
        run(['git', 'remote', 'add', 'origin', 'https://github.com/huggingface/candle.git'], cwd=candle)
        run(['git', 'fetch', '--depth', '1', '--filter=blob:none', 'origin', pin], cwd=candle)
        run(['git', 'checkout', '--detach', 'FETCH_HEAD'], cwd=candle)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=candle, text=True).strip()
    assert actual == pin, f'existing Candle checkout has a different revision: {actual}'
    if not shutil.which('cargo', path=ENV['PATH']):
        run(['bash', '-c', 'curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'], cwd=candle)
    if not shutil.which('uv', path=ENV['PATH']):
        run(['bash', '-c', 'curl -LsSf https://astral.sh/uv/install.sh | sh'], cwd=candle)
    run(['rustup', 'component', 'add', 'rustfmt'], cwd=candle)
    stage('unpack source and verify CUDA runtime')
    ROOT.mkdir(parents=True, exist_ok=True)
    resume = os.environ.get('KEV_GPU_RESUME') == '1'
    if not resume:
        run(['tar', '-xzf', '/content/kev-gpu-source.tar.gz', '-C', ROOT])
    result = ROOT / 'results-gpu'
    result.mkdir(exist_ok=True)
    run(['nvidia-smi'], result / 'nvidia-smi.txt')
    run(['nvcc', '--version'], result / 'nvcc.txt')
    run(['rustc', '--version'], result / 'rustc.txt')
    stage('build CUDA kernels and run CPU/CUDA unit tests')
    if not resume:
        run(['cargo', 'fmt'])
        run(['cargo', 'test', '--release', '--features', 'cuda'])
        run(['cargo', 'build', '--release', '--features', 'cuda'])
    binary = Path(ENV['CARGO_TARGET_DIR']) / 'release/kev-rust'
    stage('prepare pinned fp32 weights')
    if not (ROOT / '.venv/bin/python').exists():
        run(['uv', 'venv', '--python', '3.13', '.venv'])
    py = ROOT / '.venv/bin/python'
    run(['uv', 'pip', 'install', '--python', py, 'torch==2.8.0',
         '--index-url', 'https://download.pytorch.org/whl/cpu'])
    run(['uv', 'pip', 'install', '--python', py, 'safetensors==0.8.0', 'huggingface-hub==2.2.0', 'numpy==2.5.4'])
    if not all((ROOT / 'weights' / p).exists() for p in
               ['model.safetensors', 'head.safetensors', 'config.json', 'tokenizer.json', 'metadata.json']):
        run([py, 'scripts/prepare.py'])
    stage('validate CUDA decisions against official Python fp32 reference')
    parity = {}
    for name, flags in [('cuda-prefix', []), ('cuda-rows', ['--rows']),
                        ('cuda-unbatched', ['--unbatched'])]:
        run([binary, '--device', 'cuda', '--input', 'fixtures/parity-v2.jsonl', *flags], result / (name + '.jsonl'))
        run([py, 'scripts/compare.py', 'reference-cpu.jsonl', result / (name + '.jsonl'),
             '--out', result / (name + '.json')])
        parity[name] = json.loads((result / (name + '.json')).read_text())
        rows = [json.loads(s) for s in (result / (name + '.jsonl')).read_text().splitlines()]
        assert all(r['device'] == 'cuda' and r['dtype'] == 'float32' for r in rows)
        for q, z in rows[0]['logits'].items():
            assert max(abs(v - rows[4]['logits'][q][k]) for k, v in z.items()) < 1e-4
        for i, q in enumerate(rows[0]['logits'], start=5):
            assert max(abs(v - rows[i]['logits'][q][k]) for k, v in rows[0]['logits'][q].items()) < 1e-4
    stage('benchmark GPU and CPU on the same VM')
    benchmarks = {}
    for device in ['cuda', 'cpu']:
        file = result / (device + '-bench.jsonl')
        run([binary, '--device', device, '--input', 'fixtures/requests.jsonl', '--repeats', '5'], file)
        rows = [json.loads(s) for s in file.read_text().splitlines()]
        warm = [r['latency_ms'] for r in rows if r['iteration'] > 0]
        benchmarks[device] = {'warm_median_ms': statistics.median(warm), 'samples': len(warm)}
    stage('package binary and verification artifacts')
    dist = ROOT / 'dist'
    dist.mkdir(exist_ok=True)
    output = dist / 'kev-rust-linux-x86_64-cuda'
    output.write_bytes(binary.read_bytes())
    output.chmod(0o755)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    (dist / 'SHA256SUMS').write_text(f'{digest}  {output.name}\n')
    sources = [ROOT / 'Cargo.toml', ROOT / 'build.rs', *sorted((ROOT / 'src').glob('*.rs')),
               ROOT / 'cuda/kev.cu']
    report = {'hardware': {'gpu': 'NVIDIA Tesla T4', 'architecture': platform.machine(),
                           'gpu_used': True, 'cpu_count': os.cpu_count(),
                           'cpu_details': subprocess.check_output(['lscpu'], text=True)},
              'cpu_compilation_target': 'native (repository .cargo/config.toml)',
              'cuda_toolkit': (result / 'nvcc.txt').read_text(),
              'dtype': 'float32', 'tf32': False,
              'checkpoint': json.loads((ROOT / 'weights/metadata.json').read_text()),
              'parity': parity, 'benchmarks': benchmarks,
              'cpu_to_gpu_speedup': benchmarks['cpu']['warm_median_ms'] / benchmarks['cuda']['warm_median_ms'],
              'timings_seconds': timings, 'binary_sha256': digest,
              'reference_sha256': hashlib.sha256((ROOT / 'reference-cpu.jsonl').read_bytes()).hexdigest(),
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'dynamic_libraries': subprocess.check_output(['ldd', output], env=ENV, text=True),
              'question_isolation_passed': True}
    (result / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (dist / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    run(['tar', '-czf', '/content/kev-rust-linux-x86_64-cuda.tar.gz', '-C', dist,
         output.name, 'SHA256SUMS', 'build.json'])
    run(['tar', '-czf', '/content/kev-gpu-results.tar.gz', 'results-gpu', 'src', 'cuda',
         'Cargo.toml', 'Cargo.lock', 'build.rs'])
    STATE.write_text(json.dumps({'status': 'complete', 'pid': os.getpid(),
                                'started': START, 'ended': time.time()}))
    print('GPU JOB COMPLETE', flush=True)
except BaseException as e:
    STATE.write_text(json.dumps({'status': 'failed', 'pid': os.getpid(), 'error': str(e)}))
    traceback.print_exc()
    raise
