"""Build and package the final executable on a CPU Colab runtime."""
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

root = Path('/content/candle/experiments/kev-rust')
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'RAYON_NUM_THREADS': '2',
       'OMP_NUM_THREADS': '2'}
env['PATH'] = '/root/.cargo/bin:' + env['PATH']
subprocess.run(['tar', '-xzf', '/content/kev-binary-final-source.tar.gz',
                '-C', str(root)], check=True)
durations = {}
for command in [['cargo', 'test', '--release'], ['cargo', 'build', '--release']]:
    start = time.monotonic()
    subprocess.run(command, cwd=root, env=env, check=True)
    durations[' '.join(command)] = time.monotonic() - start
binary = root / 'target/release/kev-rust'
subprocess.run([str(binary), '--help'], env=env, check=True)
cuda_request = subprocess.run([str(binary), '--device', 'cuda'], env=env,
                              capture_output=True, text=True)
assert cuda_request.returncode != 0 and 'CUDA requires a build' in cuda_request.stderr
dist = root / 'dist'
dist.mkdir(exist_ok=True)
output = dist / 'kev-rust-linux-x86_64'
output.write_bytes(binary.read_bytes())
output.chmod(0o755)
digest = hashlib.sha256(output.read_bytes()).hexdigest()
(dist / 'SHA256SUMS').write_text(f'{digest}  {output.name}\n')
metadata = {
    'architecture': platform.machine(), 'gpu_used': False,
    'cpu_only_cuda_request_rejected': True,
    'build_seconds': durations,
    'rustc': subprocess.check_output(['rustc', '--version'], env=env, text=True).strip(),
    'dynamic_libraries': subprocess.check_output(['ldd', str(output)], text=True),
    'cpu': subprocess.check_output(['lscpu'], text=True),
    'sha256': digest,
    'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [root / 'Cargo.toml', root / 'build.rs', *sorted((root / 'src').glob('*.rs'))]},
}
(dist / 'build.json').write_text(json.dumps(metadata, indent=2) + '\n')
subprocess.run(['tar', '-czf', '/content/kev-rust-linux-x86_64.tar.gz',
                '-C', str(dist), output.name, 'SHA256SUMS', 'build.json'], check=True)
print(json.dumps(metadata), flush=True)
