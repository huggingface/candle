import json
import os
import subprocess
import sys
from pathlib import Path

state=Path('/content/kev-cpu-status.json')
if state.exists():
    prior=json.loads(state.read_text())
    if prior['status']=='running' and Path(f"/proc/{prior['pid']}").exists():
        raise RuntimeError('The existing CPU job is still running; do not launch twice.')
with open('/content/kev-cpu-job.log','w') as log:
    p=subprocess.Popen([sys.executable,'/content/kev_cpu_job.py'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                       env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
print(json.dumps({'launched_pid':p.pid,'log':'/content/kev-cpu-job.log','gpu_requested':False}))
