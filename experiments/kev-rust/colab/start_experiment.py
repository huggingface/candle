import os
import subprocess
import sys
with open('/content/kev-cpu-experiment.log','w') as f:
    p=subprocess.Popen([sys.executable,'/content/kev_cpu_experiment.py'],stdout=f,stderr=subprocess.STDOUT,
                       start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
print({'cpu_experiment_pid':p.pid})
