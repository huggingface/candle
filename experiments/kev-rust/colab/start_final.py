import os
import subprocess
import sys
with open('/content/kev-cpu-final.log','w') as f:
    p=subprocess.Popen([sys.executable,'/content/kev_cpu_finalize.py'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                       env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
print({'cpu_final_validation_pid':p.pid})
