import json
from pathlib import Path

state=Path('/content/kev-cpu-status.json')
if state.exists():
    data=json.loads(state.read_text())
    data['process_present']=Path(f"/proc/{data['pid']}").exists()
    if data['process_present']:
        data['process_state']=Path(f"/proc/{data['pid']}/stat").read_text().split()[2]
    print(json.dumps(data))
log=Path('/content/kev-cpu-job.log')
if log.exists():
    print('\n'.join(log.read_text(errors='replace').splitlines()[-18:]))
