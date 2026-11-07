import json
from pathlib import Path
state=Path('/content/kev-cpu-final-status.json')
if state.exists():
    d=json.loads(state.read_text());d['process_present']=Path(f"/proc/{d['pid']}").exists()
    if d['process_present']:d['process_state']=Path(f"/proc/{d['pid']}/stat").read_text().split()[2]
    print(json.dumps(d))
log=Path('/content/kev-cpu-final.log')
if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-18:]))
