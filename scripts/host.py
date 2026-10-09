"""Record non-identifying hardware fields for the actual reproduction host."""
import json
import platform
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
host={'architecture':platform.machine(),'platform':platform.platform()}
if platform.system()=='Darwin':
    proc=subprocess.run(['/usr/sbin/system_profiler','SPHardwareDataType'],capture_output=True,text=True,timeout=20)
    allowed={'Model Name','Model Identifier','Chip','Total Number of Cores','Memory'}
    for line in proc.stdout.splitlines():
        if ':' in line:
            k,v=line.strip().split(':',1)
            if k in allowed:
                host[k]=v.strip()
(ROOT/'artifacts/host.json').write_text(json.dumps(host,indent=2)+'\n')
print(json.dumps(host))
