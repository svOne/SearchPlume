"""Verify exact package versions; retain full installed distribution record."""
import importlib.metadata as md
import json
import platform
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for line in (ROOT / 'requirements.lock').read_text().splitlines():
    if line.startswith('#') or not line.strip():
        continue
    pinned = line.split()[0]
    name, version = pinned.split('==')
    assert md.version(name) == version, f'Version drift: {name}'
packages = {d.metadata['Name']: d.version for d in md.distributions()}
(ROOT / 'artifacts/environment.json').write_text(json.dumps({'python': platform.python_version(),
    'platform': platform.platform(), 'packages': dict(sorted(packages.items()))}, indent=2) + '\n')
print('Hash-locked package versions verified.')
