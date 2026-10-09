"""Check regenerated official metrics against the published reference."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
expected = json.loads((ROOT / 'results/reference/results.json').read_text())['metrics']
actual = json.loads((ROOT / 'artifacts/results.json').read_text())['metrics']
if actual != expected:
    raise SystemExit(f'Reproduction differs; investigate without tuning on test judgments. Expected {expected}, got {actual}')
print('All six official five-decimal metrics match the published reference.')
