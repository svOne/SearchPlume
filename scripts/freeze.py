"""Freeze protocol prior to downloading/reading test judgments."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
protocol = ROOT / "protocol.json"
json.loads(protocol.read_text())
digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
target = ROOT / "artifacts/protocol-freeze.json"
if target.exists():
    assert json.loads(target.read_text())["sha256"] == digest, "Frozen protocol changed"
else:
    assert not (ROOT / "data/scifact/qrels/test.tsv").exists(), "Freeze before dataset extraction"
    target.write_text(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "sha256": digest,
        "path": "protocol.json", "test_judgments_read": False,
        "note": "Created before dataset download; retrieval and fusion decisions are fixed."}, indent=2) + "\n")
print(digest)
