"""Download checksummed BEIR data and the immutable model without reading qrels."""
import argparse
import hashlib
import json
import os
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path, kind='sha256'):
    h = hashlib.new(kind)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def check_freeze():
    frozen = json.loads((ROOT / 'artifacts/protocol-freeze.json').read_text())
    assert frozen['sha256'] == digest(ROOT / 'protocol.json'), 'Protocol changed after freeze'
    return json.loads((ROOT / 'protocol.json').read_text())

def prepare():
    p = check_freeze()
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    archive = data / 'scifact.zip'
    if not archive.exists():
        temp = archive.with_suffix('.tmp')
        with urllib.request.urlopen(p['dataset']['url'], timeout=60) as src, temp.open('wb') as out:
            total = 0
            while block := src.read(1024 * 1024):
                total += len(block)
                assert total < 50 * 1024 * 1024, 'Unexpected archive size'
                out.write(block)
        temp.rename(archive)
    assert digest(archive, 'md5') == p['dataset']['archive_md5'], 'BEIR archive MD5 mismatch'
    with zipfile.ZipFile(archive) as z:
        assert sum(i.file_size for i in z.infolist()) < 100 * 1024 * 1024
        for info in z.infolist():
            target = (data / info.filename).resolve()
            assert target.is_relative_to(data.resolve()), 'Unsafe zip path'
        z.extractall(data)
    # Hash file bytes only; do not parse or expose judgment contents here.
    hashes = {str(f.relative_to(ROOT)): {'sha256': digest(f), 'bytes': f.stat().st_size}
              for f in sorted((data / 'scifact').rglob('*')) if f.is_file()}
    hashes['data/scifact.zip'] = {'sha256': digest(archive), 'md5': digest(archive, 'md5'), 'bytes': archive.stat().st_size}
    (ROOT / 'artifacts/dataset-manifest.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
        'url': p['dataset']['url'], 'hashes': hashes, 'judgments_parsed': False}, indent=2) + '\n')
    from huggingface_hub import snapshot_download
    model_dir = ROOT / 'cache/model'
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=p['dense']['model'], revision=p['dense']['revision'],
        local_dir=str(model_dir), token=False, max_workers=2,
        allow_patterns=['*.json', '1_Pooling/*.json', 'model.safetensors', 'vocab.txt', 'README.md'])
    files = {str(f.relative_to(model_dir)): {'sha256': digest(f), 'bytes': f.stat().st_size}
             for f in sorted(model_dir.rglob('*')) if f.is_file() and '.cache' not in f.parts}
    assert files['model.safetensors']['sha256'] == '53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db', 'Model weight checksum mismatch'
    assert sum(f['bytes'] for f in files.values()) < 150 * 1024 * 1024, 'Unexpected model footprint'
    (ROOT / 'artifacts/model-manifest.json').write_text(json.dumps({'model': p['dense']['model'],
        'revision': p['dense']['revision'], 'files': files}, indent=2) + '\n')
    print('Dataset and pinned model prepared; no qrels parsed.', flush=True)

if __name__ == '__main__':
    os.environ['HF_HOME'] = str(ROOT / 'cache/huggingface')
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    prepare()
