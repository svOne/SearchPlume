"""Bounded CPU orchestration; all rankings complete before judgments are parsed."""
import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['bm25', 'dense', 'rrf']

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def gate_disk():
    size = sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    assert size < 2000 * 1024**2, f'Project disk gate exceeded: {size / 1024**2:.1f} MiB'
    assert shutil.disk_usage(ROOT).free > 3 * 1024**3, 'Less than 3 GiB free disk'
    return size

def worker(method, pilot):
    import psutil
    path = ROOT / 'artifacts' / ('pilot' if pilot else 'full')
    path.mkdir(exist_ok=True)
    cmd = [sys.executable, str(ROOT / 'scripts/worker.py'), method] + (['--pilot'] if pilot else [])
    started = time.monotonic()
    max_rss = 0
    tree_sampling_available = True
    with (path / f'{method}.log').open('w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            handle = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    rss = handle.memory_info().rss
                    if tree_sampling_available:
                        try:
                            rss += sum(c.memory_info().rss for c in handle.children(recursive=True) if c.is_running())
                        except (PermissionError, psutil.AccessDenied):
                            # macOS sandbox denies global process enumeration; workers launch no child processes.
                            tree_sampling_available = False
                    max_rss = max(max_rss, rss)
                except psutil.NoSuchProcess:
                    pass
                limit_s, limit_mib = (120, 1500) if pilot else (480, 2000)
                if time.monotonic() - started > limit_s or max_rss > limit_mib * 1024**2:
                    raise RuntimeError(f'{method} resource gate exceeded: elapsed={time.monotonic()-started:.1f}s, RSS={max_rss/1024**2:.1f} MiB')
                time.sleep(0.25)
            assert proc.returncode == 0, f'{method} failed; see {path / (method + ".log")}'
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    stats = json.loads((path / f'{method}.stats.json').read_text())
    assert stats['peak_rss_mib'] < (1500 if pilot else 2000), 'Native peak RSS gate exceeded'
    assert stats['worker_wall_seconds'] < (120 if pilot else 480), 'Native wall-time gate exceeded'
    stats['parent_sampled_peak_rss_mib'] = max_rss / 1024**2
    stats['process_tree_sampling_available'] = tree_sampling_available
    stats['sampling_scope'] = 'worker and descendants' if tree_sampling_available else 'worker only; workers launch no child processes'
    stats['parent_observed_seconds'] = time.monotonic() - started
    dump(path / f'{method}.stats.json', stats)
    gate_disk()
    print(f"{method} {'pilot' if pilot else 'full'}: {stats['worker_wall_seconds']:.2f}s, peak {stats['peak_rss_mib']:.1f} MiB", flush=True)
    return stats

def all_rankings(pilot_only=False):
    from prepare import check_freeze
    p = check_freeze()
    env = {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS', 'TOKENIZERS_PARALLELISM', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']}
    assert all(env[k] == '1' for k in list(env)[:4]), 'Use run.sh for thread settings'
    assert env['TOKENIZERS_PARALLELISM'] == 'false'
    dump(ROOT / 'artifacts/runtime.json', {'utc': datetime.now(timezone.utc).isoformat(),
       'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version(),
       'logical_cpus': os.cpu_count(), 'device': 'cpu', 'environment': env,
       'seed': 42, 'project_disk_before_bytes': gate_disk(),
       'note': 'Shared machine; CPU frequency, OS cache, and other activity uncontrolled.'})
    for method in METHODS:
        s = worker(method, True)
        assert s['peak_rss_mib'] < 1500
        assert s['worker_wall_seconds'] < 120
    dump(ROOT / 'artifacts/pilot-gate.json', {'passed': True, 'utc': datetime.now(timezone.utc).isoformat(),
         'documents': 128, 'queries': 8, 'judgments_read': False})
    if pilot_only:
        return
    for method in METHODS:
        worker(method, False)
    files = {str(f.relative_to(ROOT)): {'sha256': sha(f), 'bytes': f.stat().st_size}
              for f in sorted((ROOT / 'artifacts/full').glob('*.trec.tsv.gz'))}
    dump(ROOT / 'artifacts/retrieval-complete.json', {'utc': datetime.now(timezone.utc).isoformat(),
       'protocol_sha256': sha(ROOT / 'protocol.json'), 'files': files,
       'test_judgments_previously_read': (ROOT / 'artifacts/first-qrels-read.json').exists(),
       'retrieval_code': {str(f.relative_to(ROOT)): sha(f) for f in (ROOT / 'scripts').glob('*.py')}})
    print('All fixed rankings saved and hashed; retrieval workers never parse qrels.', flush=True)

def load_rankings(path, judged_ids=None):
    corpus_ids = set()
    with (ROOT / 'data/scifact/corpus.jsonl').open() as f:
        for r in map(json.loads, f):
            corpus_ids.add(r['_id'])
    runs, rows, queries = {}, 0, set()
    last_query, expected_rank, seen_docs, previous = None, 0, set(), None
    with gzip.open(path, 'rt') as f:
        for line in f:
            qid, q0, did, rankstr, scorestr, tag = line.split()
            rank, score = int(rankstr), float(scorestr)
            if qid != last_query:
                if last_query is not None:
                    assert seen_docs == corpus_ids, f'Incomplete ranking: {last_query}'
                assert qid not in queries, 'Non-contiguous/duplicate query'
                queries.add(qid)
                last_query, expected_rank, seen_docs, previous = qid, 0, set(), None
            expected_rank += 1
            assert rank == expected_rank
            assert did in corpus_ids and did not in seen_docs
            assert q0 == 'Q0'
            assert score == score and abs(score) != float('inf')
            if previous is not None:
                assert score < previous[0] or (score == previous[0] and did < previous[1]), 'Invalid score/tie order'
            previous = (score, did)
            seen_docs.add(did)
            rows += 1
            if judged_ids is not None and qid in judged_ids and rank <= 100:
                runs.setdefault(qid, {})[did] = score
    assert seen_docs == corpus_ids
    with (ROOT / 'data/scifact/queries.jsonl').open() as f:
        query_ids = {r['_id'] for r in map(json.loads, f)}
    assert queries == query_ids
    assert rows == len(queries) * len(corpus_ids)
    return runs, {'rows': rows, 'queries': len(queries), 'documents_per_query': len(corpus_ids), 'sha256': sha(path)}

def evaluate():
    from prepare import check_freeze
    from metric_controls import run_controls
    p = check_freeze()
    completed = json.loads((ROOT / 'artifacts/retrieval-complete.json').read_text())
    for rel, info in completed['files'].items():
        assert sha(ROOT / rel) == info['sha256'], 'Ranking changed after completion'
    for rel, digest in completed['retrieval_code'].items():
        assert sha(ROOT / rel) == digest, 'Code changed between retrieval and evaluation'
    controls = run_controls()
    dump(ROOT / 'artifacts/metric-controls.json', controls)
    # First semantic read of any SciFact judgment: all retrieval decisions/results are fixed.
    if not (ROOT / 'artifacts/first-qrels-read.json').exists():
        dump(ROOT / 'artifacts/first-qrels-read.json', {'utc': datetime.now(timezone.utc).isoformat(),
           'protocol_sha256': sha(ROOT / 'protocol.json'), 'retrieval_complete_utc': completed['utc'],
           'note': 'First parsing of qrels/test.tsv follows complete rankings and synthetic metric controls.'})
    qrels = {}
    with (ROOT / 'data/scifact/qrels/test.tsv').open() as f:
        for r in csv.DictReader(f, delimiter='\t'):
            assert r['corpus-id'] not in qrels.get(r['query-id'], {}), 'Duplicate judgment pair'
            qrels.setdefault(r['query-id'], {})[r['corpus-id']] = int(r['score'])
    assert len(qrels) == 300
    assert all(any(v > 0 for v in docs.values()) for docs in qrels.values())
    import numpy as np
    import pytrec_eval
    from beir.retrieval.evaluation import EvaluateRetrieval
    judged_ids = set(qrels)
    scores, per_query, validation, runs = {}, {}, {}, {}
    for method in METHODS:
        runs[method], validation[method] = load_rankings(ROOT / f'artifacts/full/{method}.trec.tsv.gz', judged_ids)
        assert set(runs[method]) == judged_ids and all(len(v) == 100 for v in runs[method].values())
        ndcg, _, recall, _ = EvaluateRetrieval.evaluate(qrels, {q: dict(v) for q, v in runs[method].items()}, [10, 100], ignore_identical_ids=False)
        scores[method] = {'NDCG@10': ndcg['NDCG@10'], 'Recall@100': recall['Recall@100']}
        per_query[method] = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10', 'recall.100'}).evaluate(runs[method])
        assert set(per_query[method]) == judged_ids
        for metric, trec_key in [('NDCG@10', 'ndcg_cut_10'), ('Recall@100', 'recall_100')]:
            assert round(float(np.mean([v[trec_key] for v in per_query[method].values()])), 5) == scores[method][metric]
    with (ROOT / 'data/scifact/corpus.jsonl').open() as f:
        docids = {r['_id'] for r in map(json.loads, f)}
    assert all(set(v).issubset(docids) for v in qrels.values())
    paired = {}
    for method in ['dense', 'rrf']:
        paired[method] = {}
        for key in ['ndcg_cut_10', 'recall_100']:
            deltas = np.array([per_query[method][q][key] - per_query['bm25'][q][key] for q in sorted(qrels)])
            paired[method][key] = {'unrounded_mean_delta': float(deltas.mean()),
                'wins': int((deltas > 1e-12).sum()), 'ties': int((abs(deltas) <= 1e-12).sum()),
                'losses': int((deltas < -1e-12).sum())}
    complement = {}
    for depth in [10, 100]:
        counts = {'bm25_only': 0, 'dense_only': 0, 'both': 0, 'neither': 0, 'judged_relevant_pairs': 0}
        for q, rels in qrels.items():
            tops = {m: set(list(runs[m][q])[:depth]) for m in ['bm25', 'dense']}
            for d, rel in rels.items():
                if rel <= 0:
                    continue
                a, b = d in tops['bm25'], d in tops['dense']
                counts['both' if a and b else 'bm25_only' if a else 'dense_only' if b else 'neither'] += 1
                counts['judged_relevant_pairs'] += 1
        complement[str(depth)] = counts
    collisions = {q: q in docids for q in qrels if q in docids}
    selected_latency = {}
    all_qids = sorted({r['_id'] for r in map(json.loads, (ROOT / 'data/scifact/queries.jsonl').open())})
    for m in METHODS:
        values = json.loads((ROOT / f'artifacts/full/{m}.latencies.json').read_text())
        assert len(values) == len(all_qids)
        selected_latency[m] = [v for q, v in zip(all_qids, values) if q in judged_ids]
    from worker import stats
    costs = {m: json.loads((ROOT / f'artifacts/full/{m}.stats.json').read_text()) for m in METHODS}
    for m in METHODS:
        costs[m]['judged_query_latency_seconds'] = stats(selected_latency[m])
    costs['hybrid'] = {'judged_query_latency_seconds': stats(np.sum([selected_latency[m] for m in METHODS], axis=0)),
       'note': 'Sum of serial measured components for each query; not an independently timed combined service.'}
    result = {'dataset': {'documents': 5183, 'judged_queries': 300, 'all_queries_ranked': len(all_qids),
        'qrels_rows': sum(len(v) for v in qrels.values()), 'positive_judgments': sum(v > 0 for rels in qrels.values() for v in rels.values()),
        'query_document_string_id_collisions': len(collisions)},
        'metrics': scores, 'paired_vs_bm25': paired, 'relevant_pair_complementarity': complement,
        'costs': costs, 'validation': validation, 'controls_passed': True,
        'protocol_sha256': sha(ROOT / 'protocol.json'), 'utc': datetime.now(timezone.utc).isoformat()}
    dump(ROOT / 'artifacts/results.json', result)
    dump(ROOT / 'artifacts/per-query-metrics.json', per_query)
    dump(ROOT / 'artifacts/ranking-validation.json', validation)
    with (ROOT / 'artifacts/per-query-metrics.tsv').open('w') as f:
        f.write('query_id\tmethod\tndcg_at_10\trecall_at_100\n')
        for q in sorted(qrels):
            for m in METHODS:
                v = per_query[m][q]
                f.write(f"{q}\t{m}\t{v['ndcg_cut_10']:.17g}\t{v['recall_100']:.17g}\n")
    print(json.dumps(scores, indent=2), flush=True)
    print('Evaluation and full-ranking validation passed.', flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--evaluate-only', action='store_true')
    ap.add_argument('--pilot-only', action='store_true')
    args = ap.parse_args()
    if not args.evaluate_only:
        all_rankings(args.pilot_only)
    if not args.pilot_only:
        evaluate()
