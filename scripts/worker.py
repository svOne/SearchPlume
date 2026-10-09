"""Fresh single-method CPU process. This file never opens qrels."""
import argparse
import gzip
import hashlib
import io
import json
import platform
import resource
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_texts(pilot):
    with (ROOT / 'data/scifact/corpus.jsonl').open() as f:
        corpus = {r['_id']: (r.get('title', '') + ' ' + r['text']).strip() for r in map(json.loads, f)}
    with (ROOT / 'data/scifact/queries.jsonl').open() as f:
        queries = {r['_id']: r['text'] for r in map(json.loads, f)}
    ids, qids = sorted(corpus), sorted(queries)
    assert len(ids) == 5183
    if pilot:
        ids, qids = ids[:128], qids[:8]
    return ids, [corpus[d] for d in ids], qids, [queries[q] for q in qids]

def rank(scores):
    import numpy as np
    assert np.isfinite(scores).all()
    return np.lexsort((-np.arange(len(scores)), -scores))

def stats(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p95': float(np.percentile(a, 95)), 'min': float(a.min()), 'max': float(a.max())}

def writer(path):
    # Stable gzip timestamp, no source filename, scores round-trip exactly.
    return io.TextIOWrapper(gzip.GzipFile(filename='', fileobj=path.open('wb'), mode='wb', mtime=0, compresslevel=3))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('method', choices=['bm25', 'dense', 'rrf'])
    ap.add_argument('--pilot', action='store_true')
    args = ap.parse_args()
    started = time.perf_counter()
    import numpy as np
    np.random.seed(42)
    p = json.loads((ROOT / 'protocol.json').read_text())
    frozen = json.loads((ROOT / 'artifacts/protocol-freeze.json').read_text())
    assert hashlib.sha256((ROOT / 'protocol.json').read_bytes()).hexdigest() == frozen['sha256']
    ids, texts, qids, queries = load_texts(args.pilot)
    outdir = ROOT / 'artifacts' / ('pilot' if args.pilot else 'full')
    outdir.mkdir(exist_ok=True)
    summary = {'method': args.method, 'pilot': args.pilot, 'documents': len(ids), 'queries': len(qids),
               'device': 'cpu', 'threads': 1, 'protocol_sha256': frozen['sha256']}
    tie_count = 0
    if args.method == 'bm25':
        import bm25s
        t = time.perf_counter()
        tokens = bm25s.tokenize(texts, lower=True, stopwords='english', stemmer=None,
                                token_pattern=p['bm25']['token_pattern'], show_progress=False)
        retriever = bm25s.BM25(method='lucene', idf_method='lucene', k1=1.5, b=0.75,
                              dtype='float32', backend='numpy')
        retriever.index(tokens, show_progress=False)
        summary['index_seconds'] = time.perf_counter() - t
        summary['index_nonzero_entries'] = int(retriever.scores['data'].size)
        summary['stopwords_sha256'] = hashlib.sha256(json.dumps(sorted(bm25s.tokenization.STOPWORDS_EN), ensure_ascii=False).encode()).hexdigest() if hasattr(bm25s.tokenization, 'STOPWORDS_EN') else None
        def search(query):
            tok = bm25s.tokenize([query], lower=True, stopwords='english', stemmer=None,
                token_pattern=p['bm25']['token_pattern'], return_ids=False, show_progress=False)[0]
            return retriever.get_scores(tok) if tok else np.zeros(len(ids), dtype=np.float32)
    elif args.method == 'dense':
        import torch
        torch.manual_seed(42)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        from sentence_transformers import SentenceTransformer
        t = time.perf_counter()
        model = SentenceTransformer(str(ROOT / 'cache/model'), device='cpu', trust_remote_code=False)
        model.max_seq_length = 256
        assert model.get_sentence_embedding_dimension() == 384
        assert str(model.device) == 'cpu'
        summary['model_load_seconds'] = time.perf_counter() - t
        summary['torch_threads'] = torch.get_num_threads()
        summary['torch_interop_threads'] = torch.get_num_interop_threads()
        summary['max_seq_length'] = model.max_seq_length
        def lengths(seq):
            values = []
            for i in range(0, len(seq), 64):
                data = model.tokenizer(seq[i:i+64], truncation=False, padding=False, add_special_tokens=True)
                values.extend(map(len, data['input_ids']))
            return {'tokens': stats(values), 'truncated_count': sum(v > 256 for v in values),
                    'truncated_fraction': sum(v > 256 for v in values) / len(values)}
        summary['truncation'] = {'documents': lengths(texts), 'queries': lengths(queries)}
        t = time.perf_counter()
        embeddings = model.encode(texts, batch_size=16, show_progress_bar=False, normalize_embeddings=True,
                                  convert_to_numpy=True, device='cpu').astype(np.float32)
        summary['index_seconds'] = time.perf_counter() - t
        summary['embedding_shape'] = list(embeddings.shape)
        summary['embedding_bytes'] = embeddings.nbytes
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
        def search(query):
            vector = model.encode([query], batch_size=1, show_progress_bar=False, normalize_embeddings=True,
                                  convert_to_numpy=True, device='cpu')[0]
            return embeddings @ vector
    else:
        # Files were created without judgments. Fusion consumes complete base rankings.
        from contextlib import ExitStack
        warm_ranks = np.arange(1, len(ids) + 1, dtype=float)
        rank(1 / (60 + warm_ranks) + 1 / (60 + warm_ranks[::-1]))
        with ExitStack() as stack:
            readers = [stack.enter_context(gzip.open(outdir / f'{m}.trec.tsv.gz', 'rt')) for m in ['bm25', 'dense']]
            latencies = []
            with writer(outdir / 'rrf.trec.tsv.gz') as out:
                for qid in qids:
                    incoming = []
                    for reader in readers:
                        ranks = {}
                        for expected_rank in range(1, len(ids) + 1):
                            q, q0, did, r, score, tag = reader.readline().split()
                            assert q == qid
                            assert q0 == 'Q0' and int(r) == expected_rank
                            assert did not in ranks and np.isfinite(float(score))
                            ranks[did] = int(r)
                        assert set(ranks) == set(ids)
                        incoming.append(ranks)
                    t = time.perf_counter()
                    scores = np.array([1 / (60 + incoming[0][d]) + 1 / (60 + incoming[1][d]) for d in ids])
                    order = rank(scores)
                    latencies.append(time.perf_counter() - t)
                    tie_count += int(len(scores) - np.unique(scores).size)
                    for r, i in enumerate(order, 1):
                        out.write(f'{qid}\tQ0\t{ids[i]}\t{r}\t{float(scores[i]):.17g}\trrf\n')
            assert all(not reader.readline() for reader in readers)
        summary['index_seconds'] = 0.0
        summary['query_latency_seconds'] = stats(latencies)
        summary['query_latency_note'] = 'Fusion computation/ranking only; base ranking file parsing excluded.'
        save(outdir, summary, latencies, started, tie_count)
        return
    # One fixed warmup text; no query judgments or selection involved.
    rank(search('scientific evidence'))
    latencies = []
    with writer(outdir / f'{args.method}.trec.tsv.gz') as out:
        for n, (qid, query) in enumerate(zip(qids, queries)):
            t = time.perf_counter()
            scores = search(query)
            order = rank(scores)
            latencies.append(time.perf_counter() - t)
            tie_count += int(len(scores) - np.unique(scores).size)
            for r, i in enumerate(order, 1):
                out.write(f'{qid}\tQ0\t{ids[i]}\t{r}\t{float(scores[i]):.17g}\t{args.method}\n')
            if (n + 1) % 200 == 0:
                print(f'{args.method}: {n+1}/{len(qids)} queries ranked', flush=True)
    summary['query_latency_seconds'] = stats(latencies)
    save(outdir, summary, latencies, started, tie_count)

def save(outdir, summary, latencies, started, tie_count):
    from threadpoolctl import threadpool_info
    summary['native_threadpools'] = threadpool_info()
    assert all(pool['num_threads'] == 1 for pool in summary['native_threadpools']), 'Native thread limit not honored'
    summary['worker_wall_seconds'] = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary['peak_rss_mib'] = rss / (1024 * 1024 if platform.system() == 'Darwin' else 1024)
    summary['score_ties_beyond_first'] = tie_count
    (outdir / f"{summary['method']}.stats.json").write_text(json.dumps(summary, indent=2) + '\n')
    (outdir / f"{summary['method']}.latencies.json").write_text(json.dumps(latencies) + '\n')
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
