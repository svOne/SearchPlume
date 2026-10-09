# SearchPlume

A reproducible CPU comparison of keyword search, embeddings and fixed reciprocal-rank fusion on **BEIR/SciFact**. The question is practical: what does a small embedding model add to an untuned keyword baseline on 5,183 scientific abstracts?

| Fixed method | nDCG@10 | Recall@100 |
|---|---:|---:|
| BM25S | 0.66169 | 0.87589 |
| all-MiniLM-L6-v2 | 0.64508 | 0.92500 |
| Equal-weight RRF, k=60 | 0.69188 | 0.95500 |

Embeddings improved coverage while lowering standalone nDCG@10. Fixed fusion improved both observed metrics in this comparison. At cutoff 100, embeddings found 29 judged positive query/document pairs that BM25 missed; BM25 found 11 that embeddings missed. These are descriptive results for this fixed setup, not statistical significance, novel retrieval research, a competition result or a general superiority claim.

Read the [case study](docs/report.md), [fixed protocol](protocol.json), [AI disclosure](AI_DISCLOSURE.md), and [attribution](ATTRIBUTION.md). Small original outputs are in [results/reference](results/reference).

## Reproduce

The tested environment is **CPython 3.11.7 on macOS arm64**, Apple M1 / 8 GB RAM. The wheel lock requires macOS 14 or later. Linux, Windows, other Python versions and GPUs were not tested; the runner stops on unsupported environments rather than claiming portability.

From this checkout:

```sh
./run.sh
```

Use `SEARCHPLUME_PYTHON=/absolute/path/to/python3.11` if needed. The runner creates a project `.venv`, installs 56 version/hash-locked binary wheels from PyPI, downloads the official checksummed SciFact archive and pinned public MiniLM model, runs a bounded CPU pilot, generates complete rankings, runs controls and official evaluation, then compares all six rounded metrics with the original reference. No account, API key, paid inference or administrator access is needed for the benchmark.

Downloads are about 179 MiB of dependency wheels plus 91 MB of model weights and the small dataset. Allow at least 3 GiB free disk; the project must remain below 2,000 MiB. The pilot uses 128 documents/eight available query texts, with per-worker limits of 120 seconds and 1,500 MiB peak RSS. Full workers have limits of 480 seconds and 2,000 MiB RSS. Original dense indexing took 160 seconds; the full dense worker peaked at about 711 MiB RSS. All workers use CPU only and one Torch/BLAS/OMP thread.

Published reference outputs and the case study are immutable inputs to reproduction. Fresh outputs go into ignored `artifacts/`; downloaded inputs go into ignored `data/` and `cache/`. A reproduction generates all 5,747,947 ranks per method locally in `artifacts/full/*.trec.tsv.gz`. Full rankings, downloaded data/model weights, environments, credentials and the original large audit archive are not included in Git.

After dependencies are installed, run the lightweight release checks:

```sh
./check.sh
```

These check the frozen protocol, exact published metric evidence, original core-source hashes, local documentation links and synthetic official-evaluator controls. They do not run a new retrieval experiment.

## Method and provenance

BM25S uses Lucene defaults `k1=1.5`, `b=0.75`, lowercase English stopwords and no stemming. MiniLM uses revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, normalized 384-dimensional vectors, a 256-token cap and exact cosine retrieval. Both use title plus abstract. RRF sums `1/(60+rank)` over both complete rankings with equal weights. Ties use descending string document ID. The official BEIR evaluator computes nDCG@10 and Recall@100 for all 300 judged retrieval queries.

The protocol was frozen before the original test judgments were parsed. All 1,109 available query texts were ranked first; train/dev judgments were never used. Its original experiment identifier is preserved byte-for-byte because it is part of the frozen SHA256. The public project name is **SearchPlume**. See [historical provenance](results/reference/provenance.json) for original source/output hashes and the two absolute native-library paths reduced to filenames in public evidence. No scores, metric values, timings, parameters or ranking hashes changed.

A reproduction of already published results is not a fresh blind evaluation. Historical timestamps describe the original run; new executions record their own timestamps in `artifacts/`. Local hashes are audit evidence, not independent external attestation. The public release has packaging and control checks; it does not claim an additional full run or cross-platform validation.

## Limits

- One small scientific/biomedical collection, one general-purpose encoder and one untuned lexical configuration.
- MiniLM truncated 3,681 of 5,183 documents (71.0%) to its prefix; BM25 indexed complete title/abstract text.
- Sparse judgments can treat useful unjudged documents as nonrelevant. Retrieval does not determine whether a scientific claim is true.
- MiniLM training sources include S2ORC, as does SciFact's corpus. Training-data independence is unverified.
- Latency is a single serial measurement on a shared M1 host. The 8.49 ms hybrid median is a sum of measured components, not a separately timed combined service.
- This is BEIR retrieval evaluation, distinct from SciFact's original blind claim-verification task.

Source code and project documentation are MIT licensed; third-party data, model and packages retain their own licenses. SciFact claims/annotations are CC BY 4.0, abstracts ODC-By 1.0 and original code Apache 2.0. This repository provides a downloader rather than redistributing the corpus. See [ATTRIBUTION.md](ATTRIBUTION.md) and [verified sources](SOURCES.md), including the [original BEIR paper](https://arxiv.org/abs/2104.08663).
