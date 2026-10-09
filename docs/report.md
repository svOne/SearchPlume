# SearchPlume: keyword search, embeddings and fixed fusion

SearchPlume is a reproducible CPU retrieval case study on BEIR/SciFact, 2026-10-09. Fixed methods, local CPU inference, no test-judgment tuning. This is a practical comparison of established retrieval methods; it does not introduce a new hybrid algorithm.

| Method | nDCG@10 | Recall@100 |
|---|---:|---:|
| BM25S | 0.66169 | 0.87589 |
| MiniLM embeddings | 0.64508 | 0.92500 |
| Fixed RRF, k=60 | 0.69188 | 0.95500 |

Embedding-only changes nDCG@10 by -0.01661 and Recall@100 by +0.04911 versus this keyword baseline. Fixed fusion changes them by +0.03019 and +0.07911, respectively. These are absolute metric differences on a 0–1 scale.

The selected embeddings have lower nDCG@10 on this test. Complementary relevant-document retrieval, Recall@100 and the fixed hybrid result describe what they add.

## Complementarity and query variation

| Cutoff | Relevant pairs found by both | BM25 only | Embeddings only | Neither |
|---|---:|---:|---:|---:|
| 10 | 228 | 30 | 37 | 44 |
| 100 | 287 | 11 | 29 | 12 |

Counts refer to judged positive query/document pairs, not unique abstracts or all relevant scientific literature.

| Compared with BM25 | Metric | Query wins | Ties | Losses |
|---|---|---:|---:|---:|
| dense | ndcg_cut_10 | 67 | 160 | 73 |
| dense | recall_100 | 29 | 260 | 11 |
| rrf | ndcg_cut_10 | 67 | 198 | 35 |
| rrf | recall_100 | 27 | 273 | 0 |

These are descriptive paired results with numerical ties within 1e-12. No significance test or general advantage claim is made.

## Operational cost

| Method | Corpus index seconds | Judged-query median ms | Judged-query p95 ms | Fresh-worker peak RSS MiB | Worker elapsed seconds |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.482415 | 0.286542 | 0.413060 | 112.203 | 6.509626 |
| dense | 159.999363 | 6.981020 | 8.528298 | 710.672 | 182.253832 |
| rrf | 0.000000 | 1.201125 | 1.425002 | 50.969 | 22.668217 |

Hybrid serial component sum: median 8.490458 ms, p95 10.124906 ms. This sum is not an independently timed combined search service. Hybrid corpus index work is the sum of the BM25 and dense index times; a combined service's peak memory was not measured.

Dense model loading took 0.091647 seconds separately from corpus encoding. The 5,183 × 384 float32 normalized matrix occupies 7.592285 MiB.

Hardware: Apple M1 MacBook Pro, 8 GB RAM. Runtime: macOS-26.6.2-arm64-arm-64bit, CPython 3.11.7, one Torch compute/inter-op thread, BLAS/OMP/Accelerate thread limits of one, tokenizer parallelism disabled, CPU only. Actual native threadpool counts are asserted and retained. No MPS/GPU or paid inference. Every method runs in a fresh worker. Native ru_maxrss includes imports, model loading and ranking/output; macOS reports bytes, converted to MiB. Parent polling additionally samples RSS; its process-tree access/scope is recorded for each worker. An earlier sandboxed pilot could sample only workers. Workers launch no descendants.

Index time covers corpus tokenization plus BM25 sparse construction, or MiniLM corpus encoding plus matrix construction. It excludes model download/load and diagnostic token-length counting. Query timings include query tokenization/encoding, exact corpus scores and deterministic sorting, after one unjudged warmup; gzip output and RRF input-file parsing are excluded. Timings were recorded for all available query texts and then subset to the 300 judged IDs. They are one serial run on a shared machine with uncontrolled OS caches, clock frequency and other activity. Per-query latency arrays remain in the private audit archive; public evidence retains all-query summaries and the 300-query latency summaries.

## Fixed protocol and judgment isolation

BM25S 0.3.13: Lucene BM25/IDF, k1=1.5, b=0.75, float32 NumPy backend, lowercase, built-in English stopwords, regex `(?u)\b\w\w+\b`, no stemming. Dense: sentence-transformers 3.4.1, `sentence-transformers/all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 384 dimensions, max sequence length 256, normalized vectors, exact cosine via dot product, corpus batch size 16 and query batch size one, no prompt. Input is title + space + abstract; no chunking. Equal-weight RRF sums `1/(60+rank)` over both complete 5,183-document rankings. All score ties use descending string document ID, independently verified against pinned pytrec_eval.

The protocol was frozen at 2026-10-09T10:49:47.515402+00:00 with SHA256 `3b67238210276599574456d6ec29670f980f40361cf52b1344ea7bc2b8289fc0`, before download/extraction. Every available query was ranked before the first test-qrels parse at 2026-10-09T11:05:08.908055+00:00; the first retrieval-complete timestamp was 2026-10-09T11:05:08.707069+00:00. Corpus/query text and file-byte checksums were read before qrels parsing, but no judgments were semantically inspected. Synthetic controls preceded real evaluation. No train/dev judgments were parsed and no retrieval settings were changed in response to test results. Freeze events and source/output hashes provide local audit evidence, not independent external attestation.

## Validation and data provenance

Verified local counts: 5,183 documents, 300 judged test queries, 1109 available queries ranked, 339 judgment rows / 339 positive pairs. Each output has 5,747,947 rows with contiguous complete ranks, unique valid corpus IDs, finite scores, deterministic order and exact query coverage. Evaluation uses the top 100 scores, sufficient for both requested cutoffs; regenerated local runs retain every rank; full ranked outputs are not distributed in this repository.

Official `beir==2.2.0` `EvaluateRetrieval.evaluate` computes nDCG@10 and Recall@100 with pinned `pytrec-eval-terrier==0.5.10`. Every one of the 300 query IDs must return 100 retrieved documents and an evaluator score. Five-decimal official aggregates are independently reconciled with the stored unrounded per-query evaluator outputs. `ignore_identical_ids=False`: there are 0 judged query-ID strings also present as document-ID strings, but claims and abstracts use independent namespaces. Removing those document IDs would be an unjustified self-match assumption.

Synthetic controls test perfect retrieval, a total miss, relevant ranks 2/100/101, multiple relevant documents, graded relevance with trec_eval linear gain, macro averaging, identical-ID preservation and missing-query behavior. The runner adds explicit query coverage guards because BEIR otherwise averages only returned query scores. Metric control inputs/results and the installed evaluator SHA256 are in `../results/reference/metric-controls.json`. Descending string-ID tie ordering was independently checked against pinned pytrec_eval (`../results/reference/tie-control.json`); complete output validation checks every rank sequence.

Official archive MD5 `5f7d1de60b170fc8027bb7898e2efca1` matched the BEIR wiki. Archive SHA256: `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`. Every extracted file and model file has a SHA256/size manifest. All 56 dependencies are version/hash locked. Install plan/receipt remain in the private audit archive; public evidence retains the lock, measured environment and dependency budget. A bounded judgment-free CPU pilot passed before the full run; all native/monitored resource gates passed. Downloaded corpus/model/environment files and the full private audit archive are excluded from Git. The official downloader and attribution are included.

## Limits on interpretation

This is one small biomedical/scientific collection, one general-purpose embedding model and one untuned lexical configuration. The baseline excludes stemming; another fixed baseline or domain-specific model could change the outcome. Sparse judgments can mark useful unjudged abstracts as nonrelevant for evaluation. BEIR retrieval labels do not establish the truth of a scientific claim, and this split is distinct from SciFact’s original blind claim-verification test. Fixed full-depth fusion is the chosen experiment, not an optimized hybrid.

MiniLM uses a prefix of each document. Truncation includes special tokens:

| Input | Over 256 tokens | Total | Fraction | Maximum original tokens |
|---|---:|---:|---:|---:|
| documents | 3681 | 5183 | 0.710206 | 1939 |
| queries | 0 | 1109 | 0.000000 | 79 |

The model card lists S2ORC scientific abstracts in its training sources; SciFact abstracts also derive from S2ORC. No test tuning occurred here, but training-data independence is unverified. Costs and scores are specific to the recorded software/platform. Hash locking covers the tested CPython 3.11 macOS arm64 wheels; another platform needs a separately documented reproduction and cannot silently reuse this lock.

## Reproduce and inspect

From the repository root, `./run.sh` performs setup, bounded pilot, fixed retrieval, controls and evaluation. Original validated evidence is retained in `results/reference/`. A reproduction writes `artifacts/results.json`, `artifacts/per-query-metrics.tsv` and complete rankings in `artifacts/full/*.trec.tsv.gz`; those generated files are ignored by Git. Published results and this report are not overwritten. Read `README.md`, `SOURCES.md` and `ATTRIBUTION.md`.

Verified sources: [original BEIR paper](https://arxiv.org/abs/2104.08663), [BEIR datasets](https://github.com/beir-cellar/beir/wiki/Datasets-available), [official evaluator](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py), [BM25S](https://github.com/xhluca/bm25s), [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [SciFact paper](https://aclanthology.org/2020.emnlp-main.609/), [original component licenses](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/LICENSE.md).

This is a practical portfolio case study of established methods. It is not a competition result, peer review, a novel hybrid algorithm, or proof of general competitive advantage. See [AI disclosure](../AI_DISCLOSURE.md).
