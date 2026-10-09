# SearchPlume attribution and data handling

This SearchPlume portfolio case study uses BEIR/SciFact for document retrieval. This repository contains code, protocol and small metric/provenance outputs. Full rankings are regenerated locally. It excludes the SciFact archive, abstracts, claims, annotations, model weights, and virtual environment. `scripts/prepare.py` downloads the data from the official BEIR mirror and verifies its published checksum.

SciFact: David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. **Fact or Fiction: Verifying Scientific Claims**. EMNLP 2020. [Paper](https://aclanthology.org/2020.emnlp-main.609/), [repository](https://github.com/allenai/scifact), [original license](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/LICENSE.md). Claims and evidence annotations: **CC BY 4.0**. Abstracts: **ODC-By 1.0**. Repository code: **Apache 2.0**. No modifications are made to the downloaded data files; retrieval combines title and abstract for model input.

Abstract source: Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, and Daniel S. Weld. **S2ORC: The Semantic Scholar Open Research Corpus**. ACL 2020. [Paper](https://aclanthology.org/2020.acl-main.447/), [repository](https://github.com/allenai/s2orc). Preserve attribution to the original scientific works when using the abstracts.

BEIR: Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models**. NeurIPS 2021 Datasets and Benchmarks. [Original paper](https://arxiv.org/abs/2104.08663), [conference paper](https://openreview.net/forum?id=wCu6T5xFjeJ), [repository and evaluator](https://github.com/beir-cellar/beir), [dataset distribution](https://github.com/beir-cellar/beir/wiki/Datasets-available). BEIR code is Apache 2.0; the original data licenses apply to SciFact components.

BM25S: Xing Han Lù. **BM25S: Orders of magnitude faster lexical search via eager sparse scoring**. [Official project](https://github.com/xhluca/bm25s), MIT license.

Embedding model: the Sentence Transformers authors' [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache 2.0. Nils Reimers and Iryna Gurevych. **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**. EMNLP-IJCNLP 2019. [Paper](https://aclanthology.org/D19-1410/). MiniLM model lineage is described in the model card. Pinned weight revision and SHA256 are in `results/reference/model-manifest.json`.

This is a practical retrieval case study, not a competition entry, peer review, claim-verification system, or proof of general competitive advantage. The benchmark uses local CPU inference with no API keys or paid inference. Publication of this repository does not publish the corpus or model weights.
