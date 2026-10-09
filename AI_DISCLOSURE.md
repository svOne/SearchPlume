# AI assistance and validation

SearchPlume was prepared with OpenAI Codex assistance for code, documentation, source verification, experiment orchestration, packaging checks and release preparation. AI assistance is disclosed because this is a practical portfolio project.

The reported numbers come from an executed local CPU benchmark using the official BEIR evaluator and pinned pytrec_eval implementation. Synthetic controls, complete ranking validation and re-evaluation of the same outputs passed. Independent AI-agent reviews checked methodology, numerical reporting and release contents. These are automated checks, not independent human peer review.

No retrieval settings were tuned using test judgments. This does not establish training-data independence: MiniLM training includes S2ORC abstracts and SciFact also derives from S2ORC. No claim of statistical significance, new retrieval research, competition performance or general superiority is made.

The measured host was an Apple M1 MacBook Pro with 8 GB RAM and macOS arm64, CPython 3.11.7. No cross-platform or GPU testing is claimed. The release preserves the original protocol, core benchmark code and six official metric values. Release changes concern naming, safe public evidence, documentation and a runner that separates generated files from published reference results.
