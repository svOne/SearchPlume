"""Meaningful distribution checks against immutable original-run evidence."""
import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseEvidence(unittest.TestCase):
    def test_frozen_protocol(self):
        sha = hashlib.sha256((ROOT / 'protocol.json').read_bytes()).hexdigest()
        self.assertEqual(sha, '3b67238210276599574456d6ec29670f980f40361cf52b1344ea7bc2b8289fc0')
        frozen = json.loads((ROOT / 'results/reference/protocol-freeze.json').read_text())
        self.assertEqual(sha, frozen['sha256'])

    def test_exact_reference_and_query_means(self):
        result = json.loads((ROOT / 'results/reference/results.json').read_text())
        self.assertEqual(result['metrics'], {
            'bm25': {'NDCG@10': 0.66169, 'Recall@100': 0.87589},
            'dense': {'NDCG@10': 0.64508, 'Recall@100': 0.925},
            'rrf': {'NDCG@10': 0.69188, 'Recall@100': 0.955}})
        queries = json.loads((ROOT / 'results/reference/per-query-metrics.json').read_text())
        for method, values in queries.items():
            self.assertEqual(len(values), 300)
            for metric, key in [('NDCG@10', 'ndcg_cut_10'), ('Recall@100', 'recall_100')]:
                self.assertEqual(round(sum(v[key] for v in values.values()) / 300, 5), result['metrics'][method][metric])
        for values in result['relevant_pair_complementarity'].values():
            self.assertEqual(sum(values[k] for k in ['both', 'bm25_only', 'dense_only', 'neither']), 339)

    def test_original_core_algorithms_preserved(self):
        evidence = json.loads((ROOT / 'results/reference/provenance.json').read_text())
        for name, original_sha in evidence['original_core_source_sha256'].items():
            actual = hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest()
            self.assertEqual(actual, original_sha, name)

    def test_local_markdown_links(self):
        for source in ROOT.rglob('*.md'):
            if '.venv' in source.parts or '.git' in source.parts or 'artifacts' in source.parts:
                continue
            for target in re.findall(r'\]\(([^\s)]+)', source.read_text()):
                if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
                    continue
                target = target.split('#', 1)[0]
                self.assertTrue((source.parent / target).exists(), f'{source.relative_to(ROOT)}: {target}')

    def test_public_evidence_has_no_private_host_paths(self):
        for path in (ROOT / 'results/reference').rglob('*'):
            if path.is_file():
                self.assertNotIn('/Users/', path.read_text(), str(path))
                self.assertNotIn('libfile_', path.read_text(), str(path))


if __name__ == '__main__':
    unittest.main()
