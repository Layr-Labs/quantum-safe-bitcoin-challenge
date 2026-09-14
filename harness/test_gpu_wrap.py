import tempfile
import unittest
from pathlib import Path

from harness import gpu_wrap


class CollectHitsTest(unittest.TestCase):
    def collect(self, bench: str, filename: str, contents: str):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            (results / filename).write_text(contents)
            return gpu_wrap.collect_hits(bench, results)

    def test_ignores_incomplete_trailing_pinning_hit(self):
        hits = self.collect(
            "pinning",
            "pinning_hit_0.txt",
            "sequence=1\nlocktime=2\nrecid=1\nsequence=3\n",
        )

        self.assertEqual(hits, [{"sequence": 1, "locktime": 2, "recid": 1}])

    def test_ignores_incomplete_trailing_subset_hit(self):
        hits = self.collect(
            "subset",
            "digest_hit_0.txt",
            "indices=0,1,2,3,4,5,6,7,8\nrecid=1\nindices=9,10,",
        )

        self.assertEqual(
            hits,
            [{
                "bench": "subset",
                "skip": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "recid": 1,
            }],
        )

    def test_rejects_incomplete_nontrailing_pinning_hit(self):
        with self.assertRaisesRegex(ValueError, "incomplete pinning hit"):
            self.collect(
                "pinning",
                "pinning_hit_0.txt",
                "sequence=1\nsequence=2\nlocktime=3\nrecid=0\n",
            )

    def test_rejects_incomplete_nontrailing_subset_hit(self):
        with self.assertRaisesRegex(ValueError, "incomplete subset hit"):
            self.collect(
                "subset",
                "digest_hit_0.txt",
                "indices=0,1\nindices=2,3\nrecid=0\n",
            )

    def test_keeps_complete_pinning_hit_for_verification(self):
        hits = self.collect(
            "pinning",
            "pinning_hit_0.txt",
            "sequence=1\nlocktime=2\nrecid=2\n",
        )

        self.assertEqual(
            hits,
            [{"bench": "pinning", "sequence": 1, "locktime": 2, "recid": 2}],
        )

    def test_keeps_complete_subset_hit_for_verification(self):
        hits = self.collect(
            "subset",
            "digest_hit_0.txt",
            "indices=0,1\nrecid=0\n",
        )

        self.assertEqual(
            hits,
            [{"bench": "subset", "skip": [0, 1], "recid": 0}],
        )


if __name__ == "__main__":
    unittest.main()
