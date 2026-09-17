"""Check the experimental helper's shared-memory budget after preprocessing."""
from pathlib import Path
import re
import subprocess
import unittest


class InverseStorageTest(unittest.TestCase):
    def test_inplace_tree_uses_at_most_16_kib(self):
        header = Path(__file__).with_name("tree_inverse.cuh")
        source = subprocess.check_output(
            ["clang++", "-E", "-P", "-Wno-pragma-once-outside-header",
             "-x", "c++", "-DZLAB_TREE=3", str(header)],
            text=True,
        )
        arrays = re.findall(
            r"__shared__\s+uint64_t\s+\w+\[(\d+)\]\[(\d+)\]", source
        )
        self.assertTrue(arrays, "No shared arrays found; budget check is invalid")
        byte_count = sum(int(rows) * int(cols) * 8 for rows, cols in arrays)
        self.assertLessEqual(byte_count, 16384)


if __name__ == "__main__":
    unittest.main()
