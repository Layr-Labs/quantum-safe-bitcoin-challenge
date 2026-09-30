"""CPU models of layout/capacity, not GPU throughput or compiled-field correctness."""
import math
from pathlib import Path
import random
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
P = (1 << 256) - (1 << 32) - 977

class CheckedArena:
    def __init__(self, size):
        self.values = [None] * size
    def put(self, index, value):
        assert 0 <= index < len(self.values), ('write_oob', index)
        self.values[index] = value
    def get(self, index):
        assert 0 <= index < len(self.values), ('read_oob', index)
        result = self.values[index]
        assert result is not None, ('uninitialized_read', index)
        return result

def packed_wave_inverse(values):
    n = len(values)
    assert n >= 32 and n & (n - 1) == 0
    products, inverses = CheckedArena(2 * n), CheckedArena(n)
    for i, value in enumerate(values):
        products.put(i, value % P)
    offset, count = 0, n
    while count > 16:
        half = count // 2
        for t in range(half):
            products.put(offset + count + t,
                         products.get(offset + t) * products.get(offset + half + t) % P)
        offset += count
        count //= 2
    l8, l4, l2 = offset + 16, offset + 24, offset + 28
    for t in range(8):
        products.put(l8 + t, products.get(offset + t) * products.get(offset + 8 + t) % P)
    cofactors = [products.get(offset + (t ^ 8)) * products.get(l8 + ((t & 7) ^ 4)) % P
                 for t in range(16)]
    for t in range(4):
        products.put(l4 + t, products.get(l8 + t) * products.get(l8 + 4 + t) % P)
    cofactors = [c * products.get(l4 + ((t & 3) ^ 2)) % P for t, c in enumerate(cofactors)]
    for t in range(2):
        products.put(l2 + t, products.get(l4 + t) * products.get(l4 + 2 + t) % P)
    cofactors = [c * products.get(l2 + ((t & 1) ^ 1)) % P for t, c in enumerate(cofactors)]
    root = products.get(l2) * products.get(l2 + 1) % P
    root_inverse = pow(root, -1, P)
    for t, c in enumerate(cofactors):
        inverses.put(offset - n + t, root_inverse * c % P)
    offset -= 32
    count = 32
    while count < n:
        half = count // 2
        for t in range(count):
            inverses.put(offset - n + t,
                         inverses.get(offset + count - n + (t & (half - 1)))
                         * products.get(offset + (t ^ half)) % P)
        offset -= count * 2
        count *= 2
    assert offset == 0
    return [inverses.get(t & (n // 2 - 1)) * products.get(t ^ (n // 2)) % P for t in range(n)]

class Invariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = (ROOT/'candidates/subset/tests/gpu_epochs/tree.cu').read_text()
        cls.inverse = (ROOT/'candidates/subset/tests/gpu_epochs/tree_inverse.cuh').read_text()
        cls.carrier = (ROOT/'candidates/subset/QsbCarrier.h').read_text()
    def test_wave_tree_random(self):
        rng = random.Random(20260930)
        for n in (32, 64, 128, 256, 512):
            for seed in range(12):
                values = [rng.randrange(1, P) for _ in range(n)]
                with self.subTest(n=n, seed=seed):
                    self.assertEqual(packed_wave_inverse(values), [pow(x, -1, P) for x in values])
    def test_wave_tree_edge_and_tail(self):
        for n in (32, 64, 128, 256, 512):
            for live in (0, 1, n//2-1, n-1, n):
                values = [((P-1, P-2, 2, 1)[i%4] if i < live else 1) for i in range(n)]
                with self.subTest(n=n, live=live):
                    self.assertEqual(packed_wave_inverse(values), [pow(x, -1, P) for x in values])
    def test_paired_epoch_enumeration(self):
        sizes = list(range(1, 65)) + [127,128,129,255,256,257,511,512,513,1023,1024,1025,2047,2048,4097]
        for block in (256,512):
            pair_mul = 2 * (block//128)
            for epochs in sizes:
                seen = set()
                for bx in range((epochs+pair_mul-1)//pair_mul):
                    for tid in range(block):
                        lane, half = tid & 127, tid//128
                        epoch_a = pair_mul*bx+2*half
                        for epoch in (epoch_a,epoch_a+1):
                            if epoch < epochs:
                                key = (epoch,lane)
                                self.assertNotIn(key,seen)
                                seen.add(key)
                self.assertEqual(len(seen),epochs*128)
    def test_capacity_and_shared_memory(self):
        for block in (256,512):
            launches = 262144*256//block
            pair_mul = 2*(block//128)
            self.assertEqual(launches*pair_mul,1048576)
            self.assertEqual(launches*pair_mul*128,134217728)
        static, dynamic = 4*(2*512+512)*8, 6*512*16
        self.assertEqual((static,dynamic,static+dynamic),(49152,49152,98304))
        self.assertLess(static+dynamic,100*1024)
    def test_vector_parking_layout(self):
        offsets = set()
        for row in range(6):
            for tid in range(512):
                offset = (row*512+tid)*16
                self.assertEqual(offset%16,0)
                self.assertNotIn(offset,offsets)
                self.assertLessEqual(offset+16,49152)
                offsets.add(offset)
        self.assertEqual(len(offsets),3072)
    def test_all_digest_launches_pass_dynamic_bytes(self):
        launches = re.findall(r'kernel_digest<<<([^>]+)>>>',self.tree)
        self.assertEqual(len(launches),4)
        self.assertTrue(all('QSB_DIGEST_DYN_SMEM_BYTES' in x for x in launches))
        self.assertIn('qsb_carrier_try_smem(kernel_digest, QK_DIG',self.tree)
        self.assertIn('g, b, argv, dynamic_smem, st',self.carrier)
        self.assertGreaterEqual(self.tree.count('cudaFuncAttributeMaxDynamicSharedMemorySize'),2)
    def test_geometry_and_preserved_promoted_features(self):
        self.assertIn('#define QSB_SE_BLOCK   512',self.tree)
        self.assertIn('__launch_bounds__(512, 1) kernel_digest',self.tree)
        self.assertIn('#define QSB_SE_LAUNCH_BLOCKS ((ZLAB_LAUNCH_BLOCKS * 256) / QSB_SE_BLOCK)',self.tree)
        for exact in ('#define QSB_PARK128 1','#define QSB_TREE_WAVE_TOP 1',
                      '#define QSB_SHA_SCHED_V4 1','#define QSB_S3_UNIFORM_G 1'):
            self.assertIn(exact,self.tree)
        self.assertIn('ulonglong2 (*parkA2)[QSB_SE_BLOCK]',self.tree)
        self.assertIn('qsb_sc_products[4][2*QSB_SE_BLOCK]',self.inverse)
        self.assertIn('inverses[4][QSB_SE_BLOCK]',self.inverse)
    def test_unqualified_optional_paths_fail_at_compile_time(self):
        self.assertIn('#if QSB_SE_BLOCK > 256 && (ZLAB_TREE != 2',self.inverse)
        for knob in ('QSB_ROOT_LUT_SMEM','QSB_PRE3_ROOT','QSB_TREE_UNROLL','QSB_Q_SPREAD','QSB_TAIL_WEAVE'):
            self.assertIn(knob,self.inverse.split('#if QSB_SE_BLOCK > 256',1)[1].split('#endif',1)[0])

if __name__ == '__main__':
    unittest.main(verbosity=2)
