#!/usr/bin/env python3
import copy
import json
from pathlib import Path
import random
import unittest
from sass_order import controls, validate_bundle


class BundleChecks(unittest.TestCase):
    def setUp(self):
        self.bundle = json.loads(Path(__file__).with_name('sass-order-audit.json').read_text())['original']

    def test_real_bundle(self):
        validate_bundle(self.bundle)

    def test_address_alias_rejected(self):
        self.bundle[0]['base'] = self.bundle[0]['dest']
        for x in self.bundle[1:]: x['base'] = self.bundle[0]['base']
        with self.assertRaises(ValueError): validate_bundle(self.bundle)

    def test_destination_overlap_rejected(self):
        self.bundle[2]['dest'] = self.bundle[1]['dest']
        with self.assertRaises(ValueError): validate_bundle(self.bundle)

    def test_pending_wait_rejected(self):
        self.bundle[2]['hi'] |= 1 << (41+11)
        with self.assertRaises(ValueError): validate_bundle(self.bundle)

    def test_shared_completion_tag_rejected(self):
        tag = controls(self.bundle[0]['hi'])['write']
        self.bundle[2]['hi'] = (self.bundle[2]['hi'] & ~(7 << 46)) | (tag << 46)
        with self.assertRaises(ValueError): validate_bundle(self.bundle)

    def test_address_read_barrier_rejected(self):
        self.bundle[3]['hi'] |= 7 << 49
        with self.assertRaises(ValueError): validate_bundle(self.bundle)

    def test_missing_record_chunk_rejected(self):
        self.bundle[3]['offset'] = 32
        with self.assertRaises(ValueError): validate_bundle(self.bundle)

    def test_value_and_destination_tag_identity(self):
        rng = random.Random(20260925)
        for _ in range(10000):
            memory = [rng.getrandbits(32) for _ in range(16)]
            reference = {}; candidate = {}; tags_a = {}; tags_b = {}
            for order, registers, tags in [(range(4),reference,tags_a),([2,3,0,1],candidate,tags_b)]:
                for i in order:
                    instruction = self.bundle[i]
                    for limb in range(4):
                        registers[instruction['dest']+limb] = memory[i*4+limb]
                    tags[controls(instruction['hi'])['write']] = instruction['dest']
            self.assertEqual(reference,candidate)
            self.assertEqual(tags_a,tags_b)


if __name__ == '__main__': unittest.main()
