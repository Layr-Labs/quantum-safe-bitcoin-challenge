import hashlib
import itertools
from pathlib import Path
import random
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'harness'))
from crypto import sha256_midstate


def windows_for(size):
    result = []
    for first, second, third in itertools.combinations(range(13), 3):
        if size == 128:
            keep = first >= 6 or (first <= 5 and second >= 7) or (first == 0 and second == 1 and 8 <= third <= 10)
        else:
            keep = not (first >= 1 and third <= 7 and not (first == 1 and second == 2))
        if keep:
            result.append((first, second, third))
    assert len(result) == size
    return result


def run():
    rng = random.Random(20260925)
    checks = 0
    for size, slots, old_slots, expected_classes in ((128, 8, 16, 8), (256, 64, 64, 54)):
        windows = windows_for(size)
        for fixture in range(3):
            rows = [rng.randbytes(10) for _ in range(13)] if fixture else [bytes([index]) * 10 for index in range(13)]
            blocks = [b''.join(row for index, row in enumerate(rows) if index not in window)[:56] for window in windows]
            distinct = list(dict.fromkeys(blocks))
            assert len(distinct) == expected_classes
            classes = [distinct.index(block) for block in blocks]
            epochs = 67
            padding = 0xA5A5A5A5
            old = [padding] * (epochs * old_slots * 8)
            new = [padding] * (epochs * slots * 8)
            references = []
            for epoch in range(epochs):
                prefix, remainder = rng.randbytes(64), rng.randbytes(8)
                states = [sha256_midstate(prefix + remainder + block) for block in distinct]
                references.append(states)
                for class_index, words in enumerate(states):
                    for word_index, value in enumerate(words):
                        old[(epoch * old_slots + class_index) * 8 + word_index] = value
                        new[epoch * slots * 8 + word_index * slots + class_index] = value
                message = prefix + remainder + blocks[0]
                padded = message + b'\x80' + b'\0' * 55 + struct.pack('>Q', len(message) * 8)
                assert struct.pack('>8I', *sha256_midstate(padded)) == hashlib.sha256(message).digest()
            for epoch in range(epochs):
                for class_index in classes:
                    for word_index in range(8):
                        value = new[epoch * slots * 8 + word_index * slots + class_index]
                        assert value == old[(epoch * old_slots + class_index) * 8 + word_index]
                        assert value == references[epoch][class_index][word_index]
                        checks += 1
                for class_index in range(expected_classes, slots):
                    assert all(new[epoch * slots * 8 + word * slots + class_index] == padding for word in range(8))
            print(f'{size} windows: {expected_classes} classes; 67 epochs; exact SHA state and padding checks passed')
    print(f'PASS: {checks} state-word comparisons; CPU-only, no GPU timing claim')


if __name__ == '__main__':
    run()
