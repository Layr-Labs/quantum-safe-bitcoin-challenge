# Bounded root integration check

PASS on exact nineteen-file source
`f47ad8241adac7aac14e63ad947735ada3bf3565ae60266fa7153b6744d5583a`.
The unbounded6cb9 reports remain untouched.

The selected EC192 tree body is extracted unchanged except for instrumentation
and an instrumented scratch-array parameter. HM43 executes its actual bounded
header using the separately checked32-host-thread collectives. The scalar
fallback executes actual `_ModInv`, `_DivStep62`, `_IMult`, `_IMultC`, `_MulP`,
`_MatrixVecMul`, `_MatrixVecMulHalf`, `_AddCh`, both `_ShiftR62` functions and
source macros extracted from the candidate GPUMath. Their VanitySearch/Jean Luc
Pons GPL provenance is retained in the candidate and applies to these extracted
functions. This is not an OpenSSL or pow substitution for the fallback.

The legacy host projection replaces CUDA scalar carry/borrow, mulhi and madhi
operators with explicit uint128 semantics. The two signed coefficient left
shifts use unsigned bit-preserving shifts to avoid host C++ undefined behavior;
`-fwrapv` retains signed wrapping semantics. The actual loop/state algorithm is
otherwise preserved. Before each integration executable,520 independent roots
(including zero, one, near-p and512 random canonical roots) match Python's
modular inverse and preserve output canaries. Function/macro/projection hashes
are in `results.json`. This is finite independent validation of a disclosed
host projection, not a proof of the original scalar GPU implementation.

Natural caps16,0,1 each test tails0,1,31,32,33,191,192:4,032 output inverses total.
Cap16 completes cooperatively for these roots; cap0/1 each invoke the real
projected scalar fallback once per cohort. Every case retains669 tree field
multiplications, five EC subgroup joins, eleven tree warp joins,192 physical
owners64..255 and32 root participants64..95. All192 outputs and enclosing
canaries are checked even for the all-identity tail. Only tree multiplication
uses OpenSSL. Python pow is the independent expected-output oracle.

Two additional positive configurations poison all five private root words on
false return, at caps0/1. The actual caller reload and explicit root[4]=0 recover
correct results in all seven tails, adding2,688 output comparisons. Total value
canaries across all five positive configurations are13,440.

The false-return poison is a deliberate fault injection. The real helper's
contract leaves its caller output unchanged on failure. Deleting only a reload
or clear would consequently be observationally redundant under current natural
cap0/1 behavior; no natural defect is claimed. With the disclosed poison:

- removing the four-word reload produces an independent inverse mismatch;
- removing the fifth-word clear violates the scalar fallback input contract;
- publishing the incomplete private root instead of invoking fallback produces
  an independent inverse mismatch.

All three projected mutations compile and exit86 with the recorded diagnostic.
The publication and reload controls fail numerical checks. The fifth-word
control is intentionally a precondition check, not mislabeled a numerical race.
The actual candidate remains unchanged.

Reproduction from the benchmark root:

```
python3 -B candidates/subset/research/warp_root/bounded/check_tree.py
```

`--source` and `--output` are supported; source must match the frozen manifest.
Generated C++, build/run logs and individual receipts remain in this directory.
All source/support hashes are checked before/after. EC subgroup synchronization
is still a192-thread rendezvous projection of an unchanged protocol; this test
is not GPU memory-model, CUDA execution, scheduling, throughput or occupancy
evidence. No compiler VM, stage mutation or submission action was performed.
