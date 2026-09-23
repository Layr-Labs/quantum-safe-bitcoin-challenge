# Affine worker inverse tree

`inverse_tree.cuh` supplies a complete binary batch inverse for 32, 64, 128,
or 256 lanes. It adapts the checkpoint helpers already present in the pinned
candidate; the new architecture is how resident affine workers can use these
helpers with a separate inversion service.

For `N=128`, a caller-owned product array `uint64_t products[4][256]` takes
8 KiB, and an inverse array `uint64_t inverses[4][128]` takes 4 KiB. Each lane
supplies a nonzero effective leaf, replacing unusable or singular inputs by 1
while keeping its own validity mask.

```cpp
qsb_affine_tree_prepare_shared<128>(leaf, root, products);
// Only thread zero's root is valid. Obtain 1/root from the inversion service.
qsb_affine_tree_finish_shared<128>(inverse, root_inverse, products, inverses);
```

All lanes call both helpers. The finish helper only reads lane zero's
`root_inverse`; its first block barrier publishes the value. Product scratch
must survive the wait. The root inverse must be unweighted: the production
recovery hierarchy's optional ISO scaling does not belong here.

The upward pass uses `N-1` multiplications. The complete downward pass uses
`2N-2`, including each final leaf product. The pair therefore costs `3N-3`
multiplications and one externally supplied inverse. This count is the basis
for affine addition costing approximately `5M+1S` per candidate before service,
synchronization and field-normalization overheads. It is not a throughput
measurement.

The shared tree defaults to the carry-complete `qsb_field_mul`; a product error
in a shared tree can affect multiple candidates. `QSB_AFFINE_TREE_EXACT=0`
explicitly selects the inherited short-carry arithmetic for a separate
experiment. It has not received an error-amplification or recall audit.

Checkpoint alternatives are also available as `qsb_affine_tree_prepare<N>` and
`qsb_affine_tree_finish<N>`. They store exactly `4*(N-2)` words per block,
excluding leaves and the root. Finish requires each original effective leaf
again and a supplied root inverse. Arrays of four words are sufficient for
every public leaf/root argument; the helpers never write element four.

## CPU execution audit

`test_inverse_tree.cpp` executes the **actual header** using one CPU thread per
CUDA lane, block/warp barriers, and independent Boost big-integer field
arithmetic. It exercises both shared and checkpoint APIs across all four
supported widths, canonical/random/raw operands, inactive and singular masks,
in-place leaf outputs, multi-block checkpoint offsets, and allocation sentinels.
Each leaf is compared with an independent modular exponentiation inverse.
Multiplication counters assert `3N-3` for each block.

The run in `inverse_tree_cpu_result.json` passed **112 blocks, 13,440 leaf
inverses and 39,984 tree multiplications**, with zero mismatches. Of the fixture
inputs, 2,034 were masked to identity. This checks indexing, synchronization
structure and modular algebra; it does not execute CUDA PTX, check GPU memory
ordering in a service mailbox, or establish GPU performance.

From this directory, reproduce with:

```sh
rtk proxy g++ -std=c++20 -O2 -pthread -I/tmp/qsb-boost-audit/usr/include test_inverse_tree.cpp -o test_inverse_tree
rtk proxy ./test_inverse_tree
```

The include override is unnecessary where Boost headers are already installed.
