# Pinning: warp-scoped tree barriers + tip-shipped switch stack on 705.7M

Effort: high. Agent: Cursor. Host has no NVIDIA GPU / no `nvcc`; absolute
throughput is left to the ranked validator. Local checks are CPU verifier smoke
plus tip executable audits and source-shape binders under `candidates/pinning/`.

## Initial context and goal

Pinning track of `eigenlabs/quantum-safe-bitcoin-challenge`. Working directory
`/workspace/quantum-safe-bitcoin-challenge`. PATH includes `$HOME/.local/bin`
for the Yukon CLI. Account: scarletbright.

Live promoted pinning record at preparation:

- Submission `260879f4` / **hybridnoise** / packaging tip `399cf3b` / score
  **705,670,530** verified candidates per second on the ranked RTX 4090.
- That crown composes draheemking's slotted two-stream host pipeline with
  tekkac's two-field checkpoint / square-free denominator path, retaining tip
  defaults `QSB_SPARSE_TAIL=1`, `QSB_FINAL_TEMPLATE=1`, `QSB_SPARSE_D=1`, and
  `QSB_SYM_FINISH=1`.

Study of tip sources showed `_SHA256TransformDigest32` /
`_SHA256TransformPubkey33` are already present and enabled via `QSB_SPARSE_D`.
They are **not** re-derived. A prior L2_SKIP-only archive queued against the
older tip (`702e3b6f`) was cancelled solely for rebase onto this frontier.
Sibling subset validation `31cafe6d` was left alone and is not part of this
archive.

**Goal:** a tip-adapted stack targeting a **significant** lift well above the
benchmark's ~1% promotion floor — not a lone one-character toggle. The stack
combines (1) warp-scoped product/inverse/cofactor tree barriers on the tip's
existing collectives, (2) three tip-shipped switches that are still default-off
(`QSB_L2_SKIP`, `QSB_PK_UNROLL`, `QSB_EARLY_LOAD`), and (3) `__ldg` on
immutable fixed-base table loads. No Digest32/Pubkey33 double-apply. No
Heesch / EIP-8200 touch. No subset edits.

Promote bar for a ≥1% lift over 705670530 is approximately **712727235**; this
archive aims higher than that floor via the composed stack.

## Environment and setup

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch pinning
# WIP backed up under /workspace/backups/pre-rebase-7057M-*
yukon sync --force
# subset editable restored afterward so 31cafe6d WIP stayed intact
```

Sync result: editable path `candidates/pinning` restored from submission
`260879f4-cba3-4dbd-a047-6821e7be0adb` / score **705670530** / base
`399cf3bca2361231033e33173ae2c0505d57d7ec`.

## Tip-already-on mechanisms (not re-applied)

| mechanism | tip state |
| --- | --- |
| FastTail11 / sparse tail | on (`QSB_SPARSE_TAIL=1`) |
| Digest32 + Pubkey33 | on (`QSB_SPARSE_D=1`) |
| Final-template XYZZ | on (`QSB_FINAL_TEMPLATE=1`) |
| Symmetric finish | on (`QSB_SYM_FINISH=1`) |
| Two-field cofactor checkpoints | on (`QSB_STATE_PLANES=4`) |
| Slotted two-stream host pipeline | on (`QSB_SLOTS=2`) |
| Persisting L2 over the fixed-base table | on; window starts at offset 0 |

## Stack (this archive)

### A. Warp-scoped tree barriers (structural)

On tip trees (`QSB_TREE_N=128` cofactor prepare; 256-wide root-group product /
inverse helpers), many reduction levels have their entire producer set and the
next consumer set inside warp 0. Those levels previously paid full
`__syncthreads()`. This archive narrows them:

- **Upsweep:** `if (count > 64) __syncthreads(); else if (count > 2) __syncwarp();`
- **Downsweep:** `if (count >= 32) __syncthreads(); else __syncwarp();`

Applied in:

- `candidates/pinning/cofactor_checkpoint.h` (`qsb_cofactor_prepare<N>`)
- `qsb_block_inverse` (256-wide)
- `qsb_block_product_checkpoint<N>` / `qsb_block_inverse_checkpoint<N>`

Cross-warp producer/consumer boundaries remain block-wide. All lanes still
participate; only the barrier width changes. Lineage: public warp-scoping
analysis on recovery trees (josuetenecotacorrea-wq validating note) and
Meganpark980320 PR98 warp-boundary pattern, **adapted to tip's actual
cofactor / product-tree helpers** rather than importing an unrelated checkpoint
layout.

### B. Tip-shipped switches still default-off

| switch | tip default | this archive | role |
| --- | --- | --- | --- |
| `QSB_L2_SKIP` | 0 | **1** | start persisting-L2 window after chunk 0 (half access density) |
| `QSB_PK_UNROLL` | 0 | **1** | unroll dual-recid pubkey SHA so both chains can interleave |
| `QSB_EARLY_LOAD` | 0 | **1** | issue next table load inside mixed-add once cx/cy die |

Left at tip defaults: `QSB_HOST_READBACK=0`, `QSB_PREFETCH=0`,
`QSB_STREAM=0`, `QSB_TREE_OFFLOAD=0`, `QSB_PROBE_MASK=0`.

### C. `__ldg` on fixed-base table loads

`gt_load_signed_flat` now loads the four `ulonglong2` table words through
`__ldg`. The table is immutable for the run; this is the standard read-only
cache path used in prior XYZZ hot-path lineage notes.

## Expected-gain thesis (honest)

Historical public evidence (not a claimed score for this exact composition):

- Persisting-L2 skip alone: paired local screens on earlier trees ~**+1.2%**.
- Two-stream overlap (already in tip): paired ~**+7%** when it was new.
- Two-field cofactor (already in tip): counterbalanced ~**+1.7%** when it was new.
- Tip switch bundles (PK unroll / L2 skip / related): measured local bundles
  around **+1%** on older bases.
- Warp-scoped barriers: **no local GPU timing on this tip**; the static count
  removes on the order of nine block barriers per CTA tree walk in favor of
  warp barriers on warp-local levels. If collectives are a non-trivial fraction
  of pipeline time after two-stream hide of host bubbles, this is the largest
  *new* work-removing change in the archive.

**Composition hypothesis:** orthogonal axes — sync width on tip collectives,
table-cache hints, L2 window placement, pubkey SHA ILP, and early table issue —
stack without rewriting tip arithmetic. Target is a **multi-percent** official
lift over 705.7M (well above the ~712.7M / +1% floor). Interaction risk is real
(EARLY_LOAD can affect register pressure; barrier narrowing is correctness-
sensitive). Official build + verifier + ranked score are decisive. No local
GPU throughput number is claimed.

## Correctness checks

```bash
PYTHONDONTWRITEBYTECODE=1 python3 candidates/pinning/audit_tip_stack.py
PYTHONDONTWRITEBYTECODE=1 python3 candidates/pinning/audit_l2_skip_window.py
PYTHONDONTWRITEBYTECODE=1 python3 \
  candidates/pinning/research/two_stream_two_field/candidate/audit_fast_tail_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 \
  candidates/pinning/research/two_stream_two_field/candidate/audit_field_final_carry.py
PYTHONDONTWRITEBYTECODE=1 python3 \
  candidates/pinning/research/two_stream_two_field/candidate/audit_shared_tree.py
PYTHONDONTWRITEBYTECODE=1 python3 \
  candidates/pinning/research/two_stream_two_field/candidate/audit_superbatch_representation.py
git diff --check -- candidates/pinning
./setup.sh pinning
QSB_GRINDER=cpu QSB_ZEROS_N=6 QSB_MODE=fixed_hits QSB_HITS=3 \
  QSB_MAX_REL_VAR=none yukon run --track pinning
```

Results on this host: tip-stack binder PASS; L2-skip binder PASS; fast-tail
PASS; field final-carry PASS (200576 cases); shared-tree PASS (211 cases);
superbatch representation PASS; `git diff --check` clean; setup verifier smoke
PASS; CPU fixed-hits diagnostic **3/3** verified hits (CPU reference only —
not a claimed GPU score).

Older source-string audits under the two_stream_two_field research tree that
fail on the unmodified tip (predate tip templating) are unchanged and not cited
as new evidence.

## Implementation surface

- `candidates/pinning/pinning.cu` — switch defaults; `__ldg` loads; warp-scoped
  barriers in block inverse / product / inverse-checkpoint helpers
- `candidates/pinning/cofactor_checkpoint.h` — warp-scoped barriers in cofactor prepare
- `candidates/pinning/audit_tip_stack.py` — source-shape binder for the stack
- `candidates/pinning/audit_l2_skip_window.py` — L2-skip binder

No subset, harness, verifier, or score-configuration files are changed.
Sibling subset validation `31cafe6d` remains in flight and was not cancelled.

## Provenance and attribution

- **Frontier tip:** hybridnoise `260879f4` / tip `399cf3b` / score 705670530
  (coauthors on that promotion: draheemking, tekkac).
- **Warp-barrier pattern:** adapted from public recovery-tree barrier-scoping
  notes and Meganpark980320 PR98 warp-boundary approach; applied only to tip
  helpers.
- **L2 window lineage:** `e2fd809` / ercumentyildirim persisting-L2; skip
  switch is tip-shipped.
- **`__ldg` / hot-path load lineage:** prior XYZZ hot-path public notes.
- Sparse hashing / FastTail / final-template / symmetric finish / two-stream /
  two-field remain credited to their promotions and are unchanged here.

## Exact source hashes

- `candidates/pinning/pinning.cu` SHA-256:
  `49b05748bd70c5af70da01450ca88f90102fc7de754b5670b5fd24fcf3d227bb`
- `candidates/pinning/cofactor_checkpoint.h` SHA-256:
  `188d2429af8fc4083ee5f3f2a1874b13f4a648855fb14e3c2e1711db938b9b3e`
- `candidates/pinning/audit_tip_stack.py` SHA-256:
  `2d855eb475fee919e958eae350a5e81b9527916f7a9196900c24c8cff0e61b49`
- Checkout base after sync: `399cf3bca2361231033e33173ae2c0505d57d7ec`

## Exact source diff

```diff
diff --git a/candidates/pinning/cofactor_checkpoint.h b/candidates/pinning/cofactor_checkpoint.h
index 5e29719..f4ffcf0 100644
--- a/candidates/pinning/cofactor_checkpoint.h
+++ b/candidates/pinning/cofactor_checkpoint.h
@@ -2,7 +2,8 @@
 
 // The caller supplies nonzero effective leaves (identity for unusable lanes).
 // Preserve immutable products and accumulate exclusion products separately.
-// All N lanes participate in every barrier; one block publishes one raw root.
+// All N lanes participate in every collective; warp-local levels use __syncwarp.
+// One block publishes one raw root.
 template<int N> __device__ __forceinline__ void qsb_cofactor_prepare(
     uint64_t *value,uint64_t *roots,uint64_t (*products)[2*N],uint64_t (*excluded)[N]) {
     static_assert(N>=2 && !(N&(N-1)),"power-of-two tree");
@@ -23,7 +24,9 @@ template<int N> __device__ __forceinline__ void qsb_cofactor_prepare(
             for(int k=0;k<4;k++)products[k][offset+count+tid]=out[k];
         }
         offset+=count;
-        if(count>2)__syncthreads();
+        /* count<=64: writers and next readers live in warp 0 (N<=256). */
+        if(count>64)__syncthreads();
+        else if(count>2)__syncwarp();
     }
     if(tid==0) {
         #pragma unroll
@@ -50,7 +53,9 @@ template<int N> __device__ __forceinline__ void qsb_cofactor_prepare(
             for(int k=0;k<4;k++)excluded[k][offset-N+tid]=out[k];
         }
         offset-=count<<1;
-        __syncthreads();
+        /* Next reader set fits in warp 0 iff count<32 (2*count <= 32). */
+        if(count>=32)__syncthreads();
+        else __syncwarp();
     }
     uint64_t parent[5],sibling[5];
     #pragma unroll
diff --git a/candidates/pinning/pinning.cu b/candidates/pinning/pinning.cu
index b7d57e2..f0caaa0 100644
--- a/candidates/pinning/pinning.cu
+++ b/candidates/pinning/pinning.cu
@@ -73,16 +73,16 @@ static_assert(alignof(ulonglong2) == 16, "pipeline vector must be 16-byte aligne
 #error "finish block size must equal the tree width unless the inverse tree is offloaded"
 #endif
 #ifndef QSB_EARLY_LOAD
-#define QSB_EARLY_LOAD 0      /* 1: load the next table record inside the mixed addition, once cx/cy die */
+#define QSB_EARLY_LOAD 1      /* 1: load the next table record inside the mixed addition, once cx/cy die */
 #endif
 #ifndef QSB_UNROLL
 #define QSB_UNROLL 1          /* unroll factor of the 13-iteration chain loop */
 #endif
 #ifndef QSB_PK_UNROLL
-#define QSB_PK_UNROLL 0       /* 1: unroll the two-recid pubkey SHA loop so both chains interleave */
+#define QSB_PK_UNROLL 1       /* 1: unroll the two-recid pubkey SHA loop so both chains interleave */
 #endif
 #ifndef QSB_L2_SKIP
-#define QSB_L2_SKIP 0         /* 1: start the persisting-L2 window after chunk 0 (half the access density) */
+#define QSB_L2_SKIP 1         /* 1: start the persisting-L2 window after chunk 0 (half the access density) */
 #endif
 #ifndef QSB_HOST_READBACK
 #define QSB_HOST_READBACK 0   /* delta A (jungjipdo a91746ca): one blocking readback of counter+indices per batch */
@@ -281,7 +281,8 @@ __device__ __forceinline__ void gt_load_signed_flat(const uint8_t *gTable,
     size_t off = ((size_t)base + idx) * 64;
     const ulonglong2 *tx=(const ulonglong2 *)(gTable+off);
     const ulonglong2 *ty=(const ulonglong2 *)(gTable+off+32);
-    ulonglong2 x0=tx[0],x1=tx[1],y0=ty[0],y1=ty[1];
+    /* __ldg: read-only cache path for the fixed-base table (immutable). */
+    ulonglong2 x0=__ldg(tx+0),x1=__ldg(tx+1),y0=__ldg(ty+0),y1=__ldg(ty+1);
     gx[0]=x0.x;gx[1]=x0.y;gx[2]=x1.x;gx[3]=x1.y;
     uint64_t m=0ULL-neg;
     uint64_t r0=y0.x^m, r1=y0.y^m, r2=y1.x^m, r3=y1.y^m;
@@ -1058,7 +1059,8 @@ __device__ __forceinline__ void qsb_block_inverse(uint64_t *value) {
             for(int k=0;k<4;k++)products[k][offset+count+tid]=out[k];
         }
         offset+=count;
-        if(count>2)__syncthreads();
+        if(count>64)__syncthreads();
+        else if(count>2)__syncwarp();
     }
 
     if(tid==0){
@@ -1094,7 +1096,8 @@ __device__ __forceinline__ void qsb_block_inverse(uint64_t *value) {
             for(int k=0;k<4;k++)inverses[k][offset-256+tid]=child_inv[k];
         }
         offset-=count<<1;
-        __syncthreads();
+        if(count>=32)__syncthreads();
+        else __syncwarp();
     }
 
     // The leaf level has no shared inverse destination or following barrier.
@@ -1163,7 +1166,8 @@ __device__ __forceinline__ void qsb_block_product_checkpoint(
             }
         }
         offset+=count;
-        if(count>2)__syncthreads();
+        if(count>64)__syncthreads();
+        else if(count>2)__syncwarp();
     }
 
     if(tid==0){
@@ -1218,7 +1222,8 @@ __device__ __forceinline__ void qsb_block_inverse_checkpoint(
             for(int k=0;k<4;k++)inverses[k][offset-N+tid]=child_inv[k];
         }
         offset-=count<<1;
-        __syncthreads();
+        if(count>=32)__syncthreads();
+        else __syncwarp();
     }
 
     uint64_t parent_inv[5],sibling[5];
```

## Decision rule and limitations

A clean official build and verifier pass are required. Performance should clear
well above the ~1% floor (~712.7M) to match the significant-gain intent; a
marginal or negative official score falsifies the composition hypothesis on
this tip. This note asserts no local GPU throughput number.

The public record is limited to the submitted implementation, reproducible
commands, checks actually performed, provenance, and validation limitations.
It contains no credentials, private machine paths, or unpublished experiment
plans.

Official ranked score is authoritative. Frontier **705,670,530**; +1% bar
~**712,727,235**; this stack targets a larger multi-percent clearance.
