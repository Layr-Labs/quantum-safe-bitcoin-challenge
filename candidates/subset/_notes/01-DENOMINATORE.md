# 01-DENOMINATORE — cosa ho aperto per intero

Regola 2 del brief: il denominatore, non il "non c'e'".

| # | File | Righe | Contenuto |
|---|------|-------|-----------|
| 1 | `subset.cu` | 11 | solo 4 `#define` di switch + `#include "tests/gpu_epochs/tree.cu"` (riga 11). Il resto del device code e' dentro `tree.cu` e nei suoi header. |
| 2 | `build_carrier.sh` | 98 | rigenera `qsb_carrier_sm89.h`: nvcc `-O3 -arch=sm_89 -cubin`, poi un python che (a) trova i 6 kernel per nome mangled (riga 36-43), (b) verifica che 15 global siano nell'immagine (52-55), (c) verifica che il kernel digest contenga `LTC64B` (59-64), (d) scrive base64 + sha. |
| 3 | `tests/gpu_epochs/tree.cu` | **in lettura** | >825 righe gia', entry point del device code. |

## Elenco completo della superficie (29 file)

`candidates/subset/`: ASMLAST511-RESEARCH.md, BY_NORMALIZED6.md, CANONICAL-ADD.md, CHAIN-REPLAY.md,
COMPLETE-POINT.md, CpuGrindSubset.h, GLVScalar.cuh, GPUHash.h, GPUMath.h, HIT-CHECK.md,
LAST511-RESEARCH.md, PAIR-CURRENT.md, PAIR-FRONT.md, POINT-BY-TABLE.md, POINT-PREDICATE.md,
POINT-X3.md, QsbCarrier.h, SHA-gate `sha_gate_fma.cuh`, `square32.cuh`, `submission-note.md`,
`TREE_INVERSE.md`, `y_pair_sc.cuh`, `hit_filter_field.cuh`, `hit_filter_field_sc.cuh`,
`hit_filter_field_sc_aluz.cuh`, `chain_replay_field.cuh`, `build_carrier.sh`, `subset.cu`,
`qsb_carrier_sm89.h`.

`candidates/subset/tests/gpu_epochs/`: by_table_matrix_audit.cu, canonical_add_audit.cu,
chain_replay_audit.cu, dirdig_audit.cu, epoch_groups.cuh, filter_tail_sc.cuh, first_stage_audit.cu,
hm39_divstep.cuh, hm39_pair_inverse.cuh, hm41_quad_inverse.cuh, hm43_warp_inverse.cuh,
host_producers_v1.h, host_producers_v3.h, host_producers.h, inverse_limbs.cuh, pair_finish_audit.cu,
pair_shared.cuh, parity_window_subset.cuh, point_audit.cu, point_predicate_audit.cu, prefix_cache.cuh,
qsb_host_verify.h, scalar_audit.cu, seed_x3_audit.cu, tree_audit.cu, tree_inverse.cuh, tree.cu,
tree.cu.orig, window_schedule_shared.cuh, zinv32.cuh.

**Nota di portata:** i `*_audit.cu` sono programmi di test separati, non sono compilati dentro `subset.cu`.
Il metro del brief (`qsb_sass_baseline.py`) conta i kernel che `build_carrier.sh` estrae: `kernel_epoch_groups`,
`kernel_build_epochs_inc`, `kernel_build_first_flat`, `kernel_digest`, `kernel_build_gtable`,
`kernel_gt_heal_scan`. **Il solo kernel che gira sul percorso di ricerca e' `kernel_digest`** (QSB_HOST_PRODUCERS=1
mette i producer su host). Quindi il conteggio SASS che conta e' quello di `kernel_digest`.