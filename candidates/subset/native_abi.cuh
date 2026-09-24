#pragma once
#include "native_types.cuh"
// Names identify loaded native entries, never source function addresses.
namespace qsb_native {
template<class... P> struct KernelTag { const char *name; };
template<class T> struct SymbolTag { const char *name; };
namespace abi {
using u8 = uint8_t; using u32 = uint32_t; using u64 = uint64_t;
static constexpr KernelTag<const epoch_desc_t *, u32 *, unsigned, unsigned>
    kernel_build_first_flat{"kernel_build_first_flat"};
static constexpr KernelTag<u64, u64, int, int, const u32 *, const u8 *, int, const u8 *, epoch_desc_t *, u32 *>
    kernel_build_epochs{"kernel_build_epochs"};
static constexpr KernelTag<u64, u32, int, int, const u32 *, const u8 *, int, const u8 *, qsb_group_t *, u32 *, u64, u64>
    kernel_epoch_groups{"kernel_epoch_groups"};
static constexpr KernelTag<u64, u64, int, int, const u8 *, const qsb_group_t *, u64, epoch_desc_t *, u32 *, const u32 *>
    kernel_build_epochs_inc{"kernel_build_epochs_inc"};
static constexpr KernelTag<const u8 *, u8 *, const epoch_desc_t *, const u32 *, const u8 *, int>
    kernel_verify_pair_hits{"kernel_verify_pair_hits"};
static constexpr KernelTag<
    const u8 *, int, int, const u32 *, const u8 *, int, const u8 *, const u8 *, int, const u8 *, int, int,
    const u64 *, const u64 *, const u64 *, const u64 *, const u64 *, u8 *, u32 *, u32 *,
    u8 *, u8 *, u8 *, u8 *, u8 *, u8 *, int, int, int, int, int, u64, int, int, const u8 *, int,
    const u32 *, const epoch_desc_t *, const u32 *, int> kernel_digest{"kernel_digest"};
static constexpr KernelTag<const u64 *, const u64 *, u8 *> kernel_build_gtable{"kernel_build_gtable"};
static constexpr SymbolTag<u64[151][10]> BINOM_C{"BINOM_C"};
static constexpr SymbolTag<u32[64]> K{"K"};
static constexpr SymbolTag<u32[4][64]> QSB_CONST_SCHEDULE{"QSB_CONST_SCHEDULE"};
static constexpr SymbolTag<uint4[151]> QSB_PUSH_WORDS{"QSB_PUSH_WORDS"};
static constexpr SymbolTag<u64[8]> QSB_U2R{"QSB_U2R"};
static constexpr SymbolTag<u64[4]> QSB_U2R_C{"QSB_U2R_C"};
static constexpr SymbolTag<u8[128][3]> WIN3{"WIN3"};
static constexpr SymbolTag<u32[14][128]> QSB_WINDOW_FIRST{"QSB_WINDOW_FIRST"};
static constexpr SymbolTag<u32[64][128]> QSB_WINDOW_SECOND{"QSB_WINDOW_SECOND"};
static constexpr SymbolTag<u32[128]> QSB_WINDOW_CLASS{"QSB_WINDOW_CLASS"};
static constexpr SymbolTag<u32[128]> QSB_FIRST_CLASS{"QSB_FIRST_CLASS"};
static constexpr SymbolTag<u32[14][16]> QSB_FIRST_UNIQUE{"QSB_FIRST_UNIQUE"};
static constexpr SymbolTag<int> QSB_FIRST_COUNT{"QSB_FIRST_COUNT"};
} // namespace abi
} // namespace qsb_native
