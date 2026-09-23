# Subset: updated paired first-state reads

## Package identity

This package supplies the subset candidate at implementation revision `495e12254bcc153fc5951862d45af73ca31dd0e7`. It contains the grouped fixed-base table path, ranked chain and exact-replay integration, a separate offset-form table for the ranked filter path, and the current scalar and parity helpers with a register seed handoff in the ranked grouped-chain source. The package retains the host window layout helper, schedule storage declaration, root-result handoff, and root sign and canonical-output source. This revision updates the paired first-state reads. The source inventory below identifies every submitted file.

This package does not edit the benchmark, setup script, scoring code, problem generator, workflow, repository verifier, or the pinning candidate. The executable entry point remains `candidates/subset/subset.cu`. CUDA, OpenSSL, and the math-library link surface remain the ordinary repository dependencies.

## Source ancestry and attribution

The current reviewed frontier is `b59484345df5208f5caffc82c25a4a3b50cbe523`. Its subset candidate and harness trees are identical to the historical base `9f239c386c7e99f8815103d9c6cc4465d7c5a9ba`; this package retains the existing subset implementation lineage and uses the current frontier as its package base.

The public GLV scalar and grouped fixed-base donor is may93182's unpromoted source commit `4b77964fdd47499644c44a15e717a70b4a208b41`, associated with submission `5ffaef34-a958-4e80-887d-893f6b933605`. That donor supplied the original scalar helper and table structure adapted by this package. The implementation includes the later grouped-chain integration and exact-replay wiring recorded in the implementation lineage.

The offset-filter table source is adapted from kayu052's public unpromoted commit `7ae0542ad3e33f80cfff2c76d4b0e1181aa7c094`, associated with submission `41c3490e-bc19-4306-b914-5e6171453043`. The package retains a raw table for exact replay and a separately prepared table for ranked filtering. Ranked digest launch sites use the filter table; exact verification launch sites use the raw table.

The inherited pipeline includes source associated with Akashneelesh, Saviour1001, and terrapinelf. These substantially reused unpromoted contributions are credited alongside may93182 and kayu052. The guarded parity-helper adaptation credits Portablelle's public PR965 contribution, as preserved in the donor source. Existing copyright and license files are retained. `COPYING-secp256k1` preserves the supplemental upstream secp256k1 notice, and `COPYING` preserves the repository license notice.

The scalar seed handoff adapts the corresponding portion of DrCleverHans’s unpromoted public submission `a149d7b4-4ddf-408c-b6c1-1f4e15bfd41d`, published in PR #1164 at source revision `c3ac7754a0050e22d661e69969d3968ad57d8061`. DrCleverHans and fkiene are credited as coauthors for the donor contribution. Other changes in that donor package are not included here.

The paired first-state consumer adapts the corresponding read block from DrCleverHans’s public PR #1189, submission `2edbddb8-6934-4aa5-9e2f-b2e991b48080`, at source revision `08e961f7075ed7ca456bd810820fe3963d099b98`. This package includes its four aligned vector reads in the paired consumer. DrCleverHans remains credited as a coauthor. The donor’s class-table packing and additional selector scaffolding are not included.

## Additional source update

The candidate also carries the guarded paired-consumer update in `candidates/subset/tests/gpu_epochs/tree.cu` and `candidates/subset/tests/gpu_epochs/pair_shared.cuh`. The existing paired first-state read update and all retained source inventory remain in the package. The exact replay path, verifier interface, problem input contract and sibling pinning directory remain unchanged.

## Changed and retained source surface

| Item | Package content |
| --- | --- |
| Track | `subset` |
| Current reviewed frontier | `b59484345df5208f5caffc82c25a4a3b50cbe523` |
| Historical subset base | `9f239c386c7e99f8815103d9c6cc4465d7c5a9ba` |
| Implementation commit | `495e12254bcc153fc5951862d45af73ca31dd0e7` |
| Direct implementation parent | `38fd55e9b167156bf5e3f09e6a7404b2aa514ed7` |
| Grouped fixed-base integration | `f93b62761c5d95ec955cb22c5c636db0ca83f1e6` |
| Exact residual helper revision | `bac6abd9a139ad7e5bf2a35eb13e1d040367b19a` |
| Candidate entry point | `candidates/subset/subset.cu` |
| Scalar helper | `candidates/subset/GLVScalar.cuh` |
| Grouped-chain source | `candidates/subset/glv14_chain.cuh` |
| Table geometry source | `candidates/subset/glv14_geometry.cuh` |
| Host table builder | `candidates/subset/glv14_host.cuh` |
| Ranked CUDA source | `candidates/subset/tests/gpu_epochs/tree.cu` |
| Exact replay source | `candidates/subset/tests/gpu_epochs/tree.cu` and included headers |
| Host window helper | `candidates/subset/tests/gpu_epochs/window_lane_pack.h` |
| Root inverse source | `candidates/subset/tests/gpu_epochs/zinv32.cuh` |
| Schedule source | `candidates/subset/tests/gpu_epochs/window_schedule_shared.cuh` |
| Source manifest | `candidates/subset/SOURCE-MANIFEST.json` |
| Sibling candidate directories | Unmodified by this package |
| Organizer-owned files | Unmodified by this package |

The grouped fixed-base implementation replaces the inherited scalar table geometry and associated fixed-base chain source. Other inherited subset mechanisms remain represented by their submitted source files. The source manifest records the package lineage and byte-for-byte inventory. The note and manifest are packaging metadata and are excluded from the manifest's self-referential hash list.

## Standard build and repository entry points

The package uses the repository's standard subset build surface:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o candidates/subset/subset candidates/subset/subset.cu -lcrypto -lm
```

The ordinary repository commands remain:

```sh
./setup.sh subset
./benchmark.sh subset
```

The organizer supplies the trusted problem, benchmark process, and score calculation. The archive requires no local problem, score file, recorded hit set, precompiled device binary, or private experiment log.

## Packaged source inventory

The following table lists every tracked file in `candidates/subset` except the two packaging metadata files. Paths are relative to the repository root. The table identifies package contents; it does not state that every inherited file changed in this candidate.

| SHA-256 | Bytes | Mode | Path |
| --- | ---: | --- | --- |
| `ea7c302a47588f00406e8bc66d567f3d4f734882d3dfdf73e818b28a365f4ebd` | 617 | `100644` | `candidates/subset/ASMLAST511-RESEARCH.md` |
| `c3ba57bdf16ea0275b68e7bcd912f402b7acb6c9919aa6ae1336f3efaf5c8384` | 3395 | `100644` | `candidates/subset/BY_NORMALIZED6.md` |
| `7c3f88d6a29938e932e4c39cd56f6e409e3074b6dbacce5b66c6dc0909f710e3` | 2275 | `100644` | `candidates/subset/CANONICAL-ADD.md` |
| `d814443627bb374c09162cab0065f93880232facbaed3f008434c7b80e3c2dba` | 1893 | `100644` | `candidates/subset/CHAIN-REPLAY.md` |
| `095d328d7c9d2debe51378706356d9ba3f87a28dcd388ecabb0a50ccf01eb500` | 2534 | `100644` | `candidates/subset/COMPLETE-POINT.md` |
| `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` | 35149 | `100644` | `candidates/subset/COPYING` |
| `a735999c7e5649df6fcda6fb06ab97435851c392b1b93494ae8725f37441632f` | 1057 | `100644` | `candidates/subset/COPYING-secp256k1` |
| `c587e6cd72a528c73cdcca002c86ede6f4ab455e653f80604569ce137c4696ae` | 20416 | `100644` | `candidates/subset/GLVScalar.cuh` |
| `c8415e1ddc839e078421a1ee347a9db5036f84714b03edb700aa2e635e05fedd` | 35134 | `100644` | `candidates/subset/GPUHash.h` |
| `3205120d961990dc55e7e3c622411c2c581d506fa730f568a122530038b84ba6` | 59571 | `100644` | `candidates/subset/GPUMath.h` |
| `1951a41236e8f3bb6b28a435bb4ee1c570dd243829a16e09d06829951a8424ad` | 1763 | `100644` | `candidates/subset/HIT-CHECK.md` |
| `6418806393eb4957ec2efa9539ee65450bc7a98e9888501d99ab64fafe29c7e6` | 875 | `100644` | `candidates/subset/LAST511-RESEARCH.md` |
| `8fa5b2e2ae22ae35d4e7a5503e353ff8875264f5fda22fd82568d511bb98e494` | 1728 | `100644` | `candidates/subset/PAIR-CURRENT.md` |
| `3648d13065e9382736e0402957f834c448ebb3e86ae3cd2c197d1faf370f4254` | 925 | `100644` | `candidates/subset/PAIR-FRONT.md` |
| `4c78fed13f9fbe97a8476204c7f03b8fabfb7f4709f55b2ca455aeadccf445fa` | 1250 | `100644` | `candidates/subset/POINT-BY-TABLE.md` |
| `306bc77ee79a1d8a283477af30c9ace31d446ce33ab8bcc535dac4d6c447e286` | 2697 | `100644` | `candidates/subset/POINT-PREDICATE.md` |
| `4424c2217c88fff84f1a8e36b3267cc0e6faa10bac189b627c3fec9be8361bc0` | 2648 | `100644` | `candidates/subset/POINT-X3.md` |
| `0b1798a37a17d8e882604784a375068bc1bc1bd5b4ae98da8f903d876961cdab` | 14433 | `100644` | `candidates/subset/TREE_INVERSE.md` |
| `7390f456f1f8c4cdbf0df2d9e42e8f88935b09e26e1f0be6af6ddbfa59ce4bd9` | 115980 | `100644` | `candidates/subset/chain_replay_field.cuh` |
| `095511b9383f9da6ae20aed7afd994e33ef2803399696dfab2000c1cf2ba8526` | 11565 | `100644` | `candidates/subset/glv14_chain.cuh` |
| `2029eb754de97d1f6596277a4983069a35a77cf5712d8f28ab9375cd15b7e87d` | 690 | `100644` | `candidates/subset/glv14_geometry.cuh` |
| `1589fd7899d9670917f397e988257aff7ef76d9b5f6207ea498fc77b9969175e` | 6782 | `100644` | `candidates/subset/glv14_host.cuh` |
| `1efd238ee80935f8a7c0559eafce15e7d1e5a813a04173efd54e0f87074e3a45` | 117328 | `100644` | `candidates/subset/hit_filter_field.cuh` |
| `4dd068a4f9fa286c19858c7c4d931d98e5f8ff0eab73b9226ac5f0ef1dbd5f50` | 209560 | `100644` | `candidates/subset/hit_filter_field_sc.cuh` |
| `a973d1dd58bf760ce5cbff79e69cbe8cd725d8ad1a642741e936fb2d948be771` | 9893 | `100644` | `candidates/subset/square32.cuh` |
| `103101573be38d528994aa8c35c35a06ae1a23b0d2d856dcfd5bc606392976a9` | 159 | `100644` | `candidates/subset/subset.cu` |
| `5b498531bb83c83e9faa4ca233364a3bf736632ec47e6611b1c6585645f77eab` | 2123 | `100644` | `candidates/subset/test_leader_composite.py` |
| `940cb53feb731f966adb79d08c415c51e119c0f544ac26944bc580c2f287df4e` | 2151 | `100644` | `candidates/subset/tests/gpu_epochs/by_table_matrix_audit.cu` |
| `fd0bb33786a042b6e830428cb464a31e7804a83042bfbb4b92e555d73208728e` | 1885 | `100644` | `candidates/subset/tests/gpu_epochs/canonical_add_audit.cu` |
| `dd1e7aa486a4895e431ccefbabc064db8646bc2a202a83410f4850210f910a00` | 3471 | `100644` | `candidates/subset/tests/gpu_epochs/chain_replay_audit.cu` |
| `9b4a85157256cb32170ad26ab94ec09a80706b9219e956a82d4c83f40c02e69b` | 2371 | `100644` | `candidates/subset/tests/gpu_epochs/dirdig_audit.cu` |
| `ae0b03eae7d0c8036020a3af0d7965530372789ed4ee4bdbde7cc44be22a5c10` | 11140 | `100644` | `candidates/subset/tests/gpu_epochs/epoch_groups.cuh` |
| `890752b2dfb632e56232c1a148a4842354e461f04d23628b1581d1413e7471fc` | 4709 | `100644` | `candidates/subset/tests/gpu_epochs/filter_tail_sc.cuh` |
| `c9405ec70d3dc3f2572a10c5831dd1137179b8182a3983009dbc660d54e472de` | 3505 | `100644` | `candidates/subset/tests/gpu_epochs/first_stage_audit.cu` |
| `59cb33f81db4605bde47377fd8b8381ea3102479df68c3223e83eca6427d9bc8` | 2536 | `100644` | `candidates/subset/tests/gpu_epochs/hm39_divstep.cuh` |
| `ab3856bd2a3f59ead639571da34d6604882e4e864b35e57e67b676b0ff1ad474` | 2386 | `100644` | `candidates/subset/tests/gpu_epochs/hm39_pair_inverse.cuh` |
| `4a238168917e34084db4a9f2372f47b947a62cfcec89c66f4fc830300f3f7a15` | 3033 | `100644` | `candidates/subset/tests/gpu_epochs/hm41_quad_inverse.cuh` |
| `a1f5040c6f2ca7eab6e1719c0283ba2fca4849fffe87bbf1bea20cb164b2ee2f` | 7825 | `100644` | `candidates/subset/tests/gpu_epochs/hm43_warp_inverse.cuh` |
| `ce76d53a0ae76bfaaca1bbd12ff87d7f268ca7dea187c1f733da14cebb9b261a` | 5946 | `100644` | `candidates/subset/tests/gpu_epochs/pair_finish_audit.cu` |
| `dd21dd449799659a2d08120427c40a15ad29f2bf223325a971bd141f59195146` | 22160 | `100644` | `candidates/subset/tests/gpu_epochs/pair_shared.cuh` |
| `482e6e05ff134a82b48f32b27b32524cb7139c7daa0f0d877a9512315c021e0c` | 9168 | `100644` | `candidates/subset/tests/gpu_epochs/parity_window_subset.cuh` |
| `abc998ea5fc4f2180fc4690b209fc6ecb27a1189a8f884ead4607f06e82b315e` | 5465 | `100644` | `candidates/subset/tests/gpu_epochs/point_audit.cu` |
| `b8427c62a23c55070839548f28ce89c8309a47a6dc38d33ba594d3a120cc5c63` | 6256 | `100644` | `candidates/subset/tests/gpu_epochs/point_predicate_audit.cu` |
| `8fbcb6bfa47ba4cdf79a4c65d4510b0cd1a2c7c57f9bd800e89c15dff1c43cb8` | 3864 | `100644` | `candidates/subset/tests/gpu_epochs/prefix_cache.cuh` |
| `439f21c09c9b4fbf8e199eece7b439774d701f26c955544f9de8f3f7156966c4` | 2360 | `100644` | `candidates/subset/tests/gpu_epochs/scalar_audit.cu` |
| `a8129be10bcf4435bdfb4fd34c1939a8f7c68a5b98429fd58b0dcb93e23f0764` | 2533 | `100644` | `candidates/subset/tests/gpu_epochs/seed_x3_audit.cu` |
| `05bb34dc57c3acbf6b33a437c0febfee0cdb4d1dfc25e6a106f689434158d966` | 212256 | `100644` | `candidates/subset/tests/gpu_epochs/tree.cu` |
| `f839ab3f24dca4e49ac4670044b6a2eaeabb2ea0a7e74741df8408ef7104d2de` | 7968 | `100644` | `candidates/subset/tests/gpu_epochs/tree_audit.cu` |
| `c84ddd75164b90a553533813f5e77296ec9e1cf74cab97b0fe0882bb3f72b0b5` | 12866 | `100644` | `candidates/subset/tests/gpu_epochs/tree_inverse.cuh` |
| `84ff818c25fadf694b175879dd0d7b0c358b55501b0128e86cd8234862ed8d6d` | 2262 | `100644` | `candidates/subset/tests/gpu_epochs/window_lane_pack.h` |
| `561c0819884bb0dff88d19668b82d6aa9b910e98546cf31618707221fd63c217` | 13270 | `100644` | `candidates/subset/tests/gpu_epochs/window_schedule_shared.cuh` |
| `f2146371b5e2cb6fe291491f28d85dc3724ca96642b5352607f563278a1e982f` | 30434 | `100644` | `candidates/subset/tests/gpu_epochs/zinv32.cuh` |

The authoritative implementation is the source in this package. Retained research files, audit sources, and test helpers are inventory items; their presence does not alter the organizer's benchmark or scoring contract.

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 160 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*