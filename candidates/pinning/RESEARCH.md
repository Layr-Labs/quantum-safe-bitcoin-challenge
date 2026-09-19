# Pinning: complete final fold and independent host reports

This change improves the public Yukon QSB pinning implementation on organizer-generated inputs. The organizer generator, verifier, search domain and scoring contract are unchanged.

The CUDA point-chain multiply and square retain the second fold's final carry and add carry*(2^32+977) through three 32-bit limbs. Recovery multiplication and the fused square retain their existing correction. For B=2^256, K=2^32+977 and p=B-K, a product a*b=L+B*H has first fold F=L+K*H<(K+1)*B. The next fold T=(F mod B)+K*floor(F/B) is below B+K^2. On overflow, T-B+K<K^2+K<2^65, so three limbs capture the complete correction. The known primitive example a=b=p-65537 produces 0x100020001. It is an arithmetic-domain test, not a public whole-program exclusion.

Each of three streams owns separate reports, midstate uploads, events and metadata for two parities. The next batch is queued before the older parity is read. An armed parity cannot be reused; stream ordering protects device working storage, and sequence boundaries drain every remaining report. Checked persistent output and the complete table-sign guard/fallback are retained. PIPELINE-LIFETIME.md describes the lifetime argument.

Exact native evidence passes 2,444 full-width primitive pairs with alias checks, 14,436 loader cases, 66,080 affine-point comparisons and eight partial-pipeline modes including a 10ms delayed CPU read. The packaged audit_final_fold.cu / .py additionally passes 4,377 input pairs and 21,885 outputs. Original source/input/binary hashes and raw arithmetic outputs were independently rechecked locally. The final-fold, fused-field and loader auditors are included for reproduction. The affine-point and partial-pipeline sources, inputs and raw outputs are retained in the separately recorded native-table-guard-complete-fold-0240 evidence.

A fresh-seed repeated long comparison against then-pending public leader 60f37bbd measured 741.632/741.897 M/s for this source and 736.727/737.187 M/s for the peer. The mean gain is 0.652322%; minimum observed separation is 0.602945%. Every one of 12,058 reported tuples per invocation was independently verified. The accepted frontier and other current queued sources were measured in complete mirrored short cohorts. A next-table-load variant lost 2.297% to this source and was not adopted.

The unchanged production wrapper completed 1,200.142 artifact seconds on fresh generated seed 743178005. All 105,476 reported hits passed independent OpenSSL checks on the Pod and again locally. This is a full correctness qualification, not an official ranked result. All GPU execution used the shared pinning-b lease with CUDA 12.8.93 on RTX 4090. The full-run hit estimate divided by the outer wall time is about 736.346 M/s, a local measurement rather than an official score. The artifact's 761.6 M/s is a peak-progress advisory value.

Runtime identity and measured evidence are in SOURCE-MANIFEST.json. The live source-bound gate refreshes all pending public branches before upload. The late 54aacac8 source was measured separately at 718.077 M/s versus this candidate at 745.352 M/s. Its entire 12-invocation cohort reproduced all 858 independently verified tuples. A further six-arm cohort measured new arithmetic peer 327b18d0 at 715.375 M/s and this source at 745.950 M/s, with all 16 invocations reproducing 858 verified tuples. The exact ff1c44d1 source averaged 716.610 M/s versus this source at 744.970 M/s; all 12 invocations reproduced 858 verified tuples. A subsequent fresh four-arm long comparison passed all 11,679 tuples per invocation and measured this source at 741.021 M/s versus the exact frontier at 717.209 M/s (+3.320058% mean, +2.913429% minimum observed). Sources 54aac and fc2f were slower in both directions. A further fresh long cohort measured this source at 742.320 M/s, 3c21 at 713.792 M/s and aff38 at 719.943 M/s, with all 11,910 tuples per invocation independently verified; no automatic source alias is used. Local comparisons do not guarantee official promotion.


A later exact-source screen included the direct-digit implementation 96d8a95c
(runtime digest cf6c251a56c389282b013fd8b475a3f1928e038a825c4e27137b311314998ed5).
Its direct-digit path is enabled by default. This candidate averaged
746.433090M/s and that peer averaged 707.143249M/s in the matched screen.
The frontier and aff38 source were included in the same cohort; all twelve
invocations reproduced the same 858 independently verified tuples.
The new source was strictly below an already long-tested public comparator
in both directions, so the conditional plan selected no additional long run.
