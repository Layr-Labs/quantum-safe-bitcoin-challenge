# Shared pre-inverse function for paired epochs

Parent e21661fcdc3ea76ca6224a819a5e636dd3e62564. Both epochs now call one noinline pre-inverse function. Its existing arithmetic helper is unchanged; eight scalar reference-point words enter by value, and twelve result words plus the usable flag return by value. Caller output arrays never escape. The denominator top word is explicitlyzero; only its four field limbs enter the tree. Each epoch retains its original active and odd-tail guards.

Complete point replay, canonical field operations, normalized6 bounded inverse, compact tree storage/barriers, producer and finite mapping are byte-identical to the audited parent. All attribution remains. This change targets duplicated emitted code; no source operation or correctness check is removed. Actual default-device integration and finite output checks plus same-host timing are required. No measured speedup is claimed.
