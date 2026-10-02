# Unsigned bias continuation: diagnostic-backed emitter change

Signed fuse screened -1.99%. Same-tool cuobjdump counts14512vs14520default
(-8), not+9againstthe inconsistent prior14503count. Unsigned14504(-16). Avoid its signed
high word using host precomputed 3p-c² = low+high*2^256. Square's first-fold z
then adds this constant, subtracts p1, and uses the ORIGINAL unsigned second
fold with a carry ending at z2. Since p1<2^256 and c²<p,
3p-c²-p1 > p-(2^32+977) > 0; the unbiased signed high cannot be negative.
Using2p would NOT establish this for noncanonical p1. high fits2bits (1or2).

IMPORTANT: z8's carry out is deliberately dropped, matching inherited unsigned
split3p-X3. It is a new rare carry hazard, not proof of exact arithmetic. Existing
full benchmark's successful-hit checks and scored throughput decide adoption.
Default0 and mode1 constants remain eightwords byte-identical; mode2 expands
QSB_U2R_C to9words, uploaded by checked BN operations. All allocations/event
handling/cleanup unchanged. Unsupported exact diagnostic route explicitly fails
compilation rather than accidentally interpreting biased lowword as c².
BNencoding errors keep existing fatal startup behavior; no success is reported.
No synthetic oracle is substituted for the existing benchmark.
