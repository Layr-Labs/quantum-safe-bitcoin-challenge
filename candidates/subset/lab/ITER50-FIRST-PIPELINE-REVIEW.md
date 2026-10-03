# Iter50: first-state producer/consumer isolation

Following the digest phase census, test reuse of eight first SHA states per epoch.
All performance evidence uses the existing N24 benchmark verifier, synthetic seed
1789110211, 120-second fixed windows, graceful SIGTERM + complete drain. Hardware
is one RTX3090, not the benchmark JSON metadata RTX4090. No score native prediction.

## Preregistered distinct hypotheses and screens

1. FIRST_LOCAL: sixteen SHA compressions per paired CTA in512Bexistingparking;
   two fullCTA barriers publish/retire before reuse. Hostcompute remains.
   388.675150 ->365.742002 M/s (-5.900338%), exactPASS10935hits. Disabled.
2. FIRST_DEVICE: densely build globalstates once per batch, skip512MiBfirstupload;
   digest and ALLother kernelinstructions identicalsm86/sm89. Hostcompute remains.
   387.229028 ->384.217482 M/s (-0.7777%), exactPASS11211hits. Disabled.
3. FIRST_HOSTLESS+DEVICE: omit now-unused CPUfirstcompute andsteady pinnedfirstarray;
   descriptor ring640->128MiBper slot; independentbatch0checker unchanged fullCPU
   andGPU2,097,152descriptors+16,777,216firststates.Screen371.196026->372.622976
   M/s (+0.3844195%), exactPASS10779hits. Too small to claim repeatability.

## Compiler-route exclusion

Initial screens all comparecompute52PTX/JIT arms (the harness fixedbuild).
Retained actualpromotedleaderfaf5422a/efef868 binary is native86. Initial ninearm
alternation therefore mixedroutes: numerical+5.239756%leader,+1.212369%production,
48,336exacthits, EXCLUDED as qualification. Found byfatbin headers/buildlogs before
acceptance. Corrective matched-native-sm86ninearms preregistered, same retained
leader hash, three runs per arm, L/H/P,P/H/L,L/H/P. Gate>=4%leaderANDpositive
productionincrementalityANDexactverification. No submission pendingmatchedoutcome.

## Safety

CTA-local states map16writers to128distinctwords, exactsameSHAtransform/epoch mid
andclass words. MissingB aliasesA; invalidAaliases0; unchangedactivemasks.
Readerretirebarrier precedesfrontparking. Descriptors copiedbeforedeviceprepass
on same stream. copiedevent fences sourcepinnedreuse, device slotcompletion
fencesd_ep/d_fi reuse. Hostless nullablefi skipsfirstwork onlyafterdescriptoremitted
inbothSHA-NI/OpenSSLtails. Checkerfi nonnull, checkerfailure GPUfallback intact.
Off86/89cubins byteexact production; hostlessdeviceimage byteexactfirstprepass.
Native/local digest128regs24KiBsharedzero stack/spills.

## Package transport

Lossless nativecarrier LZ4HC473504->174909B; ownboundedhostdecoder, no runnerlib.
Original image SHA a724399a04db515f8bebd836b1af866e6e7a1b778a98ce73a384c6f7704de147.
Actualproductiondecoder and buildscriptregeneration restorebyteexactimage.
Buildscriptoptionaldevelopmentliblz4 only; otherwise emits legacyrawbase64.
Initial standaloneheader testfailed staleb64linecount thenfixed1944, decoderexactPASS.
Regeneration needed explicitcuobjdump/nvdisasm PATH on thisdevmachine.
Originalheaders/buildlogs/compilerscreens retainedlosslessly. This is packaging,
not cryptographic verification or measured throughput benefit.

## Next

Matchedqualification governs disposition, never mixedroutes. Ifhostlessnonpositive
or gatefailure, keep allswitchesoff and stopthispipeline axis; gather-versus-field
front isolation remains nextstrong distinctexperiment. No noise-rescue rerun.

## Matched qualification: local gate PASS

L/H/P native86 means354.856128/375.346464/356.215766M/s; leaderdelta+5.774266%,
prod+5.370536%, exactPASS47123hits. Pairleader+6.005354/+4.698640/+6.637861%;
prod+2.181795/+2.856576/+11.589193%. Thirdcontroldropped to336.055682, so
meanincrement is not a stablecost estimate; firsttwo positive. Candidate375.0-375.7
M/s. Allthree leadergates>=4%. EnableDEVICE/HOSTLESSonly;LOCALremains0. Board
recheckedfaf5422a753.571538M/s. Native4090performance unknown; submitperorders.
