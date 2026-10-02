# Iteration 5: extend qualified launch-fill direction to launch8

Qualified baseline1048576 blocks,128lane,2epochs/thread; probe2097152blocks.
No digest/arithmetic/geometry changes, no centered square. Previously launch4
passed explicit promoted-source3pair gate; full denominator relocation qualified
above it. This is the next grid-fill point, not a retry of fixed failed knob.
Buffers:4194304epochs per stream; firststate16slots*8u32=512bytes ->2GiB per
stream4GiB both, plus descriptors/group indices. Native ranked table unchanged.
Tag index up to4194304*128=2^29<2^30, batch_pos268435456fits signedint;
all per-launch descriptor/firststate offsets are size_t or validunsigned.
Fixed-time drain overhead/yield and VRAM pressure are risks, not gain claims.
Probe requires unchanged exact benchmark; 120s screening pair initially only.

Local screened PASS both376.295943 vs364.053370M/s,+3.362851%. Rejected for
ranked24GiB before any submit: table22688113472+bothfirststates4294967296=
26983080768bytes >25769803776. Even perfectzero-groupallocationcannotfit.
No native image build/confirmation spending for an unrankable grid.
