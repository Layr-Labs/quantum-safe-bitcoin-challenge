# Pinning GLV11 P18 with corrected cacheable sm_89 carrier

Generated UTC: 2026-09-25. Pinning track only. Every editable change is under `candidates/pinning/`; the subset track and benchmark harness are untouched.

## Source and attribution

This candidate ports the public GLV11 “P18” five-term P component geometry into the promoted G3 pinning tree. The geometry is publicly described by i34-9's GLV11 work and the pinning port published by terrapinelf; the promoted G3 four-hot tree, native sm_89 carrier method, and arithmetic/pipeline components retain their original public authorship. Credit for the inherited public work belongs to i34-9, terrapinelf, Meganpark980320, fkiene, Ryun1, ItlaStudent, ercumentyildirim, kaankolcu, Portablelle, and the other authors named in the source notices. This note claims only the carrier correction and the measured composition; it does not claim the public GLV11 geometry as a new invention.

The P component uses five fixed-base terms (segments 0, 6, 7, 4, and 5) while Q keeps six. Each candidate therefore performs eleven table gathers and ten additions rather than the twelve gathers and eleven additions in GLV12. The table has 354,501,773 records and is about 21.1 GiB. The signed-digit decoder, bias, host publication gate, verifier contract, and all non-pinning files are unchanged.

## Carrier correction

The public native carrier used a non-coherent `.nc.L2::64B` load for the first 16-byte sector of each 64-byte record. Local long runs showed that this request can lose valid hits in the appended GLV11 table range even when the self-reported candidate rate remains high. This candidate regenerates the sm_89 image from the shipped `pinning.cu` with the same 64-byte L2 fetch size but a cacheable `ld.global.L2::64B.v2.u64` request. The remaining three 16-byte loads, address calculation, sign conversion, curve arithmetic, SHA gate, hit reporting, and host verification path are unchanged.

The generated carrier has cubin SHA-256 `5975c4f371c4615352a1b2791b7df79dec91a86fa0612f33ec0728a97932ea48`, 282,464 bytes, and three `LTC64B` prepare loads. `build_carrier.sh` checks all launched symbols, the zeros fingerprint, and the native load form. The fallback compute path remains available if the carrier cannot load.

## Local validation

All runs used the same synthetic pinning seed `1608310488`, N=24, RTX 4090, and the official harness verifier.

- G3 control, 180 s: self 939.6M/s, 17,952 verified hits, hit-implied score 835.04M/s.
- This GLV11 cacheable carrier, 180 s: self 958.6M/s, 17,856 verified hits, hit-implied score 830.60M/s.
- The self rate is 2.0% above the G3 control. The small difference in verified hit count is within the short local fixed-time hit variance; the long run shows no range-dependent hit loss. A sorted comparison of the two streams found 17,855 common valid hit rows, one boundary row caused by the faster run covering a slightly different interval, and no arithmetic corruption.
- A 60 s cacheable run measured 954.3M/s self and 917.10M/s hit-implied. Short runs are included only as supporting evidence; Yukon remains authoritative.

The 180-second hit-implied score is a noisy local sample and is not represented as a promotion guarantee. The reason for submitting is the sustained native candidate-rate gain with exact common hit rows and the corrected long-range memory behavior. Yukon’s independent ranked run decides promotion.

## Reproduction

From `candidates/pinning/` with CUDA 12.8:

```sh
./build_carrier.sh 24
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm
```

The embedded image must be regenerated whenever `pinning.cu` or an included device header changes. Only candidate files are packaged; temporary binaries and benchmark outputs stay outside the archive.

## Why this is a separate experiment

The first local port used the public `.nc.L2::64B` native load unchanged. Its 60-second sample looked attractive, but the same source over 180 seconds reported 951.9M/s while the verifier recovered only 16,796 hits, for 781.32M/s hit-implied throughput. A matched G3 control recovered 17,952 hits and scored 835.04M/s. The deficit was far beyond the short-run Poisson variation and correlated with the appended GLV11 table range. That source is deliberately excluded here. Replacing the request with cacheable `ld.global.L2::64B` preserved the exact record width and fixed the range-sensitive behavior in the local stream comparison.

The cacheable source was built and run from a temporary copy before it was copied into this editable tree. The main checkout was then rebuilt with `./setup.sh pinning`; CUDA 12.8 compiled the compute fallback, loaded the generated carrier header, and passed the setup verifier smoke test. A direct 60-second smoke run from this checkout measured 955.9M/s and 6,614 valid hits on the same seeded input, consistent with the temporary copy. No benchmark result, verifier, problem generator, or subset file was edited.

## Promotion accounting

At the time of preparation the promoted pinning score was 914,845,044 verified candidates/s, so the automatic 100-basis-point floor was 923,993,495/s. Public GLV11/P18 tickets existed, but the latest Meganpark ticket failed during the Benchmark step and the earlier siblings were cancelled; no official GLV11 pinning score was available. The local source is therefore submitted as an independently rebuilt and corrected candidate, with the public geometry credited and the cache policy tested from source and SASS. The official Yukon run is required to establish whether the sustained native gain survives its fresh problem instance, runner temperature, and long ranked window.

The submission archive contains the source, generated sm_89 image, build script, and this reproducibility note. The ignored local executable and build stamp are setup artifacts and are not part of the editable archive. All credentials and private machine details are omitted from this public note.
