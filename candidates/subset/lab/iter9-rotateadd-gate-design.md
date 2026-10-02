# Isolated next extension of the positive scoped SHA lead

EnableQSB_SHA_LEA_GATE1 AND SHA_LEA1,PARTS0, all other current production knobs.
Only ordinary QSB_RL uses already-reviewed upstream QSB_LEA_RLA. The generic
GPUHash port remains on. No specialized IV round1 or last round63, no constant-d
round2/3, no FMA form, no LEA_PARTS behavior is ported. The old round computes
T1=h+S1(e)+Ch+kw,T2=S0(a)+Maj, outputs d+T1 and T1+T2. RLA computes
T=h+kw+Ch+Maj+S1(e), outputs d+(T-Maj) and T+S0(a): exact same32bit words.
Nothing reads changedh before the assembly has consumed inputs. Fingerprint
optional enabled-only gatekey; OFF retains measured SHA1image82b9868e...
and SHA0qualifiedimage9c9aab2... . No default change.

First measured before/after must compare scopedSHA1 as control, not SHA0, to
isolate ordinarygate effect. ExistingN24/120s benchmark under sharedGPUlock;
existing verifier unchanged. Directpromoted gate only if positive extension.

Build passed1604. OFF ordinary-gate extension matches scoped SHA native image
82b9868e... byte-for-byte; extensionONc3f6caec... . Digest14256->14128(-128),
LEA.HI+244,SHF64-732,IADD3-26,IMAD.IADD-98. Native128registers24KiB0spills.
No score yet at this checkpoint, don't infer throughput from SASS change.

## Actual exact-screen outcome

1604completedPASSboth: scopedA401.315271M/s5797/5797hits vs ordinarygateextension
B359.765211M/s5206/5206hits,-10.353471%. One screen is thermally interleaved with
otherlockedgates,notconfirmedregression;neverthelessstrongnegative andnoadoption.
Despite128fewerinstructions andnoresourcechange,thisextensionisnotnextbest.
KeepordinarygateOFF. ScopedGPUHashleadcontinues1602qualificationunchanged.
