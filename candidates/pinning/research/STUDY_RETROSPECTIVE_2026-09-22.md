# STUDY RETROSPECTIVE — el estudio comparativo completo QSB (2026-09-22)

Documento maestro del estudio pedido por el usuario: *"determinar de nuestros
procesos anteriores qué hizo que fuéramos creciendo, compararlos contra los
demás, y hacer una comparativa completa contra nuestro mejor… dejar de
adivinar y ver exactamente qué es lo que ha estado haciendo crecer el floor,
para atacarlo."*

> Si esto funciona, descubrimos oro. Si no funciona, también es oro.
> — el conocimiento es el activo; los resultados oficiales son solo su validación.

## 0. Índice de la evidencia

| Documento | Contenido | Origen |
|---|---|---|
| `OUR_TRAJECTORY_AUDIT.md` | Auditoría de nuestras 18 submissions (2 identidades): qué cambió, delta oficial, categoría de cada salto | agente 2 |
| `FIELD_CENSUS.md` | Censo de 777 submissions pinning + 538 subset: timeline completo del floor, tabla de frecuencia de mecanismos, perfiles de solver, análisis hits/s, cuantificación del ruido | agente 3 |
| `FRONTIER_ANATOMY.md` | Diff línea-por-línea del commit coronado `9f239c38` y del cluster 810M contra nuestro árbol v11 (`git fetch` por SHA — diffs exactos, no inferidos) | agente 1 |
| `COMPARATIVE_SYNTHESIS.md` | Síntesis + plan de ataque rankeado por EV | principal |
| `../ITERATIONS.md` | Registro por-iteración de cada submission propia | continuo |
| `../DEAD-ENDS.md` | Callejones falsificados con evidencia | continuo |

Datos brutos: `../../../allsubs_tmp.json` (777 subs pinning),
`../../../allsubs_subset_tmp.json` (538 subs subset) — dump de
`yukon submissions --all --json` ~2026-09-22/23, en la raíz de `qsb-pinning/`.

---

## 1. LO BUENO — qué realmente hizo crecer nuestro score

**Patrón sin excepciones en 18 submissions: cada salto real = adoptar o
componer trabajo público ya medido.** Ningún salto vino de un mecanismo
propio no medido.

| Evento | Resultado | Categoría |
|---|---|---|
| Corona `260879f4` → **705.67M PROMOTED** (única corona propia, era hybridnoise) | +2.8% sobre frontier previo | Composición de 2 mecanismos públicos pending |
| v6 → 760.58M | +5.2% | Rebase a tip público `03e399c` (CHAIN_PIPE propio aportó +0.8-1.4% self-rep — nuestro único mecanismo propio positivo medido) |
| v11 → **804.81M** (mejor oficial histórico) | +5.8% | ~100% el rebase a `0ace23d4` (809.95M público medido); nuestros 5 levers netearon −0.63% = ruido |

**Mecanismos propios que sí valen** (presentes en nuestro árbol, ausentes del
frontier coronado): `QSB_CHAIN_PIPE` (pipeline de tabla con prefetch a
registros muertos — la variante con rotación de buffers producía 48B/76B de
spills en sm_89; la variante prefetch-only cuesta +3 regs), `QSB_DEC_REP`,
`QSB_SUM_2U`, `QSB_PREP_MASK`, `QSB_TREE_FLAT`. Ninguno supera el ruido
individualmente, pero todos son gratis (124-126 regs, 0 spills).

**Lo que también fue oro:**

- La infraestructura de oráculos propia (`check_chain_pipe.py` etc.) — detecta
  regresiones semánticas sin gastar submissions oficiales. El extractor
  string/comment-aware (que sobrevive llaves dentro de strings PTX) es ahora
  patrón estándar.
- La disciplina de flags con kill-switch (`QSB_* =0` restaura el clásico) —
  permitió portar el mecanismo del frontier sin romper los paths validados.
- La mitigación de payout: statement público on-chain `SUB-QR2E70VJ` con la
  designación verbatim de wallet, autenticado por las credenciales Yukon —
  sustituto funcional del gist GitHub suspendido.
- La regla §5.9 (check de canales de mensajería en cada lanzamiento) — XMTP
  está muerto server-side (412) y lo sabemos con certeza.

## 2. LO MALO — fallos medidos (cada uno compró una lección)

| Fallo | Costo oficial | Lección comprada |
|---|---:|---|
| RAW_DIFF bundle | candidato inválido | La resta cruda mod 2^256 no es congruente mod p en borrow (error = offset K); el bundle además borró el `_ModMult(Q,R)` requerido. **Nunca bundle matemático sin oráculo.** |
| v12 init-cut bundle (`JIT_WARM`/`LAD_THREADS`/`SPOT_DEV`) | **−4.06%** (772.37M vs 804.81M) | `CUDA_MODULE_LOADING=EAGER` compila el módulo entero y bloquea el primer launch real — el warmup thread no puede esconder trabajo que está en el critical path. **Nunca 3 cambios sin aislar en un solo slot de validación.** |
| Interleave de issue-order (sm_89) | −27.2% / −44% en variantes | Rangos vivos extendidos + spills. Falsificado, en DEAD-ENDS. |
| first-fold de fkiene | −2.7%/−0.77% | Medido por el propio campo — a DEAD-ENDS sin gastar slot propio. |
| Mecanismos propios agregados | −0.3%, −1.24%, −1.4% | Micro-opts nunca superan el ruido σ≈2.6%. |
| Oráculos stale (check_fused_finish/fused_inverse/group_invert) | 0 (catch temprano) | Los oráculos de mecanismos borrados por el rebase 0ace23d4 fallan en extracción — pre-existente, documentado, no confundir con regresión. |

**La lección meta:** nuestros fracasos casi siempre fueron *bundles* —
varios cambios no aislados en un solo slot. El campo valida un mecanismo por
submission o porta verbatim; nosotros perdíamos slots mezclando.

## 3. La verdad sobre el frontier (el hallazgo que cambia la estrategia)

**El floor no subió por un mecanismo — subió por un draw.**

- `a671f274` (813.65M, ACCEPTED) es un **repackage byte-idéntico** de Ryun1
  `093fd97f` que solo midió 778.38M — swing de **+4.53% en bytes idénticos**.
- El solver coronado (`anamdongparkjinhyeong`) es un **bot de repackages**:
  62/65 submissions son "Public artifact submission" automatizados. Ganó
  re-tirando los dados sobre bytes ajenos.
- Nuestros bytes v11 (804.81M, 95.94 hits/s) son **genuinamente más rápidos
  que los bytes coronados** (~778-780M-class). La corona es suerte, no
  código.
- Ruido medido del runner: σ≈2.6%, rango −4.4%/+4.9% (49 pares). Winner's
  curse: re-medidas caen ~2.6-3.1% bajo el padre promovido.
- Score = hit-rate exacto: `verified_hits × 2²³ / elapsed`. Top-15 convergido
  en 95.9-97.0 hits/s (spread 1.14%) — la arquitectura está convergida, todo
  el margen está en draws y micro-levers.

**~60% de todo el crecimiento histórico del floor fue UN evento:**
nullforest8200 portó byte-exacto el end-state del dev-challenge (+395.4M).
El único otro salto real: ercumentyildirim +24.8M (stray-carry + SHORT_CARRY).

## 4. La estrategia implementada (y por qué)

**De adivinar → a portar lo medido.** v13 = unión de las dos líneas:

```
nuestro stack probado (CHAIN_PIPE, DEC_REP, SUM_2U, PREP_MASK, TREE_FLAT)
+ QSB_NEG_Y_MAC verbatim del frontier (negative_y_mac.cuh byte-idéntico,
  sha256 95732177…, 8 sitios, slope-swap en packed_finish preservando
  nuestros arms SUM_2U/PREP_MASK)
− QSB_X3_TAIL (skippeado: dead code en el frontier, +0.41% = varianza)
```

- Validación: sm_89 126 regs/0 spills; chain-pipe oracle PASS (1500 bitwise
  + 120 OpenSSL-affine bajo convención −y); suite completa PASS.
- Enviada: `a3749d4c` (validating). Esperado real ~806-810M; promoción
  depende del draw (~26%/slot según el modelo σ=2.6%).
- Nota pública con designación de wallet verbatim + referencia a
  SUB-QR2E70VJ (regla §5.9 cumplida).

**Por qué submitir aunque no "pone la barra alta" todavía:** el pool reparte
por promotion points, no winner-take-all. Un candidato validado sin enviar
es EV perdido. La barra alta es el *siguiente* escalón, no prerequisito.

## 5. El plan de "barra alta" (durable, no de draw)

Para que hasta un mal draw nuestro promocione y los repackagers queden
fuera (sus bytes más lentos necesitarían +4-5% de draw):

| Escalón | Lever | Estado | Est. |
|---|---|---|---|
| 1 | `QSB_NEG_Y_MAC` port | ✅ enviado en v13 | +0.21-0.36% (medido por donor) |
| 2 | fkiene rare-carry family (`RP_MUL_F8`/`RP_SQR_F8`/`MUL_SFQ` + `LAZY_ADD_FINISH`) | pendiente, fuentes `6206fb1d`/`3b67cf84` | +0.2-0.5% |
| 3 | **`_ModSqr`/`_ModSqrAddSub2` lane-form 64-bit** | **el único lever fuerte nunca medido por nadie** — el campo murió por ENOSPC, no por lentitud. Gate: ≤128 regs, 0 spills, sm_89 | +0.5-1.5% |
| — | i34-9 geometry (BATCH 16M/SLOTS 4/S2_BLOCKS 8) | skip — medido sin ganancia | 0 |
| — | más init work | skip — v12 dead-ended | − |

Nivel verdadero objetivo ≥ ~825M ⇒ mediano oficial supera el floor Y el
siguiente floor queda fuera del alcance de bytes ~778M-class incluso con
draws de +4%.

## 6. Meta-lecciones transferibles (más allá de QSB)

1. **En competencias con ruido > barra de mejora, el throughput real y la
   suerte de draw son variables separables.** Medir σ del sistema antes de
   atribuir deltas.
2. **El ganador visible puede ser un bot de repackages** — verificar siempre
   si el frontier es código o suerte antes de copiarlo.
3. **Portar verbatim > reinventar.** El mecanismo del frontier se copió
   byte-idéntico (hash verificado) en vez de reimplementarlo.
4. **Un slot de validación = una variable.** Bundles sin aislar queman slots
   y no enseñan nada (v12).
5. **Los oráculos propios pagan por sí mismos:** cada falsificación local es
   una submission oficial ahorrada (~2h de slot + ruido de interpretación).
6. **Documentar fallos con el mismo rigor que éxitos** (regla 2 de AGENTS) —
   DEAD-ENDS.md es lo que permite que la iteración N+1 no repita la N.
7. **La evidencia pública firmada sustituye canales rotos** — el statement
   on-chain cumple el binding solver→wallet que el gist suspendido no puede.

## 7a. Post-cierre (2026-09-23 00:40 UTC): escalón 2 enviado

- **v14 `9305981e` validating** — familia rare-carry de fkiene portada verbatim (4 flags).
- Verificación de expansión: 5 cuerpos asm == donor; flags-off == original (scripts en research/).
- sm_89: 124 regs (-2 vs v13), 0 spills; oráculos PASS. Esperado ~812-815M real.
- Escalón 3 pendiente: lane-form 64-bit `_ModSqr`/`_ModSqrAddSub2`.

## 7. Estado al cierre del documento

| Ítem | Estado |
|---|---|
| v13 `a3749d4c` (pinning, union) | **REJECTED 811,231,199** / 96.71 hits/s — +0.79% real sobre v11, #2 medido del campo; faltó +1.30% al floor (dentro de σ=2.6%) |
| subset v1 `7d345ff3` | validating (1ª submission propia del track) |
| Frontier pinning | 813,651,852 (a671f274); floor ~821.79M |
| Frontier subset | 623,518,629 |
| GitHub ItlaStudent | suspendido; appeal #4775447 respondido por usuario 09-21, en revisión |
| Statement requester | SUB-QR2E70VJ on-chain, tx 0x330a9b80, 0 USDC |
| Wallet designada | 0x4D2a5410f0d0733c0448E91E47608ebdC918A806 (verbatim en todas las notas) |
| Próximo trabajo | escalones 2-3 de §5 + veredictos |
