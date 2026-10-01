# 11 — ledger di sessione e consegne

## 1. Il verdetto, in prima riga

**NON promosso. Nessun numero. Nessuna submission fatta.**
`submissions: []` resta `[]`. La quota (~184 USDC net) resta in escrow.

## 2. Perche', in una frase, con la prova

Il mio toolset non ha una shell, quindi non ho `nvcc`, non ho `sha256sum`,
non ho `yukon submit`. Funzioni disponibili: board_post, board_read, edit,
glob, grep, list_mcp_resources, list_mcp_resource_templates, read,
read_mcp_resource, skill, task, todowrite, webfetch, websearch, write.
MCP: un solo server (`memory`, knowledge graph), nessuno che esegua.

**Questo non e' il verdetto di 04.** 04 concludeva "nessun compilatore".
Il compilatore c'e' (`glob` trova `/home/j-ai/cuda-12.8/bin/ptxas` e
`/home/j-ai/cuda128/cuda_nvcc-linux-x86_64-12.8.93-archive/bin/ptxas`;
`qsb-env.sh:7-9` punta la PATH li'). Io non ho il braccio per premerlo.
Ho verificato la distinzione e non cito 04 come ancora valido per la sua causa.

## 3. SHA

- cubin prodotto da me: **nessuno**. Non ho compilato.
- sha del cubin congelato che ho **letto** (`qsb_carrier_sm89.h:3,6,7`):
  `913a97b2a8e6354e7632a2f41a997a05bf0e61e5a8b9a5c5c793f931e983b463`,
  `476832` byte.
- sha del mio header rigenerato: **nessuno**.

Il numero che 05 riportava (cubin 477 344 B, sha `cd2cc1b7...`) viene da
`nvcc -O3 -arch=sm_89 -cubin pinning.cu` **senza** `-DQSB_CARRIER_BUILD=1`.
Quello non e' il cubin punteggiato: e' il prodotto di
`research/build_sm89_cubin.sh`, che `QsbCarrier.h` non legge. **I 512 byte
di differenza che 05 chiama "la prova che la compilazione e' viva" provano
che `research/build_sm89_cubin.sh` gira — non che il carrier e' vivo.**
Vedi `08-CORRECTIONE-PERCORSO-BUILD.md`.

## 4. File aperti per intero, e cosa ho cambiato

Aperti per intero (read tool, file completo):

| file | righe | cosa ho cambiato |
|---|---:|---|
| `_notes/05-COMPILER-UNBLOCKED.md` | 103 | niente |
| `_notes/03-GEOMETRY-COST-CURVE.md` | 143 | niente |
| `_notes/01-TREE-IDENTITY.md` | 116 | niente |
| `_notes/02-NOISE-VS-GATE.md` | 113 | niente |
| `DEAD-ENDS.md` | 31 | niente |
| `ITERATIONS.md` | 1708 | niente (3 letture: 1-879, 880-1579, 1580-1708) |
| `QsbCarrier.h` | 216 | niente |
| `build_carrier.sh` | 75 | niente |
| `research/build_sm89_cubin.sh` | 38 | niente |
| `SUBMISSION-CODEX-20260925-G3.md` | 80 | niente |
| `_notes/04-RESULT.md` | — | no (grep) |
| `.pinning.build` | 1 | niente |

Letti in sezione (grep/read mirato, non per intero): `pinning.cu` (6 580
righe — 20 finestre, incluse 158-247, 644-692, 1091-1178, 1766-1911, 2000-2019,
3643-3650, 4096-4104, 4417-4424), `GLVScalar.cuh` (1-1189 di ~1200).

**Modificati: 5 file, tutti nuovi, tutti in `_notes/`.** Nessun file di
produzione toccato. `GLVScalar.cuh` e `pinning.cu` sono **intatti** — e
questa e' una scelta, non un incidente: vedi §6.

Creati: `06-SESSION-OPEN-20260930.md`, `07-N8-VERDICT.md`,
`08-CORRECTIONE-PERCORSO-BUILD.md`, `09-HEADER-FROZEN-NON-MATCHA.md`,
`10-N8-E-IL-BASELINE.md`, `11-LEDGER.md` (questo).

## 5. Mutanti scritti

**Nessuno.** Il brief li chiedeva "con l'output del rosso". Un rosso richiede
di eseguire un test, e non ho un esecutore. Scrivere un mutante che non posso
far girare e riportarlo come "rosso" sarebbe falso. Zero mutanti, zero
output, dichiarato.

L'unico test che ho potuto fare e' **statisco**, e l'ho fatto: la
coerenza interna dell'header congelato. `qsb_carrier_b64_lines = 5299`
(`:20`) e il file ha 5 321 righe; 5 321 - 5 299 = 22 righe di preambolo, che
contando riga per riga (3 commento + pragma + include + 2 costanti + `static
const char *const qsb_carrier_kernel_names[] = {` + 10 nomi + `};` +
dichiarazione `b64_lines` + `static const char *const qsb_carrier_b64[] = {`)
**chiude a 22**. L'header non e' troncato. PASS.

## 6. Cosa NON ho fatto, e perche' — con la riga

**Non ho scritto la patch N=8.** Motivo, con riferimenti:

- `01-TREE-IDENTITY.md:98-101` + `pinning.cu:161`: l'albero **e' gia'** la
  geometria a 11 termini promossa a 948,9 M. N=8 la sostituisce con una a
  16 gathers / 14 adds: piu' add su una tabella promossa.
- `pinning.cu:1885,1888,2016-2017`: la catena e' `#pragma unroll 1`. L'argomento
  del "muro dei registri" di `03:138-140` non regge: N=8 aggiunge iterazioni,
  non termini vivi.
- `GLVScalar.cuh:135,180`: larghezze e record per chunk sono **costanti
  hard-coded**. N=8 non e' un flag: e' riscrivere cinque funzioni di decode
  con tre varianti PTX (`GLVScalar.cuh:247,283-307,320-327`) piu' il bias di
  telescoping piu' tre `static_assert` (`pinning.cu:665-676`).
- `ITERATIONS.md:24-33`: GLV10, stessa classe di movimento, **399 697 199**,
  -55.7%, "Table geometry in the fewer-adds direction is CLOSED".
- `03:110-116`: lo stesso 03 dichiara che il suo unico vantaggio e'
  "an inference, not a measurement".
- `pinning.cu:183`: `QSB_PMIX12 65536` attivo, che `03` non include nel suo
  modello.

**Non ho rigenerato l'header.** `build_carrier.sh` e' uno script bash: senza
shell non gira.

**Non ho sottoposto.** Il brief: "Non dichiarare migliore nessuna versione
se non hai lo sha del cubin cambiato in mano." Non ce l'ho. E l'unica
submission del task, spesa su un artefatto non verificabile, non e' una
scommessa che il brief consente.

**Non ho rieseguito il verdetto di 04.** Era corretto per la sua causa; la
causa e' cambiata; l'ho verificato e non lo propago.

## 7. I due controlli da fare per primi, in ordine

Entrambi gratis, entrambi di una riga, entrambi con shell:

    cd /home/j-ai/quantum-safe-bitcoin-challenge/candidates/pinning
    ./build_carrier.sh 24        # -> sha in riga 72

Se lo sha e' `913a97b2a8e63...`, l'header e' allineato al sorgente e il ciclo
e' vivo. Se e' diverso **a sorgente invariato**, allora l'header era stale
prima di me (vedi `09`) e il carrier era probabilmente gia' spento su ogni
run: `QsbCarrier.h:143` chiama `qsb_carrier_off` e si torna al compute_52
JIT-ato dentro la finestra, in silenzio.

    git -C /home/j-ai/quantum-safe-bitcoin-challenge diff 8d07d3e --stat -- candidates/pinning/

Se vuoto: l'albero **e'** il record, la submission richiesta e' un re-roll,
che il brief vieta. Se non vuoto: il diff dice quanto c'e' da guadagnare.
Questo era gia' **OPEN — blocking** in `01-TREE-IDENTITY.md:115` e resta il
primo ostacolo, prima di qualunque geometria.

## 8. Cosa ho lasciato di meglio

Il candidato vero non e' N=8. E' il **tempo**, in un kernel che spende la
sua vita su 21,1 GiB di tabella a 6 cold read per candidato, dentro una
finestra L2 di 42 MiB. `02-NOISE-VS-GATE.md:110-113` dice che l'unica
classe che ha mai mosso questo track oltre il rumore e' la sostituzione
totale di geometria, e che l'istanza migliore **e' gia' in questo albero**.
Il prossimo passo non e' scegliere una geometria: e' misurare dove va il tempo,
e questo richiede il cubin rigenerato piu' un contatore di profilo che
questa sessione non puo' produrre.

## Ledger

| # | item | stato | evidenza |
|---|---|---|---|
| 1 | toolchain presente ma non eseguibile | **BLOCKED** | toolset senza shell; `glob` su `cuda-12.8/bin/ptxas` |
| 2 | carrier congelato: 14 launch, 0 `cuLaunchKernel` | **DONE** | `pinning.cu:4075-5663` (14 match); `QsbCarrier.h:169` usa `cudaLaunchKernel` |
| 3 | `ITERATIONS.md` + `DEAD-ENDS.md` letti per intero | **DONE** | 1 708 + 31 righe |
| 4 | geometria GLV11 verificata sul sorgente | **DONE** | `pinning.cu:161,649-652,665-673,183`; `GLVScalar.cuh:99-161` |
| 5 | N=8 valutata e **rifiutata** | **DONE** | `07`, `10` |
| 6 | percorso di build corretto | **DONE** | `08`: `build_carrier.sh`, non `research/build_sm89_cubin.sh` |
| 7 | symbol QK_S0 dell'header non matcha la firma attuale | **DONE (da confermare con shell)** | `qsb_carrier_sm89.h:9-10` vs `pinning.cu:3647,4100,4421`; `07`/`09` |
| 8 | header internamente coerente (non troncato) | **DONE** | 5 321 righe - 5 299 dichiarate = 22 di preambolo, chiuso |
| 9 | baseline compile + sha | **BLOCKED** | serve shell |
| 10 | rigenerazione header | **BLOCKED** | serve shell |
| 11 | `yukon submit` | **BLOCKED** | serve shell + rete |
| 12 | **albero = record o indietro?** | **OPEN — blocking** | `01-TREE-IDENTITY.md:115`; serve `git diff 8d07d3e` |
| 13 | mutanti | **NON FATTI** | serve esecutore; dichiarato, non simulato |
| 14 | **promozione** | **NO** | nessun punteggio ottenuto |
