# 05-VERDETTO.md — consegna

## 1. VERDETTO, in prima riga

**NO — non ho rotto qualcosa che vale un punto. E non ho misurato niente.**

Il motivo e' uno solo e va detto subito: **non ho una shell in questa sessione**.
Il mio toolset e' Read/Write/Edit/Glob/Grep/Task/Skill/board. Non c'e' `Bash`, non
c'e' `exec`. Evidenza: `list_mcp_resources` -> unica risorsa `memory://knowledge-graph`
(servizio `memory`); `list_mcp_resource_templates` -> lista vuota. Nessun server
terminale.

Quindi `nvcc`, `./build_carrier.sh`, `cuobjdump`, `ptxas -v`,
`qsb_sass_baseline.py` **non li ho lanciati**. Nessun cubin e' stato prodotto, nessun
sha e' stato ricalcolato, nessun conteggio SASS e' stato preso. Il mio contributo e'
**lettura integrale e una divergenza commento/codice con la riga**, non una misura.

## 2. IL DENOMINATORE — quanti file aperti per intero, e cosa c'era

**Aperti per intero da ME: 3.**

| File | Righe | Cosa c'era |
|------|-------|-----------|
| `subset.cu` | 11 | 4 `#define` di switch + `#include "tests/gpu_epochs/tree.cu"` (riga 11). Tutto il device code e' dentro `tree.cu` e nei suoi header: questo file non e' il programma. |
| `build_carrier.sh` | 98 | rigenera `qsb_carrier_sm89.h`: nvcc `-O3 -arch=sm_89 -cubin` (riga 28), poi un python che trova i 6 kernel per nome mangled (36-43), verifica 15 global (52-55), pretende `LTC64B` nel kernel digest (59-64), scrive base64+sha. **Sviluppo, mai eseguito dall'harness ranked.** |
| `filter_tail_sc.cuh` | 93 | i due helper di campo speculativi `qsb_fsub`/`qsb_fadd`, i dispatch `QSB_FMUL/X_FMUL`, e la contraddizione di C3. |

**Letti per intero da 6 subagent `explore` (11 file, ~14.000 righe), che ho poi
controllato pezzo per pezzo dove le loro conclusioni contavano:** `tree.cu` (7.298
righe, letto in strisce da me: 1-1725, 1726-2325, 2989-3688, 5285-5334),
`hit_filter_field_sc.cuh` (4.128), `pair_shared.cuh` (775), `tree_inverse.cuh`
(483), `zinv32.cuh` (432), `inverse_limbs.cuh` (148), `GLVScalar.cuh` (954),
`sha_gate_fma.cuh` (422), `y_pair_sc.cuh` (1.164), `square32.cuh` (234),
`window_schedule_shared.cuh` (396), `GPUHash.h` (1.168), `GPUMath.h` (1.412),
`chain_replay_field.cuh` (1.754), `hit_filter_field.cuh` (1.864),
`hit_filter_field_sc_aluz.cuh` (3.969).

**Aperti parzialmente, NON conta nel denominatore:** `qsb_carrier_sm89.h` (prime 12
righe: intestazione e fingerprint), i 14 `.md` (titoli, e `PAIR-CURRENT.md` per
intero — 9 righe).

**Non aperti:** i 12 `*_audit.cu` (programmi di test separati, non compilati dentro
`subset.cu`) e `tree.cu.orig`.

## 3. Cosa ho rotto

**Nessun difetto di correttezza confermato.** I due "difetti SHA" piu' citati di
questa superficie sono **falsi** e li ho smentiti io, riga per riga, in
`02-CORREZIONI.md`:

- `pair_shared.cuh:428-429` (feed-forward) — corretto, e' `output[i] += stato`, come
  `GPUHash.h:258-265`. La citazione del subagent **non coincide col file**.
- `pair_shared.cuh:457-458` (riga 63 del gate H0) — corretto, stessi termini di
  `sha_gate_fma.cuh:409` in ordine diverso. Nessun "doppio T1".

Quello che resta e' una **divergenza commento/codice**, non un bug: `FINDING-A` in
`04-FINDING-A.md`. Sintesi: `QSB_GATHER_ONE_FORM=1` (default, `tree.cu:272`) scarta
con `(void)ef;` il predicato caldo/freddo che `tree.cu:2061` calcola, e di conseguenza
**i 48 MiB che la finestra L2 persistente dovrebbe fissare vengono caricati con
l'istruzione evict-first** pensata per i 4 GiB e 5,08 GiB dei segmenti freddi. Il
commento a `tree.cu:223` dice "default 0" e a `tree.cu:222` promette ancora "the
predicated hot/cold pair". Meccanismo e metro ready-to-run in `04-FINDING-A.md`.

**Precondizione esatta, per prima:** `QSB_GATHER_ONE_FORM == 1` **e** la coppia
calcolata e poi scartata. Con `=0` il ramo caldo torna e il predicato torna vivo;
con `=2` tutti i record prendono il ramo caldo. La riga e' `tree.cu:1828`.

## 4. I MUTANTI

**zero mutanti.** Non ne ho scritti. Non ho un interprete Python e non ho un
compilatore: un mutante che non posso eseguire non e' un mutante, e' una promessa.
Il brief lo dice esplicitamente: *"Se non hai scritto mutanti, scrivi `zero
mutanti` — non simularli, non raccontarli."* Non li racconto.

Quello che ho al loro posto e' il **metro ready-to-run** di `04-FINDING-A.md` §"Il
metro": tre build a una riga di differenza, con l'istruzione esatta di come leggere
lo sha e cosa significa se non cambia.

## 5. Cosa NON sono riuscito a rompere, e perche' — con il numero di riga

1. **L'inverse a blocchi: non l'ho toccato e non potevo romperlo.** Con
   `ZLAB_TREE 2` + `QSB_TREE_WAVE_TOP 1` e n=256, ho ricostruito a mano la sequenza:
   up-sweep `tree_inverse.cuh:232`, quattro ondate sul warp 0 (`:305-345`), root
   invertito a 32 lane (`:351`), down-sweep `:429`, foglia `:459`. **12 barriere in
   tutto: 6 `__syncthreads` (`:223`, `:250`x2, `:450`x3) + 6 `__syncwarp`.** Non ho
   trovato una lettura di shared protetta da una barriera mancante: ogni scrittura
   cross-warp e' separata dalla lettura da un `:250` o `:450`. Il vincolo
   `products` write-once (`:247`) e mai riscritto nel down-sweep e' cio' che rende
   sicura la lettura della foglia. **Non e' un muro** — e' "non ho eseguito
   `compute-sanitizer --tool racecheck`", che e' la prova vera e qui non e' possibile.

2. **Il divisstep del root: non misurabile, e il tempo e' strutturale.**
   `inverse_limbs.cuh:57-126` fa al massimo 32 batch da 30 passi (tetto
   `ZI_ROOT_MAX_BATCHES`, `zinv32.cuh:385`): 960 divisstep, ognuno con 5 lookup in
   tabella da 832 parole. Il commento `tree.cu:630-645` dice che 4 dei 5 lookup di
   ogni batch stanno sulla catena di decisione **seriale** del root e costano 63-66
   cicli. Non e' un bug, e' l'architettura. Il rimedio gia' scritto nella tree
   (`QSB_ROOT_LUT_SMEM`, tabella in shared) e' **spento** di default a `tree.cu:647`
   e secondo il suo stesso commento **non ha passato il gate dei registri**.

3. **I carry scartati: non li chiamo difetti.** Ce ne sono molti e sono **voluti**:
   `filter_tail_sc.cuh:39` e `:60` (con `QSB_SHORT_CARRY4 1` scartano il riporto su
   limb-1), e in `hit_filter_field_sc.cuh` le dieci pieghe a `:774, 1084, 1427, 1752,
   2017, 2318, 2619, 2876, 3324, 3670`. Il contratto e' scritto: sono sul percorso
   **filter-only**, e ogni hit tentativo viene ricalcolato esatto sull'host prima di
   essere pubblicato (`QSB_HOST_VERIFY 1`, `tree.cu:53-55`). Un carry perso fa
   perdere un hit, non ne pubblica uno falso. Non e' il mio difetto perche' **non e'
   un difetto**.

4. **I due SHA "sbagliati" del gate: gia' smentiti, vedi §3.**

## 6. IL METRO — sha256 e conteggio SASS, prima e dopo

| | valore |
|---|---|
| sha256 cubin **prima** | `e0c0897f799baf81df92f777f89adb4b351cf224df4a6a6c6d8a8cabf1631fea` (473.376 B) — **letto dall'intestazione di `qsb_carrier_sm89.h:3`, NON ricalcolato** |
| sha256 cubin **dopo** | **`NON MISURATO`** — nessuna build eseguita |
| conteggio SASS per kernel, prima | **`NON MISURATO`** — `qsb_sass_baseline.py` non lanciato |
| conteggio SASS per kernel, dopo | **`NON MISURATO`** |
| registri / stack | **`NON MISURATO`** — `ptxas -v` non lanciato |

**Il numero di SASS resterebbe identico?** Se qualcuno applica `FINDING-A`, **no**:
cambiare `QSB_GATHER_ONE_FORM` da 1 a 0 rimette il ramo caldo di `tree.cu:1861` e
cambia le classi di `LDG` nel kernel digest. E' esattamente il caso in cui uno sha
**uguale** a `e0c0897f799baf81` significherebbe che la modifica non e' arrivata al
device code, e va rifatta. Lo dico perche' e' la trappola numero uno di questo track.

## 7. RISCHI E PASSI AVANTI

1. **Il primo controllo che spetta a chi ha la rete:** i rilievi pubblici esterni
   (9 Blackthorn + 12 ChainSecurity citati nel brief). Non ho accesso a quei testi,
   quindi **non dichiaro che `FINDING-A` non sia gia' coperto**. Se lo e', il mio
   valore e' zero e va detto.
2. **`FINDING-A` e' un'ipotesi con meccanismo, non un risultato.** Se la build `=2` o
   `=0` non cambia lo sha, o se il conteggio SASS non si muove, la mia consegna e'
   `zero`. Lo scrivo prima di misurare, non dopo.
3. **Nessuna submission.** Non ho usato `yukon submit`, non dichiaro un punteggio,
   non ho scritto `RISULTATO.md` firmato Twin, e il codice del cubin non e' in questa
   consegna.

**Nessun file di questa consegna contiene il sorgente del cubin, e la nota pubblica
resta a chi la firma.**