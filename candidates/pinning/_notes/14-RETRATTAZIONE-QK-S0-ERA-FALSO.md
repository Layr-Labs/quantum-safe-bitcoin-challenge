# 14 — RETRATTazione. L'header NON e' stale. E cosa resta vero.

Questa nota era stata scritta come "il simbolo QK_S0 non puo' venire da
questo sorgente". **Era falsa.** Sotto la retraction, e poi l'unica cosa che
ho prodotto che regge.

## 1. La retraction, prima di tutto

Il mio argomento era: l'header dichiara 22 parametri, il sorgente ne ha 23,
manca `uint64_t *tree`, quindi l'header e' stale e il carrier e' spento.

**Il conteggio era sbagliato.** Ho letto `PmSA_12qsb_tail_pre` come **un**
tipo di parametro. Sono **due**:

    P10ulonglong2   ->  saved
    Pm              ->  roots
    SA_             ->  tree      (uint64_t*, stessa di roots: sostituzione)
    12qsb_tail_pre  ->  tp         (by-value, struct a pinning.cu:2204)

= **23 parametri**, che e' esattamente `pinning.cu:3633-3647`. L'header
**quadra**.

E la prova che non ho costruito io, gia' nel repo:

`_notes/10-CORREZIONE-QK-S0.md:15-24` riporta una sessione **che aveva la
shell, ha compilato `pinning.cu` da zero** e ha letto il simbolo dal cubin
prodotto:

    _Z23kernel_pinning_pipelineILb1ELi0EEvPKjPKhiiiijjPKmS5_S5_S5_S5_PhPjS7_iiiP10ulonglong2PmSA_12qsb_tail_pre

"IDENTICI, carattere per carattere" con l'header congelato.

**Quindi: header allineato al sorgente, ciclo vivo, riferimento `913a97b2`
valido.** Non ho niente che lo contraddica, e ho una riga che lo conferma.

### La trappola che avevo costruito, e che non esiste

Avevo scritto che la regola di `09-CICLO-VIVO-MISURATO.md` ("uno sha diverso da
`913a97b2` = la modifica e' arrivata") "inverte il segno". **No.** Il metro e'
esatto. `build_carrier.sh` rigenera dall'attuale sorgente, l'attuale sorgente
produce `913a97b2`, quindi la regola vale.

### L'errore di fondo, e l'ho gia' commesso una volta in questo track

Ho ricostruito l'ABI Itanium a memoria, senza un caso reale che lo controllasse,
e ne ho fatto una struttura che chiudeva il resto. E' esattamente l'errore che
`10-CORREZIONE-QK-S0.md:63-71` descrive, e che `09-HEADER-FROZEN-NON-MATCHA.md`
aveva gia' commesso prima di me: *"Il suffisso `12qsb_tail_pre` ... e' un
parametro di template. In Itanium ABI un tipo non-template non entra mai nel
nome mangled di una funzione."*

E' falso, e si smentisce con il repo in mano: `qsb_root_group_prepare` ha
quattro parametri di funzione e tutti e quattro sono nel suo nome mangled
(`qsb_carrier_sm89.h:11`). Se un tipo classe non entrasse mai, quel nome non
esisterebbe.

**Il contro-probro che avrei dovuto fare prima di scrivere, e che esiste gia'
a `:190` di questa nota:** cercare la stringa `PmSA_` nei dump di `cuobjdump`
veri del repo. L'ho fatto **dopo** aver scritto, e mi ha smentito.

## 2. Il contro-probro, e cosa mi dice davvero

`PmSA_` (22 parametri, senza `tp`) compare in **otto** dump reali di build
sm_89 — `research/{fused_reduction,pk_unroll,two_stream_two_field,
wide_windows,native-base-comparison}` — e in
`research/build_sm89_cubin.sh:28-29`, che lo elenca come simbolo atteso.

Sono **revisioni piu' vecchie**, prima che `qsb_tail_pre tp` entrasse nella
firma (la "constant-bank tail"). Quindi la differenza 22/23 fra quei dump e
l'header e' **attesa**, e non e' una traccia di staleness.

Il mio errore e' stato leggere quei 22 come se dessero un upper bound per
l'header. Non lo fanno: l'header ha 23, che e' il numero giusto.

## 3. Cosa resta vero, ed e' utile: `WARP=1` non e' una leva

Questo l'ho verificato per intero e **regge**. Il brief e la nota 13 puntano
`pinning.cu:201` (`QSB_PMIX12_WARP` 0 -> 1).

    pinning.cu:149   QSB_TREE_N 128      -> 4 warps per blocco
    pinning.cu:183   QSB_PMIX12 65536
    pinning.cu:211   QSB_PMIX12_N 1
    pinning.cu:232-238   la selezione

    WARP=0 :  (blockIdx.x & 65535) == 0
              -> i blocchi 0, 65536, 131072, ...   tutti i 4 warps
              -> **1 candidato su 65536**

    WARP=1 :  ((4*blockIdx.x + warp) & 65535) < 1
              -> warp deve essere 0 (0..3), quindi i blocchi 0, 16384, 32768
              -> 1 warp di 4 in 1 blocco su 16384
              -> (1/16384)*(1/4) = **1 candidato su 65536**

**Le due opzioni selezionano la stessa frazione: 1/65536 = 0.0015 %.** Non e'
un interruttore di carico, e' lo stesso identico sottoinsieme di candidati
distribuito in modo diverso. Il delta atteso sul tempo e' **zero**.

E la nota 13 misura il costo (`pinning.cu:201` in sua compilazione): 8 byte
di stack spill aggiunti al kernel da 128 registri. Quello spill lo paga **il
100 % dei candidati**; il beneficio — distribuire meglio lo 0.0015 % — lo
riceve lo 0.0015 %.

Con il sigma di run-ufficiale (~2.6 %, `ITERATIONS.md:1311`) e la soglia a
+1 %, questa variante e' un **ri-roll con un costo noto e un guadagno non
misurabile**. Non e' un muro: e' una non-leva. Il commento a `:191` che dice
"default 1" contraddice il codice a `:201`, ma il codice e' quello che conta.

Attenzione pero' al verso: `QSB_PMIX12_WARP=1` **non e' mai stato un vicolo
chiuso** per questa K. `DEAD-ENDS.md` non lo nomina, e `ITERATIONS.md` non
contiene la stringa `PMIX12` (verificato con grep: zero occorrenze). Le
submission GLV12 storiche usavano `WARP=1` con `QSB_PMIX12=16, N=2` — una
configurazione diversa, 1/8 dei candidati, non 1/65536.

Il vero interrogativo su questa famiglia non e' `WARP`, e' **K**: a
`QSB_PMIX12=65536` la miscelazione GLV12/GLV11 e' spenta di fatto. Il confronto
fra le due geometrie che la nota 03 usa come calibrazione e' proprio
GLV12 (960 830 125) contro GLV11 (948 943 797), cioe' **K piccolo**, non
`K=65536`. L'albero di ogni giorno gira con la miscelazione a una frequenza
che non sposta niente.

## 4. Cosa NON ho fatto, e perche' — con la riga

- **Non ho compilato.** Nessuno strumento di esecuzione nel toolset; l'MCP
  esposto e' un solo server `memory`, nessuno che esegua. Ricontrollato in
  questa sessione, non citato per inerzia.
- **Non ho rigenerato l'header.** `build_carrier.sh` e' bash.
- **Non ho sottomesso.** Non ho uno sha prodotto da me, e
  `12-CONTRATTO-SUBMIT.md:33` vieta di dichiarare un numero non misurato.
- **Non ho scritto mutanti.** Un rosso richiede di eseguire. Zero mutanti,
  zero output, dichiarato — non simulato, non raccontato.
- **Non ho toccato `pinning.cu` ne `GLVScalar.cuh`.** Intatti, di proposito:
  senza baseline non si tocca niente.

## Ledger

| # | item | stato | evidenza |
|---|---|---|---|
| 1 | header congelato allineato al sorgente | **DONE, confermato** | `10-CORREZIONE-QK-S0.md:15-24`, compilazione reale |
| 2 | "header stale / carrier spento" | **RETRATTO** | conta sbagliata: 23, non 22 |
| 3 | argomento Itanium di `09-HEADER-FROZEN-NON-MATCHA` | **CONFERMATO falso** | 4 parametri in `_Z22qsb_root_group_preparePKmiPmS1_` |
| 4 | riferimento `913a97b2` valido come metro | **DONE** | `09-CICLO-VIVO-MISURATO.md`, `11-ORDINE-DI-OPERAZIONE.md:27-34` |
| 5 | `WARP=1` seleziona 1/65536 come `WARP=0` | **DONE** | `pinning.cu:149,183,211,232-238` |
| 6 | `WARP=1` costa 8 byte di spill per il 100 % dei candidati | **DONE (da 13, non riprodotto)** | `13-CICLO-PROVATO.md:22-24` |
| 7 | `PMIX12` assente da `ITERATIONS.md` e `DEAD-ENDS.md` | **DONE** | grep: 0 riscontri in `ITERATIONS.md`; `DEAD-ENDS.md` 31 righe |
| 8 | baseline sha reale | **BLOCKED** | serve shell |
| 9 | rigenerazione header | **BLOCKED** | serve shell |
| 10 | `yukon submit` | **BLOCKED** | serve shell + rete |
| 11 | **promozione** | **NO** | nessun punteggio ottenuto |