# 04-FINDING-A — `QSB_GATHER_ONE_FORM=1` annulla la finestra L2 sui segmenti caldi

**Stato: `NON MISURATO`.** Non ho una shell (vedi `00-START.md`). Quello che segue
e' una **divergenza fra intento documentato e codice**, con il meccanismo e la riga.
Non e' una misura di throughput, e non lo chiamo tale.

## L'intento, scritto dagli stessi autori

`tests/gpu_epochs/tree.cu:1753-1755`:

    /* Table point for a code: segments 0-3 (inside the pinned L2 window) with the
     * ordinary read-only load, segments 4-5 (DRAM) with evict-first ld.global.cs
     * so the 9 GiB stream does not displace the rest of the kernel's L2 working set */

E `tree.cu:507`: *"Segments 0-3 (786,432 records = 48 MiB) are physically first:
exactly the bytes the persisting L2 window pins."*

Il progetto, insomma, ha **due pezzi**: una finestra L2 persistente
(`QSB_TABLE_L2_WINDOW`, `tree.cu:112-114`) che fissa i 48 MiB caldi, e un predicato
per-record che sceglie il caricamento caldo su quei segmenti e evict-first sui
freddi.

## Il codice

`tree.cu:271-273` — il default e' **1**:

    271 #ifndef QSB_GATHER_ONE_FORM
    272 #define QSB_GATHER_ONE_FORM 1
    273 #endif

`tree.cu:1827-1833`, dentro `qsb_s3_load_n`:

    #if QSB_GATHER_ONE_FORM == 1
        const bool cold = true;  (void)ef;    /* the cold form for every record */
    #elif QSB_GATHER_ONE_FORM == 2
        const bool cold = false; (void)ef;    /* the hot form for every record */
    #else
        const bool cold = ef;
    #endif

`cold` e' una costante di compilazione, quindi a **1** il ramo `else` (i `__ldg` caldi,
riga 1861) **viene eliminato dal DCE** e ogni raccolta del device code finisce nel
ramo evict-first di `tree.cu:1856-1859`:

    if (cold) {
        asm("{ ... ld.global.cs.nc.L2::64B.v2.u64 ... }");   // x0
        x1 = __ldcs(tx + 1); y0 = __ldcs(ty); y1 = __ldcs(ty + 1);
    }

## Il difetto

Il predicato che distingue caldo da freddo **esiste ed e' calcolato** — `ef` vale
`d.off >= GT_DENSE_ENTRIES` (per es. `tree.cu:2061`), e `GT_DENSE_ENTRIES` e'
786432 = esattamente 262144+262144+131072+131072, cioe' i segmenti 0-3
(`tree.cu:917`, `static_assert` a `tree.cu:947-949`) — ma **a `QSB_GATHER_ONE_FORM 1`
viene scartato con `(void)ef;`**.

Quindi: **i 48 MiB che la finestra L2 persistente dovrebbe tenere in cache vengono
caricati con l'istruzione di streaming evict-first**, la stessa che il progetto usa
apposta per i 4 GiB e 5,08 GiB dei segmenti 4-5 perche' non spostino il resto del
working set. Il meccanismo che protegge i caldi li marchia come freddi.

## Il commento che contraddice

`tree.cu:223` descrive lo switch come *"QSB_GATHER_ONE_FORM (screen arm, **default
0**)"*. Il codice dice `1`. E il commento di `tree.cu:222` promette ancora *"the
predicated hot/cold pair"*, che con `ef` scartato **non e' piu' quello che gira**.

Questa e' la seconda contraddizione commento/codice su uno switch di
`QSB_CARRIER_KNOBS` che ho trovato (la prima e' `QSB_SHORT_CARRY4`,
`filter_tail_sc.cuh:11` che dice "default off" e `filter_tail_sc.cuh:23` che dice `1`).
Due su due sbagliate nello stesso verso: il commento dice spento, il codice dice
acceso. **Il metro e' il numero dei non-nil, non la lettura** — e qui il numero dice
che la coppia predicata calcolata e poi scartata e' codice morto che il commento
promette come vivo.

## Cosa NON sto dicendo

- **Non** sto dicendo che `=1` e' piu' lento. Può darsi che il risparmio del predicato
  e di un ramo valga piu' della perdita di L2, e che sia un win misurato. Non lo so:
  non posso eseguire.
- **Non** sto dicendo che il codice sia scorretto. E' bit-identico (stessi byte
  caricati, stessa maschera) e il gate esatto sull'host presidia comunque.
- **Non** sto contando un +1%. Sarebbe inventare il numero che il brief vieta.

## Il metro, se qualcuno ha una shell

    cd /home/j-ai/quantum-safe-bitcoin-challenge/candidates/subset
    source /home/j-ai/concord/qsb-env.sh

    # 1. baseline: il sha congelato del brief
    ./build_carrier.sh 24
    grep -oE 'cubin sha256 [0-9a-f]{16}' qsb_carrier_sm89.h   # atteso: e0c0897f799baf81

    # 2. contare i CDBG.128 che cambiano di classe (cs vs default) nel kernel digest
    python3 /home/j-ai/concord/memory/qsb_sass_baseline.py <work>/c.cubin

    # 3. LE TRE BUILD, una sola riga di differenza ciascuna
    #    (tutte e tre dentro QSB_CARRIER_KNOBS, quindi l'host e l'immagine restano d'accordo)
    sed -i 's/#define QSB_GATHER_ONE_FORM 1/#define QSB_GATHER_ONE_FORM 0/' tests/gpu_epochs/tree.cu
    ./build_carrier.sh 24 ; grep -oE 'cubin sha256 [0-9a-f]{16}' qsb_carrier_sm89.h
    sed -i 's/#define QSB_GATHER_ONE_FORM 1/#define QSB_GATHER_ONE_FORM 2/' tests/gpu_epochs/tree.cu
    ./build_carrier.sh 24 ; grep -oE 'cubin sha256 [0-9a-f]{16}' qsb_carrier_sm89.h
    sed -i 's/#define QSB_GATHER_ONE_FORM 2/#define QSB_GATHER_ONE_FORM 1/' tests/gpu_epochs/tree.cu

Se dopo una build `grep` restituisce ancora `e0c0897f799baf81`, la modifica **non e'
arrivata al device code** e va rifatta: il numero non misura niente.