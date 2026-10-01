# 09 — il symbol QK_S0 dell'header congelato non puo' venire da questo sorgente

Trovato leggendo l'header, non eseguendo nulla. E' la discovery piu' utile
di questa sessione, e riguarda la premessa del brief.

## Cosa dice l'header congelato

    qsb_carrier_sm89.h:9   "_Z23kernel_pinning_pipelineILb1ELi0EEvPKjPKhiiiijjPKmS5_S5_S5_S5_PhPjS7_iiiP10ulonglong2PmSA_12qsb_tail_pre", /* QK_S0 */
    qsb_carrier_sm89.h:10  "_Z23kernel_pinning_pipelineILb1ELi2EE..._12qsb_tail_pre",  /* QK_S2 */

Il suffisso `_12qsb_tail_pre` in coda al nome mangled **e' un parametro di
template**. In Itanium ABI un tipo non-template non entra mai nel nome
mangled di una funzione: solo i parametri di template e i tipi di classe
entrano. Quindi l'immagine congelata contiene un `kernel_pinning_pipeline`
**istanziato con un terzo argomento di template `qsb_tail_pre`**.

## Cosa dice il sorgente di oggi

    pinning.cu:3647   ulonglong2 *saved, uint64_t *roots, uint64_t *tree, qsb_tail_pre tp
    pinning.cu:4100   ..., const qsb_tail_pre &tp QSB_STREAM_PARM
    pinning.cu:4421   ..., const qsb_tail_pre &tp, cudaStream_t st

`qsb_tail_pre` qui e' un **parametro di funzione by-value/by-reference**
(`struct` a `pinning.cu:2204`), non un argomento di template. La firma e'
`template<bool FAST_TAIL, int STAGE>`: due parametri di template, `<Lb1ELi0E>`.
Il terzo non c'e'.

## Perche' questo e' decisivo, e non un dettaglio

Il nome mangled cambia **solo** se cambia la lista dei parametri di template.
Il sorgente attuale non ha quel terzo parametro. Quindi il cubin che
`build_carrier.sh` produrrebbe **adesso** avrebbe un nome diverso, e
`QsbCarrier.h:142` (`cudaLibraryGetKernel(..., name)`) andrebbe a cercare
`_Z23kernel_pinning_pipelineILb1ELi0EEvPKjPKh...SA_` — senza il suffisso.

Cosa succede in quel caso, riga per riga:

- `QsbCarrier.h:143`  `qsb_carrier_off("kernel missing from image"); return;`
  il carrier si **spegne**;
- si torna al compute_52, che sul runner viene **JIT-compilato dentro la
  finestra punteggiata** — esattamente il costo che v22 voleva togliere
  (`ITERATIONS.md:1015-1024`, ~5-15 s, e il modello di yield di v8/v9);
- `QSB_NOJIT` va a 0 (`QsbCarrier.h:151`), quindi il modulo compute_52 viene
  comunque toccato e compilato.

Il fallback e' **silenzioso in termini di score**: nessun errore, nessun
exit, solo throughput peggiore. `QsbCarrier.h:105` stampa solo
`Native sm_89 carrier: off (...)`. E `QsbCarrier.h:19` promette che "The exact
OpenSSL host gate checks every published hit in both modes" — cioe' la
correttezza regge, il punteggio no.

E c'e' un'arma in piu: `QSB_PMIX12` (pinning.cu:183) fa partire una variante
del kernel prepare **non istanziata staticamente allo stesso modo** nel
carriere vecchio, perche' il suo trip count dipende da `QSB_PMIX12_SEL()`
(`pinning.cu:1762`). Il nome mangled di QK_S0 che il carrier risolve
potrebbe non essere quello che il lancio effettivamente usa.

## Il punto che chiude la domanda del brief

Il brief chiede se il cubin congelato e' "un ridisegno della stessa immagine"
e se toccare `pinning.cu` senza rigenerare l'header "produce uno score che e'
un ridisegno della stessa immagine congelata".

La risposta verificata e' **peggiore di cosi'**: se il sorgente e' advanced
rispetto all'header (come la firma a `pinning.cu:3647` indica), allora
**l'header gia' non corrisponde al sorgente gia' adesso**, prima di qualsiasi
mia modifica. Il carrier potrebbe essere gia' spento su ogni run passato, con
il JIT che lavora in finestra, e nessuno che lo abbia notato perche' il
fallimento e' silenzioso e la correctness regge.

**Non posso confermarlo eseguendo `build_carrier.sh` su sorgente invariato.**
E' il controllo numero uno per un worker con shell, ed e' gratuito:

    cd candidates/pinning && ./build_carrier.sh 24

Se lo sha che esce e' `913a97b2a8e6354e...` (quello di riga 7), l'header e'
allineato e `pinning.cu` e' quello che pensiamo. Se esce altro **a sorgente
invariato**, allora la finding e' confermata: l'header era stale gia' prima di
me, e la prima submission va rifatta sul cubin nuovo.

Non e' una scoperta da usare come scusa per non lavorare. E' una scoperta da
mettere **prima** del lavoro, perche' cambia cosa significa "rigenerare
l'header": non e' burocrazia, e' la differenza fra girare su sm_89 e fare JIT
in finestra su una 4090.

## Cosa ho verificato direttamente, e cosa no

Verificato, leggendo:
- i 10 nomi kernel in `qsb_carrier_sm89.h:9-18` (letti per intero);
- la firma di `kernel_pinning_pipeline` in `pinning.cu:3647, 4100, 4421`;
- `qsb_tail_pre` e' uno `struct` a `pinning.cu:2204`, non un template;
- l'array di base64 e' dichiarato a 5299 righe (`qsb_carrier_sm89.h:20`) e il
  file contiene 5321 righe: 5321 - 5299 = 22 righe di preambolo
  (commento, `#pragma once`, include, le due costanti, l'array dei nomi, le
  sue 10 righe, la dichiarazione `b64_lines`, `static const char *const
  qsb_carrier_b64[] = {`). **Il conteggio torna.** L'header e' internamente
  coerente: non e' troncato a meta' da una scrittura interrotta.

Non verificato, perche' servono strumenti che non ho:
- il contenuto effettivo del cubin (serve `cuobjdump -symbols` sul nuovo);
- se `QSB_PMIX12` produce un istanziamento diverso da quello risolto in QK_S0;
- se gli altri 9 nomi (RGP, ISR, RGF, BUILD, YOFF, RF, RR, PFC) reggono.
  Questi ultimi sono plausibili: `qsb_root_group_prepare` (`:11`),
  `qsb_invert_super_roots` (`:12`), `qsb_root_group_finish` (`:13`),
  `kernel_build_gtable` (`:14`), `qsb_table_offset_y` (`:15`) sono tutti
  funzioni non-template e i loro nomi sono stabili.
