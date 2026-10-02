# 07 — N=8 valutato. Non regge. Con i numeri di riga.

Questa e' la valutazione che il brief mi chiedeva di fare su
`03-GEOMETRY-COST-CURVE.md`. L'ho fatta sul sorgente, non in astratto.
**Esito: non la implemento.** Tre errori, tutti con riga.

Prima la geometria, verificata, cosi' il resto e' leggibile.

## La geometria che gira davvero (verificata, non da note)

    GLVScalar.cuh:5    QSB_BIGTBL   1
    GLVScalar.cuh:14   QSB_FOUR_HOT 1
    GLVScalar.cuh:17   QSB_DIGIT_LEAN 1
    GLVScalar.cuh:41   QSB_GLV_GLUE 15
    GLVScalar.cuh:65   QSB_DECODE_CUT 3
    GLVScalar.cuh:107  QSB_GT_RADIX_BITS 14
    pinning.cu:161     QSB_GLV11 1     <- sovrascrive il default 0 di GLVScalar.cuh:82
    pinning.cu:168     QSB_QGLV5 0
    pinning.cu:183     QSB_PMIX12 65536   <- ATTIVO
    pinning.cu:183     QSB_PMIX12_WARP 0  (default a :201)
    pinning.cu:345     QSB_CHAIN_PIPE 1
    pinning.cu:360     QSB_PHI_HOIST 1
    pinning.cu:363     QSB_CHAIN_PEEL 1
    pinning.cu:6       QSB_PERSIST_WINDOW_CAP 42 MiB

`GT_Q_TERMS 6`, `GT_CHUNKS 6`, `GT_GLV_TERMS = 6+6-1 = 11` (pinning.cu:649-652).
Totale record 354 501 773 x 64 B = 22 688 113 472 B = 21.13 GiB, e il
`static_assert` di `pinning.cu:665-667` lo richiama esattamente. Cio' la
tabella di `03` e' quella giusta. Fin qui `03` e' corretto.

## Errore 1 — il "muro dei registri" non esiste, perche' la catena e' rolled

`03:138-140`: *"N=8 adds three live chain terms"*, e da li' il verdetto
che 128 registri + 3 termini = spill.

Il codice dice il contrario. I due loop che eseguono la catena:

    pinning.cu:1885    #pragma unroll 1
    pinning.cu:1886    for(;;) {
    pinning.cu:1888        #pragma unroll 1
    pinning.cu:1889        for(;term<stop;term++) {
    pinning.cu:2016    #pragma unroll 1
    pinning.cu:2017    for(int term=first+2;term<last;term++) {

`#pragma unroll 1` = **un solo corpo di loop, non srotolato**. I termini GLV
non sono variabili live simultanee: sono *iterazioni dello stesso corpo*.
Il vivo e' lo stato del punto, `(X, Y, Rp, U, V, x1, y1, y0)` — otto word
64-bit piu' i ruotati `y0`/`y1`, un numero **fisso** indipendente da quanti
termini ha la catena.

Quindi N=8 non aggiunge tre termini vivi. Aggiunge tre *iterazioni* di un
corpo che resta identico. Il costo e' tempo, non pressione di registro.
L'argomento che chiude `03` non e' verificato sul sorgente, e senza di esso
non resta una ragione tecnica per dichiarare N=8 un muro.

Onesto il verso opposto: questo NON significa che N=8 sia buono. Significa
che il muro che `03` oppone non e' quello. La verifica vera (ptxas -v)
resterebbe da fare — e non posso farla. Ma un argomento refutato non e' un
argomento.

## Errore 2 — la geometria analizzata non e' quella che gira

`03` analizza "GLV11: 11 gathers, 5 hot, 6 cold" e conclude da li'. Ma
`pinning.cu:183` ha `QSB_PMIX12 65536` **attivo di default**. Il suo
commento a `pinning.cu:173-181` lo dice: un blocco su 65536 decodifica P
con i 6 termini GLV12 invece dei 5, e `pinning.cu:1887` (`stop`) e
`pinning.cu:1762` (`last`) lo realizzano nel trip count.

L'effetto e' minuscolo (1/65536 dei candidati) e **non** invalida la tabella
di `03`. Ma la premessa "questa e' la geometria che gira" e' falsa, e una
analisi di costo che non sa quale meccanismo e' attivo non e' una base per
prevedere un ">5%".

## Errore 3 — "N=8" non e' affatto N=8 in questo codice

Qui la nota e' semplicemente sbagliata sul nome, e il nome e' il punto.

`GLVScalar.cuh:107` ha `QSB_GT_RADIX_BITS 14`, e `pinning.cu:674` lo
usa in un `static_assert` sulla scala H del segmento piu' largo. La
larghezza di un chunk non e' un parametro libero: i chunk hanno larghezze
**fisse** in `q9_bigtbl_code` (`GLVScalar.cuh:180`):

    const unsigned bits = c==1 ? 19u : c<4 ? 18u : 27u;

e i record per chunk sono fissi in `q9_bigtbl_entries` (`GLVScalar.cuh:135`):

    return c<2 ? 262144u : c<4 ? 131072u : c==4 ? 67108864u : 85279885u;

Passare a "8 termini da 16 bit" non e' cambiare una costante: e' riscrivere
`q9_bigtbl_entries`, `q9_bigtbl_offset`, `q9_bigtbl_shift`, `q9_bigtbl_code`,
`q9_bigtbl_code_lean` **e** `q9_bigtbl_code_z` (le tre varianti di decode,
con tre scritture PTX distinte: `GLVScalar.cuh:247, 283-307, 320-327`),
piu' `q11_bigtbl_code`, `q11_bigtbl_code_z`, `q11_radix_code_z`, la
ricostruzione del bias di telescoping, `QSB_GT_TOTAL`, `QSB_GT_TOP_CENTER`,
`QSB_GT_TOP_SHIFT`, la scala H e i suoi due `static_assert`
(`pinning.cu:665-676`).

`03:123-126` chiama N=8 "a whole-geometry replacement" e lo dà per fatto.
E' una sottovalutazione: e' una riscrittura di **tutta** la superficie di
decode, con tre implementazioni parallele che devono restare bit-identiche
fra loro. E' esattamente la classe di lavoro che le 31 versioni passate
hanno gia' consumato senza mai produrre uno sha.

## Errore 4 — il conteggio "adds" non conta le istruzioni che girano

`03:85-87` stima il costo di N=8: *"+3 adds and +4 gathers to remove 4 cold
reads ... in exchange for three extra serial mixed additions (~7.5 field
multiplies each, ~22 field multiplies, in a chain that is the program's
critical path)"*.

Ma "11 adds / 10 adds" non e' il conteggio del lavoro reale. `pinning.cu:1683`
dice che la catena copre **10 o 4 trip**, e `pinning.cu:2002-2004` mostra che
**ogni trip e' un `_PointAddXYZZ_pair`**. Il termine "add" di `03` e' un
termine di chunk, non un'istruzione: un chunk costa un'intera
`PointAddXYZZ`, non un'addizione.

Questo e' lo stesso errore che `03:59-66` rimprovera a terrapinelf per il
-66.9% ("those two numbers cannot both be right"). Lo stesso errore, sulla
stima che regge la proposta. Una previsione ">5%" costruita su un conteggio
di istruzioni che non e' il conteggio delle istruzioni non e' una previsione.

## Il verdetto, e perche' non implemento

Non implemento N=8. Tre ragioni, in ordine di peso:

1. **Il mandato e' di +1% sul record, non "provare la cosa di 03".** La
   soglia e' 1 018 288 896. `03` promette ">5%", ma la promessa poggia su un
   conteggio di istruzioni errato (errore 4) e su un muro di registri che il
   sorgente non conferma (errore 1).

2. **La classe di movimento e' gia' stata smentita ufficialmente.**
   `ITERATIONS.md:24-33` (v23 GLV10): **399 697 199**, -55.7%. La voce:
   *"Table geometry in the fewer-adds direction is CLOSED — falsified
   officially by us and independently by measurement."* E `DEAD-ENDS.md:19`:
   *"GLV is closed."* N=8 e' piu' add a tabella piu' piccola: e' quel
   movimento. L'unica differenza e' che la tabella si rimpicciolisce, il che
   e' il caso *favorevole* — ed e' anche l'unica ragione per cui valuterei
   N=8 contro GLV10, pur senza poterlo misurare.

3. **Non posso verificarlo, e il brief e' esplicito.** "Non dichiarare
   migliore nessuna versione se non hai lo sha del cubin cambiato in mano."
   Qui non ho lo sha. Non ho il cubin. Non ho la shell per produrli.

Se un worker con shell rilegge questo: il punto 2 e' l'unico che cambia la
decisione, e per cambiarlo serve una misura, non un argomento. Il test
onesto sarebbe GLV14/N=8 a tabella 16 MiB, *non* N=8 a 42 MiB, perche' e'
l'unico modo di separare "meno DRAM" da "meno istruzioni" — ma e' anche
il caso peggiore per la finestra di persistenza, e va detto prima di spenderci.
