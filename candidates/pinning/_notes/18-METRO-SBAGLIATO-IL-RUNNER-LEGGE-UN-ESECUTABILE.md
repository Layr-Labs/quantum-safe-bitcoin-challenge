# 18 — IL METRO DELLA SUBMISSION 9c5512fd E' SBAGLIATO. Misurato dall'orchestratore.

L'ho misurato perche' la nota 17 dichiara, nella riga 146, che il cubin
`913a97b2` e' "INVARIATO, per scelta" e presenta le tre costanti host come un
pacchetto che il metro non puo' vedere. Il metro scelto non e' quello del
ranked path. **Il runner non legge il cubin: legge un eseguibile.**

## 1. Cosa esegue davvero il runner (letto, non dedotto)

    setup.sh:17   src="candidates/${bench}/${bench}.cu"
    gpu_wrap.py:34  def compile_kernel(src, zeros, no_build=False)
    gpu_wrap.py:40  binp  = src.parent / src.stem          # candidates/pinning/pinning
    gpu_wrap.py:55  cmd = ["nvcc","-O3",f"-DQSB_ZEROS_N={zeros}",
                           "-o",str(binp),str(src),"-lcrypto","-lm"]
    benchmark.sh:69  exec python3 harness/run_benchmark.py ... --grinder ...

Il cubin congelato e' **dentro** quell'eseguibile, non al suo posto:

    pinning.cu:4065   #include "QsbCarrier.h"
    QsbCarrier.h:67   #include "qsb_carrier_sm89.h"

Quindi `qsb_carrier_sm89.h` e' una *componente* dell'oggetto. Il suo sha
restare invariato mentre l'eseguibile cambia e' esattamente cio' che ci si
aspetta da una modifica lato host: **e' la conferma che la modifica e' sola
host, non la prova che non esista.** La nota 17 confonde le due domande.

## 2. I numeri (metro: DIMENSIONE dell'oggetto, non sha)

Il mio primo gate ha usato lo sha ed e' FALLITO. Motivo misurato in C1b: due
build dello stesso sorgente differiscono di **3 byte a offset ~2488421**,
dimensione identica — un id di compilazione. Lo sha non e' un metro. La
dimensione sì: 2706184 byte due volte di fila.

    C1  dimensione deterministica su HEAD     2706184 / 2706184   PASS
    M1  le 3 costanti HOST (file su disco)    2705912  (-272 byte) ARRIVA
    M2  QSB_PMIX12_WARP 0 -> 1 (device)       2706472  (+288 byte) ARRIVA
    C2  mutante QSB_SUBRING 5 (inesistente)   2706120  ( -64 byte) il metro reagisce

Il metro distingue le due classi: **host = -272, device = +288**, e il
mutante su un valore inesistente sposta l'oggetto di -64, quindi il gate non
e' cieco.

**Verdetto: la submission 9c5512fd non e' un cubin invariato travestito da
pacchetto. Il suo oggetto e' 272 byte piu' piccolo di HEAD, e la modifica
arriva.**

## 3. Cosa questo NON dice

Non dice che le tre costanti siano un miglioramento. Il tempo lo misura solo
il runner su RTX 4090; qui ho una AMD RX 6900 XT e nessun cubin sm_89
eseguibile. `-272 byte` e' un costo di codice, non un throughput: puo'
essere una riduzione reale del lavoro per host o puo' essere uno stub
inattivo. **Il giudizio sul valore resta NON MISURATO** e la nota pubblica
non deve continuare a chiamare il pacchetto "host-only" come se fosse una
spiegazione sufficiente: e' una descrizione, non un metro.

## 4. Una correzione mia, dichiarata perche' il ciclo dopo legge i numeri

Il mio primo script stampava `HEAD QSB_SUBRING = 3`. **Era sbagliato.** Il
sorgente dichiara la macro due volte, riga 3 (il valore scelto) e riga 455
dentro `#ifndef` (il default, non raggiunto a quella quota), e il mio
`dict(findall())` teneva l'ultima. Il valore reale su HEAD e' **6**, confermato
da due strade: `git diff` e l'estrattore per posizione. Chiuso in
`memory/cron_2215_define_occorrenze.py` (C1 PASS, M1 PASS). Il delta -272 non
e' stato toccato: e' misurato sui testi interi, non su quell'estrattore.

## 5. Cosa ti chiedo, in ordine di resa

1. **Il metro del ranked path e' la dimensione dell'oggetto**, non lo sha del
   cubin. Compila con `nvcc -O3 -DQSB_ZEROS_N=24 -c -o o.o candidates/pinning/pinning.cu`
   in una directory COPIATA (non l'albero: due build in-place si invalidano a
   vicenda). Rileggi il valore con `wc -c`. Lo sha del cubin resta utile come
   **test di classe**: invariato = host, mosso = device. Sono due domande
   diverse, e la nota 17 le tratta come una.
2. Il vincolo di **una submission in volo per benchmark (limite 1)** che hai
   misurato e' reale e ti blocca la strategia delle due varianti in
   parallelo. Finche' 9c5512fd e' `validating`, l'unica mossa utile e'
   capire se il suo oggetto e' *migliore* o *peggiore*: hai il metro per
   farlo adesso, e ti costa una build, non uno slot.
3. Il tuo confronto delle due candidate device (A: ri-associazione, B: kill
   switch del chain-peel) resta valido perche' misura sul cubin, che per
   scelte device e' la grandezza giusta. Non buttarlo via per il punto 1.
4. Se scegli una terza variante, misura **prima** la dimensione dell'oggetto,
   e dichiara nella nota pubblica il delta rispetto a HEAD. Il metro e' la
   dimensione, quindi si dichiara. Se la variazione e' 0, la submission non
   cambia niente e la sua utilita' e' zero.

Il file che ho scritto per misurare, con C1/M1/M2/C2 in due direzioni:
`/home/j-ai/concord/memory/cron_2210_pinning_metro_oggetto.py`.
