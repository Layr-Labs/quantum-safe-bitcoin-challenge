# 18 — LE CORREZIONI. Otto affermazioni di questo track che non reggono il confronto.

Ogni riga qui e' stata verificata per esecuzione, non riportata. Se una di
queste ti fa perdere tempo, e' la tua ora a buttarla.

## 1. `submissions: []` era falso: la submission c'era

Tre fonti, stesso id `368ac8dc-390e-4650-bf2b-9b5084224e74`, `status:
validating`, `officialScore: null`. Il brief partiva da `submissions: []` e
proseguiva come se la prima submission fosse ancora da fare. Non era da fare.

## 2. Il gate `readelf` del brief non puo' passare: il file non esiste

    readelf -h pinning_sm89.cubin | grep -q "Flags:.*0x59"

`pinning_sm89.cubin` **non esiste** da nessuna parte (`ls` exit 2, `find` 0
risultati). `build_carrier.sh:9-11` compila dentro `$(mktemp -d)` e scrive
`c.cubin`. Il gate corretto, sul cubin effettivamente prodotto, **passa**:

    Flags:                             0x590559
    cuobjdump -elf: sm=89, toolkit=128, flags = 0x590559

`LANG` era gia' `C`: la nota 05 ha ragione sul locale e torto sul nome del file.

## 3. L'albero **e'** il record. Non c'e' margine da scoprire li'.

    git diff --name-status 8d07d3e -- candidates/pinning/     -> (vuoto)

`8d07d3e` e' "Accept submission b9736ce1" = kaankolcu = **1.008.206.828**. Il
worktree e' quei byte, `qsb_carrier_sm89.h` compreso, quindi il cubin
`913a97b2...` **e' l'immagine device che ha prodotto il record**.

Questo chiude per sempre due voci aperte nelle note precedenti
(`_notes/01-TREE-IDENTITY.md:115`, `_notes/10-N8-E-IL-BASELINE.md:59-70`) e ne
ribalta la premessa: **non c'e' una baseline da costruire, la baseline e' il
record.** Ogni "migliora di N" che nonsia sul record e' una speranza, non un
progresso. Il riferimento di confronto e' 1.008.206.828, non 995.3M.

## 4. La soglia e' un passo solo, e i migliori rifiutati ci arrivano a 0,78 %

    minScoreImprovementBips: 100        ->  1.008.206.828 x 1.01 = 1.018.288.896

Dalle 2101 submission, le migliori **rifiutate**:

    ercumentyildirim  1.016.064.891   (+0,78 %)  "fell short of the required 100 bips"
    cefika            1.015.939.991   (+0,77 %)  stesso testo

`improved: true` **non basta**: il gate e' 100 bips in un passo. E il best non si
alza con loro: si alza solo con una promossa, quindi il record e' fermo al 28
settembre. 1485 righe `rejected` e zero promosse dopo quella data.

Nota aritmetica: `1,008,206,828 x 1,01 = 1,018,288,896,28`, quindi l'intero
minimo che passa e' **1.018.288.897**.

## 5. **Una sola submission in volo per account.** Cambia il piano delle note 15

    {"error":{"code":"conflict",
      "message":"account already has 1 submission(s) in flight for this benchmark (limit 1)"}}

La nota 15 prescrive "prepara la variante dopo, cosi' che quando arriva il primo
risultato ci sia subito la seconda da mandare". **Impossibile**: due in volo non
esistono. Lo slot e' una risorsa scarsa e la domanda non e' "posso mandare anche
la seconda?" ma **"quale delle due merita lo slot?"**.

## 6. WARP=1 **non e' una leva**. Le note 13 e 15 indicano la direzione sbagliata

Letto in `pinning.cu:232-238`, non da una nota:

    #if QSB_PMIX12_WARP
      #define QSB_PMIX12_SEL()  (((blockIdx.x*(QSB_TREE_N/32)+(threadIdx.x>>5))*N) & (K-1)) < N)
    #else
      #define QSB_PMIX12_SEL()  ((blockIdx.x&(K-1u))==0u)
    #endif

Con `QSB_PMIX12 65536` (`:183`) e `QSB_TREE_N 256` (8 warps per blocco):

- **WARP=0** -> un blocco su 65536 = 8 warps = **1/8192** dei candidati
- **WARP=1** -> un warp su 65536 = **1/65536** dei candidati

WARP=1 **riduce di 8x** una quota gia' sotto lo 0,002 %, e paga 8 byte di stack
spill sul kernel caldo al 100 % dei candidati. La promessa del commento
(`:191`, "i blocchi residenti di ogni SM portino la stessa quota invece che blocchi
interi su pochi SM") e' un argomento di bilanciamento che ha senso a `K=16` e a
`K=65536` non ha soggetto: non ci sono blocchi da distribuire.

**"Hai 128 registri e uno spill di 8 byte: quella e' la direzione con piu'
spazio" (nota 15:77) e' falso per costruzione aritmetica.** Quella submission e'
stata annullata e sostituita: vedi nota 17.

## 7. `-maxrregcount` non e' mai stato usato, ma **ogni suo effetto fisico e' gia' misurato e perde**

Zero occorrenze in sorgente e script (le 3 hit sono in note, come
*raccomandazione*). Ma la direzione non e' vergine: `__maxnreg__ 108` ->
**911,2M vs 923,2M** di controllo; 5 blocchi/SM -> **96 registri + 44 B di
spill**; 8 blocchi/SM su stage-2 -> nessun guadagno; PR #700 five-block ->
**-1,99 %**. E la regola che emerge e' piu forte di ciascun dato:

> **Ogni meccanismo di questo track che ha ridotto i registri del kernel caldo
> senza aggiungere spill e' risultato piu' lento o neutro.**

Una leva "non provata" i cui effetti sono tutti chiusi non e' una leva aperta.

## 8. `DEAD-ENDS.md` e `NEXT-OPTIMIZATIONS.md` sono scritti contro il frontier 778.624.395

Entrambi ancorati a una frontier di tre settimane fa. La voce "GLV/joint/radix
CLOSED" di `NEXT-OPTIMIZATIONS.md:50` si riferisce alla variante **random-access**,
mentre il **dense** GLV e' stato promosso tre volte (`ITERATIONS.md:1513-1516`).
Chi legge quei due file per decidere cosa non fare, chiude cose che sono invece
la strada giusta.

## Non risolto, e lo dichiaro

La coppia di taratura del track e **contraddittoria e non risolta**:

    GLV12 (4 letture fredde)  960.830.125   promosso
    GLV11 (6 letture fredde)  948.943.797   promosso   ->  1 lettura fredda ~0,6 %
    -66,9 % per +1 lettura sparsa 64 B   (ITERATIONS.md:1701, da una nota altrui)

Non possono essere entrambe vere per lo stesso programma. Io agirei sulla coppia
stessa-progetto, che e' in banda; il -66,9 % e' stato prezzo di una perturbazione
marginale su un'altra geometria. **Chiunque si appoggi al -66,9 % sta appoggiandosi
a una cifra che nessuno di noi ha riprodotto.**

## Lo stato, in una riga

    sottomesso  9c5512fd-b0a7-4b2e-aec8-f02125002d38   status validating, score null
    annullato   368ac8dc-390e-4650-bf2b-9b5084224e74   "cancelled by submitter"
    cubin       913a97b2...463  476832 B   invariato di proposito (pacchetto host-only)
    finestra    2026-10-07T23:00:00Z

Il numero non c'e' ancora: la coda era satta (9 in volo, il piu' vecchio da
19:01Z) e la mia era 7a per `createdAt`. Nessun tempo locale riportato come score:
qui non gira CUDA.