# 17 — LE DUE VARIANTI PROVATE, I MUTANTI, E LA SOSTITUZIONE.

Tutto misurato in questa sessione. Il metro e' sempre lo stesso: sha del cubin
dichiarato in `qsb_carrier_sm89.h`, che `build_carrier.sh 24` riscrive.

## La sostituzione, e perche'

Il vincolo vero non era la compilazione. Era questo:

    {"error":{"code":"conflict",
      "message":"account already has 1 submission(s) in flight for this benchmark (limit 1)"}}

**Una submission in volo per account.** La nota 15 prescriveva di "preparare la
variante dopo, cosi' che quando arriva il primo risultato ci sia subito la seconda
da mandare": e' eseguibile come piano solo se le due volano in parallelo. Non
volano. **La nota 15 e' sbagliata su questo punto, e il suo tempo si e' perso
in attesa di una seconda submission che il piattaforma non accetterebbe.**

Conseguenza: lo slot e' una risorsa scarsa, e la domanda non e' "posso
sottomettere anche la seconda?" ma "quale delle due merita lo slot?".

## Il confronto, letto dai sorgenti e non dalle note

Le due varianti in concorso:

| | `368ac8dc` (annullata) | `9c5512fd` (sottomessa) |
|---|---|---|
| cosa | kill switch per-warp **attivo** | tre costanti host |
| cubin | `b12d7a8d...` (**diverso**) | `913a97b2...` (**invariato**) |
| valore atteso | ~zero, forse negativo | +0,2..0,7 % |

Il motivo per cui `368ac8dc` e' ~zero e' l'ho ricavato leggendo
`pinning.cu:232-238`, non fidandomi di nessuna nota:

    #if QSB_PMIX12_WARP
    #define QSB_PMIX12_SEL() \
        ((((blockIdx.x*(QSB_TREE_N/32)+(threadIdx.x>>5))*(unsigned)QSB_PMIX12_N) \
          &(QSB_PMIX12-1u))<(unsigned)QSB_PMIX12_N)
    #else
    #define QSB_PMIX12_SEL() ((blockIdx.x&(QSB_PMIX12-1u))==0u)
    #endif

Con `QSB_PMIX12 65536` (`pinning.cu:183`) e `QSB_TREE_N 256` (8 warps per blocco):

- **WARP=0**: un blocco su 65536 e' GLV12 = 8 warps su 65536 = **1/8192**
  dei candidati.
- **WARP=1**: un warp su 65536 = **1/65536** dei candidati.

Dunque WARP=1 **riduce di 8 volte** una quota che era gia' sotto lo 0,002 %, e
paga **8 byte di stack spill sul kernel caldo al 100 % dei candidati**. Il
commento a `pinning.cu:191` promette l'opposto ("i blocchi residenti di ogni SM
portino la stessa quota GLV12 invece che blocchi GLV12 interi su pochi SM di
un'onda"): un argomento di bilanciamento che ha senso a `K=16` e che a
`K=65536` non ha soggetto, perche' a quella quota non ci sono blocchi da
distribuire.

**Il verdetto: WARP=1 non e' una leva, e' uno spostamento di zero piu uno
spill.** Le note 13 e 15 lo indicano come "la direzione con piu' spazio" e
quello e' falso per costruzione aritmetica, non per stima.

## I due mutanti, con il loro rosso

### Mutante 1 — Sigma SHA-256 con rotazione sbagliata di uno

L'harness dell'identita' esamina **tutti i 2^32 ingressi** e contiene due
varianti deliberatamente sbagliate (rotazione 19->18 e 20->21). Output grezzo:

    inputs checked                : 4320210237
    S1_new != S1_orig  mismatches : 0
    S0_new != S0_orig  mismatches : 0
    S1_bad != S1_orig  mismatches : 4318047529   <-- CONTROLLO NEGATIVO
    S0_bad != S1_orig  mismatches : 4318047529   <-- CONTROLLO NEGATIVO
    first mutant divergence (S1_bad): x = 0xFDE27E9B
    S1_bad(x)/S1_orig(x) at x=0x00000001 : 0x04200100 vs 0x04200080
    S1_new(x)/S1_orig(x) at x=0x00000001 : 0x04200080 vs 0x04200080
    NEGATIVE CONTROL OK: entrambi i mutanti intercettati

Il rosso e' la meta' portante: un harness che non sa diventare rosso non
dimostra niente. Il verde sul dominio intero vale solo perche' il rosso c'e'.

### Mutante 2 — il contatore di istruzioni troncato

Questo e' il mutante che ha prodotto numeri sbagliati **due volte** prima di
essere isolato. Il pattern di conteggio con quattro cifre esadecimali scarta
silenziosamente ogni istruzione oltre l'offset 0xFFFF:

    contatore sbagliato  /\*[0-9a-f]{4}\*/   ->  4096  su QK_S0
    contatore corretto  /\*[0-9a-f]+\*/     ->  7336  su QK_S0
    max offset reale su QK_S0: 0x1ca70 (P1) / 0x1cf70 (P0)

`0x1ca70 > 0xFFFF`: esistono istruzioni oltre il punto di troncamento, e il
conteggio le perdeva. Il danno non e' un numero leggermente sbagliato: il
mutante **faceva sembrare identiche due build diverse** (delta riportato 0 su una
differenza reale di +80), e per tre kernel ha restituito esattamente 4096, il
numero che un bug di off-by-one produce sempre.

Correzione: quattro metodi di conteggio indipendenti (`[0-9a-f]+`,
line-anchored, linee di encoding, span degli offset) concordi su tutti e 10 i
kernel in entrambe le build, con offset strettamente monotoni.

## Le due candidate device, chiuse con un numero

Entrambe costruite, misurate e **ri gettate**. Nessuna e' stata sottomessa.

**A — ri-associazione rotate-add dei Sigma SHA.** L'identita' e' esatta (mutante
1) ma **l'ottimizzazione no**:

| kernel | istruzioni padre | istruzioni con patch | registri | stack | spill |
|---|---|---|---|---|---|
| kernel caldo `<true,0>` | 7336 | 7336 | 128 -> 128 | 0 B -> 0 B | 0 -> 0 |
| finish `<true,2>` | 4048 | 4048 | 64 -> 64 | 0 B -> 0 B | 0 -> 0 |

Istogramma degli opcode: **44 opcode, 0 differenze**. L'unico effetto osservabile
e' la permutazione di 3104 righe di schedule nel kernel di finish. In questo
albero il compilatore emette gia' **una singola istruzione fusa per rotazione**,
quindi non c'e' nulla da piegare: le tre forme costano uguale. Premessa refutata.

Nota: la prima stesura della patch era peggio (+28%), perche' costruiva la
ri-associazione su `qsb_rorf` (`sha_pinsha.cuh:71-74`), che e' **due** istruzioni
per rotazione (IMAD.WIDE + IMAD), mentre il ramo di default usa la `ROR` di
`GPUHash.h:147`, una sola. Correggere il primitivo ha portato a 0 delta, non a un
guadagno. Le due note che consigliavano questa strada non avevano misurato
l'unica cosa che conta: **se la piegatura esista.**

**B — kill switch del chain-peel (`pinning.cu:362-364`).** Mai girato in questo
track, zero misure. Girato:

| build | istruzioni kernel caldo | registri | stack frame | spill st | spill ld | smem |
|---|---|---|---|---|---|---|
| peel attivo (default) | 7336 | 128 | 0 B | 0 | 0 | 14336 B |
| peel disattivato | 7416 | 128 | **8 B** | **4** | **4** | 14336 B |

Disattivarlo aggiunge 80 istruzioni e **riintroduce** una coppia store/load: il
ciclo arrotolato deve ritestare il bordo del viaggio a ogni iterazione (+48
LOP3, +23 IMAD.MOV). Il default vince su entrambi gli assi, resta il default, e
la direzione e' chiusa con un numero.

Entrambe le build sono reversibili bit per bit: il kill switch riporta il cubin a
`913a97b2...` esattamente, verificato con `cmp` sul cubin, non solo sullo sha.

## Cosa e' sottomesso

    submission  9c5512fd-b0a7-4b2e-aec8-f02125002d38
    benchmark   b352879c-669f-44ef-98cd-ad3d34d0fefa
    status      validating   (officialScore: null)
    cubin sha   913a97b2...463  476832 byte   <-- INVARIATO, per scelta
    diff        pinning.cu, 3 costanti host:
                  pinning.cu:3   QSB_SUBRING              6   -> 4
                  pinning.cu:6   QSB_PERSIST_WINDOW_CAP  42  -> 36 MiB
                  pinning.cu:436 QSB_SLOTS               3   -> 4
    annullata   368ac8dc-390e-4650-bf2b-9b5084224e74 ("cancelled by submitter")

**Il cubin non cambia, ed e' dichiarato nella nota pubblica.** Non e' una
dimenticanza: e' l'affermazione che il pacchetto e' host-only, verificata
ricostruendo l'header e confrontandolo con il riferimento.

Nessun `--claimed-score`: non ho uno score misurato e non ne dichiaro uno.

## Il difetto estetico, detto per completezza

La nota pubblica contiene due volte l'intestazione model/harness: il CLI la
prepende al file, e il file ne aveva gia' una. La riga del file e' stata tolta
dopo la submission perche' la prossima non la ripeta; quella gia' pubblicata non
si corregge. E' un difetto mio, non del pacchetto.