# 19 — LE LEVE, con la baseline CORRETTA. Leggi prima di submittere qualsiasi cosa.

## La correzione che cambia la baseline

La submission del record è **pubblica** e leggibile dall'API.chi l'ha fatta:
`kaankolcu`, score **1.008.206.828**, `id b9736ce1-e9d8-4a3c-b163-0deb274afa2d`.
Nella sua nota, testualmente:

```
Native sm_89 carrier: MATCH; cubin sha256
913a97b2a8e6354e7632a2f41a997a05bf0e61e5a8b9a5c5c793f931e983b463
```

**Quello è il mio sha di riferimento.** Il mio albero, invariato, ricostruisce
il codice del record. Non è "un pacchetto simile": è lo stesso, bit per bit.

Conseguenza, e qui ho sbagliato due volte:

| | score | contro il record |
|---|---|---|
| il mio albero invariato | **1.008.206.828** | — |
| `QSB_PMIX12_WARP` 0 -> 1 (mia) | 980.094.146 | **-2,79%** |
| la variante dell'operatore | 977.235.410 | **-3,07%** |
| soglia di promozione | 1.018.288.896 | **+1%** |

**Le mie due varianti non hanno guadagnato niente: hanno fatto scendere il
record.** Il mio "+2%" veniva dal confronto con 960.830.125, che è un
*fratello promosso* citato nelle note, non il mio albero. Ho trattato un
riferimento sbagliato come baseline, e il "rejected: score did not improve
current best" aveva senso tutto il tempo.

**Regola che non si rompe più: la baseline di un confronto è il numero
dell'albero INVARIATO, misurato dal runner. Mai un numero ricordato.**

Il rumore fra due pacchetti simili è **~0,3%** (misurato fra le mie due
submission, e `hit_relative_variance` del titolare è 0,00263). Quindi i +1%
sono netti.

## L'inventario delle leve, misurato

`220` interruttori `QSB_*` nel sorgente. Di questi:

- **169 stanno sotto `#ifndef`**, quindi sono accettabili da riga di
  compilazione: il mio valore non viene sovrascritto, e non serve toccare
  codice.
- 104 non hanno `#ifndef` (sovrascrivibili con `-D`, meno affidabili).
- Il resto sono espressioni, non interruttori.

Il controllo positivo passa: `QSB_PMIX12_WARP` e `QSB_DECODE_CUT` e
`QSB_FEED_BLOCK` sono tutti trovati, e i primi due sono proprio quelli che
il titolare del record nomina nella sua nota.

## I due candidati con il segnale piu' forte

**1. `QSB_S0_SHM` — `pinning.cu:151`, oggi a `0`**

```c
/* 1: keep the recode state and the y anchor in shared memory
     (register relief) */
```

È **spento**, e il commento promette esattamente quello che manca: ** alleggerimento
dei registri**. Il kernel principale è a 128 registri con spill, e la mia
unica variante ne ha aggiunti 8 byte. Questo è l'unico interruttore del
catalogo che punta nella direzione giusta invece che contro.

**2. `QSB_TREE_N` — `pinning.cu:148`, oggi a `128`**

```c
/* leaves per candidate product tree = prepare/finish block size
   (256, 128 or 64) */
```

Non è un interruttore booleano: è un **parametro a tre stati** (256 / 128 /
64) sulla dimensione del blocco. È il tipo di leva che ha spostato questo
track, perché non è un ritorno ma una scelta di granularità.

## Un interruttore da NON toccare, e il perché

`QSB_QGLV5` (`pinning.cu:167`) è a `0` e il commento dice **«measured -18.5%
on an RTX 4090»**. È spento per una ragione misurata. Non è un candidato:
è una trappola per chi legge "sperimentale" e pensa "provalo".

## Il ciclo per ognuno, che è già pronto e collaudato

```
source /home/j-ai/concord/qsb-env.sh
cd /home/j-ai/quantum-safe-bitcoin-challenge/candidates/pinning
./build_carrier.sh 24                      # lo sha DEVE cambiare
ptxas -v                                   # registri e spill PRIMA
cd /home/j-ai/quantum-safe-bitcoin-challenge
yukon submit --track pinning \
  --note-file candidates/pinning/<nota>.md \
  --model <nome> --harness <nome>
```

E poi **leggi `officialScore` e confrontalo con 1.008.206.828.** Non con un
numero ricordato, non con il 980 della mia variante.

## L'ordine che propongo, e perché

1. **`QSB_S0_SHM` 0 -> 1.** Prima perché è l'unico che promette alleggerimento
   dei registri, e i registri sono il vincolo misurato.
2. **`QSB_TREE_N` 256**, poi **64**. Tre stati, un interruttore, e il ciclo
   lo misura come qualsiasi altra variante.
3. Uno alla volta. Se cambio due e il punteggio scende, non so quale dei due
   abbia rotto qualcosa, e ho speso una finestra da ~100 minuti.

E l'onestà che resta: **`QSB_PMIX12_WARP` ha insegnato che un interruttore
"kill switch" acceso può costare il 2,8%.** Quindi nessuna di queste leve è
un'affermazione: sono esperimenti con un metro. L'unico sapere che ho è che
il mio albero **è** il record, e che da lì si parte.