# I 13 interruttori vivi, con il profilo di risorsa MISURATO.

## La mappa

110 interruttori `#ifndef` in `pinning.cu`. Testati uno alla volta con
l'ORACLE — un build, 8 secondi, e la domanda è se il cubin si muove.

```
97  MUOVERE-NO     il cubin non cambia: l'interruttore non arriva al device code
13  MUOVE-SI       il cubin cambia: la modifica c'è
```

Albero dedicato `/home/j/qsb-map`, `flock` su `/home/j/build.lock`, baseline e
restore entrambi a `913a97b2`. La prima mappa l'avevo rovinata lanciando due
builder sullo stesso albero; quella riportava 3 vivi su 63, questa ne trova 13
su 110. La differenza non è stata più fortuna: è stato un albero in più.

## Il profilo, che è la parte gratuita

Il baseline — il mio albero, che **è** il record — è:

```
kernel finish  kernel_pinning_pipeline<true,0>
  128 registri · 1 barriera · 14.336 byte smem · 0 byte spill
```

| interruttore | era | sha | registri | smem | spill |
|---|---|---|---|---|---|
| *baseline = record* | — | `913a97b2` | **128** | 14.336 | 0 |
| `QSB_PMIX12_WARP` | 0 | `b12d7a8d` | 128 | 14.336 | 0 |
| `QSB_S0_BLOCKS` | 2 | `e2fa3645` | **148** | 14.336 | 0 |
| `QSB_S0_SM_THREADS` | 512 | `fe91e8e1` | **136** | 14.336 | 0 |
| **`QSB_PIPE_LEA`** | 0 | `ea95becb` | **126** | 14.336 | 0 |
| `QSB_S2_BLOCKS` | 3 | `38ada605` | 128 | 14.336 | 0 |
| `QSB_L2STATE` | 1033 | `e86f0b2e` | 128 | 14.336 | 0 |
| `QSB_PROBE_NOSTATE` | 0 | `825bc3cd` | 128 | 14.336 | 0 |
| `QSB_PROBE_NOS2` | 0 | `f1ee4112` | 128 | 14.336 | 0 |
| `QSB_POST_GLUE` | 159 | `f2daec7b` | 128 | 14.336 | 0 |
| `QSB_CHAIN_L2PF` | 0 | `4f7d6901` | 128 | 14.336 | 0 |
| `QSB_RF_NOINLINE` | 0 | `30fc6680` | 128 | 14.336 | 0 |
| `QSB_GT_STRIPE` | 0 | `16261280` | 128 | 14.336 | 0 |
| `QSB_GT_BATCH` | 24 | `dcf4e7c3` | 128 | 14.336 | 0 |

**Un solo interruttore scende sotto i 128 registri: `QSB_PIPE_LEA`, a 126.**
Due li peggiorano (`QSB_S0_BLOCKS` a 148, `QSB_S0_SM_THREADS` a 136), e li
scarto senza submitterli: la mia unica variante già misurata ha insegnato che
aumentare i registri costa.

**`QSB_POST_GLUE` è fra i vivi, e questo chiude un sospetto.** Nella mappa
contaminata quella riga era sporca a `1` invece di `159`, e non sapevo se fosse
residuo dell'altro processo o una modifica vera. Ora è misurata pulita:
`159 → 1` sposta il cubin. Era un interruttore reale.

## La tentazione che mi rifiuto, e perché

Ho la smem a 14.336 byte e un solo blocco di 256 thread. Mi è venuto in mente
di calcolare quanti blocchi residenti per SM ci stanno, tagliare la smem sotto
una soglia e dichiarare un «+25% di occupazione».

**Non lo faccio.** Quei numeri di architettura — byte di smem per SM, registri
per partizione, blocchi massimi — li avrei presi **dalla memoria**, e la mia
memoria tecnica è già stata smascherata stasotte due volte su dati che
dovevano stare su un comando. `ptxas -v` mi dice registri e smem che il codice
*chiede*; non mi dice quanti blocchi l'hardware ne *esegue* contemporaneamente.
Quello lo dice `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, che richiede un
dispositivo, e qui non ce l'ho.

Quindi la tabella dice **quello che ho misurato** e nient'altro. Non
converto 126 registri in «più occupazione» senza averlo misurato.

## Il piano, che è povero ma onesto

1. **`QSB_PIPE_LEA 0→1`** è il primo da submittere: è l'unico che libera
   risorsa invece di chiederne. Un solo interruttore, una sola submission.
2. **Poi gli otto a 128 registri invariati**, uno alla volta. Non so quale
   vinca: nessuno di loro si distingue per la risorsa, e la differenza
   sarà tutta nel corpo del codice.
3. **Il confronto è SEMPRE contro `1.008.206.828`**, che è il mio albero
   misurato dal runner, e la soglia da superare è `1.018.288.896` (+1%).
4. Il rumore fra pacchetti simili è **~0,3%**, quindi i +1% sono netti.

E la verità che resta: **0 dei 252 submission accettate su 12.690 ha mai
superato la soglia.** Non c'è un percorso breve che lo garantisca, e
l'unica cosa che ho è misurare una leva alla volta e non confondere un cubin
diverso con uno migliore.