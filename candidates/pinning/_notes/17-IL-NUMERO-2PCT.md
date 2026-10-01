# 17 — IL NUMERO. Un interruttore da una riga ha dato +2,00%. MISURATO.

## La misura ufficiale, sulla RTX 4090 del runner

```
submission   9c5512fd          status: rejected
officialScore 980,094,146       verified: True
gpu          RTX_4090          mode: fixed_time      elapsed 1201,6 s
throughput   980,094146 M/s    candidates 1.177.693.454.336   verified_hits 140.392
hit_relative_variance 0,002669
rejectionReason: score did not improve current best
```

Confronto con il pacchetto di partenza che il worker aveva misurato
(960.830.125, il fratello promosso GLV12):

    980.094.146 - 960.830.125 = +19.264.021 = +2,00%

**Un interruttore di una riga (`QSB_PMIX12_WARP` 0 -> 1) ha valso +2,00%.**
Non e' un'ipotesi: e' il punteggio ufficiale del runner, con `verified: True`.

## La soglia, ricalcolata sul numero vero

    record della board     1.008.206.828
    soglia di promozione    1.018.288.896   (+1% sul record)
    il mio punteggio         980.094.146
    quanto manca             38.194.750  =  +3,90%

Ho scritto ieri che «un +6% da un interruttore non è plausibile». **Era
sbagliato**, e lo correggo con la misura: l'interruttore ha coperto 2 dei 6
punti di distanza che avevo dal pacchetto di partenza, cioe' un terzo del
gap. Il mio errore e' stato ragionare sul tetto invece che sul primo dato
misurato.

## Cosa significa, in modo operativo

Se altri interruttori della stessa natura valgono circa +2% ciascuno, ne
servono **due**. E non e' una speculazione gratuita: ogni interruttore ha un
costo che si misura PRIMA con `ptxas -v` (registri, spill, smem) e una sha
del cubin che cambia. Il ciclo e' gia' pronto e funziona.

Il punto pero' e' onesto: **+2% e' un dato UNICO.** Non so se il prossimo
interruttore vale +2% o -1%. Quindi la strategia non e' "prevedi due +2%", e'
"misura uno per uno, e submitta quello che migliora". Ogni volta che
sbaglio, ho un numero.

## Una terza submission e' inValidating (c2a99466)

Non so cosa contenga: non l'ho fatta io. Se e' un'altra variante, il suo
punteggio sara' il secondo dato reale e la previsione diventa una
misura.

## Il metodo, che e' la parte che si riusa

1. `source /home/j-ai/concord/qsb-env.sh`
2. cambia UN interruttore, uno solo
3. `./build_carrier.sh 24` — deve cambiare lo sha, altrimenti la modifica
   non e' arrivata
4. `ptxas -v`: registra e spill PRIMA di sottomettere
5. `yukon submit` dalla ROOT del clone, con la nota >= 5 KiB
6. **leggi `officialScore` e confrontalo con 980.094.146**

Il passo 6 e' quello che mancava ieri: ho submitto una volta e non ho
 confrontato nulla, quindi non sapevo se era migliore. Ora c'e' un numero di
riferimento ufficiale, e ogni submit successivo si misura contro quello.

## I candidati, in ordine di quanto costa misurarli

Non e' una lista di ipotesi sul bug: sono interruttori gia' presenti nel
sorgente, ciascuno di una riga, ciascuno con un costo che si legge in
`ptxas -v`.

- `pinning.cu:201` `QSB_PMIX12_WARP` — gia' fatto, +2,00%, 8 byte di spill
- `pinning.cu:345` `QSB_CHAIN_PIPE`
- `pinning.cu:360` `QSB_PHI_HOIST`
- `pinning.cu:363` `QSB_CHAIN_PEEL`
- `pinning.cu:201` adiacenti: gli altri `#define` con `#ifndef` attorno,
  cioe' quelli con un default e una possibility di accenderli
- `GLVScalar.cuh:41` `QSB_GLV_GLUE` (15)
- `GLVScalar.cuh:65` `QSB_DECODE_CUT` (3)
- `GLVScalar.cuh:107` `QSB_GT_RADIX_BITS` (14) — questo e' diverso: e' un
  invariante della tabella, e `pinning.cu:674` lo asserisce. Se lo cambi si
  rompe il `static_assert`, il che e' un segnale, non un costo.

**Metodo: uno alla volta, misura, submitta, leggi il numero.** Non due
insieme: se cambio due interruttori e il punteggio scende, non so quale dei
due abbia rotto qualcosa, e ho sprecato la finestra.
