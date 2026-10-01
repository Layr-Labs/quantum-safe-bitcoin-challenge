# 10 — il fatto che chiude la proposta N=8: questo albero E' gia' GLV11

Questo e' il punto che mancava, e ribalta la domanda. Nota scritta prima di
qualunque altra conclusione, cosi' non la si puo' perdere.

## Il brief, e `01-TREE-IDENTITY.md:91-104`

Il brief chiede di valutare N=8 come "la proposta concreta" di 03. Ma
`01-TREE-IDENTITY.md:98-101`, scritto prima di me e verificato sul sorgente,
dice gia':

    QSB_GLV11 1 (pinning.cu:161), eleven-term
    pinning.cu:666 mette la tabella a 22,688,113,472 B
    786,432 hot records (QSB_HOT_RECS, pinning.cu:1416) pinati dalla
    finestra L2 persistente
    SUBMISSION-GLV12-V2.md:51 registra la geometria a 11 termini
    promossa a 948,943,797

E lo dice nella sezione intitolata *"The +7.5% prize is the baseline, not a
lever"* — cioe' **il premio che 03 chiama "l'unico design con la giusta
grandezza" e' gia' quello che sto compilando.**

## Perche' questo ribalta N=8 da "esperimento" a "regressione"

Confrontiamo le due geometrie sullo stesso piano, con i numeri di 03 e i
numeri del sorgente:

| | termini/componente | gathers | adds | tabella | termine |
|---|---:|---:|---:|---:|---|
| questo albero (`QSB_GLV11 1`) | 6 (Q) / 5 (P) | 11 | 10 | 21.1 GiB | promossa a 948,9 M |
| GLV12 (`QSB_GLV11 0`) | 6 / 6 | 12 | 11 | 9.4 GiB | promossa a 960,8 M |
| N=8 (proposta di 03) | 8 / 8 | 16 | 14 | 16 MiB | mai misurata |

N=8 non e' "una geometria nuova da provare". E' **sostituire la geometria
promossa con una piu' grossa in termini di add, sperando che la tabella
piccola la ripaghi.** E la tabella piccola e' l'unica cosa su cui 03 stesso
ammette di basarsi su un'inferenza, non su una misura (03:110-116: *"this is
an inference, not a measurement"*).

E il precedente piu' vicino e' negativo: `ITERATIONS.md:24-33`, v23 GLV10,
**399 697 199**, -55.7%, con la voce *"Table geometry in the fewer-adds
direction is CLOSED — falsified officially by us and independently by
measurement."* Piu' add, tabella diversa: e' esattamente li'.

## Il verdetto finale su N=8

Non lo implemento, e adesso il motivo non e' piu' "non ho la shell".
Il motivo e' che **N=8 e' una sostituzione a mano di una geometria gia'
promossa e misurata**, il cui unico vantaggio teorico poggia su un'inferenza
che lo stesso 03 dichiara non verificata, e la cui classe di movimento ha un
precedente ufficiale di -55.7%.

Se il mandato fosse "fai crescere questo albero di 1%", la domanda giusta non
e' "che geometria metto". E' "dove va il tempo nel kernel a 21.1 GiB di
tabella". Sono due domande diverse, e 03 risponde alla prima.

## Cosa resta aperto davvero, e l'ho verificato io

`01-TREE-IDENTITY.md:115` lascia un item **OPEN — blocking**:

    "is the tree identical to, or behind, the 1.008B record?"
    "needs git diff 8d07d3e -- candidates/pinning/pinning.cu; no git in this
     session"

**Questo e' il vero punto di partenza, prima di N=8, prima di tutto.** Se
l'albero in mano *e'* gia' i byte del record 1.008B, allora la submission
richiesta non e' un esperimento: e' un re-roll, che il brief vieta
esplicitamente, e che `00-SESSION-OPEN.md` / `02-NOISE-VS-GATE.md` mostrano
essere rumore (844,9 M -> 882,1 M su byte identici). Se invece e' *dentro*,
la promozione richiede +1% reale e serve una leva nuova.

Non posso risolverlo: non ho `git` (e quindi nemmeno shell). Ma e' una
domanda di **una riga**, e chi ha la shell la risponde in trenta secondi:

    git -C /home/j-ai/quantum-safe-bitcoin-challenge diff 8d07d3e --stat -- candidates/pinning/

Se il diff e' vuoto, il brief e' v20-obsoleto e la risposta e' "re-roll,
vietato". Se non e' vuoto, il diff dice esattamente quanto c'e' da guadagnare.
