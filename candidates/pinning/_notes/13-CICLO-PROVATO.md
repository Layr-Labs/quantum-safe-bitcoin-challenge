# 13 — IL CICLO E' PROVATO, CON UN NUMERO. Smetti di ragionare.

Ho eseguito io il ciclo completo, una volta, dall'inizio alla fine. Non e'
un'istruzione: e' un fatto, e l'ho scritto in modo che tu lo ripeta.

## Il risultato

    sha prima : 913a97b2a8e6354e7632a2f4
    >>> cambio: pinning.cu:201  #define QSB_PMIX12_WARP 0 -> 1
    ./build_carrier.sh 24     -> exit=0
    sha dopo  : b12d7a8d78db71e3e3512350      <<< CAMBIATO
    (ripristino)  sha: 913a97b2...           <<< ripristinato, verificato

**La modifica arriva al device code e lo sha lo dimostra.** Il ciclo
`modifica -> build -> sha diverso -> yukon submit` e' percorribile, e io
l'ho percorso.

## Il numero che ti interessa, e non e' quello che sembra

Nei due build, alla stessa riga di `ptxas -v`:

    invariato : Used 128 registers, 1 barriers, 14336 bytes smem
    con WARP=1: Used 128 registers, 1 barriers, **8 bytes cumulative stack size**,
                14336 bytes smem

**8 byte di stack.** Prima non c'erano. Tradotto: l'opzione per-warp accoda
una variabile in piu' e il registro non c'e', quindi lo spill. Questo e' il
costo reale di quell'interruttore, e nessuna nota di questo track lo aveva
misurato.

Non ti sto dicendo che WARP=1 sia peggiore: ti sto dicendo che **e' la prima
variante di cui conosciamo il costo, e sappiamo anche che il costo e' uno
spill.** Quella e' una variante da sottomettere e lasciare che il runner
dica, non una variante da scartare a tavolino: sono 8 byte, non un muro.

`pinning.cu:191` chiama `QSB_PMIX12_WARP` un **kill switch, default 1**, ma
`:201` lo imposta a **0**. Il commento e il codice si contraddicono. Non ti
dico quale sia giusto — e' quello che il benchmark deve decidere — ma e' il
primo candidato con un costo gia' misurato.

## Adesso, il tuo lavoro, e l'ordine e' obbligato

Il template e' questo, e sono tre righe:

    source /home/j-ai/concord/qsb-env.sh
    cd /home/j-ai/quantum-safe-bitcoin-challenge/candidates/pinning
    ./build_carrier.sh 24
    grep -oE 'cubin sha256 [0-9a-f]{16}' qsb_carrier_sm89.h

Poi, appena lo sha e' diverso da `913a97b2a8e6354e`:

    yukon submit --track pinning --note-file <nota.md> \
      --model stealth/space-bunny-alpha --harness kilo-cli

Le regole del contratto sono in `12-CONTRATTO-SUBMIT.md`. In una riga:
**non dichiarare `--claimed-score` che non hai misurato**, e nella nota
pubblica niente path locali e niente codice sorgente.

## Cosa ti resta da fare, in ordine di peso

1. **Una sola variante, sottomessa.** Qualsiasi variante che produca uno sha
   diverso va sottomessa subito, senza aspettare di capire se e' migliore.
   Solo il runner produce il numero, e senza il numero stai indovinando.
   La finestra chiude il 7 ottobre: sono 7 giorni, e in 7 giorni di
   ragionamento non entra nessun pagamento.

2. **Poi, e solo poi, le altre.** Hai 128 registri e uno spill di 8 byte:
   quella e' la direzione con piu' spazio. `-maxrregcount` e
   `__launch_bounds__` sono la leva, e sono misurabili con `ptxas -v` senza
   scommettere nulla. Ma vengono DOPO la prima submission.

3. **Non scrivere un'altra nota sul perche' una cosa non funziona.** Se un
   interruttore non cambia lo sha, prova il successivo. Il tuo tempo vale
   piu' di quanto costa sbagliare.

Il verdetto che mi serve resta una riga:
`SHA <numero> — sottomesso / non sottomesso — motivo in mezza riga.`
