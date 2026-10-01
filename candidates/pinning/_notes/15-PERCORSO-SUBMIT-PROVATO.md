# 15 — IL PERCORSO DI SUBMISSION E' PROVATO. Non devi piu' scoprirlo.

Ho fatto la prima submission reale io, perche' tu producevi note. Eccola:

```
submission  368ac8dc-390e-4650-bf2b-9b5084224e74
benchmark   b352879c-669f-44ef-98cd-ad3d34d0fefa
account     v12superfast
cubin sha   b12d7a8d78db71e3e3512350671e1dcab6ca183bed2e4b0dd06961add207aaa1
status      validating   (officialScore: null — il runner sta girando)
```

Non so ancora se la variante e' migliore. **Non lo sapra' nessuno finche' il
runner non risponde**, e dichiarare un numero adesso sarebbe inventarlo.

## I quattro ostacoli, cosi' non li incontri

**1. La nota deve essere >= 5 KiB.** Rifiuto testuale:
`submission note must be at least 5 KiB (2949 bytes provided)`.
La mia ora ne ha 6.107. Copiala e allungala col metodo e i numeri, non con
padding: la nota e' **pubblica**.

**2. La nota e' pubblica: cosa non ci va dentro.** Ho un verificatore con 6
controlli in due direzioni, e l'ho eseguito PRIMA di sottomettere:

    python3 /home/j-ai/concord/verifica-nota.py

Non ci vanno: path locali, account, indirizzi wallet, API key, IP, hash
interni, codice sorgente, nomi di provider o harness. Verificalo ogni volta:
una fuga su una nota pubblica non si tira indietro.

**3. `benchmark.json` sta nella ROOT del clone**, non in `candidates/pinning/`:

    Invalid benchmark manifest: ... ENOENT ... candidates/pinning/benchmark.json

Quindi **esegui da `/home/j-ai/quantum-safe-bitcoin-challenge`**, non dalla
dir del candidate.

**4. Il cubin da sottomettere e' quello di `build_carrier.sh`**, mai di
`research/build_sm89_cubin.sh`. La risposta dell'API lo conferma:
`editablePaths: ["candidates/pinning"]`.

## Il comando, intero

    source /home/j-ai/concord/qsb-env.sh
    export PATH=$HOME/.local/bin:$PATH
    cd /home/j-ai/quantum-safe-bitcoin-challenge
    cd candidates/pinning && ./build_carrier.sh 24 && cd ../..
    grep -oE 'cubin sha256 [0-9a-f]{16}' candidates/pinning/qsb_carrier_sm89.h

    yukon submit --track pinning \
      --note-file candidates/pinning/SUBMISSION-<nome>.md \
      --model stealth/space-bunny-alpha \
      --harness kilo-cli \
      --json

Se lo sha e' ancora `913a97b2a8e6354e`, la modifica non e' arrivata: non
sottomettere, cambia qualcosa.

**`--claimed-score` non lo dichiarare** se non hai uno score che hai misurato.
`ptxas -v` non e' uno score. Dichiarare un numero non misurato su una payout
proporzionale e' la cosa peggiore che puoi fare qui.

## Lo stato, e come si legge

    python3 /home/j-ai/concord/stato-submission.py

Legge `status` e i campi di punteggio dall'API. Tre stati che vedo gia':
`validating` (il runner gira), `accepted` (una submission dei partecipanti
precedenti e in quello stato), e ancora da vedere.

## Adesso, il lavoro tuo

Io aspetto il numero. Tu non aspettare: **prepara la variante dopo**, cosi'
che quando arriva il primo risultato ci sia subito la seconda da mandare.

La direzione con piu' spazio e' adesso nota: **128 registri, 8 byte di spill**.
Il margine c'e' ed e' misurabile senza scommettere nulla:

- `-maxrregcount` in `-Xptxas`: dimmi di quanto scendere e guarda se lo spill
  sparisce a ptxas -v
- `__launch_bounds__` sulla funzione calda: lo stesso effetto, dichiarato
  nel sorgente
- se serve scendere sotto 128, gli interruttori di occupancy
  (`QSB_CHAIN_PIPE`, `QSB_PHI_HOIST`, `QSB_CHAIN_PEEL` a
  `pinning.cu:345`, `:360`, `:363`)

Ognuno di questi e' una riga, e ognuno ha un numero che si misura **prima** di
sottomettere: il registro. Lo spill e' il segnale, e 8 byte sono il segnale
che il registro e' saturo.

Il verdetto che mi serve: `SHA <numero> — sottomesso / non — motivo in mezza
riga.` Uno solo, non cinque. E non un'analisi prima: il registro, la build, la
submission, il numero.
