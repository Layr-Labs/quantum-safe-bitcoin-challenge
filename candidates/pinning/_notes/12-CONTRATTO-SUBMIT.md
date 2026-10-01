# 12 — il contratto di `yukon submit`, verificato, e cosa manca

Ho letto `yukon submit --help`. Ecco il contratto reale, perche' sbagliare
un flag qui significa una submission respinta e una versione bruciata.

## I flag OBBLIGATORI

    yukon submit --track pinning \
      --note-file <path.md> \
      --model <nome> \
      --harness <nome> \
      [--claimed-score <score>] [--json]

- `--track pinning` — confermato: `yukon tracks` marca `*` sul track pinning.
- `--note-file` — **richiesto**, "required public markdown note to attach to
  the submission". Pubblica il testo sulla leaderboard: **niente segreti, niente
  path locali, niente account**.
- `--model` — richiesto. Io sono `stealth/space-bunny-alpha` su openrouter.
- `--harness` — richiesto, "coding agent or harness used". Per noi:
  `kilo-cli` con `qsb-cuda-1`. Non inventarlo: se il comando lo chiede e non
  lo sai, fermati e scrivilo.
- `--claimed-score` — "local score; required when claimed scores prefilter
  submissions". **Qui non possiamo produrre uno score locale**: `yukon run`
  chiama `/opt/starkware-challenge/bench-exec.sh`, che non esiste su questa
  macchina (GPU AMD, serve sudo). Quindi **non dichiarare uno score che non
  hai misurato**: e' il numero che decide se sei onesto, ed e' la stessa
  regola che vale per i bounty.

## Cosa NON fare

- **Non dichiarare uno score inventato.** Un `--claimed-score` falso e' una
  dichiarazione falsa, e il 199 USDC sono proporzionali a un numero che
  qualcuno dovrà verificare. Se il flag e' obbligatorio e non hai uno score,
  fermati e scrivi "non posso dichiararlo" invece di riempirlo.
- **Non pubblicare path locali nella nota.** `/home/j-ai/...` non va nella
  nota pubblica. La mia identita' di operatore non appare da nessuna parte:
  solo Twin, nome e indirizzo.
- **Non pubblicare il codice del cubin.** La nota descrive il cosa, non
  contiene il sorgente. Un pacchetto con il codice visibile lo prende chi e'
  piu' veloce e non lo paga.

## Il template della nota

    # <cosa hai cambiato, in una riga>

    **Risultato:** <lo score che hai misurato, o "non misurabile qui: il
    benchmark richiede una GPU NVIDIA e questa macchina ha una AMD">

    **Cosa ho cambiato e perche':**
    <2-6 righe: il file, la riga, il motivo tecnico>

    **Costo atteso:** <le gathers/adds contate, e cosa ho misurato con
    ptxas -v: registri e spill>

    **Cosa NON ho verificato:** <separa sempre questo. Il benchmark officiale
    gira su GitHub Actions con una RTX 4090; qui ho solo compilato e letto i
    flag dell'architettura.>

Quella ultima sezione non e' umiltà e' accuratezza: se la ometti, la nota
racconta una misura che non hai fatto.

## L'ordine, di nuovo, e senza piu' note

Il metro e' l'sha. Quando `build_carrier.sh` restituisce un numero diverso da
`913a97b2a8e6354e7632a2f41a997a05bf0e61e5a8b9a5c5c793f931e983b463`:

1. scrivi la nota (su file, non inline: `--note-file`)
2. `yukon submit --track pinning --note-file nota.md --model stealth/space-bunny-alpha --harness kilo-cli`
3. riporta **l'output del comando**, non un riassunto: l'id della submission
   o l'errore esatto

Se il comando fallisce, l'errore e' il risultato. Riportalo testuale e non
interpretarlo: "non so perche' fallisce" vale piu' di un motivo inventato.
