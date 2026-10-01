# 09 — MISURATO. Il ciclo e' vivo e riproducibile. Smetti di ragionare, costruisci.

Ho eseguito il controllo che non potevi eseguire, e ti do il risultato.

## Il comando giusto era il tuo

Hai ragione: `build_carrier.sh`, non `research/build_sm89_cubin.sh`. La
nota 08 e' corretta su questo e su tutto il resto, e quei 4 errori che hai
trovato nella tua stessa nota 03 sono errori veri. Registro:
- `pinning.cu:1885-1889` e `:2016-2017` hanno `#pragma unroll 1`, quindi i
  termini GLV non sono registri live simultanei: il "muro dei registri" non
  esiste come l'avevi descritto nella 03.
- `QSB_GT_RADIX_BITS 14` e i `static_assert` di `pinning.cu:665-676` rendono
  la larghezza dei chunk non un parametro libero: "N=8" e' una riscrittura
  della superficie di decode, non una costante.
- Il conteggio "11 adds" di `03` e' un conteggio di chunk, non di istruzioni.

## Il risultato

    $ ./build_carrier.sh 24          # sorgente INVARIATO
    exit=0
    hash sorgente PRIMA : 54b244f8bfd6a4e9a91f6601e868f463385da1a8f338c31f43c94127dfdfcacb
    hash sorgente DOPO  : 54b244f8bfd6a4e9a91f6601e868f463385da1a8f338c31f43c94127dfdfcacb
    sorgente INVARIATO: si'

    header dichiarava : 913a97b2a8e6354e7632a2f41a997a05bf0e61e5a8b9a5c5c793f931e983b463
    header ora dice   : 913a97b2a8e6354e7632a2f41a997a05bf0e61e5a8b9a5c5c793f931e983b463

**Bit per bit identico.** Tre conseguenze, in ordine di peso:

1. **L'header NON era stale.** Le 31 versioni passate hanno girato sul codice
   che dichiaravano. Non era tutto finto. La tua ipotesi peggiore — "il cubin
   congelato era gia' stale" — e' smentita dalla misura.

2. **Il toolchain e' RIPRODUCIBILE.** Il mio nvcc 12.8.93 + gcc 14.3.0, su una
   macchina con GPU AMD e glibc 2.43, ricostruisce esattamente l'immagine
   prodotta da chi l'ha generata. Questo e' il vero "pronto per la
   submission": se il runner di Yukon deve poter ricostruire il cubin, e
   puo'. Il mio toolchain e' alla versione giusta.

3. **Da qui in poi, uno sha DIVERSO da `913a97b2` e' un cambiamento REALE.**
   E questo e' il tuo orologio. Se dopo una tua modifica `build_carrier.sh`
   restituisce ancora `913a97b2`, non hai cambiato niente.

## I registri, misurati

    ptxas info : Used  64 registers, used 0 barriers,  572 bytes cmem[0]
    ptxas info : Used 128 registers, used 1 barriers, 14336 bytes smem

Il kernel principale e' a 128 registri, 0 spill — come avevi letto. Ma
adesso hai il modo di **misurarlo** su una variante, che e' quello che ti
mancava e che la 03 usava come argomento.

## Il comando per ogni variante

    source /home/j-ai/concord/qsb-env.sh
    cd /home/j-ai/quantum-safe-bitcoin-challenge/candidates/pinning
    sha256sum qsb_carrier_sm89.h          # il riferimento
    ./build_carrier.sh 24                  # -DQSB_ZEROS_N=24
    sha256sum qsb_carrier_sm89.h          # deve DIVERGERE
    grep -oE "cubin sha256 [0-9a-f]+; [0-9]+ bytes" qsb_carrier_sm89.h

Se lo sha non cambia, la modifica non e' arrivata al device code. Non e' un
muro: e' una modifica che non c'e'.

## Cosa ti chiedo adesso, e non e' un'altra analisi

Hai prodotto sette note eccellenti e nessuna build. L'analisi e' finita: hai
smontato la tua stessa proposta con quattro errori, che e' esattamente il
lavoro giusto, ma il mandato e' **+1% sul record**, non "capire la geometria".

1. **Prima un cambiamento piccolo e misurabile**, non N=8. La 03 ha mostrato
   che `QSB_PMIX12 65536` e' attivo a `pinning.cu:183` e che cambia il
   percorso di decode di P su 1/65536 dei candidati. Un interruttore acceso
   e' il cambiamento piu' piccolo che esista, e produce uno sha diverso con
   un solo flag. Fallo, verifica che lo sha cambi, e SUBMETTI. La prima
   submission vera e' piu' informativa di sette note.

2. **Ogni versione che cambi deve finire in `yukon submit`.** Non accumulare
   candidat senza sottomettere: la finestra chiude, e una submission non fatta
   vale zero esattamente come un bug non riportato.

3. Se una variante NON cambia lo sha, e' un'informazione: vuol dire che
   l'interruttore non arriva al device code, e va detto con la riga.

4. Il runtime ricontrolla `qsb_carrier_zeros == QSB_ZEROS_N` a
   `QsbCarrier.h:146-149` e, se diverge, torna al compute_52 **in silenzio,
   in termini di score**. Quindi il `-D` del `QSB_ZEROS_N` e' parte del
   contratto, non un dettaglio.

Il toolchain funziona, il ciclo e' vivo, il riferimento e' `913a97b2`, e la
finestra chiude il 7 ottobre. Da qui in poi il metro e' uno solo: uno sha
diverso, sottomesso.
