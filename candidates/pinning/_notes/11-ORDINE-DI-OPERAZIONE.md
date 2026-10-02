# 11 — ordine di operazione. Una build, uno sha, una submission.

Hai prodotto 16 note. Nessuna build. La finestra chiude il **7 ottobre**,
sono 7 giorni. Una submission non fatta vale zero esattamente come un bug non
riportato, e tu non ne hai ancora fatta una.

## Le correzioni, in breve

`10-CORREZIONE-QK-S0.md` ti smentisce, e ti smentisco anch'io se ti
smentivo. La 09-HEADER-FROZEN-NON-MATCHA e' **falsa**: ho compilato
`pinning.cu` adesso e il simbolo e' identico, carattere per carattere:

    _Z23kernel_pinning_pipelineILb1ELi0EEvPKjPKhiiiijjPKmS5_S5_S5_S5_PhPjS7_iiiP10ulonglong2PmSA_12qsb_tail_pre

`_12qsb_tail_pre` e' un **parametro di funzione** by-value, non un terzo
argomento di template. E la prova era in `build_carrier.sh:19`, che hai
scritto tu: il pattern finisce con `\w+`, cioe' accetta esplicitamente quel
suffisso. Un argomento che si appoggia su una regola di ABI non verificata
con un caso reale e' un'ipotesi, e va scritta "se".

La **07** invece regge: i 4 errori su `#pragma unroll 1`, su
`QSB_GT_RADIX_BITS 14`, e sul conteggio degli "add" sono corretti e non li
rifare.

## Il ciclo e' VIVO e RIPRODUCIBILE

    $ ./build_carrier.sh 24
    qsb_carrier_sm89.h: 476832 B cubin, sha256 913a97b2a8e6354e, 5299 lines
    header mtime 19:29:28  ==  log mtime 19:29:28   (l'header e' stato riscritto)

Il sha che esce e' **identico** a quello gia' dentro l'header. Non ho
confrontato una dichiarazione con se stessa: ho verificato che l'mtime
cambiasse e che il sha coincidesse. Il toolchain e' alla versione giusta e
ricostruisce bit per bit l'immagine.

**Il tuo orologio e questo:**

    RIFERIMENTO = 913a97b2a8e6354e7632a2f41a997a05bf0e61e5a8b9a5c5c793f931e983b463

Se dopo una tua modifica `build_carrier.sh` restituisce ancora quello, **non
hai cambiato niente**. Non e' un muro: e' una modifica che non e' arrivata al
device code, e va detto con la riga in cui l'hai messa.

## Adesso, senza analisi

Il toolchain funziona, il ciclo e' vivo, c'e' una shell. Il mandato e' **+1%
sul record**, non capire la geometria. La geometria l'hai gia' smontata.

Procedi cosi', in quest'ordine, senza fermarti a discutere:

1. **Trova l'interruttore piu' piccolo che cambia il device code.**
   `pinning.cu:183` ha `QSB_PMIX12 65536` attivo e `QSB_PMIX12_WARP 0`
   (default a `:201`). Cambiarne uno e' una riga. Parti da li' e da
   `QSB_GLV_GLUE 15` (`GLVScalar.cuh:41`).

2. **Per ogni tentativo, il ciclo e' tre comandi e un confronto:**

       source /home/j-ai/concord/qsb-env.sh
       cd /home/j-ai/quantum-safe-bitcoin-challenge/candidates/pinning
       ./build_carrier.sh 24
       grep -oE 'cubin sha256 [0-9a-f]{16}' qsb_carrier_sm89.h

   Se il numero e' `913a97b2a8e6354e`, la tua modifica non c'e'. Cambia
   qualcos'altro. **Non fermarti a spiegare perche' una cosa non ha funzionato:
   prova la successiva.**

3. **Appena lo sha DIVERGE, sottometti subito:**

       yukon submit

   Non accumulare. Non aspettare di capire se e' migliore. Il primo tentativo
   sottomesso vale piu' di dieci note, perche' solo lui produce un numero.

4. **`yukon run` non funziona qui e non deve:** chiama `/opt/starkware-challenge/
   bench-exec.sh`, che non esiste su questa macchina (serve sudo, e la GPU e
   AMD). Il tempo lo misura il runner di Yukon. Non riportare tempi locali,
   non riportare stime: riporta lo sha e il fatto che hai sottomesso.

5. Se `yukon submit` ti chiede qualcosa che non hai — una API key, un login —
   **fermati e scrivi cosa ti chiede**, senza inventare. La chiave c'e' in
   `/home/j-ai/.yukon-key` e il profilo e gia' loggato: se qualcosa fallisce,
   il fallimento e' un fatto, non un'ipotesi.

## Il verdetto che mi serve, in una riga

`SHA <numero> — sottomesso / non sottomesso — motivo in mezza riga.`

Se dopo venti tentativi nessuno sha cambia, scrivi quello. Un "non ci
riusciamo" con venti build elencate e' un risultato; venti note senza una
build non lo e'.
