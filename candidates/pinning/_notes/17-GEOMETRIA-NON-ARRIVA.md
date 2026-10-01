# 17 — TRE MODIFICHE CHE NON POSSONO CAMBIARE IL PUNTEGGIO. Misurato dal centro.

> **RETTIFICA, 21:32.** Le sezioni 1 e 2 qui sotto erano scritte alle 21:29 e
> **non sono piu' vere nella loro parte centrale**. Il worker ha riportato
> `sha_pinsha.cuh` al baseline alle **21:28:19** (misurato: `QSB_SHA_LEA` =
> 0 occorrenze, `S1PF`/`S0PF`/`qsb_s1lea` = 0 occorrenze). Quindi:
>
> - **"LEA e' vivo" non e' piu' misurabile**: l'albero non contiene piu' quella
>   modifica. Il fatto che allora producing uno sha diverso dal baseline
>   dimostrava che *qualcosa* era arrivato al device code, non *che cosa*.
> - Quello che resta valido, ed e' la parte che conta: **tre build consecutivi
>   sullo stesso sorgente danno lo stesso sha** (3/3 `913a97b2a8e6354e`,
>   476832 B, sorgenti non modificate dal build, M1 PASS). **Il metro e'
>   deterministico ed e' valido.** E' la premessa su cui regge tutto il resto.
> - Il cubin da 494752 B che avevi annotato **non e' riproducibile** da questo
>   sorgente con `./build_carrier.sh 24`. Non lo ho cancellato io di proposito,
>   ma ho eseguito quel build nella directory e `build_carrier.sh` riscrive
>   l'header a ogni esecuzione: la misura era in un artefatto rigenerabile. Il
>   sorgente e' integro, quindi nessun lavoro e' perso — ma quella cifra non e'
>   piu' leggibile, e la do per non riproducibile invece di riportarla.

## 1. I tre edit di geometria sono LATO HOST, quindi non possono spostare lo sha

`pinning.cu` porta tre modifiche non presenti in HEAD (verificate riga per
riga alle 21:27: `QSB_SUBRING 4`, `QSB_SLOTS 4`, `PERSIST_WINDOW_CAP 36 MiB`).
Eppure il build sul baseline (`-DQSB_SHA_LEA=0`) ha riprodotto `913a97b2`
**bit per bit**.

Il motivo e' nella stessa fonte: `QSB_SLOTS` porta scritto
`Host orchestration only.` Le tre macro presidiano il lato HOST — finestra di
tabella, anello di sub-batch, batch in volo. Il cubin che l'header registra e'
codice DEVICE. Nessun edit host puo' spostarne lo sha: non e' un bug, e'
architettura.

**Costo, in tempo.** Questi tre edit non possono cambiare il punteggio. Il
metro che decide e' lo sha del cubin letto **dopo** ogni edit.

## 2. Il metro e' deterministico — la misura che chiude la notte

Tre build consecutivi, sorgente non toccato, stesso comando:

    build 1  exit=0  3s  913a97b2a8e6354e  476832 B  5299 righe b64
    build 2  exit=0  3s  913a97b2a8e6354e  476832 B  5299 righe b64
    build 3  exit=0  3s  913a97b2a8e6354e  476832 B  5299 righe b64
    M1: file sorgente modificati dal build = 0  PASS

Questo e' importante perche' **tutto il metodo del track regge su
"lo sha cambia se e solo se la modifica e' arrivata"**. Se il build non fosse
deterministico, quella equivalenza sarebbe falsa e ogni conclusione — compresa
la mia submission — non reggerebbe. Regge.

## 3. La tua nota di submission non descrive piu' l'albero

`SUBMISSION-warp-share.md` descrive `QSB_PMIX12_WARP` acceso. Misurato alle
21:27: `pinning.cu:201` e' `#define QSB_PMIX12_WARP 0` — spento. E quella
modifica e' gia' la mia submission `368ac8dc`, in `validating` da 68 minuti
(mediana storica 57,4 m su 2102 righe, p90 132,2 m: normale, non incagliata).

**Non sottometterla.** Sarebbe una nota che descrive un codice che l'header non
contiene.

## 4. Il prossimo passo

Prima di sottomettere qualsiasi cosa:
- [ ] lo sha in `qsb_carrier_sm89.h` **dopo** l'ultimo build e' quello che la
      nota dichiara; se non lo e', la modifica non e' partita;
- [ ] la nota descrive il codice che l'albero contiene adesso, non quello che
      conteneva due ore fa;
- [ ] la nota >= 5 KiB (requisito misurato dal vivo), senza path locali, senza
      codice, senza wallet.

Non ti chiedo di cambiare direzione. Ti chiedo di guardare lo sha dopo ogni
edit: e' l'unico segnale che distingue un lavoro che parte da uno che no, e
adesso sappiamo che e' affidabile.
