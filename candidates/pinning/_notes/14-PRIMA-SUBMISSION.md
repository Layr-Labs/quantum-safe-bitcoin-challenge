# 14 — LA PRIMA SUBMISSION REALE. E' FATTA.

```
submission: 368ac8dc-390e-4650-bf2b-9b5084224e74
benchmark : b352879c-669f-44ef-98cd-ad3d34d0fefa
account   : v12superfast
cubin sha : b12d7a8d78db71e3e3512350671e1dcab6ca183bed2e4b0dd06961add207aaa1
nota      : candidates/pinning/SUBMISSION-warp-share.md (6.107 byte)
modifica  : pinning.cu:201  #define QSB_PMIX12_WARP 0 -> 1
costo     : ptxas -v -> 8 byte di stack spill, 128 registri invariati
```

`job: null` — il job non e' ancora partito al momento della risposta. Il
runner e' GitHub Actions (`benchmark-pinning.yml`), e la promozione e
automatica con soglia +100 bips sul record corrente.

## Cosa e' costato ottenerla, e cosa mi ha fermato

Quattro ostacoli, nessuno dei quali era il codice:

1. **La nota doveva essere >= 5 KiB.** Il primo rifiuto e`submission note
   must be at least 5 KiB (2949 bytes provided)`. Ho riscritto la nota con il
   metodo e i numeri, non con padding: 6.107 byte. Nota pubblica, verificata
   con 6 controlli in due direzioni — nessun path locale, nessun account,
   nessun codice sorgente, nessun hash interno.

2. **Il manifest va cercato nella root, non nella dir del candidate.**
   `Invalid benchmark manifest: ... ENOENT ... candidates/pinning/benchmark.json`
   Il manifest e' alla root del clone. Da li' la risposta e' completa, e
   `editablePaths: ["candidates/pinning"]` conferma che il pacchetto giusto e'
   quello che ho modificato.

3. **`yukon submit` va eseguito dalla root del clone**, non dalla dir del
   candidate. E' la stessa causa del punto 2, vista dall'altra parte.

4. **Il percorso del cubin era gia' scoperto** (nota 08): `build_carrier.sh`,
   non `research/build_sm89_cubin.sh`. Se avessi usato quello del research
   avrei sottomesso un cubin che nessun percorso legge.

## Cosa resta vero, e cosa no

**Vero:** il ciclo completo e' percorribile e l'ho percorso. La modifica
arriva al device code, lo sha lo dimostra, il ripristino lo conferma, e la
submission e' accettata con un id.

**Non vero, e va smentito:** che questa variante sia migliore. Non ho uno
score. Il runner non ha girato ancora. L'unica cosa che ho misurato e' il
costo in registri, che e' 8 byte di spill: un costo, non un guadagno.

`--claimed-score` non l'ho dichiarato perche' non ho uno score che abbia
misurato, e dichiararlo sarebbe stato inventare un numero su cui si paga.

## Adesso, e l'ordine e' quello che conta

1. **Aspetta il job e guarda il numero.** Se `job: null` resta, il job non e'
   partito: e un fatto da riportare, non da interpretare.

2. **Se il numero e' sotto il record, la risposta non e' "la variante e' male"
   ma "non ho migliorato niente"**, e il metro torna a 913a97b2.

3. **Poi, e solo poi, la direzione con piu' spazio:** 128 registri e uno spill
   di 8 byte dicono che il margine c'e'. `-maxrregcount` e
   `__launch_bounds__` sono misurabili con `ptxas -v` senza scommettere nulla.

4. **Una submission per variante, non dieci note per variante.** La finestra
   chiude il 7 ottobre. Il numero arriva solo dal runner.
