# 08 — correzione: due script di build, e il brief usa quello sbagliato

Il brief, punto 4 del ciclo: *"l'header si rigenera con
`research/build_sm89_cubin.sh`"*. **Quello script non rigenera l'header.**

## I due script

**`research/build_sm89_cubin.sh`** (38 righe):
- riga 20: `nvcc -O3 -arch=sm_89 -cubin pinning.cu -o pinning_sm89.cubin`
- riga 24: verifica `readelf -h ... | grep -q "Flags:.*0x59"`
- riga 26-37: verifica 6 kernel + 5 simboli
- **non tocca `qsb_carrier_sm89.h`**

**`build_carrier.sh`** (75 righe, nella dir del candidate):
- riga 10-11: `nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_CARRIER_BUILD=1 -arch=sm_89 -cubin`
  con `-Xptxas -v`
- riga 12-13: `cuobjdump -symbols` e `-sass`
- riga 56: `open("qsb_carrier_sm89.h", "w")` <- **scrive l'header**
- riga 59: scrive `cubin sha256 {sha}; {len(img)} bytes`
- riga 72: stampa `sha256 {sha[:16]}`

## Perche' la distinzione e' cio' che tiene insieme tutto

`QsbCarrier.h:67` include `qsb_carrier_sm89.h`, e `QsbCarrier.h:111-113` lo
decodifica e lo carica con `cudaLibraryLoadData`. E' **quella** immagine che
i 14 `qsb_carrier_launch` di `pinning.cu` lanciano. `pinning_sm89.cubin` e'
un file diverso, prodotto da uno script diverso, che il percorso carrier non
legge.

Quindi, per la regola del brief ("nessuna versione e' migliore finche'
l'header non e' stato rigenerato e il suo sha256 cambia"), il comando e':

    ./build_carrier.sh 24

non `research/build_sm89_cubin.sh`. Il `-D` che conta e' `-DQSB_CARRIER_BUILD=1`
(build_carrier.sh:10), assente dallo script di research: senza quello
l'immagine non ha il `ld.global.nc.L2::64B` e `build_carrier.sh:49-50` fallisce
il controllo `LTC64B`.

C'e' anche un controllo che merita una riga: `build_carrier.sh:40-44` pretende 11
globali nel simboli, fra cui `qsb_carrier_zeros`; e `QsbCarrier.h:146-149`
ri-controlla a runtime che `qsb_carrier_zeros == QSB_ZEROS_N` prima di accendere
il carrier. Se i due divergono il carrier si spegne e si torna al compute_52 —
silenziosamente, in termini di score. Vale la pena sapere che esiste.

## Il numero del cubin, dichiarato nell'header congelato

    qsb_carrier_sm89.h:3   cubin sha256 913a97b2a8e6354e7632a2f41a997a05bf0e61e5a8b9a5c5c793f931e983b463
    qsb_carrier_sm89.h:3   476832 bytes
    qsb_carrier_sm89.h:6   static const size_t qsb_carrier_cubin_bytes = 476832;

Questo e' lo sha da battere. Se `build_carrier.sh` su sorgente invariato
restituisce `913a97b2...`, allora l'header e' gia' allineato al sorgente e il
ciclo e' vivo. Se restituisce qualcos'altro **a sorgente invariato**, allora il
cubin congelato era gia' stale prima di qualsiasi mia modifica — e questo
sarebbe il finding piu' importante del track, perche' significherebbe che
alcuna delle 31 versioni ha mai girato sul codice che dichiarava di girare.

**Non ho potuto eseguire questo controllo.** E' il primo comando che un
worker con shell deve fare, e va fatto *prima* di toccare `GLVScalar.cuh`,
perche' e' l'unico modo di distinguere "l'ho rotto io" da "era gia' rotto".
