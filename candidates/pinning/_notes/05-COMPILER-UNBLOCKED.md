# 05 — IL BLOCCO E' RIMOSSO. Il compilatore c'è.

Questa nota risponde a `04-RESULT.md`, che ha dichiarato:

> "That is a coherent >5 % story. It is also, unavoidably: 1. a change to
> GLVScalar.cuh + pinning.cu device code, hence 2. **inert** unless
> qsb_carrier_sm89.h is regenerated (note 01), which needs `nvcc -arch=sm_89`
> + `cuobjdump` + a shell, and 3. unverifiable here... Writing that patch
> without a compiler would be exactly the 'manufacture a patch to look
> productive' failure the brief warns against. **I am not writing it.**"

Quelo era corretto quando e' stato scritto. Il blocco esiste piu'.

## Cosa e' stato misurato, con l'output

    $ source /home/j-ai/concord/qsb-env.sh
    $ nvcc --version | grep release
    Cuda compilation tools, release 12.8, V12.8.93
    $ gcc --version | head -1
    gcc-14 (Ubuntu 14.3.0-14ubuntu1) 14.3.0

Compilazione del **`pinning.cu` REALE** (298.024 byte, 6.759 righe), non di un
file di prova:

    $ nvcc -O3 -arch=sm_89 -cubin pinning.cu -o pinning_sm89.cubin
    $ ls -la pinning_sm89.cubin
    477344 byte
    $ sha256sum pinning_sm89.cubin
    cd2cc1b776803c05f616770cb1fd132f8ba77db3208b7873adc9dba006885b2a
    $ readelf -h pinning_sm89.cubin | grep -i flag
    Flag:  0x590559        # 0x59 = sm_89, la 4090

Confronto con l'header congelato:

    header dichiara : 476832 byte, sha 913a97b2...e983b463
    il mio cubin    : 477344 byte, sha cd2cc1b7...
    differenza      : 512 byte

**I 512 byte sono la prova che la compilazione e' viva.** Se il cubin fosse
stato identico a quello congelato, la firma non sarebbe cambiata e le 31
versioni passate avrebbero significato di nuovo nulla.

## Le tre condizioni della nota 03, una per una

| # | La condizione | Stato |
|---|---|---|
| 1 | la patch tocca device code | **non e' un ostacolo**, e' il punto |
| 2 | serve `nvcc -arch=sm_89` + `cuobjdump` + shell | **RIMOSSO** — vedi sopra |
| 3 | "unverifiable here" | **RIMOSSO**: compili e leggi i registri |

Il punto 3 era il vero nodo, e non perche' non si potesse scrivere la patch:
perche' il vincolo che avevi citato tu — 128 registri, 0 spill, con N=8 che
aggiunge 3 termini vivi — era un'affermazione non verificata. **Adesso si
misura con `ptxas -v`.**

## Cosa fare adesso, in ordine

1. **Baseline, prima di tutto.** Compila il `pinning.cu` **invariato** con
   `-arch=sm_89`, registra byte, sha256, i simboli (`readelf -sW`) e i registri
   (`ptxas -v`). Senza questa baseline, un test rosso non distingue "il codice
   e' rotto" da "l'ho rotto io".

2. **Poi la proposta N=8.** Cambia la geometria in `GLVScalar.cuh` secondo la
   tua stessa analisi della nota 03 (N=8 termini per componente, 16 gathers,
   14 adds, ~16 MiB per componente, dentro il `QSB_PERSIST_WINDOW_CAP` da
   42 MiB). Compila e misura.

3. **Il giudizio lo dai tu, e lo dichiari.** Se `ptxas -v` mostra spill, la
   variante e' un muro: si dichiara con il numero di riga e l'output. Un muro
   dichiarato vale piu' di una patch consegnata. Non ritoccare i registri a
   mano per far passare la variante: quello e' spostare il problema.

4. **Solo se lo sha del cubin cambia**, rigenera l'header con
   `research/build_sm89_cubin.sh` e poi `yukon submit`.

## Due avvertenze oneste

- **Qui non puoi eseguire il cubin.** Questa macchina ha una AMD, nessun
  device NVIDIA. Compili e verifichi l'architettura; il *tempo* lo misura il
  runner di Yukon su GitHub Actions con una RTX 4090. Un tempo locale non e'
  uno score e non va riportato come tale.
- **`LANG=C` e nel profilo per una ragione precisa:** `readelf` in italiano
  scrive `Flag:` e il `grep "Flags:"` dello script del repo non aggancia. Un
  controllo che non parte produce un fallimento che sembra un difetto del
  codice. Lo stesso vale per te: se un tuo controllo restituisce qualcosa di
  inatteso, prima leggi il grezzo.

## Il punto di partenza, corretto

La gerarchia delle prove, dalla piu' forte alla piu' debole:

    - glibc 2.43 dichiara cospi/sinpi in math.h con firma diversa da CUDA
      -> conflitto in cuda_runtime.h, che nvcc pre-include SEMPRE
      -> rimedio: ho messo un guardia su quelle dichiarazioni nel MIO
         math_functions.h, cioe' in un file che ho scaricato io, non nel
         codice del progetto. Non cambia il cubin che esce.
    - CUDA 12.8 accetta gcc <= 14, il sistema ha gcc 15 -> gcc-14 userland
    - CUDA 13.0 non spedisce piu' cicc in un pacchetto scaricabile da solo
      (verificato cercando cicc in 46 componenti: 0 risultati), quindi il
      12.8 con gcc-14 e' la strada, non un ripiego

Nessuna di queste e' una scelta che ti riguarda: sono risolte. Il tuo lavoro
riparte dalla nota 03, che e' la parte migliore mai uscita da questo track.
