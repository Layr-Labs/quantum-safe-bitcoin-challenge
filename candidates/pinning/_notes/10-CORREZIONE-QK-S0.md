# 10 — il symbol QK_S0: la scoperta di 09-HEADER-FROZEN-NON-MATCHA e' FALSA

## La mia conclusione era giusta, e perche' la prova e'keiworker stesso

`build_carrier.sh:19` contiene il pattern:

    ("QK_S0", r"_Z23kernel_pinning_pipelineILb1ELi0EE\w+"),

Quel `\w+` finale accetta **esplicitamente** un suffisso dopo `Li0E`.
Uno script che volesse escluderlo scriverebbe `$` lì. Quindi il pattern non
e' una prova di un terzo argomento di template: e' scritto per tollerarlo.

## La misura, non l'argomento

Ho compilato `pinning.cu` adesso, da zero, con lo stesso comando dello
script, e ho letto il simbolo dal cubin prodotto:

    _Z23kernel_pinning_pipelineILb1ELi0EEvPKjPKhiiiijjPKmS5_S5_S5_S5_PhPjS7_iiiP10ulonglong2PmSA_12qsb_tail_pre

e dall'header congelato:

    _Z23kernel_pinning_pipelineILb1ELi0EEvPKjPKhiiiijjPKmS5_S5_S5_S5_PhPjS7_iiiP10ulonglong2PmSA_12qsb_tail_pre

    IDENTICI, carattere per carattere

## L'errore di fondo: leggere un ABI senza compilare

Il ragionamento della 09 era: *"in Itanium ABI un tipo non-template non entra
mai nel nome mangled di una funzione, quindi `_12qsb_tail_pre` in coda
dimostra un terzo argomento di template."*

La premessa e' falsa. Nel mangling di Itanium i **parametri di funzione by
value entrano nel nome**. Lo schema e:

    _Z <len>Nome <elenza-tipi-dei-parametri> <parametri>

Qui: `_Z23kernel_pinning_pipeline` poi `ILb1ELi0E` (i due parametri di
template: `bool FAST_TAIL`, `int STAGE`), poi
`vPKjPKhiiiijjPKmS5_S5_S5_S5_PhPjS7_iiiP10ulonglong2PmSA_12qsb_tail_pre`
(tutti i parametri di funzione, e in coda `S_12qsb_tail_pre` = substitution
che riusa `qsb_tail_pre`).

Quindi `qsb_tail_pre` e' un **parametro di funzione**, esattamente come
hanno letto `pinning.cu:3647`, `:4100` e `:4421` — solo che la 09 ha
concluso il contrario, e l'ha scritto come "la discovery piu' utile della
sessione" e "decisiva".

## Cosa resta vero, e cosa no

**Vero:** il ciclo e' vivo e bit-riproducibile. `build_carrier.sh` su
sorgente invariato restituisce `913a97b2a8e6354e7632a2f41a997a05bf0e61e5a8b9a5c5c793f931e983b463`,
identico all'header congelato, con l'mtime dell'header aggiornato (19:29:28,
uguale al mio log). Non ho confrontato una dichiarazione con se stessa: ho
verificato che l'header e' stato riscritto e che il sha che vi e' dentro
coincide. Il toolchain e' alla versione giusta e riproduce bit per bit
l'immagine di chi l'ha prodotta.

**Falso:** che l'immagine congelata non possa venire da questo sorgente.

**Vera anche la 07** (i 4 errori su `#pragma unroll 1`, su
`QSB_GT_RADIX_BITS`, sul conteggio degli "add"). Quelle restano.

## La lezione, che vale piu' di questa nota

Una conclusione costruita su un'ABI letta a memoria, senza compilare, si
presenta con la stessa autorevolezza di una misura. Questa diceva "il simbolo
non puo' venire da qui", che e' la forma piu' pericolosa di un errore: non un
numero sbagliato, ma una struttura che regge tutto quello che viene dopo.

**Un argomento che si appoggia su una regola che non hai verificato con un
caso reale e' un'ipotesi, e va scritta "se" finche' non hai compilato.**

Costo di questa ora: una nota che si annunciava come la discovery piu'
importante del track e che chiudeva la strada. Se fosse rimasta, il worker
avrebbe trattato l'header congelato come un artefatto alieno e non avrebbe
costruito nulla.
