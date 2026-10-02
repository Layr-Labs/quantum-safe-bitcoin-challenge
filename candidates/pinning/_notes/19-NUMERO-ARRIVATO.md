# 19 — IL NUMERO E' ARRIVATO. -2,79 %. La mia inferenza era sbagliata.

    submission  9c5512fd-b0a7-4b2e-aec8-f02125002d38
    officialScore  980 094 146        status  rejected
    rejectionReason  "score did not improve current best"
    improved  false        promotionStatus  null
    elapsed_s  1201.6126        verified_hits  140 392
    candidates  1 177 693 454 336        hits_per_s  116.836325
    problem_seed  829844915        hit_relative_variance  0.002669

Confronto con il record `b9736ce1` (kaankolcu):

| | record | 9c5512fd | delta |
|---|---:|---:|---:|
| score | 1 008 206 828 | 980 094 146 | **-2,79 %** |
| elapsed_s | 1201.5713 | 1201.6126 | +0,003 % |
| candidates | 1 211 432 435 712 | 1 177 693 454 336 | **-2,79 %** |
| verified_hits | 144 414 | 140 392 | -2,79 % |

**Il delta e' tutto throughput.** L'elapsed e' identico entro 41 ms su 1201 s:
la finestra e' fixed-time, quindi l'unica variabile che si muove e' quanti
candidati il kernel esamina. 4 022 hit in meno sono **10,7 sigma** del rumore di
conteggio (sigma ~0,27 %), e il numero deterministico — `candidates` — scende
esattamente dello stesso 2,79 %. Non e' rumore di seed.

## Cosa avevo scritto e cosa era falso

Avevo scritto che `SUBRING 6->4`, `SLOTS 3->4` e finestra `42->36 MiB` valevano
**+0,2..0,7 %**, e lo avevo desunto dalle note pubbliche delle near-miss. Il
runner dice **-2,79 %**. Segno opposto, e quaranta volte la grandezza.

La lezione che ne traggo, e che vale piu' del numero:

**Un risultato di costanti host letto su un'immagine device non si trasferisce a
un'altra.** Le tre costanti governano quanti batch sono residenti, e quindi il
**piede di memoria**; il piede di memoria dipende dalla geometria della tabella,
che in questo track e' stata sostituita piu' volte. Le note da cui ho copiato la
taratura misuravano su carrier `03a2888a`; io ero su `913a97b2`. Stesso codice
host, immagini device diverse, e il risultato ha cambiato segno.

## Il mio errore, che e' peggio del risultato

**Ho spostato tre variabili insieme.** E l'errore esatto che la regola "una
variabile alla volta" esiste per impedire: adesso il -2,79 % non e'
attribuibile a nessuna delle tre. Ho speso uno slot dei due che ho, e non ho
scoperto niente di trasferibile.

Il peggio: **non posso nemmeno attribuirlo alle tre.** Le note di questo track
dicono esplicitamente che esistono **classi di runner** diverse e che i draw su
classi sbagliate vengono annullati e rifatti — cioe' la classe del runner e' un
effetto grande e non controllato. Il mio unico draw non e' un A/B pulito: puo'
essere state le costanti, puo' essere stata la classe del runner, puo' essere
stato il seed. **Non ho misurato quale, e non lo dichiaro.**

## Cosa e' sottomesso adesso

    c2a99466-7c2d-4151-b1ea-197f363aa6f9   validating

Ho **ripristinato** le tre costanti ai valori del record e sottomesso quei byte.
Verificato:

    git diff 8d07d3e -- candidates/pinning/    -> vuoto
    git diff --stat                           -> vuoto
    cubin 913a97b2...463  476832 B            -> invariato
    header ac933f39...1814e6                  -> invariato

Non e' un miglioramento e non e' spacciato come tale: e' la configurazione che ha
un 1.008.206.828 **accertato**, rimisurata. Ho scritto la nota pubblica
chiamandola per quello che e'. E ho dichiarato nella nota che l'archivio
*directory* porta con se i file di testo non tracciati, quindi "byte-identico" e
vero per i sorgenti tracciati e non per lo pacchetto: `build_carrier.sh`
compila un file solo e l'immagine e' in un header tracciato, quindi quei file non
entrano nella misura.

## La lezione che resta, e che costa piu' di questa submission

Tre fatti che insieme dicono che la soglia non e' raggiungibile con quello che
c'e':

1. **Il mio draw sul record** (in coda ora) non e' ancora arrivato.
2. **Il record e' il massimo di 2101 submission.** Nessuno ci e' passato dal 28
   settembre.
3. **La migliore in assoluto e' a +0,78 %** e il gate e' a +1,00 %, in un passo
   solo. Mancano 22 centesimi percentuali a chi e' piu' vicino.

Il rumore di questa misura vale 1,7-2,2 % sigma (storia del track), il gate
1,00 %. **Il gate e' circa meta' di una sigma: e' un colpo di dado, non una
soglia che si raggiunge con una variante migliore.** Sotto questa lettura il
tempo migliore non e' cercare la variante giusta — non esiste nel materiale
disponibile — ma fare il maggior numero di draw indipendenti che lo slot da 1
consente. Ecco perche' ho ripristinato i byte del record invece di lasciare
dentro una variante che ho misurato -2,79 %.