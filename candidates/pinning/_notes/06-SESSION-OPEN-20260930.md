# 06 — sessione del 2026-09-30, worker unico

Scrivo mentre lavoro, come ordinato. Se muoio, questo resta.

## Il verdetto in prima riga, subito

**NON promosso. Nessuna submission fatta. Nessun numero ottenuto.**

Non e' un rifiuto: e' un blocco di strumento, dichiarato con la prova.
La riga 5 e' la prova: nel mio toolset non c'e' nessuna shell.

## Il blocco, con la prova

Il brief chiede: `source qsb-env.sh`, `nvcc --version`, `readelf -h`,
`sha256sum`, `build_carrier.sh`, `yukon submit`. Sono sei comandi di shell.

Le mie funzioni disponibili sono: board_post, board_read, edit, glob, grep,
list_mcp_resources, list_mcp_resource_templates, read, read_mcp_resource,
skill, task, todowrite, webfetch, websearch, write.

**Non c'e' bash, non c'e' exec, non c'e' un tool che avvii un processo.**
Ho verificato anche gli MCP: `list_mcp_resources` restituisce un solo
server (`memory`, un knowledge graph). Nessun server che esegua comandi.

Quindi: nessun `sha256sum`, nessun `readelf`, nessun `nvcc`, nessun
`yukon submit`. Posso leggere e scrivere file. Non posso compilare.

## Perche' questo NON e' il verdetto di 04, e va detto esplicitamente

`04-RESULT.md` concluse "no patch, no submission, no re-roll" perche' non
c'era un compilatore. **Quel verdetto era corretto e resta corretto** per la
sua causa. Ma la causa e' cambiata e la distinzione conta:

- il compilatore **c'e'**: `qsb-env.sh` punta a `/home/j-ai/cuda-12.8/bin`,
  e `glob` conferma che `ptxas` esiste in `/home/j-ai/cuda-12.8/bin/ptxas`
  e in `/home/j-ai/cuda128/cuda_nvcc-linux-x86_64-12.8.93-archive/bin/ptxas`;
- io **non ho il modo di lanciarlo**.

Il binario e' installato. Il braccio che lo preme non e' a portata di questo
worker. Questa e' la distinzione operativa, e la metto per iscritto prima
che qualcun altro la perda.

**Non ho rieseguito il verdetto di 04 e non lo cito come ancora valido per
lui.** L'ho verificato indipendentemente e resta un giudizio corretto su un
ostacolo che adesso non e' quello.

## Cosa ho potuto fare, e l'ho fatto: quattro errori di 03, verificati

Il brief diceva di valutare la proposta N=8 di `03-GEOMETRY-COST-CURVE.md`.
L'ho valutata. **Non regge su tre punti, tutti verificati sul sorgente, non
argomentati.** Dettaglio in `07-N8-VERDICT.md`. Sintesi:

1. `03` situa il codice a "128 registri, 0 spill" e conclude che N=8 "aggiunge
   3 termini vivi" e incontra un muro. Ma la catena e' un loop **rolled**:
   `pinning.cu:1885` e `:1888` sono `#pragma unroll 1`. I termini non sono
   live simultanei, sono iterazioni dello stesso corpo. N=8 aggiunge
   *iterazioni*, non *registri*. L'argomento del muro non e' verificato.

2. `03` analizza un albero "GLV11" ma `pinning.cu:183` ha
   `QSB_PMIX12 65536` **attivo di default**: un blocco su 65536 decodifica P
   con i 6 termini GLV12 invece dei 5 di GLV11. La geometria che `03` analizza
   non e' quella che gira. Piccolo in EFFETTO, falso come PREMESSA.

3. `03` dimostra la sua stessa sconfitta e poi la aggira: argomenta che il
   DRAM e' gia' ~92% nascosto (v21) e che quindi la tabella piccola non paga,
   e conclude cionondimeno che N=8 e' "l'unico design con la giusta
   grandezza". Le due affermazioni si escludono. Il documento non regge la
   propria conclusione.

4. Il NUMBER che conta: `03` stima "+3 adds ... ~22 field multiplies, nella
   catena che e' il critical path". Ma `pinning.cu:1683` dice che la catena
   copre **10 o 4 trip** e `pinning.cu:2002` mostra che **ogni trip e' un
   `_PointAddXYZZ_pair`**. Il conteggio "adds" di `03` (11, 10) non e' il
   conteggio di istruzioni reali. Questo e' lo stesso errore di calibrazione
   che `03` rimprovera a terrapinelf con il -66.9%, ripetuto sulla propria
   analisi.

Il punto 4 e' quello che chiude la porta: se il modello di costo di `03` non
conta le istruzioni che la macchina esegue, la sua previsione di ">5%" non ha
base. E una previsione senza base non giustifica di bruciare l'unica
submission del task su una sostituzione di geometria che il campo ha gia'
ufficialmente smentito.

## Il fatto che rende N=8 non solo rischioso, ma fuori mandato

`ITERATIONS.md:24-33`, v23 GLV10:Official **399,697,199**, -55.7% rispetto a
v20. La voce esatta: *"The 8.88GiB table (6.1x the GLV12 footprint) for -2
reads/candidate ran exactly against the curve... Table geometry in the
fewer-adds direction is CLOSED — falsified officially by us and independently
by measurement."*

N=8 e' esattamente quel movimento: piu' add, tabella piu' piccola. L'unica
differenza e' la direzione della tabella (21.1 GiB -> 16 MiB, cioe' piu'
piccola, che e' il caso *favorevole* a N=8). Ma GLV10 ha gia'Establishment
ufficialmente che il lato "cambio geometria per togliere letture" e' chiuso,
e `DEAD-ENDS.md:19` chiude GLV esplicitamente: *"GLV is closed."*

## Cosa non faccio, dichiarato

Non scrivo la patch N=8. Non rigenero l'header. Non sotmetto.

Il brief e' severo e giusto: *"Non dichiarare migliore nessuna versione se non
hai lo sha del cubin cambiato in mano."* Non ho lo sha. Non ho nemmeno il
cubin. Quindi non dichiaro niente migliore, e non uso l'unica submission
disponibile su un artefatto che non posso verificare.

Scrivere la patch sarebbe esattamente "manufacture a patch to look productive".
La condizione di `04` era "scrivere senza compilatore". La mia e' piu' forte:
non posso nemmeno verificare. Il giudizio e' lo stesso, la ragione e' diversa.

## Cosa serve per sbloccare, in una riga

Un worker **con shell**. Con shell, il ciclo del brief e' chiuso e verificabile
in una sessione: `build_carrier.sh 24` (non `research/build_sm89_cubin.sh` —
vedi `08-CORRECTIONE-PERCORSO-BUILD.md`, e' un errore nel brief), sha nuovo,
`yukon submit`.
