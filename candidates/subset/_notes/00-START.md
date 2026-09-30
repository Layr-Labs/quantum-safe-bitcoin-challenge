# 00-START — journal del worker `subset`

## VERDETTO (prima riga, in testa)

**NON MISURATO.** Non ho uno strumento di esecuzione in questa sessione: il mio
toolset e' Read / Write / Edit / Glob / Grep / Task(subagent) / Skill / board.
**Non c'e' Bash, non c'e' shell, non c'e' `exec`.** Quindi `nvcc`, `./build_carrier.sh`,
`ptxas -v`, `cuobjdump`, `yukon` e `python3 qsb_sass_baseline.py` **non li posso
lanciare**. Non ho misurato throughput, non ho misurato SASS, non ho prodotto un
cubin nuovo, e non dichiaro nessun numero che non sia uscito da un file che ho
letto per intero.

Evidenza del toolset: `list_mcp_resources` -> unica risorsa `memory://knowledge-graph`
(servizio `memory`), `list_mcp_resource_templates` -> lista vuota. Nessun server
terminale. Quindi: lettura statica integrale, e dichiarazione esplicita di
`NON MISURATO` su tutto il resto.

## Cosa mi resta e cosa ci faccio

Resta la lettura integrale: 43 sorgenti CUDA/header secondo il brief, e qui se ne
vedono 26 nella directory. Quello che POSSO fare davvero e' trovare difetti e
ottimizzazioni **leggendo**, con la riga. Quello che NON posso fare e' confermare
che una modifica cambi il conteggio SASS. Quindi le mie uscite saranno:
- difetti con **riga e precondizione**, verificabili da chi ha una shell;
- una patch pronta, con l'istruzione esatta del metro da eseguire.

## Log cronologico

- T0: letto elenco directory `candidates/subset/`. 32 voce (28 file + `tests/` + `COPYING*`).
- T0: verificato toolset. Nessuna shell. Scritto questo file per primo, come da regola 2.