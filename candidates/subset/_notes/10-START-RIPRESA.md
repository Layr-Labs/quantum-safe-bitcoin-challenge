# 10-START-RIPRESA — worker di ripresa, T0

## VERDETTO (prima riga, provvisorio — aggiorno quando ho numeri)

**IN CORSO. Zero misure eseguite al momento della scrittura di questa riga.**
Non dichiaro niente che non sia uscito da un comando che ho lanciato io.

## Cosa cambio rispetto al worker precedente (05-VERDETTO.md)

Il precedente ha consegnato `NON MISURATO` su tutto dichiarando "non ho una shell".
**Quella conclusione era sbagliata e il brief me lo dice.** Io ho
Read/Write/Edit/Glob/Grep/Task/Skill/board e `list_mcp_resources` -> solo
`memory://knowledge-graph`. Ma **il subagent `Task` di tipo General Agent ha una
shell e la usa.** Quindi la misura e' possibile: la eseguo via subagent, e registro
qui **l'output del comando**, non la sua descrizione.

Regola che mi obbliga: un report di subagent e' una **dichiara[zione]**, non un fatto.
Quindi per ogni numero che metto in questa consegna io **rileggo il file di output**
o **rilancio io il comando**. Il subagent esegue, ma la verifica e' mia.

## Perche' ho scelto la **sandbox copiata** invece di costruire in `candidates/subset/`

L'orchestratore sta misurando `tests/gpu_epochs/tree.cu` **adesso**, in parallelo
(ciclo baseline -> `=0` -> `=2` -> restore). Due problemi, non uno:

1. **`tree.cu` e' in uso.** Non lo tocco. Regola del brief, e il motivo e' giusto:
   perdere una misura vale piu' di un candidato in piu'.
2. **`build_carrier.sh` scrive `qsb_carrier_sm89.h` e il `.cubin` nella directory
   del candidato.** Se costruissi li' mentre l'orchestratore costruisce li', non
   e' solo che il mio sha e'rumoroso: **e' rumore anche il suo**, e il suo e' il ciclo
   che sta decidendo se `FINDING-A` regge. Io romperei una misura che non e' mia per
   cercarne una che lo e'. Quindi **copio**.

La copia porta con se' una **prova di fedelta' gratuita**: se la sandbox, su sorgente
non modificato, produce esattamente `e0c0897f799baf81`, allora la sandbox e' equivalente
byte per byte e ogni delta mio e' attribuibile alla mia modifica e non all'ambiente.
Se non produce quello, la sandbox non vale niente e lo dichiaro invece di misurarci.

## Denominatore di partenza (ereditato, non mio)

Letto per intero io **adesso**: `00-START.md`, `01-DENOMINATORE.md`, `02-CORREZIONI.md`,
`03-RILIEVI-NOTI.md`, `04-FINDING-A.md`, `05-VERDETTO.md` — 6 file, 341 righe.

Da quelle note, e **da rifare prima di crederci**:
- i due "difetti SHA" di `pair_shared.cuh:428-429` e `:457-458` sono **smentiti**, non li riapro;
- `FINDING-A` (`QSB_GATHER_ONE_FORM`, `tree.cu:272`) e' **in uso dall'orchestratore**, non lo misuro;
- `QSB_SHORT_CARRY4` (`filter_tail_sc.cuh:11` vs `:23`) e' gia' registrato, non e' mio spazio;
- `zero mutanti` finora.

## La regola 4 l'ho gia' eseguita (l'unica cosa che il precedente ha fatto bene)

`03-RILIEVI-NOTI.md` elenca i **20 titoli .md** della root del candidato e iloro
"cosa NON copre". Spazio libero che risulta: **l'interleaving dei due hash SHA del
gate** (`02-CORREZIONI.md` C4: `QSB_GATE_H0_FMA` default 1 esegue due compressioni
**in sequenza**, e la versione interleaved round-per-round esiste ma sta nel
`#else`). **Ed e' esattamente la superficie che scelgo**, perche':
- non e' `tree.cu`;
- e' gia' scritto nel codice come `QSB_CARRIER_KNOBS`, quindi l'host e l'immagine
  restano d'accordo e il contratto di test non si rompe;
- il commento del progetto stesso (`pair_shared.cuh:374-376`) dichiara che
  l'interleaving e' **lo scopo** della knob.

Il **rischio di sovrapposizione con i 9 Blackthorn + 12 ChainSecurity resta aperto**
e lo dichiaro `NON MISURATO`: non ho la rete per quei testi. Se quel titolo esiste
gia', il mio valore e' zero e va detto.

## Cosa devo ancora verificare prima di chiamarlo candidato (e non l'ho ancora fatto)

1. `qsb_sha256_gate_h0_pair` (la forma interleaved) **e' davvero sul percorso di
   ricerca**? Il solo kernel che gira e' `kernel_digest` (`01-DENOMINATORE.md`, nota
   di portata). Se il gate non ci finisce, la mia candidatura vale zero e lo dico.
2. Il gate e' dentro `kernel_digest` **hot** (per ogni candidato) o una volta per
   batch? Se e' per batch, il risparmio e' diluito e probabilmente non arriva a +1%.
3. `QSB_GATE_H0_FMA 0` **cambia davvero lo sha del cubin**? Se non lo cambia, non e'
   arrivata al device code e la misura non misura niente (trappola numero uno).
4. Il conteggio **non-nil returns** della funzione calda prima e dopo, non l'opinione.

## Log

- T0: letti i 6 file del precedente per intero. Scritto questo file per primo.
