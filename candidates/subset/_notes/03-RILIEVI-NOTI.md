# 03-RILIEVI-NOTI — cosa coprono i provini noti (regola 4, PRIMA della misura)

Il brief e' esplicito: **prima** di scrivere un test, elenca i rilievi noti del
bersaglio e leggi i titoli. Se il mio test sta per misurare qualcosa che un
titolo gia' descrive, e' una riscoperta e vale zero.

## I rilievi noti di `candidates/subset/` (20 file .md nella root)

| # | File | Titolo (riga 1) | Cosa copre | Cosa NON copre |
|---|------|-----------------|-----------|----------------|
| 1 | `POINT-PREDICATE.md` | Point-local predicate accumulation with exact chain replay | predicato a livello di punto + replay esatto | il gate SHA, il tree, la finestra |
| 2 | `POINT-BY-TABLE.md` | Combined point predicate and bounded table inverse | predicato + inverse con tabella limitata | il gate, la raccolta |
| 3 | `CHAIN-REPLAY.md` | Exact replay for point-chain field boundaries | replay esatto dei confini di campo | tutto il resto |
| 4 | `BY_NORMALIZED6.md` | Six divsteps indexed by an odd-normalized ratio | 6 divstep su ratio normalizzata | il gate SHA |
| 5 | `PAIR-FRONT.md` | Shared pre-inverse function for paired epochs | funzione pre-inverse condivisa | i due hash separati |
| 6 | `LAST511-RESEARCH.md` | PR511 composition: last511 | composizione ultimo-add | — |
| 7 | `ASMLAST511-RESEARCH.md` | Packed PTX last-add filter experiment | esperimento PTX impacchettato | — |
| 8 | `CANONICAL-ADD.md` | Canonical sparse-prime addition | addizione canonica | — |
| 9 | `COMPLETE-POINT.md` | Complete final addition on the ordinary fifteen-term chain | add finale su catena a 15 termini | — |
| 10 | `HIT-CHECK.md` | Speculative point filter with exact hit recomputation | filtro speculativo + ricalcolo esatto | **il gate H0 e i suoi due hash** |
| 11 | `TREE_INVERSE.md` | Work-efficient block inversion for the GPU-epoch subset grinder | **l'inverse a blocchi** (sezioni 89-243: implementazione, verifiche indipendenti, risultati misurati) | il front, il gate |
| 12 | `PAIR-CURRENT.md` | Corrected current point and inverse with a compact paired pipeline | pipeline a coppie, tree compatto 16 KiB | **l'interleaving dei due hash del gate** |
| 13 | `POINT-X3.md` | Fused raw X3 inside the guarded point trial | X3 fuso | — |
| 14 | `submission-note.md` | record `e6715658` + co-grinder host-only | il record e la nota di submission | il device code |

**Conclusione della regola 4:** nessuno dei 14 titoli descrive **l'interleaving dei
due hash SHA del gate**. `HIT-CHECK.md` copre il *predicato* del filtro; `PAIR-FRONT.md`
copre la *funzione* pre-inverse; `PAIR-CURRENT.md` copre la pipeline a coppie e il
tree. Nessuno copre "i due hash del gate girano in sequenza invece che
interleaved". Questo e' il mio spazio legittimo.

**Cosa invece e' GIAA' coperto e NON rifaccio** (elencato per onesta'):
- l'inverse a blocchi (`TREE_INVERSE.md` ha gia' "Measured results", riga 177) — non
  ritocco `tree_inverse.cuh`;
- la forma della catena a punti, i 15/11/10 termini GLV, il seed, l'ultimo add
  (nove note dedicate) — non ritocco `qsb_filter_chain_trial`;
- il layout GLV11/P18 e il mix Q (`PAIR-CURRENT.md`, `POINT-BY-TABLE.md`) — non tocco
  `QSB_GLV11`/`QSB_Q_P18`/`QSB_Q_MIX`;
- i divisstep del root (`BY_NORMALIZED6.md`, `hm39/41/43`, `zinv32.cuh`) — non tocco.

**Nota sui rilievi pubblici esterni** (9 Blackthorn + 12 ChainSecurity citati nel
brief): non ho accesso a quei testi in questa sessione, quindi **non dichiaro la
mia non sovrapposizione con loro**. Lo dichiaro `NON MISURATO` e lo segnalo come
il primo controllo che chi ha la rete deve fare prima di credere a `FINDING-A`.