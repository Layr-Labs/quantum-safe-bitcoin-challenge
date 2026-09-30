# 16 — STATO MISURATO. Due affermazioni del brief sono sbagliate.

Tutto qui sotto e' **misurato in questa sessione**, non ricordato. Due righe
del brief non reggono il confronto con la macchina.

## 1. `submissions: []` e' FALSO. La submission c'e' gia'.

Tre fonti indipendenti, stesso id:

    yukon submissions b352879c-669f-44ef-98cd-ad3d34d0fefa --json
      -> submissions[] length 1
    GET /api/submissions/368ac8dc-390e-4650-bf2b-9b5084224e74   -> HTTP 200
    python3 /home/j-ai/concord/stato-submission.py
      -> giro 1/2/3: status = validating

    id       368ac8dc-390e-4650-bf2b-9b5084224e74
    account  v12superfast / d3bc8fa3-8295-4bb0-a0a9-6baa1aa29477
    cubin    b12d7a8d78db71e3e3512350671e1dcab6ca183bed2e4b0dd06961add207aaa1
    status   validating
    score    officialScore: null   improved: null   promotionStatus: null
    created  2026-09-30T18:14:49.319Z
    updated  2026-09-30T18:14:49.319Z   <-- identico: nessuna transizione

`createdAt == updatedAt` significa che dal momento della creazione nessuna
transizione di stato e' stata scritta. Il runner non ha ancora prodotto un
numero. **Questo non e' "zero submission", e' "una submission in corso".**
La differenza conta: non devo produrre la prima, devo produrre la *prossima*.

## 2. Il gate `readelf` del brief non puo' passare. File inesistente.

Il brief prescrive, come "prova che funziona":

    readelf -h pinning_sm89.cubin | grep -q "Flags:.*0x59" && echo PASS

**Fallisce, perche' `pinning_sm89.cubin` non esiste.** Non e' un problema di
locale:

    ls: cannot access 'pinning_sm89.cubin': No such file or directory   (exit 2)
    find /home/j-ai/quantum-safe-bitcoin-challenge -name pinning_sm89.cubin
      -> 0 risultati

`LANG` era gia' `C` per tutta la sessione, quindi la nota 05 e la nota 13
hanno ragione sul locale e torto sul filename. Causa: `build_carrier.sh:9-11`
compila dentro `$(mktemp -d)` e scrive `c.cubin`; quel nome non viene mai
prodotto da nessun percorso del repo.

**Il gate corretto, e PASSA:**

    LANG=C readelf -h <cubin prodotto da build_carrier.sh> | grep -q "Flags:.*0x59"
    -> PASS
       Flags:                             0x590559
       cuobjdump -elf: 64bit elf: type=2, abi=7, sm=89, toolkit=128, flags = 0x590559

Un controllo che non parte non e' un controllo che non ha misurato: e' un
controllo che non esiste.

## 3. Baseline: riprodotta, tre volte

    $ ./build_carrier.sh 24            # sorgente INVARIATO
    qsb_carrier_sm89.h: 476832 B cubin, sha256 913a97b2a8e6354e, ...

    sha256sum del c.cubin prodotto
      913a97b2a8e6354e7632a2f41a997a05bf0e61e5a8b9a5c5c793f931e983b463
    sha256sum qsb_carrier_sm89.h  PRIMA  ac933f39eb46720fea8c61b61b897a657dfd05f1f8b05d3c564d2b4a231814e6
    sha256sum qsb_carrier_sm89.h  DOPO   ac933f39eb46720fea8c61b61b897a657dfd05f1f8b05d3c564d2b4a231814e6
    git status --porcelain           -> nessun file tracciato modificato

**Identico bit per bit.** Il riferimento `913a97b2` e' vivo e il toolchain
e' quello giusto (nvcc 12.8.93, gcc 14.3.0).

Stato degli interruttori, misurato:

    pinning.cu:201   #define QSB_PMIX12_WARP 0     <-- OFF
    pinning.cu:191   /* kill switch, default 1 */  <-- il commento mente

Il tree e' al **baseline**, non allo stato sottomesso. La submission ha
pacchettizzato `QSB_PMIX12_WARP 1` in un commit; il worktree e' tornato a HEAD.

## 4. Registri, misurati (ptxas -v, immagine intera)

| kernel | reg | stack | spill | smem |
|---|---:|---:|---:|---:|
| `kernel_pinning_pipeline<true,0>`  (QK_S0, il caldo) | **128** | 0 | **0** | 14336 |
| `kernel_pinning_pipeline<true,2>`  (QK_S2) | 64 | 0 | 0 | – |
| `kernel_build_gtable` | 162 | 944 | 0 | – |
| `qsb_root_fused<8>` | 112 | 120 | 0 | 12288 |

**Spill stores = 0 e spill loads = 0 in tutte le funzioni.** L'unico kernel a
128 registri e' quello caldo, ed e' saturo: e' il posto dove un livello in piu'
va in spill.

## 5. La soglia, ricalcolata dal campo dell'API

L'API **non** restituisce mai una soglia. Restituisce:

    minScoreImprovementBips: 100
    currentBestScore:       1008206828

    1 008 206 828 x 1.01 = 1 018 288 896.28  ->  1 018 288 896

Il numero del brief e' **corretto**, ma e' una costante scritta a mano in
`stato-submission.py:19` (`SOGLIA = 1018288896`), non una lettura. Quello che
la API dichiara e' `100 bips`, e il testo di rifiuto degli altri lo conferma:
`score improved but fell short of the required 100 bips improvement`.

**Nota di calibrazione:** `benchmark.closesAt = 2026-10-07T23:00:00.000Z`.
Non il 7 ottobre in generale: le 23:00 UTC del 7.

## 6. Il fatto che conta piu' di tutti: la soglia e' vicinissima

Dalle 2101 submission `--all`, le migliori **rifiutate**:

    ercumentyildirim  officialScore=1016064891  "fell short of the required 100 bips"
    cefika            officialScore=1015939991  "fell short of the required 100 bips"

La migliore in assoluto, e rifiutata, e a **+0,78 %** sul record. La soglia e'
a **+1,00 %**. Manca mezzo punto percentuale a chi e' piu' vicino, e quel
mezzo punto e' costato 1485 righe di `rejected`.

Conseguenza operativa: **`improved=True` non basta.** Il gate e' UN passo
solo, e deve valere >= 1,00 % sulla standings. Qualsiasi cosa che produca
"+qualcosa" non paga. E serve il numero vero dal runner: qui non gira CUDA.

**Nota tecnica che spiega il rifiuto di quelle due:** hanno migliorato il
record ma non di 100 bips, quindi `promotionStatus` resta null e
`currentBestScore` resta 1 008 206 828 — il best **non** si alza con loro.
Il best si alza solo con una promossa. Nessuno ci e' arrivato dal 28
settembre (record di kaankolcu). 9 submission sono in `validating` ora.