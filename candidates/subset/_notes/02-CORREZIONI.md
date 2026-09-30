# 02-CORREZIONI — cosa ho smentito io, non i subagent

Regola: un report di un subagent e' una **dichiiarazione**, non un fatto. Qui ho
aperto io i file e ho confrontato riga per riga.

## C1. SMENTITO: il feed-forward di `qsb_sha256_init_transform_pair` NON e' rotto

Il subagent `pair_shared.cuh` ha scritto che `pair_shared.cuh:428-429` aggiunge un
termine spostato per parola. Ho aperto io `pair_shared.cuh:428`:

    428: o0[0]=I[0]+a0;o0[1]=I[1]+b0;o0[2]=I[2]+c0;o0[3]=I[3]+d0;o0[4]=I[4]+e0;...

Confronto con `GPUHash.h:258-265` (`output[i] += <reg>`). Sono **la stessa cosa**:
l'output parte da IV e si aggiunge lo stato. La citazione del subagent nel suo
report **non coincide con il file**: lui riporta `o0[0]=I[0]+a0+b0`, che nel file
non c'e'. Non e' un difetto. **zero mutanti perche' non c'e' niente da smentire.**

## C2. SMENTITO: `qsb_sha256_gate_h0_pair:457` NON e' rotto

`pair_shared.cuh:457`:

    *o0=I[0]+a0+S1(f0)+Ch(f0,g0,h0)+K[63]+w0[15]+S0(b0)+Maj(b0,c0,d0);

Confronto con `sha_gate_fma.cuh:409`:

    return a + S1(f) + Ch(f,g,h) + w[15] + (qsb_klit(63) + QSB_IV0) + S0(b) + Maj(b,c,d);

Stessi termini, ordine diverso. Somma e' commutativa: **identico**. Il subagent ha
concluso "2*T1 + 2*T2 - a_63 invece di T1 + T2": falso, non c'e' alcun doppio.

**Perche' importa:** i due "difetti SHA" piu' citati di questa superficie sono
falsi. Se li avessi passati in alto senza aprire il file, avrei bruciato un posto
nella valutazione su due riscoperte.

## C3. CONFERMATO (mia lettura): `QSB_SHORT_CARRY4` contraddice il proprio commento

`tests/gpu_epochs/filter_tail_sc.cuh:11` dice *"kill switch, **default off**"*;
`filter_tail_sc.cuh:22-24` fa `#ifndef / #define QSB_SHORT_CARRY4 1`. Quindi
**spiega OFF, e' ON**. Le righe 38-43 (`qsb_fsub`) e 59-64 (`qsb_fadd`) sono
l'unica forma compilata, e sono quella che **scarta il riporto del carry su
limb-1**. Il commento del file stesso (riga 15-16) dice che questo alza
l'esposizione da `<= 2^-95` alla classe `2^-31`.

Non e' un difetto di correttezza: il contratto "filter-only, l'host ricalcola
esatto" regge. E' una **contraddizione commento/codice su uno switch che e'
dentro `QSB_CARRIER_KNOBS`** e che il commento dichiara spento. Per il bounty:
rumore. Lo registro perche' il brief chiede la superficie rotta, non solo i difetti.

## C4. VERIFICATO: il gate NON e' interleaved, e questa e' la pista A

`pair_shared.cuh:469-483`, `qsb_k2s_gate_h0`:

    474 #if QSB_GATE_H0_FMA        (default 1, riga 462-464)
    475     h0=_SHA256Pubkey33H0(pb0);
    476     h1=_SHA256Pubkey33H0(pb1);

Due compressioni SHA-256 **intere, in sequenza**. Ma il commento di
`QSB_GATE_PAIR` (riga 374-376) dichiara esattamente il contrario come suo scopo:

    "1 = hash both recovery-id pubkeys in one interleaved SHA-256 block
     (two independent dependency chains -> ILP)"

E `qsb_sha256_gate_h0_pair` (riga 435-459) **esiste proprio per fare
quell'interleaving**, round per round, su due catene indipendenti — ma e' nel
`#else` (riga 478), che `QSB_GATE_H0_FMA 1` esclude. Dettaglio in `03-RILIEVI.md`.