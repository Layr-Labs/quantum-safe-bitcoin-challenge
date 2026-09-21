#!/usr/bin/env python3
"""Static audit for the composed pinning candidate."""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    math = (HERE / "GPUMath.h").read_text()
    cu = (HERE / "pinning.cu").read_text()
    tree = (HERE / "cofactor_checkpoint.h").read_text()

    required = {
        "host_gate": "#define QSB_HOST_GATE 1" in cu,
        "c31": "#define QSB_C31 1" in cu,
        "rp_sqr": "#define QSB_RP_SQR 1" in cu,
        "sas_z9": "#define QSB_SAS_Z9SUB_ALL 1" in math,
        "sas_no_z9_add": '#define QSB_SAS_Z9ADD_TAIL ""' in math,
        "sas_no_z9_sub": '#define QSB_SAS_SUB_TAIL ""' in math,
        "sas_direct_fold": '#define QSB_SAS_SFQ "mov.u32 sfq, z8;"' in math,
        "top16": "#define QSB_TOP16 1" in tree and "qsb_cofactor_top16" in tree,
        "lazy_finish": "#define QSB_LAZY_ADD_FINISH 1" in cu,
        "lazy_xminus": "QSB_FINISH_ADD(x_minus, f, h);" in cu,
        "lazy_parity": "QSB_FINISH_ADD(h, u, v);" in cu,
    }
    assert all(required.values()), required
    assert math.count("QSB_SECOND_FOLD_TAIL") == 7
    print({"test": "pinning composite source audit", "checks": required})


if __name__ == "__main__":
    main()
