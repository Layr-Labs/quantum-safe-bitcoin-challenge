# Portable arithmetic checks

Run from `candidates/pinning`:

```
python3 warp_checks/check_warp_inverse.py
python3 warp_checks/check_balanced_roots.py
```

These scripts run independent Python integer/address models, never native
compilers or CUDA. The two promoted Subset source files are reference data
from a137e289b236c3622eba80f1ad5e9a0c8a91eb67; they are not included in the
production build. Their retained comments carry original attribution.

The official executable independently checks the new kernel with OpenSSL at
startup. That native check was implemented, not run on the development host.
A passing Python model is not a GPU compilation, timing or race-detector result.
