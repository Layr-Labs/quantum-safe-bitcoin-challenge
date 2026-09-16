# Pinning candidate verification

Only edit candidates/pinning for the pinning track. Run Yukon commands from the repository root, not this subdirectory.

The new shared_finish.cuh helper has a CPU algebra audit that includes the actual helper and uses OpenSSL field-operation adapters:

```sh
clang++ -std=c++11 -O2 -Wall -Wextra $(pkg-config --cflags openssl) candidates/pinning/test_shared_finish.cpp -o candidates/pinning/.test_shared_finish $(pkg-config --libs openssl)
candidates/pinning/.test_shared_finish
```

For a host memory/undefined-behavior audit, replace -O2 with -O1 -fsanitize=address,undefined -fno-omit-frame-pointer. Keep assertions enabled; NDEBUG is rejected. Expected output is 4096 projective cases, 8192 recovered points/hashes and four fallback cases passing, with 11 multiplications, two squares and one inverse on each normal input.

This test does not validate CUDA compilation, inline PTX field arithmetic or GPU performance. Official GPU verification still requires yukon setup --track pinning and yukon run --track pinning. A CPU-reference harness score must not be represented as a CUDA-candidate score. Exclude generated host test binaries and debug symbols from submissions. Preserve the existing GPL notices and COPYING.
