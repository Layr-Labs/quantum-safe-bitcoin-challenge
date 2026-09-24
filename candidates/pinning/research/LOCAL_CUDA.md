# CUDA compilation on the Mac

The local Mac now has a task-specific ARM Linux VM with NVIDIA's CUDA compiler.
It has no NVIDIA GPU. This closes the CUDA compilation/assembly evidence gap,
but does not provide GPU correctness, runtime, timing or driver-JIT results.
Do not treat an ARM host executable as the exact Linux x86 judge executable.

The environment uses portable Lima 2.2.0, Apple VZ, Ubuntu 24.04 ARM, four CPUs,
4 GiB RAM and a 12 GiB sparse disk. Containerd is disabled. The only host mount
is the dedicated temporary build directory, writable by the VM. No home or
credential directories are mounted. CUDA 12.8.93 compiler, runtime headers,
cuobjdump and nvdisasm were installed from NVIDIA's official Ubuntu 24.04 SBSA
repository alongside build-essential and libssl-dev. No GPU driver was added.

The local paths below are operational notes, not text for public submission
notes. Temporary storage can be cleaned by the operating system; preserve
source-bound reports in the selected candidate directory.

```sh
LIMA_HOME=/tmp/qsb-lima-state /tmp/qsb-lima-bin/bin/limactl start qsb-cuda --tty=false
python3 candidates/pinning/research/compile_local.py --default-build --report /tmp/pinning-native.json
LIMA_HOME=/tmp/qsb-lima-state /tmp/qsb-lima-bin/bin/limactl stop qsb-cuda
```

The helper snapshots the quoted-include closure into a fresh build directory,
compiles and links for `sm_89`, captures ptxas register/shared/stack/spill data,
disassembles the cubin and records source hashes. `--default-build` also builds
without an explicit architecture, matching the official compile flags. It
does not alter the source, install software, start the VM or submit anything.
Its `--source` and `--entry` options support immutable staged audit programs.
Source and guest mount paths are mapped explicitly: macOS resolves `/tmp` to
`/private/tmp`, which is not the Linux mount path.

Keep the VM stopped between compilation sessions to release its memory. Native
resource reports help reject instruction expansion, spills or build errors;
they cannot establish an expected major gain by themselves. Check the exact
ranked kernel instantiation instead of quoting fallback-kernel resources.

Official installation references:

- https://lima-vm.io/docs/installation/
- https://lima-vm.io/docs/config/vmtype/vz/
- https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/
