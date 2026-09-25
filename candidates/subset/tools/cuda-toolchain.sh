#!/usr/bin/env bash
# Assemble a user-local CUDA 12.3 build toolchain for hosts without a system
# CUDA install (no root CUDA packages, gcc newer than nvcc supports), so that
# ./setup.sh subset can prebuild the kernel with the harness's own nvcc command.
#
#   candidates/subset/tools/cuda-toolchain.sh [PREFIX]    # default ~/.cache/qsb-cuda
#   eval "$(candidates/subset/tools/cuda-toolchain.sh --env [PREFIX])"
#   ./setup.sh subset
#
# Sources: NVIDIA's CUDA redistributable archives (nvcc, cudart, curand, cccl).
# nvcc 12.3 accepts gcc <= 12, so a gcc-12/g++-12 shim directory is put first on
# PATH when the default gcc is newer (install g++-12 from your distro first).
#
# The same prefix also works with the open-source clang CUDA frontend:
#   clang++ -x cuda --cuda-path="$PREFIX" --cuda-gpu-arch=sm_89 -O3 -DQSB_ZEROS_N=24 \
#     -o subset candidates/subset/subset.cu -L"$PREFIX/lib64" -lcudart -lcrypto -lm
#
# Building needs no GPU; running the kernel needs an NVIDIA GPU and driver.
set -euo pipefail

env_only=0
if [[ "${1:-}" == "--env" ]]; then env_only=1; shift; fi
prefix="${1:-${HOME}/.cache/qsb-cuda}"
redist="https://developer.download.nvidia.com/compute/cuda/redist"
nvcc_ver="12.3.107"
cudart_ver="12.3.101"
curand_ver="10.3.4.107"
cccl_ver="12.3.101"

print_env() {
  local path="${prefix}/bin"
  [[ -d "${prefix}/gcc12" ]] && path="${prefix}/gcc12:${path}"
  echo "export PATH=\"${path}:\${PATH}\""
  echo "export LD_LIBRARY_PATH=\"${prefix}/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\""
}
if ((env_only)); then print_env; exit 0; fi

fetch() {  # component version
  local name="$1-linux-x86_64-$2-archive"
  [[ -d "${prefix}/.src/${name}" ]] && return
  mkdir -p "${prefix}/.src"
  echo "cuda-toolchain: fetching ${name}" >&2
  curl -fsSL "${redist}/$1/linux-x86_64/${name}.tar.xz" | tar -xJ -C "${prefix}/.src"
}

[[ "$(uname -m)" == "x86_64" ]] || { echo "cuda-toolchain: x86_64 only" >&2; exit 1; }
fetch cuda_nvcc "${nvcc_ver}"
fetch cuda_cudart "${cudart_ver}"
fetch libcurand "${curand_ver}"   # clang's CUDA wrapper includes curand headers
fetch cuda_cccl "${cccl_ver}"     # ...which include <nv/target>

mkdir -p "${prefix}"
for d in "${prefix}"/.src/*/; do
  for sub in bin include lib nvvm; do
    [[ -d "${d}${sub}" ]] || continue
    dst="${sub}"; [[ "${sub}" == lib ]] && dst=lib64
    mkdir -p "${prefix}/${dst}"
    cp -a "${d}${sub}/." "${prefix}/${dst}/"
  done
done
echo "CUDA Version ${nvcc_ver%.*}.0" > "${prefix}/version.txt"

gcc_major="$(gcc -dumpversion 2>/dev/null | cut -d. -f1 || echo 0)"
if ((gcc_major > 12)); then
  command -v g++-12 >/dev/null 2>&1 \
    || { echo "cuda-toolchain: gcc ${gcc_major} is too new for nvcc 12.3; install g++-12" >&2; exit 1; }
  mkdir -p "${prefix}/gcc12"
  for n in gcc cc; do ln -sf "$(command -v gcc-12)" "${prefix}/gcc12/${n}"; done
  for n in g++ c++; do ln -sf "$(command -v g++-12)" "${prefix}/gcc12/${n}"; done
fi

"${prefix}/bin/nvcc" --version | tail -2 >&2
echo "cuda-toolchain: ready; run: eval \"\$($0 --env ${prefix})\"" >&2
print_env
