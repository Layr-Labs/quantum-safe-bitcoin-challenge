# Pinning 제출 후보: CUDA Graph로 서브배치 실행 재사용

기준 커밋: f0e453daaf8b1af848e0bf4afd42fb730018c041 (2026-09-27 origin/main과 일치 확인).

로컬 개발 제어군에서 반복 개선을 확인한 **공식 평가용 후보**다.
RTX 4090의 개선 폭과 승격 여부는 아직 확인하지 않았다.
benchmark.json의 승격 하한은 100 bips = 1%이며, 로컬 결과는 이보다 작다.
3% 이상의 개선을 달성했다는 주장은 하지 않는다.

## 변경과 가설

128K 후보마다 prepare → root inversion → finish 커널을 다시 실행하고
CUDA 이벤트로 연결하는 호스트 비용을 줄인다.

- 상태 버퍼 링마다 세 커널의 CUDA Graph를 한 번 캡처한다.
- 다음 실행에는 prepare/finish 인자만 갱신한다. root 인자는 마지막 짧은
  배치의 크기가 달라질 때만 갱신하고, 다음 전체 배치에서 복구한다.
- 링별 스트림 순서로 상태 버퍼 재사용을 보호한다. 출력 복사는 모든 링의
  완료를 기다려 기존 hit buffer, sequence, locktime 대응을 유지한다.
- 캡처한 green context, kernel priority, access-policy window를 사용한다.
  green context는 cuCtxFromGreenCtx로 변환한 뒤, context를 명시하는
  cuGraphExecNodeSetParams로 갱신한다.
- QSB_SUBGRAPH_OFF=1이면 원래 실행 경로를 사용한다. 필요한 native kernel
  또는 드라이버 API가 없으면 기존 스트림 경로를 유지한다.

변경 코드: pinning.cu의 호스트 실행 부분과 새 QsbSubGraph.h.
공식 GLV11 테이블 크기, 128-SM 장치 조건, green partition 20/shared 8,
GPU 연산, 후보 열거, CPU co-grinder, OpenSSL gate는 기존 설정이다.

## GPU 코드 동일성 및 빌드

CUDA 12.8.93으로 기준과 후보를 각각 sm_89 CUBIN으로 빌드했다.
두 결과는 기존에 포함된 native image와도 바이트 단위로 동일하다.

SHA-256: 625c22c4298276a77064a5570a38821e8f97a7f6620596f961e945708d5a9bbd

| 커널 | 레지스터/스레드 | 공유 메모리/블록 | 스택 |
|---|---:|---:|---:|
| Prepare | 128 | 14,336 B | 0 B |
| Finish | 64 | 0 B | 0 B |
| Fused root | 112 | 12,288 B | 120 B |

기존 setup.sh pinning으로 최종 엔트리포인트 빌드와 verifier smoke를 통과했다.
로컬 Ubuntu 26.04의 glibc 2.43 헤더는 CUDA 12.8과 충돌하므로, 공식 Ubuntu
24.04 glibc 2.39 개발 패키지를 /tmp/qsb-noble-sysroot에 추출하고 GCC 13을
사용했다. 설치된 시스템 라이브러리는 교체하지 않았다. 사용한 호환 옵션:

    PATH=/usr/local/cuda-12.8/bin:$PATH
    NVCC_PREPEND_FLAGS='-ccbin=/usr/bin/g++-13 -isystem=/tmp/qsb-noble-sysroot/usr/include/x86_64-linux-gnu -isystem=/tmp/qsb-noble-sysroot/usr/include'

고정 nvcc 명령의 -O3, N=24, 링크 옵션은 유지했다.
이 로컬 호환 옵션은 제출 코드 또는 공식 빌드 스크립트에 넣지 않았다.

## 로컬 성능: 4060 Ti 개발 제어군

장비: Ryzen 5 7500F, RTX 4060 Ti 8GB, WSL2.
21.13 GiB 공식 테이블이 로컬 VRAM보다 크므로, 성능 비교에는 기존
GLV12 3-hot 테이블(1,465,193,024 B)을 양쪽에 동일하게 사용했다.
green partition도 양쪽 모두 prepare/root 30 SM, finish 6 SM, shared 2 SM이다.

동일 바이너리에서 Graph OFF/ON만 바꾸고, CUDA 12.8 호스트 빌드와
동일한 GLV12 native image를 사용했다. CPU co-grinder는 양쪽 모두 활성화했다.
N=24, 각 120초, 순서는 A/B/B/A다. 구현을 고정한 뒤 두 seed를 무작위로
선택했으며 각 비교쌍은 같은 입력을 처음부터 다시 계산했다.
하네스, CPU verifier, 시간 측정, 분산 기준을 변경하지 않았다.
점수는 검증된 hit와 하네스의 실제 경과 시간으로 계산했다.

| 실행 | 경로 | Seed | 하네스 초 | 검증 hit | M candidates/s |
|---|---|---:|---:|---:|---:|
| a1 | 기준 OFF | 873050785 | 120.1755 | 2,583 | 180.301055 |
| b1 | Graph ON | 873050785 | 120.2310 | 2,599 | 181.334197 |
| b2 | Graph ON | 1405381414 | 120.2139 | 2,764 | 192.873856 |
| a2 | 기준 OFF | 1405381414 | 120.1711 | 2,748 | 191.825681 |

- 합산 기준: 186.063249 M/s.
- 합산 후보: 187.103593 M/s.
- 전체 점수 개선: **+0.5591%**.
- 비교쌍별 개선: +0.5730%, +0.5464%.
- GPU hit만 분리한 개선: +0.6317%.
- **10,694/10,694 hit 검증 통과**.

이는 작은 개선이 두 입력에서 같은 방향으로 관측된 결과다.
독립적인 4090 성능 증명 또는 3% 개선의 증거로 해석하지 않는다.
원본 JSON의 GPU 필드는 하네스 config에서 가져온 RTX_4090 문자열이다.
실제 측정 장치는 모든 로컬 실행에서 RTX 4060 Ti다. 원본 결과는 고치지
않았고 별도 manifest에 실제 장치를 기록했다.

## 경계 처리와 실제 제출 설정 검증

CPU co-grinder를 끈 별도 검사에서 양쪽 GPU가 같은 3개 sequence,
총 3,733,800,000개 후보를 모두 처리했다. 두 결과의 436개 hit가 완전히
일치했고 중복은 없었다. 마지막 host batch 3,086,016개 및 마지막 sub-batch
71,360개, 다음 sequence에서 전체 크기로 복귀하는 경로를 포함한다.

최종 제출 설정 그대로의 4060 Ti 실행도 N=24, 180초로 검사했다:
220/220 hit 검증 통과,
하네스 181.2647초, 10.181211 M/s.
34-SM 장치에서는 원래 구현과 같이 monolithic fallback을 사용하므로
이 실행은 graph 성능 증거가 아니다. 앞서 원본 main은 1200초 실행에서
11.487155 M/s였다. 두 실행은 시간 길이·seed·호스트 툴체인이 다르고,
VRAM oversubscription의 영향을 받아 직접적인 성능 비교로 쓰지 않는다.

CPU smoke는 N=6, fixed_hits=3으로 3/3 통과했다.
CPU smoke의 분산 게이트 완화는 문서화된 smoke 옵션이며 성능 평가에 사용하지 않았다.

## RTX 4090에서 남는 위험

공식 실행은 작은 개발 테이블 대신 21.13 GiB GLV11 테이블과 더 많은 SM,
더 큰 L2, 기존 116/20 SM(shared 8) 구성을 사용한다. 호스트 실행 비용의
비중과 링별 그래프의 스케줄링 차이가 달라져 개선이 커지거나 사라질 수
있고, 성능 회귀도 배제하지 않는다. 공식 1200초 실행과 1% 승격 기준
충족 여부는 서버 평가가 필요하다.

따라서 이 패키지는 로컬 검증을 통과한 평가 후보이며,
공식 기준을 이미 넘긴 확정 개선품이라는 주장을 하지 않는다.

## 파일 범위와 증거

harness/, problems/, spec/, benchmark.json, setup.sh, benchmark.sh,
GitHub Actions 파일은 기존 해시와 동일하다. GPLv3 및 secp256k1 고지,
기존 산술/해시 헤더, 포함된 native image를 유지했다.
생성된 .pinning.build 캐시 변화와 로컬 실험 파일은 제출 패치에 넣지 않는다.

증거 폴더: benchmark-results/graph-20260927/
주요 파일: paired-abba/summary.json, 각 실행의 artifact.json 및 command.json,
fixed-work/comparison.json, device-base.cubin, device-graph.cubin,
device-resources.txt, final-setup.log, production-fallback/artifact.json,
final-cpu-smoke/artifact.json.

원본 기준 자료: benchmark-results/baseline-20260927-wvL6fv/
작업 트리에 선택한 코드가 이미 적용되어 있다. 별도 patch는 깨끗한 기준
커밋에 적용하기 위한 것이다. 공개 제출, push, commit, PR은 실행하지 않았다.

API 참고:
- [NVIDIA green contexts and CUDA Graphs](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/green-contexts.html)
- [CUDA 12.8 driver graph API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__GRAPH.html)
- [CUDA 12.8 green context API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)
