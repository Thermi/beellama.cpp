# Nanbeige 90k Baseline

## Environment

- GPU: NVIDIA GeForce RTX 5060, 8150 MiB, compute capability 12.0
- Driver: 610.88
- CUDA toolkit: 13.3.33
- Source: `016eb263d`
- Model: `Nanbeige4.2-3B-heretic-Q4_K_M.gguf`
- Model size reported by `llama-bench`: 2.39 GiB
- KV cache: `q4_0` K / `q4_1` V
- Runtime: `-ngl 999 -b 512 -ub 256 -t 6`
- Focused build: Release CUDA, `CMAKE_CUDA_ARCHITECTURES=120`, `GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_KVARN=OFF`

`GGML_CUDA_KVARN=OFF` is an explicit supported build option. It excludes KVarN-only kernels while retaining standard Q4 FlashAttention, which is the cache path used by this Nanbeige benchmark.

## Valid Serial Measurements

| Prompt | Decode | Repetitions | Decode tok/s | Prompt tok/s | Notes |
|---:|---:|---:|---:|---:|---|
| 0 | 128 | 3 | 70.61 +/- 0.17 | 2735.89 +/- 36.93 | Initial focused build |
| 0 | 128 | 3 | 73.79 +/- 0.68 | 2769.28 +/- 17.14 | Clean serial rerun |
| 8000 | 32 | 3 | 75.18 +/- 0.42 | 2793.17 +/- 16.68 | Clean representative long-context run |
| 40000 | 32 | 3 | unavailable | 653.40 +/- 0.57 | `llama-bench` emitted only the prompt row before completion; decode row unavailable |
| 90000 | 32 | 2 | unavailable | unavailable | Timed out after 40 minutes without a result row |

The 8k run is the valid profiling workload used for kernel attribution. The 40k and 90k values are not extrapolated.

## Invalid Measurements Excluded

- 40k and 90k runs launched concurrently: excluded because both processes contended for the same GPU.
- Candidate attention benchmark launched concurrently with CTest/Nsight: excluded because the GPU and profiling driver resources were shared.
- A 90k run that exceeded the timeout: recorded as unavailable, not as zero throughput.

## Build Verification

The full all-quant CUDA build could not complete on this host within the available build window. The focused standard-Q4 build completed and produced `llama-bench.exe` and `llama-server.exe`. The required KVarN test is not applicable to the `GGML_CUDA_KVARN=OFF` focused build because its graph intentionally contains no KVarN operations.
