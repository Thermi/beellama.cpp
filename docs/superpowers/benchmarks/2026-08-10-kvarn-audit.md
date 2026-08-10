# KVarN/Nanbeige Change Audit

## Scope

Audited branch `kvarn-native-decode` against `016eb263d`, including Nanbeige SWA/sink metadata, ISWA/KV sizing, CUDA/MSVC runtime selection, KVarN prefetch syntax, KVarN window routing, tests, and benchmark notes.

## Verified CUDA guidance

- NVIDIA CUDA C++ Best Practices Guide 13.3 recommends the APOD loop: assess the measured hotspot, apply one change, verify correctness and speed, then repeat.
- The same guide treats register pressure, occupancy, coalescing, shared-memory bank conflicts, L2 behavior, asynchronous copies, and concurrent execution as separate constraints. Occupancy changes must be validated with the resulting register/spill behavior; `__launch_bounds__` is not automatically a speedup.
- CUDA asynchronous host/device transfers require pinned host memory for overlap; repeated accesses to mapped host memory can re-transfer data, so hot data should be cached in device memory.
- CMake documents `CUDA_RUNTIME_LIBRARY` as `None`, `Shared`, or `Static`. CUDA 13.2+ changed Windows NVCC runtime selection from `-cudart=shared` to `-cudart=hybrid`; this branch now selects `Hybrid` only for CUDA >= 13.2 and `Shared` for older toolkits.

## Measured performance conclusions

- Q4 F16-MMA attention is latency-limited at 8k but DRAM-bound at 40k (95.6% DRAM, approximately 428/448 GB/s).
- Existing Q4 vector attention is direct-Q4 but approximately 12x slower for multi-token long-context; optimistic dispatch was rejected.
- KVarN `kvarn8` decode was approximately 77 t/s; `kvarn4` completed a 40k `pp40000+tg16` run at approximately 596 t/s, but KVarN prefill pays record-store cost.
- Making KVarN F16-window routing opt-in improved decode only marginally but made large-batch prefill approximately 15x slower; the window remains default-on for that reason.

## Corrections applied

- Preserve `sink_tokens` when `llama_memory_hybrid_iswa` forwards the ISWA cache constructor arguments.
- Save `LLM_KV_ATTENTION_SINK_TOKENS` in the model saver.
- Clamp the effective SWA sink span consistently in SWA masking and KVarN record-ring sizing; the record ring now accounts for `n_swa + effective_sink`.
- Make CUDA runtime selection version-aware for Windows CUDA 13.2+ hybrid runtime linking.
- Added a static sink-contract regression test covering forwarding, saver persistence, and KVarN ring sizing.

## Remaining performance work

The remaining high-value optimization is the KVarN record-store/encode path and the hybrid F16 window path. Any change there must be evaluated with separate prefill and decode metrics, Nsight Systems kernel totals, and Nsight Compute DRAM/occupancy data. Do not use aggregate `pp+tg` alone to choose a kernel because it conflates prompt encoding and token generation.

## Sources

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
- https://cmake.org/cmake/help/latest/prop_tgt/CUDA_RUNTIME_LIBRARY.html
- https://discourse.cmake.org/t/hybrid-cuda-runtime-linking/15607
- https://docs.nvidia.com/cuda/archive/13.3.0/cuda-compiler-driver-nvcc/index.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
- https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
- https://docs.nvidia.com/cuda/developer-preview/13.4/nsight-compute/ComputeTriage/index.html
