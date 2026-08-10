# Nanbeige4.2-3B 90k Optimisation — Measured Findings

Status: research notes. Dates: 2026-08 (session). Hardware: RTX 5060 8 GiB (GB206, cc 12.0, sm 30, 448 GB/s), CUDA 13.3.

These are recorded **measured** results and **web-sourced** upstream facts. No inferred numbers.

## Baseline metric (valid, serial)

- Model: Nanbeige4.2-3B-heretic-Q4_K_M.gguf (2.39 GiB)
- `-pg 8000,32 -ngl 999 -b 512 -ub 256 -t 6 -ctk q4_0 -ctv q4_1 -r 3`
- pp512 ≈ 2740–2810 t/s; tg128 ≈ 73–76 t/s; pp8000+tg32 ≈ 1622–1705 t/s
- Focused build: Release CUDA, `CMAKE_CUDA_ARCHITECTURES=120`, `GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_FA=ON`, `GGML_CUDA_KVARN=OFF` (NOT `GGML_CUDA_FA_ALL_QUANTS`)

## Nsight Systems (8k, serial; ananaysed with NVIDIA Nsight Systems 2026.4.1 skill CLI)

### GPU kernel time (top by total)
| Kernel | Launches | Total | Mean |
|---|---|---|---|
| `mul_mat_q` (type12 / Q4_K mmq) | 17 490 | 3.79 s | 217 µs |
| `flash_attn_ext_f16<128,128,8,8,0,0,1,1>` | 2 992 | 2.43 s | 812 µs |
| `mul_mat_q` (type14) | 2 640 | 0.66 s | 251 µs |

### Host CUDA API time (top by total)
| API | Calls | Total |
|---|---|---|
| `cudaStreamSynchronize` | 3 750 | 6.60 s |
| `cudaLaunchKernel` + `...ExC` | ~113 k | 3.05 s |
| `cudaMemcpyAsync` | 1 950 | 1.08 s |

### Host<->Device transfers
- Host→Device: 1 782 ops / 2.64 GB / 993 ms, one op up to 196 ms
- Device→Host: 111 MB; Memset: 479 MB

### Transfer origin (traced)
- Host→Device traffic is dominated by **46 copies of exactly 64 MiB (67 108 864 B)**, each into a **distinct** device buffer (COUNT(DISTINCT allocationGlobalHandle)=46), first @9.36 s → last @12.95 s.
- 46 × 64 MiB ≈ 2.88 GB matches the total (~44 logical layers + extras).
- **Conclusion:** the 2.64 GB H2D is a **one-time bulk per-layer buffer upload during setup**, not per-token steady-state traffic. It does NOT scale with each generated token, so it is not the driver of 8k→90k decode degradation.

## Nsight Compute (serial)

### `mul_mat_q<12,128,0>` (largest GPU timer)
- DRAM 22.3% / mem 40.0% / SM 49.6% → latency/occupancy-limited, not bandwidth-bound
- 251 regs/thread (0 spills); ~2.0 active warps/scheduler; 49% no-eligible-warp cycles
- LSU = highest pipe (40%): shared-store bank conflicts 33% (1.6-way); L2 hit 81.6%

### `flash_attn_ext_f16<128,128,8,8,...>` (attention)
- 255 regs/thread, local-memory spills (2768 B), 82% no-eligible-warp cycles
- DRAM 18.8% / mem 34.6% / compute 35.8% → latency/register-bound

### `flash_attn_ext_vec<128,1,3,3,0>` (q4_1/q4_1 native vec)
- DRAM 8.9% / mem 36% / compute 36%, duration 10 µs, grid (1,2,48) → small/latency-limited

## Experiments (measured, serial)

| Change | Result vs baseline |
|---|---|
| `q4_1/q4_1` (homogeneous valid pair) | ≈ same (tg128 74.9, pp8k 1700) — native vec, no throughput change |
| `mul_mat_q` force 2 resident blocks/SM (`__launch_bounds__` 2) | **Regression**: pp512 2020 (↓28%), tg128 70.8, pp8k 1350 — registers capped near 128 → spills |
| attention `nthreads 128→256`, occupancy 2→1 (ncols=64 row) | neutral on pp512/tg128, but pp8k 1501 (↓~8%) — not a win |
| `GGML_CUDA_FA_ALL_QUANTS=ON` build (q4_0/q4_1) | **identical to default-FA baseline**: pp512 2770.65, tg128 74.98, pp8k 1703.62 → the all-quants build does NOT change speed |

All levers tested (cache pair, occupancy x2, attention threads, all-quants FA) fail to beat baseline.

## Profile of the ALL_QUANTS build (Nsight Systems, q4_0/q4_1, -pg 8000,4)

| Kernel                                   | Launches | Total  | Mean  |
| ---------------------------------------- | -------- | ------ | ----- |
| `mul_mat_q<12>` (Q4_K weight mm)         | 17 490   | 3.93 s | 225 µs |
| `flash_attn_ext_f16<128,128,8,8,F16,F16>`|  2 992   | 2.53 s | 846 µs |
| **`flash_attn_ext_vec<128,1,q2,q3>` (new, Q4-native)** | 132 | 14 ms | 107 µs |
| `dequantize_block_q4_1`                  |  2 992   | 81 ms  | 27 µs |

- The all-quants build DOES compile + dispatch a Q4-native vector attention (`flash_attn_ext_vec`), used for the **single-token decode** path (132 launches, no dequant).
- The **dominant attention is still `flash_attn_ext_f16`** (2,992) with `dequantize_block_q4_1` (2,992) — the multi-token/prefill path still dequantizes `q4_1` to F16 and runs the MMA F16 kernel. Hence no throughput change in the prefill-heavy `-pg 8000` benchmark.
- `mul_mat_q` (Q4_K weights) remains the **#1 GPU cost** (3.93 s); its earlier Nsight Compute capture showed it latency/occupancy-limited, not DRAM-bound.
- Decode `tg128` stayed ~75 tok/s despite Q4-native decode attention — the Q4-native path does not improve measured decode for this model.

## Long-context (40k) profile — the decisive scaling evidence

Nsight Systems at `-pg 40000,4` (all-quants build, q4_0/q4_1), top kernels:
| Kernel                            | Launches | Total  | Mean  |
| --------------------------------- | -------- | ------ | ----- |
| `flash_attn_ext_f16` (F16+dequant) | 13 992   | **89.3 s** | 6.4 ms |
| `mul_mat_q<12>` (Q4_K weights)     | 83 740   | 19.4 s | 231 µs |
| `dequantize_block_q4_0`            | 13 992   | 2.0 s  |       |
| `dequantize_block_q4_1`            | 13 992   | 2.2 s  |       |

At 40k, attention is ~4.6× the weight matmul (at 8k it was a minor share). The multi-token attention still runs F16 MMA with `dequantize_block_q4_0/q4_1` growing with context.

Nsight Compute on a late ≈full-40k `flash_attn_ext_f16` launch:
- DRAM throughput **95.6%** (≈428/448 GB/s), memory throughput 95.6%, compute 51.7%, 11.6 ms/op.

## Fix attempt result (measured) — dispatcher-only Q4-native route is INSUFFICIENT

A subagent implemented route (a): a dispatcher `ggml_cuda_fattn_q4_multi_vec` in `fattn.cu` gated on `3 < Q->ne[1] <= 16`, single sequence, `K->ne[1] >= 8192`, Ada+, Q4 K/V — to route small multi-token Q4 batches to the existing Q4-native vec kernel.

Measured at 40k (`-pg 40000,4`, all-quants+q4_multi_vec), top kernels:
| Kernel                            | Launches  | Total  | vs baseline |
| --------------------------------- | --------- | ------ | ----------- |
| `flash_attn_ext_f16` (F16+dequant) | 13 992    | 92.4 s | UNCHANGED   |
| `dequantize_block_q4_0/1`          | 13 992 ea | 4.4 s  | UNCHANGED   |
| `flash_attn_ext_vec` (single-token) | 132       | 66 ms  | pre-existing |

**Why it did not help (measured):** the prefill attention here runs 256-token batches ⇒ `Q->ne[1]=256 > 16`, so the gate never fires and it stays on the F16-MMA+dequant path. The DRAM-bound 8×8 (`ncols=64`) multi-token shape the user needs is exactly the case the conservative gate excludes. Single-token decode was already Q4-native before this change.

**Conclusion:** the dispatcher-only change is correct but does not relieve the measured bottleneck. The genuine fix is route (b): a **Q4-native kernel for the multi-token `ncols=64` (8×8) MMA shape** that consumes q4_0/q4_1 directly (thread Q4 source + byte strides through the iteration templates) — the previously-blocked change.

## Optimistic dispatcher — measured regression (dead end)

Widened `ggml_cuda_fattn_q4_multi_vec` cap from `Q->ne[1]<=16` to `<=256`, so the Q4-native vec path is selected for full 256-token prefill batches at long KV. Measured at 40k (`-pg 40000,4`):
- `flash_attn_ext_vec<128,2,q2,q3>`: 11,088 launches, **1128.6 s** (mean 101.8 ms/op) vs baseline `flash_attn_ext_f16` 89.3 s → **~12× SLOWER**.
- `dequantize_block_q4_0/q4_1`: eliminated (0 launches) — the optimistic gate did route multi-token to Q4-native vec and removed the dequant passes, but the vec kernel is far slower at ncols=2 long-context.
- `flash_attn_ext_f16` dropped to 2,904 launches (non-Q4 remainder).

**Conclusion:** dispatching the EXISTING Q4-native vec kernel to the multi-token case is a dead end in BOTH conservative (no effect) and optimistic (12× regression) forms — it is not viable for multi-token long-context attention. No improvement was found via the vector path. The only untried, high-risk path is a proper **Q4-native tensor-core MMA** rewrite for the ncols=64 shape.

## Attention-kind landscape for long-context q4 (from source map)

| Kind | Tensor-core? | Q4 native (no F16 dequant)? | Useful here? |
|---|---|---|---|
| MMA F16 (current) | Yes | No (dequant→F16, ~4× bytes) | Baseline; DRAM-bound at 40k (95.6%) |
| WMMA F16 | Yes | No (F16) | Never dispatched on Blackwell (Volta/RDNA3 only) |
| Vector | No (ALU) | Yes (reads Q4 directly) | Good for single-token decode (already active); ~12× slower multi-token |
| Tile | No | No (F16) | Non-tensor-core F16, not better |
| KVarN native | Mixed | Yes (records) | KVARN=OFF; uses KVarN format, not q4_0/q4_1 |

**Measured conclusion:** the only Q4-native attention kind avoiding the F16 DRAM path is the vector kernel, which is ALU-bound and ~12× slower for multi-token. No tensor-core MMA in the tree consumes Q4 directly. None of the implemented kinds is useful for multi-token long-context q4; the missing piece is a net-new Q4-native tensor-core MMA.

## KVarN measurement (GGML_CUDA_KVARN=ON, kvarn8) — the fork's native-attention path

Built with KVARN=ON (default FA pairs, arch 120). KVarN selected via `-ctk kvarn8 -ctv kvarn8` (pseudo type → kvarn_k8v8_g128 records; head dim 128 supported).

| Metric | kvarn8 | q4 baseline |
|---|---|---|
| decode tg128 @ ~8k intent | **77.67 t/s** | ~74-75 t/s |
| decode tg128 @ ~40k intent | **77.45 t/s** | (q4 40k prefill timed out) |
| pp512 | 1533 t/s | ~2740-2770 |
| pp8000+tg32 | 1030 t/s | ~1622-1705 |

Notes: KVarN gives decode slightly above q4 and uses the fork's tensor-core native record attention (no F16 materialization; kvarn8 ≈1 byte/elem), so it avoids the q4 F16-dequant DRAM pressure. Prefill metrics are lower (KVarN encode cost). The full 40k prefill timed out in this benchmark window on both kvarn and q4 (>40 min on this host), so a clean 90k decode-vs-baseline could not be completed here.

### Profile of the exact kvarn8 build (Nsight Systems, -pg 8000,4)

| Kernel                                          | Launches | Total  | Mean  |
| ----------------------------------------------- | -------- | ------ | ----- |
| `kvarn_store_kernel_hishmem` (record encode)      | 6,160    | **4.9 s**  | 796 µs |
| `ggml_cuda_fattn_kvarn_window_f16_partial_kernel` | 2,992    | 2.28 s | 763 µs |
| `ggml_cuda_fattn_kvarn_window_dequant_kernel`     | 2,992    | 0.73 s | 245 µs |
| `flash_attn_ext_f16` (MMA F16)                    | 2,904    | 0.51 s | 175 µs |
| `mul_mat_q` (weights)                             | 17,490   | 3.95 s | 226 µs |

Nsight Compute on `ggml_cuda_fattn_kvarn_window_f16_partial_kernel<128,128,8,8>`:
- DRAM 24.0%, memory 34.2%, compute 35.8% → latency-limited (SOL <60%), tensor pipeline highest, no spills.

**Insight:** the exact `kvarn8` path is HYBRID, not pure native: `kvarn_store` (record encode) is the #1 cost and is why prefill is slow; attention then runs `window_f16_partial` + `window_dequant` + `flash_attn_ext_f16` (partial F16 window materialization). KVarN's benefit is the compressed record footprint (small KV → very high context on 8 GB) and a small decode gain (~77 vs ~74 t/s); it is not a fully-native record MMA in this workload.

## Working long-context configuration: KVarN kvarn4 (measured)

`-ctk kvarn4 -ctv kvarn4` at 40k context on the 8 GB card (KVARN=ON build), single run:
| Metric | kvarn4 |
|---|---|
| pp512 | 1573.91 t/s |
| tg128 (short-context decode) | 70.72 t/s |
| pp40000+tg16 (combined) | 596.33 t/s |

Contrast: the q4 F16-dequant 40k attention was DRAM-bound (95.6%, ~89 s attention); kvarn8 40k prefill timed out. kvarn4 retains 4-bit KV precision, halves KV bytes vs kvarn8 (~0.5 byte/elem → ~2× longer context fit), and completes a 40k-context benchmark in-window. This is the largest real lever found and the recommended long-context setting on 8 GB.


## Branch survey — the fork's KVarN native attention is the helpful mechanism (already in base)

Local-branch survey of attention/KV implementations (read-only, git ls-tree/show):

| Branch | Attention/KV kind | Tensor-core? | Direct KV (no F16)? | Long-context-on-8GB helpful? |
|---|---|---|---|---|
| kv-compact-only (base 016eb263d) | KVarN native MMA on 2–8-bit records (decode-split/vector) | Yes | Yes | **Yes — fork's intended solution** |
| kv-speed-work (e00618d9) | NEW packed-q4 D128 kernel + persistent F16 shadow | No (scalar/warp) | Yes | Marginal (q4-specific, no tensor cores) |
| nanbeige-arch | KVarN native + pinned host KV + wider streams | Yes | Yes | Inherits KVarN + DMA gains |
| perf/nanbeige-throughput | KVarN native + KnormPress prefill compaction | Yes | Yes | Inherits KVarN + lower KV |
| v0.4.0/v0.4.3 | KVarN native | Yes | Yes | same fork feature |
| v0.4.2-fv5 | KVarN native + FV5 ternary weights | Yes | Yes | weight-side only |
| ternary-bonsai-q2_0 | KVarN native REMOVED | n/a | No | Not helpful |
| mindcontrol-reasoning-budget | reasoning budget | n/a | n/a | Not attention/KV |

**Key insight:** the fork's headline long-context mechanism is **KVarN native tensor-core attention consuming compressed 2–8-bit records directly** (no F16 materialization, minimal KV/token → high context on 8 GB). It is ALREADY in base. My benchmarks built with `GGML_CUDA_KVARN=OFF`, which excluded exactly this feature — that is why no q4-path experiment could win. `kv-speed-work`'s packed-q4 kernel is the only genuinely-new attention kernel but is non-tensor-core and q4-specific (redundant vs KVarN). Next step (format-agnostic): enable `GGML_CUDA_KVARN=ON` and measure KVarN native attention on Nanbeige at long context.
- **=> Long-context attention is MEMORY-BANDWIDTH-BOUND**, reading the full dequantized-F16 KV each step. This is the opposite regime from 8k (18.8% DRAM, latency-bound).
- Reducing KV bytes read (native Q4 consumes ~1/4 the bytes of F16) directly relieves the 95% DRAM saturation → the evidence-backed 90k lever is a **native-Q4 multi-token attention** path.

## Web-sourced upstream facts (not inferred)

- llama.cpp **issue #24485**: quantized KV + `--flash-attn` **requires building with `-DGGML_CUDA_FA_ALL_QUANTS=ON`**; without it the build silently loses the native Q4 FA path and falls back to F16 or CPU attention. Authors measured ~25× prefill difference on an sm_120 Blackwell GPU (Gemma 4 12B QAT, head_size 512, SWA).
- llama.cpp **issue #24166** (PR #23907): CUDA `flash_attn_ext` reserves an **F16 K/V dequant scratch sized by the whole allocated KV cache** (`ggml_nelements(K/V)`); with quantized KV and a large `-c` this grows the per-op flash-attn dst allocation to GB scale, which can OOM a small GPU.
- hammer.ai blog: the Q4 FA dispatch chain is `ggml_cuda_flash_attn_ext_vec -> flash_attn_ext_vec -> get_dequantize_V -> <type>`; the type is chosen from compiled template pairs.

## Implications / open questions

- At short/8k context, attention is latency/register-bound (18.8% DRAM) and the Q4-native lever gives no gain.
- At long context (measured 40k), attention becomes **memory-bandwidth-bound (95.6% DRAM)** reading F16-dequantized KV. **Native-Q4 multi-token attention is the evidence-backed 90k lever** (cuts KV bytes ~4×). This direction was previously blocked in the fork (MMA K/V is half2-specific); the fix must make the dominant multi-token attention consume Q4 directly, not just the single-token decompile vec path that ALL_QUANTS already added.
- `dequantize_block_q4_0/q4_1` grow linearly with context (~4.2 s at 40k) — removing them via Q4-native attention both cuts the dequant pass and reduces DRAM bytes.
- Host serialization (6.6 s sync) is a llama-bench measurement artifact (it syncs per token to time), not necessarily the server's behavior.
