# Nanbeige Nsight Analysis

## Capture

- Tool available: Nsight Compute 2026.2.0.0
- Nsight Systems: not installed or not on `PATH`
- Workload: one serial `llama-bench` process, 8000 prompt tokens and 4 generated tokens
- Cache: `q4_0/q4_1`
- Kernel: `flash_attn_ext_f16<128, 128, 8, 8, 0, 0, 1, 1>`
- Launch: 128 threads, grid size 60, 30 SMs

## Evidence

The valid baseline Nsight Compute report is `ncu-nanbeige-fattn.csv`.

- DRAM throughput: 18.79% of peak
- Memory throughput: 34.58% of peak
- Compute throughput: 35.84% of peak
- L2 hit rate: 77.32%
- Registers per thread: 255
- Local-memory spilling requests: 2768 bytes
- Eligible warp cycles: 17.79%
- No-eligible warp cycles: 82.21%
- L1TEX scoreboard stall: 37.42% of average warp issue interval
- Average active warps per scheduler: 1.95
- Average eligible warps per scheduler: 0.20

## Classification

The low DRAM percentage is not evidence that the kernel is bandwidth-saturated. The kernel is primarily latency/register-pressure limited: it has maximal register allocation, local-memory spills, very low eligible-warp availability, and dominant L1TEX scoreboard stalls. The result supports investigating a different attention implementation or a carefully measured tile/register redesign, not simply adding more memory transfers or assuming PCIe is the bottleneck.

Nsight Systems could not be used because `nsys` is unavailable on the host. Therefore PCIe overlap, CUDA API launch gaps, and stream-level transfer attribution remain unmeasured.

## Rejected Candidate

Changing `Q_in_reg` for the wrong `ncols` specialization had no generated-code effect and was discarded. Changing the actual `ncols=64` Ampere specialization also passed focused tests but reduced the valid 8k mixed prompt/decode result from approximately 1705 tok/s to 1622 tok/s, so it was reverted. No attention behavior change is retained.
