# Nanbeige4.2-3B Benchmark Results

## Hardware
- GPU: NVIDIA GeForce RTX 5060 (8151 MiB)
- PCIe: Gen4 x8

## Model
- Model: Nanbeige4.2-3B-heretic-Q4_K_M.gguf
- Size: 2.45 GB
- Architecture: Looped Transformer (22 physical layers × 2 loops = 44 logical layers)
- KV Cache: 44 layers × 8 KV heads × 128 dim

## Baseline (nanbeige42 image, commit f7ca480)
- Build: CUDA_ARCHITECTURES=120, GGML_CUDA_FA_ALL_QUANTS=ON
- KV Cache: q4_0/q4_1

### Short prompt (~9 tokens)
- Decode: 57.8 tok/s

### 1k context (~1000 tokens)
- Decode: 57.8 tok/s

### 10k context (~10,000 tokens)
- Decode: 33.7 tok/s

### 40k context (~40,000 tokens)
- Decode: 13.0 tok/s

### 80k context (~80,000 tokens)
- Decode: 4.6 tok/s

## Performance vs Theoretical Maximum

| Context | Actual | Theoretical | Efficiency |
|---------|--------|-------------|------------|
| 9 tokens | 57.8 tok/s | 172 tok/s | 34% |
| 1k | 57.8 tok/s | ~100 tok/s | 58% |
| 10k | 33.7 tok/s | ~80 tok/s | 42% |
| 40k | 13.0 tok/s | ~50 tok/s | 26% |
| 80k | 4.6 tok/s | ~30 tok/s | 15% |

## Root Cause Analysis

The performance degrades with context length due to:

1. **Memory bandwidth bottleneck**: Each token requires reading ~4.7 GB of KV cache from VRAM
2. **Compute overhead**: The KVarN materialize kernel adds significant computation beyond pure memory reads
3. **Kernel launch overhead**: Each layer requires multiple kernel launches
4. **Lack of overlap**: KVarN materialize and attention compute run on the same stream

## Optimization Opportunities

1. **Enable CUDA graph optimization** (GGML_CUDA_GRAPH_OPT=1) to reduce kernel launch overhead
2. **Overlap KVarN materialize with attention** using concurrent CUDA streams
3. **Reduce KV cache size** using q2_0 quantization (2.0 GB vs 4.7 GB)
4. **Optimize memory access patterns** in the KVarN decode kernel

## Build Configuration

```bash
# Docker build
docker build -t beellama-server -f .devops/cuda.Dockerfile \
  --build-arg CUDA_DOCKER_ARCH=120 \
  --build-arg GGML_CUDA_FA_ALL_QUANTS=ON \
  --target server .

# Run benchmark
docker run --gpus all -v /models:/models -p 8080:8080 beellama-server \
  llama-server -m /models/model.gguf \
  --port 8080 --host 0.0.0.0 --ctx-size 92160 -ngl 999 \
  --cache-type-k q4_0 --cache-type-v q4_1 \
  --batch-size 512 --ubatch-size 256 --threads 6 --parallel 1
```

## Notes

- The nanbeige42 image was built from commit f7ca480 (main branch)
- The current codebase (c702b9f88) shows regression due to code changes between commits
- Performance at 80k+ context is limited by memory bandwidth (448 GB/s)
- Theoretical maximum at 90k context is ~61 tok/s
