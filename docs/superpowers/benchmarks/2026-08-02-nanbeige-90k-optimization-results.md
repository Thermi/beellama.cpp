# Nanbeige Optimization Results

## Retained Change

`ggml/src/ggml-cuda/fattn-kvarn-vec.cuh` now uses CUDA PTX syntax compatible with MSVC-hosted NVCC and CUDA 13.3:

```cpp
asm volatile("prefetch.global.L2 [%0];" : : "l"(record));
```

The previous GNU spelling `__asm__ __volatile__` failed under MSVC. The old lowercase `prefetch.global.l2` spelling reached PTXAS but was rejected by CUDA 13.3. The uppercase `L2` form compiles successfully in the focused CUDA build.

This is a portability/build fix, not yet a measured throughput optimization.

## Tests

Passed in the focused build:

- `test-cuda-prefetch-msvc-static`
- `test-cuda-fattn-route-policy`
- `test-cuda-fattn-vec-policy`

The focused standard-Q4 build completed and produced runnable benchmark binaries.

## Performance Conclusion

No performance candidate is accepted yet. Nsight Compute identifies register pressure, local spills, and L1TEX latency as the dominant short/representative long-context attention symptoms, but the host lacks Nsight Systems and cannot complete a 90k prefill within 40 minutes under WDDM. The next performance iteration should implement or evaluate a separate attention route that reduces the active kernel's register footprint, with serial benchmark and Nsight captures only.
