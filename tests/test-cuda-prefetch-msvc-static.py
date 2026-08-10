from pathlib import Path
import sys


def main() -> int:
    root = Path(sys.argv[1])
    source = (root / "ggml/src/ggml-cuda/fattn-kvarn-vec.cuh").read_text()

    assert source.count("__asm__ __volatile__") == 0, "KVarN CUDA prefetch must not use GNU __asm__ spelling"
    assert source.count('asm volatile("prefetch.global.L2') == 4, (
        "expected four CUDA-compatible L2 prefetch instructions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
