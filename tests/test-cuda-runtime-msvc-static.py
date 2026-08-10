from pathlib import Path
import sys


def main() -> int:
    source = (Path(sys.argv[1]) / "ggml/src/ggml-cuda/CMakeLists.txt").read_text()
    guard = "if (MSVC AND (GGML_BACKEND_DL OR BUILD_SHARED_LIBS) AND NOT GGML_STATIC)"
    assert guard in source
    assert 'if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.2")' in source
    assert "set_property(TARGET ggml-cuda PROPERTY CUDA_RUNTIME_LIBRARY Hybrid)" in source
    assert "set_property(TARGET ggml-cuda PROPERTY CUDA_RUNTIME_LIBRARY Shared)" in source
    assert "target_link_options(ggml-cuda PRIVATE /NODEFAULTLIB:LIBCMT)" in source
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
