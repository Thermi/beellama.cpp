from pathlib import Path
import sys


def main() -> int:
    source = (Path(sys.argv[1]) / "ggml/src/ggml-cuda/CMakeLists.txt").read_text()
    guard = "if (MSVC AND (GGML_BACKEND_DL OR BUILD_SHARED_LIBS) AND NOT GGML_STATIC)"
    body = source.split(guard, 1)[1].split("endif()", 1)[0]
    assert guard in source
    assert "set_property(TARGET ggml-cuda PROPERTY CUDA_RUNTIME_LIBRARY Hybrid)" in body
    assert "target_link_options(ggml-cuda PRIVATE /NODEFAULTLIB:LIBCMT)" in body
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

