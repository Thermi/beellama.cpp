from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1])
    args = (root / "common/arg.cpp").read_text()
    common = (root / "common/common.cpp").read_text()

    assert '"-ctk", "--cache-type-k"' in args
    assert '"-ctv", "--cache-type-v"' in args
    assert 'params.no_kv_offload = !value;' in args
    assert "cparams.offload_kqv       = !params.no_kv_offload;" in common

    physical_layers = 22
    loops = 2
    context = 226_000
    kv_heads = 8
    head_dim = 128
    blocks = kv_heads * head_dim // 32
    q40_row = blocks * 18
    q41_row = blocks * 20
    q40_q41 = physical_layers * loops * context * (q40_row + q41_row)
    q40_q40 = physical_layers * loops * context * (q40_row * 2)

    assert q40_q41 == 12_091_904_000
    assert q40_q40 == 11_455_488_000
    assert q40_q40 < q40_q41
    print(json.dumps({"q4_0_q4_1_bytes": q40_q41, "q4_0_q4_0_bytes": q40_q40}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

