from pathlib import Path
import re
import sys


def main() -> int:
    root = Path(sys.argv[1])
    hybrid = (root / "src/llama-memory-hybrid-iswa.cpp").read_text()
    saver = (root / "src/llama-model-saver.cpp").read_text()
    kvarn = (root / "src/llama-kv-cache-kvarn.cpp").read_text()
    hparams = (root / "src/llama-hparams.h").read_text()
    hparams_cpp = (root / "src/llama-hparams.cpp").read_text()

    assert re.search(
        r"tail_rollback_tokens,\s*hparams\.n_swa_sink_tokens,\s*tail_native_exact_swa",
        hybrid,
    ), "hybrid iSWA must forward sink_tokens before tail_native_exact_swa"
    assert re.search(
        r"add_kv\(LLM_KV_ATTENTION_SINK_TOKENS,\s*hparams\.n_swa_sink_tokens\)", saver,
    ), (
        "model saver must preserve sink-token metadata"
    )
    assert "n_swa_pattern" in hparams, "SWA pattern period must be retained in hparams"
    assert "n_swa_pattern = n_pattern" in hparams_cpp, "SWA pattern setter must retain its period"
    assert "LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN" in saver, (
        "model saver must preserve the SWA pattern metadata"
    )
    assert re.search(
        r"kvarn_swa_visible_groups\([^)]*sink_tokens",
        kvarn,
    ), "KVarN SWA ring sizing must include sink_tokens"
    assert "n_swa + effective_sink" in kvarn, (
        "KVarN SWA visible span must include the effective sink span"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
