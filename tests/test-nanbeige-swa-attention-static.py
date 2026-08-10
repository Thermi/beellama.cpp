from pathlib import Path
import sys


def main() -> int:
    root = Path(sys.argv[1])
    model = (root / "src/models/nanbeige.cpp").read_text()
    arch = (root / "src/llama-arch.h").read_text()
    arch_cpp = (root / "src/llama-arch.cpp").read_text()
    hparams = (root / "src/llama-hparams.h").read_text()
    graph = (root / "src/llama-graph.cpp").read_text()
    cache = (root / "src/llama-kv-cache-iswa.cpp").read_text()
    cache_core = (root / "src/llama-kv-cache.cpp").read_text()
    models_h = (root / "src/models/models.h").read_text()

    assert "LLM_KV_ATTENTION_SLIDING_WINDOW" in arch
    assert "ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW" in model
    assert "set_swa_pattern" in model
    assert "hparams.n_swa = swa_window" in model
    assert "hparams.n_swa = 0" in model
    assert "streamingllm" in model.lower()
    assert "sink_tokens" in model
    assert "LLM_KV_ATTENTION_SINK_TOKENS" in arch
    assert '"%s.attention.sink_tokens"' in arch_cpp
    assert "n_swa_sink_tokens" in model
    assert "n_swa_sink_tokens" in hparams
    assert "n_swa_sink_tokens" in cache_core
    assert "hparams.n_swa_sink_tokens" in cache_core
    assert "template <bool iswa>" in models_h
    assert "std::conditional_t<iswa" in model
    assert "build_attn_inp_kv_iswa" in graph
    assert "size_swa = GGML_PAD" in cache
    assert "is_masked_swa" in hparams
    assert "is_swa_impl" in hparams
    assert "LLAMA_SWA_TYPE_STANDARD" in hparams
    assert "GGML_PAD" in cache
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

