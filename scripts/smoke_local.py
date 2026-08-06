"""Local hardware smoke test: Qwen/Qwen3-0.6B bf16, exactly 32 greedy tokens.

Exit codes:
    0 - success (32 tokens generated on a CUDA GPU)
    1 - runtime fault (wrong token count, model failure)
    2 - no CUDA GPU visible (clean one-line message, no traceback)

See docs/infra.md section 7.
"""

from __future__ import annotations

import sys

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "Speculative decoding works by"
NEW_TOKENS = 32


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print(
            "smoke_local: CUDA GPU unavailable - no NVIDIA GPU is visible to "
            f"torch {torch.__version__} (expected a CUDA-capable device such "
            "as the local RTX 4060); refusing to run on CPU.",
            flush=True,
        )
        return 2

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=NEW_TOKENS,
            do_sample=False,
        )
    torch.cuda.synchronize()

    generated = output_ids[0, input_ids.shape[1] :]
    token_count = int(generated.shape[0])
    text = tokenizer.decode(generated, skip_special_tokens=True)
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"model            : {MODEL_ID} (bf16)")
    print(f"device           : {torch.cuda.get_device_name(0)}")
    print(f"token_count      : {token_count}")
    print(f"decoded_text     : {text}")
    print(f"peak_vram_gb     : {peak_vram_gb:.3f}")

    if token_count != NEW_TOKENS:
        print(
            f"smoke_local: expected exactly {NEW_TOKENS} new tokens, "
            f"got {token_count} (early EOS?)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # boundary: report, never dump a raw traceback path
        print(f"smoke_local: runtime fault - {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
