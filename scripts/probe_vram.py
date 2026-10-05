"""One-off VRAM feasibility probe for the target model.

Not part of the pipeline. It answers the question the stage scripts assume the
answer to: does a 4-bit Qwen2.5-7B actually load, generate and expose layer-20
activations inside 6.44 GB, and at what throughput? The numbers it prints set
the batch sizes and the achievable scale.

    python scripts/probe_vram.py [path/to/checkpoint]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from nlaast.models.loading import cuda_memory, nf4_config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", nargs="?",
                   help="local checkpoint directory; default is the cached "
                        "Qwen2.5-7B-Instruct snapshot")
    src = p.parse_args().checkpoint
    if src is None:
        from huggingface_hub import snapshot_download

        src = snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_files_only=True)
    print("source:", src)

    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(
        src,
        quantization_config=nf4_config(),
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()
    print(f"loaded in {time.monotonic() - t0:.0f}s | {cuda_memory()}")
    print("layers:", model.config.num_hidden_layers, "d_model:", model.config.hidden_size)

    msgs = [
        {"role": "system", "content": "You are a careful mathematical reasoner. "
         "Think step by step, then give the final answer on its own last line "
         "in the form 'The answer is X.'"},
        {"role": "user", "content": "Janet's ducks lay 16 eggs per day. She eats "
         "three for breakfast and bakes muffins with four. She sells the rest at "
         "$2 per egg. How much does she make every day?"},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    for bs in (1, 2, 4):
        try:
            enc = tok([prompt] * bs, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=192, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            dt = time.monotonic() - t0
            new = (out.shape[1] - enc["input_ids"].shape[1]) * bs
            print(f"batch={bs}: {new} new tokens in {dt:.1f}s = {new/dt:.1f} tok/s "
                  f"| {cuda_memory()}")
            if bs == 1:
                print("--- sample output ---")
                print(tok.decode(out[0, enc['input_ids'].shape[1]:],
                                 skip_special_tokens=True)[:700])
                print("---------------------")
        except torch.cuda.OutOfMemoryError:
            print(f"batch={bs}: OOM")
            torch.cuda.empty_cache()
            break

    # Activation hook at layer 20.
    store = {}

    def hook(_m, _i, o):
        store["h"] = (o[0] if isinstance(o, tuple) else o).detach().float().cpu()

    h = model.model.layers[20].register_forward_hook(hook)
    enc = tok([prompt], return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        model(**enc, use_cache=False)
    h.remove()
    a = store["h"]
    print(f"layer-20 capture: shape={tuple(a.shape)} dtype={a.dtype} "
          f"last-token L2={a[0, -1].norm():.2f} (AV injects at L2=150)")
    print(f"peak VRAM: {cuda_memory()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
