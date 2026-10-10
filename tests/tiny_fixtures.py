"""Tiny stand-ins for the three 7B checkpoints, and a synthetic run directory.

The four stages that need the released autoencoder - ``nla``,
``faithfulness``, ``causal`` and everything that reads their output - cost
26 GB of download and several GB of VRAM to exercise against the real
checkpoints. That is the right thing to do once, and the wrong thing to do on
every change: a shape error, a missing key or a wrong slicing convention in a
stage script would otherwise only surface after an hour of model loading.

So this module builds the *interfaces* at 1/56th the width:

* a 2-layer ``Qwen2ForCausalLM`` at ``d_model=64`` standing in for the target
  model, the activation verbaliser and the activation reconstructor;
* the real Qwen tokeniser, because the injection convention is a property of
  the tokeniser and faking it would make the fixture prove nothing;
* an ``nla_meta.yaml`` per checkpoint whose injection token id and neighbours
  are **read off the live tokeniser**, so the sidecar is self-consistent by
  construction rather than by a copied constant;
* a ``value_head.safetensors`` for the reconstructor, which upstream's
  ``NLACritic`` refuses to run without.

What the fixture proves: every stage script runs end to end, the
``inputs_embeds`` injection path produces text, the reconstruction metric is
computed in the released convention, and the artefacts each stage writes are
the ones the next stage reads. What it cannot prove: anything scientific. The
weights are random, so every number it produces is noise.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: Width of the fixture models. Small enough to be instant, large enough that a
#: broadcasting bug cannot pass unnoticed.
D_MODEL = 64
N_LAYERS = 2

#: The AR template is verbatim from the released
#: ``kitft/nla-qwen2.5-7b-L20-ar`` sidecar. The AV template is the
#: ``<concept>{injection_char}</concept>`` core of the released AV prompt,
#: which wraps the same marker in a 125-token researcher-persona preamble; the
#: marker and its two neighbour tokens are what the injection convention
#: depends on and they are identical either way. The injection ids are not
#: copied at all - they are resolved from the live tokeniser below, so the
#: fixture sidecar is self-consistent by construction.
AV_TEMPLATE = "Here is the vector:\n\n<concept>{injection_char}</concept>\n"
AR_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
INJECTION_CHAR = "㈎"  # U+320E, the marker the released checkpoints use


def tokenizer_source() -> Path | None:
    """A directory holding a real Qwen2.5 tokeniser, or ``None``.

    Prefers the NF4 target checkpoint this project builds, then the HuggingFace
    cache. Returning ``None`` makes the end-to-end test skip rather than fail:
    the tokeniser is 11 MB of someone else's bytes and cannot be synthesised.
    """
    from nlaast import paths

    candidates = [paths.quantised_models() / "Qwen__Qwen2.5-7B-Instruct"]
    hub = paths.model_cache() / "hub"
    if hub.is_dir():
        for repo in sorted(hub.glob("models--Qwen--Qwen2.5-7B*")):
            candidates.extend(sorted(repo.glob("snapshots/*")))
    for c in candidates:
        if (c / "tokenizer.json").exists() and (c / "tokenizer_config.json").exists():
            return c
    return None


def resolve_injection_ids(tokenizer) -> dict[str, int]:
    """Find the injection site in the chat-templated AV prompt.

    This is ``nla.meta.verify_tokenizer`` run backwards: instead of checking
    pinned ids against the tokeniser, it reads the ids the tokeniser actually
    produces and writes *those* into the fixture sidecar. The fixture then
    satisfies the same consistency check the real checkpoint does, without
    hardcoding anything that could drift.
    """
    from nlaast.nla.meta import chat_template_ids

    content = AV_TEMPLATE.format(injection_char=INJECTION_CHAR)
    ids = chat_template_ids(tokenizer, content)
    marker = tokenizer.encode(INJECTION_CHAR, add_special_tokens=False)
    if len(marker) != 1:
        raise RuntimeError(
            f"the injection character tokenises to {len(marker)} tokens; the "
            f"whole injection convention assumes exactly one"
        )
    inj = marker[0]
    positions = [i for i, t in enumerate(ids) if t == inj]
    if len(positions) != 1:
        raise RuntimeError(f"expected one injection site, found {len(positions)}")
    p = positions[0]
    return {
        "injection_token_id": int(inj),
        "injection_left_neighbor_id": int(ids[p - 1]),
        "injection_right_neighbor_id": int(ids[p + 1]),
    }


def _tiny_model(vocab_size: int, n_layers: int = N_LAYERS):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=D_MODEL,
        intermediate_size=128,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(cfg).eval()


def _sidecar(role: str, ids: dict[str, int], *, injection_scale, mse_scale) -> dict:
    return {
        "kind": "nla_model",
        "schema_version": 2,
        "role": role,
        "d_model": D_MODEL,
        "extraction": {"injection_scale": injection_scale, "mse_scale": mse_scale},
        "tokens": {"injection_char": INJECTION_CHAR, "critic_suffix_ids": None, **ids},
        "prompt_templates": {"av": AV_TEMPLATE, "ar": AR_TEMPLATE},
        "extraction_layer_index": N_LAYERS - 1,
        "note": "FIXTURE - random weights, no scientific content.",
    }


def build_checkpoints(quant_dir: Path, tok_src: Path) -> dict[str, Path]:
    """Write fixture target / AV / AR checkpoints under ``quant_dir``.

    The directory names match ``nlaast.models.loading._quant_dir``, so
    ``ensure_nf4`` finds them and never reaches the network.
    """
    import yaml
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(str(tok_src))
    ids = resolve_injection_ids(tokenizer)
    vocab = max(len(tokenizer), ids["injection_token_id"] + 1)
    mse_scale = float(np.sqrt(D_MODEL))

    out: dict[str, Path] = {}
    specs = [
        ("target", "nlaast-fixture__target", None, N_LAYERS),
        ("av", "nlaast-fixture__av", _sidecar("av", ids, injection_scale=150.0,
                                              mse_scale=mse_scale), N_LAYERS),
        ("ar", "nlaast-fixture__ar", _sidecar("ar", ids, injection_scale=None,
                                              mse_scale=mse_scale), N_LAYERS),
    ]
    for name, dirname, sidecar, layers in specs:
        d = quant_dir / dirname
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "config.json").exists():
            model = _tiny_model(vocab, layers)
            model.save_pretrained(d, safe_serialization=True)
            tokenizer.save_pretrained(d)
            # ``save_pretrained`` may not emit the jinja file on every
            # transformers version; the AV cannot build its prompt without it.
            src_tpl = tok_src / "chat_template.jinja"
            if src_tpl.exists() and not (d / "chat_template.jinja").exists():
                shutil.copy2(src_tpl, d / "chat_template.jinja")
        if sidecar is not None:
            (d / "nla_meta.yaml").write_text(
                yaml.safe_dump(sidecar, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        if name == "ar" and not (d / "value_head.safetensors").exists():
            torch.manual_seed(1)
            save_file({"weight": torch.randn(D_MODEL, D_MODEL, dtype=torch.bfloat16)},
                      str(d / "value_head.safetensors"))
        out[name] = d
    return out


# --- synthetic corpus -------------------------------------------------------

#: Four traces hand-written so the AST detector has something to decide on:
#: two with a clean stable suffix, one that never stabilises, one truncated.
#: ``stable_from`` is the first chunk index whose forced answer and
#: continuations all agree with the final answer.
_TRACES = [
    {"pid": "fixture-0001", "dataset": "gsm8k", "split": "train", "level": None,
     "gold": "18", "answer": "18", "stable_from": 2, "truncated": False,
     "sentences": [
         "Janet has 16 eggs each day.",
         "She eats 3 and bakes with 4, leaving 9 to sell.",
         "Nine eggs at 2 dollars each is 18 dollars.",
         "Let me double check that arithmetic.",
         "Yes, the total is correct.",
         "The answer is 18.",
     ]},
    {"pid": "fixture-0002", "dataset": "gsm8k", "split": "eval", "level": None,
     "gold": "42", "answer": "42", "stable_from": 3, "truncated": False,
     "sentences": [
         "There are 7 boxes in the first shipment.",
         "Each box holds 6 items.",
         "So the shipment holds 42 items in total.",
         "Verifying: 7 times 6 is 42.",
         "That matches the earlier count.",
         "So the answer is 42.",
     ]},
    {"pid": "fixture-0003", "dataset": "math", "split": "eval", "level": "Level 3",
     "gold": "7", "answer": "5", "stable_from": None, "truncated": False,
     "sentences": [
         "Start by expanding the expression.",
         "That gives a quadratic in x.",
         "Solving it suggests the value 5.",
         "On reflection the discriminant was wrong.",
         "The answer is 5.",
     ]},
    {"pid": "fixture-0004", "dataset": "math", "split": "train", "level": "Level 2",
     "gold": "9", "answer": "9", "stable_from": 1, "truncated": True,
     "sentences": [
         "The sum of the first three terms is 9.",
         "Each later term repeats the pattern.",
         "So the answer is 9.",
         "Continuing the check for the next term",
     ]},
    # The next two exist so the causal stage has a fittable direction. It needs
    # at least five tail and five pre-stabilisation vectors on the *train*
    # split, which the four traces above cannot supply; without them the
    # ablate_dir / random_dir / matched_position arms never run and the
    # projection path would go untested.
    {"pid": "fixture-0005", "dataset": "gsm8k", "split": "train", "level": None,
     "gold": "24", "answer": "24", "stable_from": 5, "truncated": False,
     "sentences": [
         "The recipe needs 3 cups per batch.",
         "There are 8 batches to make today.",
         "Multiplying gives the total cups needed.",
         "Three times eight is twenty four.",
         "So twenty four cups are needed.",
         "Checking the multiplication once more.",
         "Three times eight is indeed twenty four.",
         "The answer is 24.",
     ]},
    {"pid": "fixture-0006", "dataset": "math", "split": "eval", "level": "Level 4",
     "gold": "11", "answer": "11", "stable_from": 4, "truncated": False,
     "sentences": [
         "Let the unknown be denoted by n.",
         "The constraint fixes n above ten.",
         "Testing eleven satisfies the constraint.",
         "So n equals eleven.",
         "Re-reading the constraint confirms it.",
         "Nothing larger would satisfy it.",
         "The answer is 11.",
     ]},
    # F6 (probe leakage) and F7 (layer sweep) both need at least thirty
    # probe examples in two classes; these two carry the corpus past that so
    # the leakage control is actually exercised rather than skipped.
    {"pid": "fixture-0007", "dataset": "gsm8k", "split": "train", "level": None,
     "gold": "15", "answer": "15", "stable_from": 2, "truncated": False,
     "sentences": [
         "Each shelf holds 5 books.",
         "So the answer is 15 for three shelves.",
         "Three shelves give fifteen books.",
         "The answer is 15.",
         "Confirming: five times three is fifteen.",
         "The answer is 15.",
         "No books are left over.",
         "The answer is 15.",
     ]},
    {"pid": "fixture-0008", "dataset": "math", "split": "eval", "level": "Level 1",
     "gold": "6", "answer": "4", "stable_from": 2, "truncated": False,
     "sentences": [
         "The expression simplifies in two steps.",
         "So the answer is 4 after simplification.",
         "The answer is 4.",
         "Double checking the second step.",
         "The answer is 4.",
         "That step was algebraically valid.",
         "The answer is 4.",
     ]},
]


def _build_trace_row(spec: dict, chunk_cfg, k_continuations: int) -> dict:
    """One ``traces.jsonl`` row, in exactly the schema ``generate_traces`` writes.

    Token offsets are faked at one token per four characters. Nothing in the
    fixture reads a real activation at a real position, so the only requirement
    is internal consistency: ``token_end`` must be monotone and inside the
    trace's token count.
    """
    from nlaast.trace.chunking import chunk_trace

    text = " ".join(spec["sentences"])
    n_tokens = max(1, len(text) // 4)
    offsets = [(i * 4, min(len(text), (i + 1) * 4)) for i in range(n_tokens)]
    chunks = chunk_trace(text, chunk_cfg, token_offsets=offsets)

    final = spec["answer"]
    boundaries = []
    for c in chunks:
        stable = spec["stable_from"] is not None and c.index >= spec["stable_from"]
        answers = [final if stable else None] * k_continuations
        boundaries.append({
            "index": c.index,
            "parsed_answer": final if stable else None,
            "forced_answer": final if stable else "0",
            "forced_text": f"\n\nTherefore, the answer is {final if stable else 0}.",
            "forced_matches_final": bool(stable),
            "continuation_answers": answers,
            "continuation_tails": [f"the answer is {a}." for a in answers],
            "continuations_matching": k_continuations if stable else 0,
            "n_continuations": k_continuations,
            "prefix_tokens": c.prefix_tokens,
            "evaluated": True,
        })

    return {
        "id": spec["pid"],
        "problem_id": spec["pid"],
        "dataset": spec["dataset"],
        "split": spec["split"],
        "level": spec["level"],
        "question": f"Fixture question for {spec['pid']}?",
        "gold": spec["gold"],
        "trace": text,
        "token_ids": list(range(100, 100 + n_tokens)),
        "offsets": [list(o) for o in offsets],
        "n_tokens": n_tokens,
        "prompt_tokens": 20,
        "truncated": spec["truncated"],
        "final_answer": final,
        "final_correct": final == spec["gold"],
        "n_chunks": len(chunks),
        "chunks": [c.to_dict() for c in chunks],
        "boundaries": boundaries,
        "boundary_scan_capped": False,
        "max_boundaries_evaluated": 12,
        "entropy_samples": [{"answer": final, "n_tokens": 10},
                            {"answer": final, "n_tokens": 11},
                            {"answer": "0", "n_tokens": 9}],
        "ok": True,
    }


def build_run(cfg) -> dict[str, int]:
    """Populate ``runs/<cfg.run_id>`` with synthetic ``data``/``traces``/``acts``.

    Activations are random, but *structured*: tail rows get a constant offset
    added so the causal stage's difference-in-means direction is non-degenerate
    and the probe has something to find. That makes the fixture exercise the
    code paths a flat random block would skip, and it is why none of its
    numbers mean anything.
    """
    from nlaast.activations.store import ActivationStore
    from nlaast.logging_utils import write_json, write_jsonl

    run = cfg.dir
    (run / "data").mkdir(parents=True, exist_ok=True)

    # Declare the run synthetic, in a file anything reading `runs/` can see.
    # This test writes its run *last*, so a "newest run wins" rule picks it:
    # the root-level supervised analyses once fitted on six rows of this
    # noise and wrote the result to outputs/. Comparing settings is not enough
    # on its own, because the reference run is what the comparison is made
    # against. See FIXTURE_MARKER in common_data.py.
    (run / "FIXTURE").write_text(
        "Synthetic run written by tests/test_end_to_end_tiny.py.\n\n"
        "The traces, activations and verbalisations here come from randomly\n"
        "initialised 64-dimensional models. Every number in this directory is\n"
        "noise. Nothing that pools corpus data may read it.\n",
        encoding="utf-8",
    )

    problems = [{
        "id": s["pid"], "problem_id": s["pid"], "dataset": s["dataset"],
        "split": s["split"], "level": s["level"], "gold": s["gold"],
        "question": f"Fixture question for {s['pid']}?",
    } for s in _TRACES]
    write_jsonl(run / "data" / "problems.jsonl", problems)
    write_json(run / "data" / "validation.json",
               {"n": len(problems), "ok": True, "note": "FIXTURE"})

    rows = [_build_trace_row(s, cfg.chunk, cfg.ast.k_continuations) for s in _TRACES]
    (run / "traces").mkdir(parents=True, exist_ok=True)
    write_jsonl(run / "traces" / "traces.jsonl", rows)

    store = ActivationStore(run / "acts")
    rng = np.random.default_rng(0)
    offset = rng.standard_normal(D_MODEL).astype(np.float32)
    for spec, row in zip(_TRACES, rows):
        positions = [int(c["token_end"]) for c in row["chunks"]]
        arr = rng.standard_normal((len(positions), D_MODEL)).astype(np.float32)
        if spec["stable_from"] is not None:
            arr[spec["stable_from"]:] += 3.0 * offset
        for layer in sorted({cfg.target.layer, *cfg.robustness.layer_sweep}):
            if store.has(row["problem_id"], layer):
                continue
            store.put(row["problem_id"], layer, arr, positions,
                      kinds=["chunk_boundary"] * len(positions),
                      extra={"chunk_indices": [int(c["index"]) for c in row["chunks"]],
                             "prompt_tokens": 20,
                             "trace_tokens": row["n_tokens"],
                             "resolved_positions": [20 + p for p in positions],
                             "alignment_exact": True,
                             "retokenisation_drift": 0,
                             "clamped_positions": 0,
                             "layer_convention": "FIXTURE"})
    write_json(run / "acts" / "summary.json",
               {**store.summary(), "n_failed": 0, "failures": [],
                "layers": sorted({cfg.target.layer, *cfg.robustness.layer_sweep}),
                "note": "FIXTURE"})
    return {"n_problems": len(problems), "n_blocks": len(store)}


#: Stand-in verbalisations. Shaped like the real AV's output - two or three
#: short declarative snippets - so ``split_claims`` has something to split and
#: the paraphrase null has more than one claim to pool over.
_FIXTURE_EXPLANATIONS = [
    "The state encodes a numeric answer that has already been committed to. "
    "It also carries a verification intent. The surrounding text restates the result.",
    "This activation tracks arithmetic that is complete. "
    "A confirmation step follows it. No new quantity is introduced here.",
]


def substitute_explanations(cfg) -> int:
    """Replace the fixture AV's random text with well-formed explanations.

    The integrity gate in ``nla/verbalizer.py`` requires an English
    ``<explanation>`` block, and a randomly initialised model will never emit
    one - correctly, that is the off-distribution signature the gate exists to
    catch. So the ``nla`` stage's own output is tested as generated, and then
    this rewrites the explanation fields in place so the `faithfulness` stage
    receives input of the shape the real verbaliser produces.

    Only the parsed fields change. Each sample keeps the raw text the fixture
    model actually generated, under ``raw``, and gains
    ``fixture_substituted: true`` so no reader can mistake this for a real
    verbalisation. ``reconstructions.jsonl`` is left alone and therefore
    describes the pre-substitution text; the fixture tests wiring, not
    agreement between the two files.
    """
    from nlaast.logging_utils import read_jsonl, write_jsonl

    path = cfg.dir / "nla" / "verbalisations.jsonl"
    rows = read_jsonl(path)
    # Keep what the fixture model really produced, so a test can still assert
    # on the generated output rather than on the substitution.
    shutil.copy2(path, path.with_name("verbalisations.asgenerated.jsonl"))
    for i, row in enumerate(rows):
        for group in ("samples", "noise_samples"):
            for j, s in enumerate(row.get(group, [])):
                text = _FIXTURE_EXPLANATIONS[(i + j) % len(_FIXTURE_EXPLANATIONS)]
                if group == "noise_samples":
                    # The noise arm has to overlap the real arm on some claims
                    # and not others, or the verbaliser-only control would be
                    # trivially satisfied or trivially violated.
                    text = ("The state encodes a numeric answer that has already "
                            "been committed to. Unrelated prior content appears here.")
                s["explanation"] = text
                s["text"] = text
                s["ok"] = True
                s["well_formed"] = True
                s["ascii_fraction"] = 1.0
                s["cjk_chars"] = 0
                s["fixture_substituted"] = True
        row["n_ok"] = len(row.get("samples", []))
    write_jsonl(path, rows)
    return len(rows)


def write_manifest_stub(cfg) -> None:
    """Mark ``env``/``data``/``models``/``traces``/``acts`` complete.

    The orchestrator and the stage scripts read the manifest to decide what to
    skip; without this the fixture run would try to download a dataset and a
    model for stages whose output has just been written by hand.
    """
    from nlaast import provenance

    manifest = provenance.Manifest.open(cfg.dir, cfg.to_dict(), cfg.hash())
    for stage, metrics in (
        ("env", {"note": "FIXTURE"}),
        ("data", {"n": len(_TRACES)}),
        ("models", {"prepared": ["fixture"], "checks_ok": True}),
        ("traces", {"n_traces": len(_TRACES), "n_ok": len(_TRACES)}),
        ("acts", {"n_blocks": len(_TRACES)}),
    ):
        manifest.start_stage(stage)
        manifest.finish_stage(stage, status="complete", metrics=metrics)
    (cfg.dir / "models").mkdir(parents=True, exist_ok=True)
    (cfg.dir / "models" / "resolved.json").write_text(
        json.dumps({"fixture": True}, indent=2), encoding="utf-8")
