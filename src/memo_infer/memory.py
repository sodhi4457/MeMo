"""Load the trained Memory Model M_phi and query it at inference time."""

from __future__ import annotations

from pathlib import Path


def load_memory_model(
    adapter_path: str | Path,
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
):
    """Load the LoRA adapter (saved by Step B) merged into the base model.

    The adapter directory must contain both the LoRA weights and the tokenizer
    (Step B's trainer.save_model + tokenizer.save_pretrained writes both there).

    Returns:
        (model, tokenizer) — ready to pass into query_memory().
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(adapter_path)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    # Merge LoRA weights into base → single model, zero overhead at query time
    model = model.merge_and_unload()
    model.eval()

    return model, tokenizer


def query_memory(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 256,
) -> str:
    """Ask the Memory Model one question; return the answer string.

    The model answers purely from its parametric knowledge — no source
    documents are passed.  Greedy decoding (do_sample=False) is used for
    deterministic, reproducible answers.
    """
    import torch

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()
