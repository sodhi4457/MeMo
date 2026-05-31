"""Memory Model loader: base + (optional) QLoRA, plus a CPU-only path.

We never call flash-attn directly — many Windows/Colab setups don't have
it. If you want it, install separately and pass attn_implementation.
"""

from __future__ import annotations

from typing import Literal


def _pick_device_map(prefer: str | None = None) -> str | dict | None:
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "auto"
    return None  # CPU


def load_memory_model(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    use_qlora: bool = False,
    attn_implementation: str | None = None,
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16",
    device_map: str | dict | None = None,
):
    """Load base Memory Model and tokenizer.

    - use_qlora=True : 4-bit NF4 quant via bitsandbytes (needs CUDA + the [qlora] extra).
    - use_qlora=False: load in `dtype` (bf16 by default).
    - attn_implementation: pass "flash_attention_2" only if flash-attn is installed.
    - device_map: defaults to "auto" on CUDA, CPU placement otherwise.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]
    resolved_device_map = device_map if device_map is not None else _pick_device_map()

    if use_qlora:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "QLoRA requires `transformers` + `bitsandbytes`. "
                "Install with: uv sync --extra train --extra qlora"
            ) from exc
        from peft import prepare_model_for_kbit_training

        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA needs a CUDA GPU. Disable --qlora for CPU runs.")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=resolved_device_map or "auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        kwargs = {"torch_dtype": torch_dtype}
        if resolved_device_map is not None:
            kwargs["device_map"] = resolved_device_map
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    return model, tokenizer


def wrap_with_lora(
    model,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
):
    """Attach LoRA adapters to the standard Qwen-style projection layers."""
    from peft import LoraConfig, TaskType, get_peft_model

    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
