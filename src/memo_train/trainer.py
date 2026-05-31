"""Step B: supervised fine-tuning of the Memory Model M_phi."""

from __future__ import annotations

from pathlib import Path

from .data import (
    format_qa_as_chat,
    load_qa_dataset,
    tokenize_with_answer_masking,
)
from .model import load_memory_model, wrap_with_lora


def train_memory_model(
    dataset_path: str | Path,
    output_dir: str | Path = "./memory_model_checkpoint",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    use_qlora: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    max_length: int = 2048,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 1,
    grad_accumulation: int = 16,
    warmup_ratio: float = 0.05,
    gradient_checkpointing: bool = True,
    bf16: bool = True,
    optim: str = "adamw_torch_fused",
    max_steps: int = -1,
    logging_steps: int = 20,
    save_strategy: str = "epoch",
    seed: int = 42,
):
    """End-to-end SFT pipeline matching Step B of the breakdown.

    Returns:
        (model, tokenizer, train_output): trained PEFT model, its tokenizer,
        and the HuggingFace Trainer output object.
    """
    import torch
    from datasets import Dataset
    from transformers import (
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    qa_pairs = load_qa_dataset(dataset_path)
    print(f"[data] {len(qa_pairs)} Q&A pairs loaded from {dataset_path}")

    print(f"[model] loading base: {model_name}  (qlora={use_qlora})")
    model, tokenizer = load_memory_model(model_name=model_name, use_qlora=use_qlora)
    model = wrap_with_lora(model, lora_rank=lora_rank, lora_alpha=lora_alpha)

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    print("[data] formatting + tokenizing...")
    raw = Dataset.from_list([
        {
            "text": format_qa_as_chat(qa, tokenizer),
            "question": qa["question"],
            "answer": qa["answer"],
        }
        for qa in qa_pairs
    ])
    tokenized = raw.map(
        lambda x: tokenize_with_answer_masking(x, tokenizer, max_length),
        batched=False,
        remove_columns=raw.column_names,
        desc="tokenize",
    )

    cuda = torch.cuda.is_available()
    bf16_ok = bf16 and cuda and torch.cuda.is_bf16_supported()
    fp16_ok = (not bf16_ok) and cuda  # fp16 fallback on GPUs without bf16

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        gradient_checkpointing=gradient_checkpointing,
        bf16=bf16_ok,
        fp16=fp16_ok,
        optim=optim if cuda else "adamw_torch",
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to="none",
        seed=seed,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print("[train] starting...")
    out = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train] done. adapter saved to {output_dir}")
    return model, tokenizer, out


def quick_inference(
    adapter_path: str | Path,
    question: str,
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 256,
) -> str:
    """Load the trained adapter and answer ONE question.

    Used by `memo-step-b infer` to sanity-check that fine-tuning actually
    moved the model toward the corpus knowledge.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model = model.merge_and_unload()
    model.eval()

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
