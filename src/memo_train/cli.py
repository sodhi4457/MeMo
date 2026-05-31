"""CLI for Step B: inspect dataset, train, run a quick inference probe.

Usage:
    uv run memo-step-b inspect --data data/reflection_qa.jsonl
    uv run memo-step-b train   --data data/reflection_qa.jsonl --output memory_model_ckpt
    uv run memo-step-b infer   --adapter memory_model_ckpt --question "Who was Marchetti?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Show dataset stats + a sample tokenization + masking fraction."""
    from .data import (
        format_qa_as_chat,
        load_qa_dataset,
        loss_token_fraction,
        tokenize_with_answer_masking,
    )

    qa = load_qa_dataset(args.data)
    print(f"[stats] pairs: {len(qa)}")
    q_lens = [len(p["question"]) for p in qa]
    a_lens = [len(p["answer"]) for p in qa]
    print(f"[stats] question chars  min/avg/max: {min(q_lens)} / {sum(q_lens)//len(qa)} / {max(q_lens)}")
    print(f"[stats] answer chars    min/avg/max: {min(a_lens)} / {sum(a_lens)//len(qa)} / {max(a_lens)}")

    if args.tokenizer_check:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        sample_n = min(args.sample, len(qa))
        fractions = []
        for pair in qa[:sample_n]:
            text = format_qa_as_chat(pair, tokenizer)
            tok = tokenize_with_answer_masking(
                {"text": text, "question": pair["question"], "answer": pair["answer"]},
                tokenizer,
                max_length=args.max_length,
            )
            fractions.append(loss_token_fraction(tok))

        avg = sum(fractions) / len(fractions)
        print(f"[mask]  sample={sample_n}  avg answer-token fraction: {avg:.2%}")
        print("[mask]  healthy range: ~20-40% for typical Q&A. If <5%, masking is broken.")

        print("\n[sample 0] formatted text:")
        print(format_qa_as_chat(qa[0], tokenizer))

    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .trainer import train_memory_model

    train_memory_model(
        dataset_path=args.data,
        output_dir=args.output,
        model_name=args.model,
        use_qlora=args.qlora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_length=args.max_length,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accum,
        max_steps=args.max_steps,
    )
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    from .trainer import quick_inference

    answer = quick_inference(
        adapter_path=args.adapter,
        question=args.question,
        base_model_name=args.model,
        max_new_tokens=args.max_new_tokens,
    )
    print("\nQ:", args.question)
    print("A:", answer)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="memo-step-b",
        description="MEMO Step B — Memory Model supervised fine-tuning.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # inspect
    p_ins = sub.add_parser("inspect", help="Dataset stats + masking sanity check")
    p_ins.add_argument("--data", type=Path, default=Path("data/reflection_qa.jsonl"))
    p_ins.add_argument(
        "--tokenizer-check",
        action="store_true",
        help="Load the tokenizer and verify answer-only masking on a few samples.",
    )
    p_ins.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p_ins.add_argument("--max-length", type=int, default=2048)
    p_ins.add_argument("--sample", type=int, default=10)
    p_ins.set_defaults(func=_cmd_inspect)

    # train
    p_tr = sub.add_parser("train", help="Run SFT")
    p_tr.add_argument("--data", type=Path, default=Path("data/reflection_qa.jsonl"))
    p_tr.add_argument("--output", type=Path, default=Path("memory_model_checkpoint"))
    p_tr.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p_tr.add_argument("--qlora", action="store_true", help="4-bit NF4 quantization (CUDA only)")
    p_tr.add_argument("--lora-rank", type=int, default=32)
    p_tr.add_argument("--lora-alpha", type=int, default=64)
    p_tr.add_argument("--max-length", type=int, default=2048)
    p_tr.add_argument("--epochs", type=int, default=3)
    p_tr.add_argument("--lr", type=float, default=2e-5)
    p_tr.add_argument("--batch-size", type=int, default=1)
    p_tr.add_argument("--grad-accum", type=int, default=16)
    p_tr.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Cap total optimizer steps (-1 = use epochs). Useful for smoke tests.",
    )
    p_tr.set_defaults(func=_cmd_train)

    # infer
    p_inf = sub.add_parser("infer", help="Probe the trained Memory Model with one question")
    p_inf.add_argument("--adapter", type=Path, required=True)
    p_inf.add_argument("--question", required=True)
    p_inf.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p_inf.add_argument("--max-new-tokens", type=int, default=256)
    p_inf.set_defaults(func=_cmd_infer)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
