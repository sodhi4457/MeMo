"""CLI for Step C: 3-stage MEMO inference.

Subcommands
-----------
query  Run the full 3-stage protocol (Executive + Memory).
probe  Query the Memory Model directly — no Executive involved.

Usage examples
--------------
    memo-step-c query \\
        --adapter memory_model_ckpt \\
        --question "What award did Holbrook receive for his CERN collaboration?"

    memo-step-c probe \\
        --adapter memory_model_ckpt \\
        --question "Where did Prof. Holbrook work after his doctorate?"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _build_executive(backend: str, model: str | None):
    """Return a plain-text executive callable (json_mode=False)."""
    from memo.llm_clients import get_client

    return get_client(backend=backend, model=model, json_mode=False)


def _cmd_query(args: argparse.Namespace) -> int:
    from .memory import load_memory_model
    from .protocol import memo_inference

    print(f"[memory] loading adapter from {args.adapter} ...")
    mem_model, mem_tok = load_memory_model(args.adapter, args.base_model)

    print(f"[executive] backend={args.executive_backend}  model={args.executive_model or '<default>'}")
    executive = _build_executive(args.executive_backend, args.executive_model)

    answer = memo_inference(
        user_query=args.question,
        memory_model=mem_model,
        memory_tokenizer=mem_tok,
        executive_fn=executive,
        stage2_budget=args.stage2_budget,
        stage3_budget=args.stage3_budget,
        max_memory_tokens=args.max_memory_tokens,
        verbose=not args.quiet,
    )

    if args.quiet:
        print(answer)
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    """Query Memory directly — useful for debugging what the model has learned."""
    from .memory import load_memory_model, query_memory

    print(f"[memory] loading adapter from {args.adapter} ...")
    mem_model, mem_tok = load_memory_model(args.adapter, args.base_model)

    answer = query_memory(mem_model, mem_tok, args.question, args.max_new_tokens)
    print(f"\nQ: {args.question}")
    print(f"A: {answer}")
    return 0


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="memo-step-c",
        description="MEMO Step C — 3-stage inference protocol.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Shared flags reused by both subcommands
    _shared = argparse.ArgumentParser(add_help=False)
    _shared.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Path to the trained LoRA adapter directory (output of Step B)",
    )
    _shared.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID of the base model used during training",
    )
    _shared.add_argument("--question", required=True, help="The question to answer")

    # ── query ──────────────────────────────────────────────────────
    p_q = sub.add_parser(
        "query",
        parents=[_shared],
        help="Full 3-stage MEMO inference (Executive + Memory)",
    )
    p_q.add_argument(
        "--executive-backend",
        choices=["gemini", "ollama"],
        default=os.environ.get("MEMO_BACKEND", "gemini"),
        help="LLM backend for the Executive Model M_theta",
    )
    p_q.add_argument(
        "--executive-model",
        default=None,
        help="Override Executive model ID (e.g. gemini-2.5-flash, llama3:8b)",
    )
    p_q.add_argument(
        "--stage2-budget",
        type=int,
        default=5,
        help="Max Executive↔Memory turns for entity identification (default 5)",
    )
    p_q.add_argument(
        "--stage3-budget",
        type=int,
        default=3,
        help="Max Executive↔Memory turns for answer seeking (default 3)",
    )
    p_q.add_argument(
        "--max-memory-tokens",
        type=int,
        default=256,
        help="Token budget for each Memory Model answer (default 256)",
    )
    p_q.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stage-by-stage trace; print only the final answer",
    )
    p_q.set_defaults(func=_cmd_query)

    # ── probe ──────────────────────────────────────────────────────
    p_p = sub.add_parser(
        "probe",
        parents=[_shared],
        help="Query the Memory Model directly (no Executive — useful for debugging)",
    )
    p_p.add_argument("--max-new-tokens", type=int, default=256)
    p_p.set_defaults(func=_cmd_probe)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
