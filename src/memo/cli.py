"""CLI entry point for Step A (data synthesis).

Usage examples:
    uv run memo-step-a --backend gemini
    uv run memo-step-a --backend ollama --model gemma3:4b
    uv run memo-step-a --corpus corpus --output data/reflection_qa.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .llm_clients import get_client
from .synthesis import build_reflection_dataset

SUPPORTED_EXTS = {".txt", ".md"}


def load_corpus(corpus_dir: Path) -> tuple[list[str], list[str]]:
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    files = sorted(
        p for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not files:
        raise FileNotFoundError(
            f"No {sorted(SUPPORTED_EXTS)} files in {corpus_dir}"
        )
    return [p.read_text(encoding="utf-8") for p in files], [p.name for p in files]


def smoke_test(backend: str, model: str | None) -> int:
    """Ping the configured backend with a one-shot prompt and print the result."""
    print(f"Smoke test: backend={backend} model={model or '<default>'}")
    client = get_client(backend, model)
    prompt = (
        'Return a JSON array with exactly one object: '
        '[{"question": "What is 2+2?", "answer": "4"}]. '
        "No other text."
    )
    out = client(prompt)
    print("--- raw response ---")
    print(out)
    print("--------------------")
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="MEMO Step A — build a reflection Q&A dataset from a corpus."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("corpus"),
        help="Directory containing .txt/.md documents (default: ./corpus)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reflection_qa.jsonl"),
        help="Output JSONL path (default: data/reflection_qa.jsonl)",
    )
    parser.add_argument(
        "--backend",
        choices=["gemini", "ollama"],
        default=os.environ.get("MEMO_BACKEND", "gemini"),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model id. Defaults: gemini-2.5-flash, gemma3:4b.",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Just send one prompt to the backend and exit.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        return smoke_test(args.backend, args.model)

    corpus, names = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} document(s): {names}")
    print(f"Backend: {args.backend}  Model: {args.model or '<default>'}")

    document_groups = [list(range(len(corpus)))]

    client = get_client(args.backend, args.model)
    build_reflection_dataset(
        corpus=corpus,
        document_groups=document_groups,
        generator=client,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        save_path=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
