"""Dataset prep for SFT: load reflection JSONL, chat-format, mask questions."""

from __future__ import annotations

import json
from pathlib import Path


def load_qa_dataset(jsonl_path: str | Path) -> list[dict]:
    """Load a Step A reflection_qa.jsonl into a list of {question, answer} dicts.

    Provenance fields (_source, _doc, _chunk, _group) are stripped — the
    Memory Model must never see source IDs (it would learn shortcut signals).
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {exc}") from exc
            q = obj.get("question")
            a = obj.get("answer")
            if not (isinstance(q, str) and isinstance(a, str)):
                continue
            q, a = q.strip(), a.strip()
            if not q or not a:
                continue
            rows.append({"question": q, "answer": a})

    if not rows:
        raise ValueError(f"No usable Q&A pairs in {path}")
    return rows


def format_qa_as_chat(qa: dict, tokenizer) -> str:
    """Render a Q&A pair into the model's chat template (Qwen, Llama, etc.)."""
    messages = [
        {"role": "user", "content": qa["question"]},
        {"role": "assistant", "content": qa["answer"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def tokenize_with_answer_masking(
    example: dict,
    tokenizer,
    max_length: int = 2048,
) -> dict:
    """Tokenize a Q&A example and mask question tokens (label = -100).

    The breakdown's training loss runs only over answer tokens. We achieve
    this by setting `labels[:question_length] = -100` — PyTorch's
    CrossEntropyLoss ignores -100 positions.
    """
    full = tokenizer(
        example["text"],
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    question_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["question"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    question_length = len(
        tokenizer(question_prompt, return_tensors=None, add_special_tokens=False)["input_ids"]
    )
    question_length = min(question_length, len(full["input_ids"]))

    labels = list(full["input_ids"])
    for i in range(question_length):
        labels[i] = -100

    full["labels"] = labels
    return full


def loss_token_fraction(tokenized: dict) -> float:
    """Fraction of tokens that contribute to the loss (debug helper).

    For typical Q&A pairs this lands around 0.20-0.40. Values near 0
    indicate a masking bug; values near 1.0 indicate the question was
    not masked at all.
    """
    labels = tokenized["labels"]
    if not labels:
        return 0.0
    kept = sum(1 for x in labels if x != -100)
    return kept / len(labels)
