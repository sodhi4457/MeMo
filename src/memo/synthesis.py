"""5-step MEMO data synthesis pipeline.

Steps:
  1) Dual fact extraction (direct + indirect)        -> Q_dir, Q_indir
  2) Consolidation into multi-fact pairs             -> Q_mrg
  3) Self-containment verification / rewrite         -> Q_ver
  4) Entity surfacing (reverse Q&A)                  -> Q_ent
  5) Cross-document synthesis over topical groups    -> Q_cross

Final reflection dataset:
    Q_final = Q_ver  ∪  Q_ent  ∪  Q_cross
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from .chunking import chunk_document, get_encoder
from .prompts import (
    CONSOLIDATION_PROMPT,
    CROSS_DOC_PROMPT,
    DIRECT_EXTRACTION_PROMPT,
    ENTITY_SURFACING_PROMPT,
    INDIRECT_EXTRACTION_PROMPT,
    VERIFICATION_PROMPT,
)

GeneratorFn = Callable[[str], str]


def _safe_parse_json_array(text: str) -> list[dict]:
    """Parse a JSON array from an LLM response, tolerating common noise.

    Handles: leading/trailing whitespace, markdown code fences, and
    extra prose before/after the array. Returns [] on unrecoverable
    parse failures so the pipeline degrades gracefully.
    """
    if not text:
        return []
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()
    try:
        result = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", s, flags=re.S)
        if not m:
            return []
        try:
            result = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    return result if isinstance(result, list) else []


def _clean_pairs(items: list[dict]) -> list[dict]:
    """Keep only well-formed {question, answer} dicts."""
    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        q = it.get("question")
        a = it.get("answer")
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
            out.append({"question": q.strip(), "answer": a.strip()})
    return out


def step1_dual_extraction(
    chunk: str,
    generator: GeneratorFn,
) -> tuple[list[dict], list[dict]]:
    """Step 1: extract direct + indirect Q&A pairs for one chunk."""
    q_dir = _clean_pairs(
        _safe_parse_json_array(generator(DIRECT_EXTRACTION_PROMPT.format(chunk=chunk)))
    )
    q_indir = _clean_pairs(
        _safe_parse_json_array(generator(INDIRECT_EXTRACTION_PROMPT.format(chunk=chunk)))
    )
    return q_dir, q_indir


def step2_consolidate(q_raw: list[dict], generator: GeneratorFn) -> list[dict]:
    """Step 2: merge related pairs into multi-fact composite pairs.

    Returns only the NEW merged pairs. Caller does q_con = q_raw + q_mrg.
    """
    if len(q_raw) < 2:
        return []
    pairs_str = json.dumps(q_raw, indent=2, ensure_ascii=False)
    return _clean_pairs(
        _safe_parse_json_array(generator(CONSOLIDATION_PROMPT.format(pairs=pairs_str)))
    )


def step3_verify_and_rewrite(
    q_con: list[dict],
    chunk: str,
    generator: GeneratorFn,
) -> list[dict]:
    """Step 3: enforce self-containment. Drop unfixable pairs.

    On a parse failure we fall back to the unmodified q_con so the
    pipeline doesn't lose all pairs from a chunk on one bad response.
    """
    if not q_con:
        return []
    pairs_str = json.dumps(q_con, indent=2, ensure_ascii=False)
    raw = _safe_parse_json_array(
        generator(VERIFICATION_PROMPT.format(chunk=chunk, pairs=pairs_str))
    )
    if not raw:
        return q_con

    kept: list[dict] = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        if p.get("status") not in ("ok", "rewritten"):
            continue
        q = p.get("question")
        a = p.get("answer")
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
            kept.append({"question": q.strip(), "answer": a.strip()})
    return kept


def step4_entity_surface(q_ver: list[dict], generator: GeneratorFn) -> list[dict]:
    """Step 4: generate reverse Q&A pairs (attributes -> entity name)."""
    if not q_ver:
        return []
    pairs_str = json.dumps(q_ver, indent=2, ensure_ascii=False)
    return _clean_pairs(
        _safe_parse_json_array(generator(ENTITY_SURFACING_PROMPT.format(pairs=pairs_str)))
    )


def step5_cross_document(
    doc_entity_pairs: dict[int, list[dict]],
    generator: GeneratorFn,
) -> list[dict]:
    """Step 5: cross-document synthesis over a topical group."""
    if len(doc_entity_pairs) < 2:
        return []
    pairs_by_doc = json.dumps(
        {f"document_{k}": v for k, v in doc_entity_pairs.items()},
        indent=2,
        ensure_ascii=False,
    )
    raw = _safe_parse_json_array(
        generator(CROSS_DOC_PROMPT.format(pairs_by_doc=pairs_by_doc))
    )
    out: list[dict] = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        q = p.get("question")
        a = p.get("answer")
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
            entry = {"question": q.strip(), "answer": a.strip()}
            t = p.get("type")
            if isinstance(t, str):
                entry["type"] = t
            out.append(entry)
    return out


def build_reflection_dataset(
    corpus: list[str],
    document_groups: list[list[int]],
    generator: GeneratorFn,
    chunk_size: int = 512,
    overlap: int = 64,
    save_path: str | Path = "data/reflection_qa.jsonl",
    encoder=None,
    append: bool = False,
) -> list[dict]:
    """Run the full Step A pipeline over a corpus and write Q_final to JSONL.

    Each JSONL line carries provenance fields (_source, _doc, _chunk, _group)
    that are useful for debugging but are stripped before SFT training.
    """
    encoder = encoder or get_encoder()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists() and not append:
        save_path.unlink()

    q_final: list[dict] = []
    doc_entity_pairs: dict[int, list[dict]] = {}

    print(f"Processing {len(corpus)} document(s)...")

    for doc_idx, document in enumerate(corpus):
        print(f"\n[Doc {doc_idx + 1}/{len(corpus)}]")
        chunks = chunk_document(document, chunk_size, overlap, encoder)
        print(f"  split into {len(chunks)} chunk(s)")

        q_d_ver: list[dict] = []
        for c_idx, chunk in enumerate(chunks):
            print(f"  chunk {c_idx + 1}/{len(chunks)}: ", end="", flush=True)
            q_dir, q_indir = step1_dual_extraction(chunk, generator)
            q_raw = q_dir + q_indir
            print(f"{len(q_raw)} raw", end="", flush=True)

            q_mrg = step2_consolidate(q_raw, generator)
            q_con = q_raw + q_mrg
            print(f" -> {len(q_con)} consolidated", end="", flush=True)

            q_ver = step3_verify_and_rewrite(q_con, chunk, generator)
            print(f" -> {len(q_ver)} verified")
            q_d_ver.extend(q_ver)

            with save_path.open("a", encoding="utf-8") as f:
                for p in q_ver:
                    f.write(
                        json.dumps(
                            {**p, "_source": "verified", "_doc": doc_idx, "_chunk": c_idx},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        q_ent = step4_entity_surface(q_d_ver, generator)
        doc_entity_pairs[doc_idx] = q_ent
        print(f"  entity-surfacing pairs: {len(q_ent)}")

        q_final.extend(q_d_ver)
        q_final.extend(q_ent)

        with save_path.open("a", encoding="utf-8") as f:
            for p in q_ent:
                f.write(
                    json.dumps(
                        {**p, "_source": "entity", "_doc": doc_idx},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"\nStep 5: cross-document synthesis over {len(document_groups)} group(s)...")
    for g_idx, group in enumerate(document_groups):
        if len(group) < 2:
            print(f"  group {g_idx + 1}: skipped (only {len(group)} doc)")
            continue
        group_pairs = {i: doc_entity_pairs.get(i, []) for i in group}
        q_cross = step5_cross_document(group_pairs, generator)
        print(f"  group {g_idx + 1} ({len(group)} docs): {len(q_cross)} cross-doc pairs")
        q_final.extend(q_cross)

        with save_path.open("a", encoding="utf-8") as f:
            for p in q_cross:
                f.write(
                    json.dumps(
                        {**p, "_source": "cross", "_group": g_idx},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"\nQ_final total: {len(q_final)} pair(s)")
    print(f"Saved to: {save_path}")
    return q_final
