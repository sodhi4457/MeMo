"""3-stage MEMO inference protocol.

Stages
------
1. Grounding      — Executive decomposes the query into atomic sub-questions;
                    Memory answers each independently.
2. Entity ID      — Executive iteratively narrows to a specific entity e*
                    using Memory follow-ups (budget B2).
3. Answer Seeking — Executive collects additional facts about e* from Memory
                    (budget B3), then synthesises the final answer â.

Reference: MEMO paper §4 / Memo-Complete-Breakdown Part 4.
"""

from __future__ import annotations

import json
from typing import Callable


def memo_inference(
    user_query: str,
    memory_model,
    memory_tokenizer,
    executive_fn: Callable[[str], str],
    stage2_budget: int = 5,
    stage3_budget: int = 3,
    max_memory_tokens: int = 256,
    verbose: bool = True,
) -> str:
    """Run the full 3-stage MEMO inference protocol.

    Args:
        user_query:        The user's (possibly complex / multi-hop) question.
        memory_model:      Trained Memory Model returned by load_memory_model().
        memory_tokenizer:  Its tokenizer.
        executive_fn:      Callable wrapping the Executive Model M_theta.
                           Signature: (prompt: str) -> str.
                           Any LLM works — Gemini, GPT-4, a local Ollama model.
        stage2_budget:     Max Executive↔Memory turns for entity identification.
        stage3_budget:     Max Executive↔Memory turns for answer seeking.
        max_memory_tokens: Token budget for each Memory answer.
        verbose:           Print a stage-by-stage trace to stdout.

    Returns:
        â — the final synthesised answer string.
    """
    from .memory import query_memory

    def _mem(q: str) -> str:
        return query_memory(memory_model, memory_tokenizer, q, max_memory_tokens)

    def _log(*args) -> None:
        if verbose:
            print(*args)

    _log(f"\n{'='*60}")
    _log(f"QUERY: {user_query}")
    _log(f"{'='*60}")

    # ──────────────────────────────────────────────────────────────
    # STAGE 1: GROUNDING
    # Executive decomposes the query into atomic sub-questions.
    # Memory answers each *independently* — no shared context between
    # calls — forcing parametric answers rather than contextual drift.
    # ──────────────────────────────────────────────────────────────
    _log("\n[STAGE 1: GROUNDING]")

    decompose_prompt = (
        "You are querying a Memory Model trained on a specific document corpus.\n"
        "Decompose the following question into ATOMIC sub-questions.\n"
        "Each sub-question must target exactly ONE identifying fact or constraint.\n"
        "Keep sub-questions short and self-contained.\n\n"
        f"Question: {user_query}\n\n"
        "Return ONLY a JSON array of sub-question strings. No explanation.\n"
        'Example: ["Sub-question 1?", "Sub-question 2?"]'
    )

    raw = executive_fn(decompose_prompt)
    try:
        cleaned = raw.strip().strip("```json").strip("```").strip()
        sub_questions: list[str] = json.loads(cleaned)
        if not isinstance(sub_questions, list) or not sub_questions:
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        sub_questions = [user_query]

    _log(f"  Sub-questions ({len(sub_questions)}): {sub_questions}")

    grounding_responses: dict[str, str] = {}
    for sq in sub_questions:
        ans = _mem(sq)
        grounding_responses[sq] = ans
        _log(f"  Q: {sq}")
        _log(f"  A: {ans}\n")

    grounding_context = "\n".join(
        f"Sub-question: {q}\nMemory answer: {a}"
        for q, a in grounding_responses.items()
    )

    # ──────────────────────────────────────────────────────────────
    # STAGE 2: ENTITY IDENTIFICATION
    # Executive reads grounding responses and narrows to entity e*.
    # It may ask follow-up questions (up to stage2_budget turns).
    # If no entity is found, fall back to synthesising from Stage 1.
    # ──────────────────────────────────────────────────────────────
    _log(f"\n[STAGE 2: ENTITY IDENTIFICATION (budget={stage2_budget})]")

    identified_entity: str | None = None
    followup_history: list[str] = []

    for turn in range(stage2_budget):
        history_str = (
            "\n".join(followup_history) if followup_history else "None yet."
        )

        entity_prompt = (
            "You have retrieved information from a Memory Model.\n"
            "Goal: identify the specific entity the original question is about.\n\n"
            f"Original question: {user_query}\n\n"
            f"Grounding information:\n{grounding_context}\n\n"
            f"Additional follow-up information:\n{history_str}\n\n"
            "Based on all information above, return ONE of:\n"
            '  {"entity": "<name of the entity>", "followup": null}\n'
            '  {"entity": null, "followup": "<question to ask the Memory Model>"}\n\n'
            "Return ONLY valid JSON. No explanation."
        )

        raw = executive_fn(entity_prompt)
        try:
            cleaned = raw.strip().strip("```json").strip("```").strip()
            parsed: dict = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            _log(f"  Turn {turn + 1}: JSON parse failed — stopping Stage 2.")
            break

        if parsed.get("entity"):
            identified_entity = str(parsed["entity"])
            _log(f"  Entity identified: {identified_entity}")
            break

        followup_q = parsed.get("followup")
        if followup_q:
            followup_a = _mem(str(followup_q))
            followup_history.append(
                f"Follow-up Q: {followup_q}\nMemory answer: {followup_a}"
            )
            _log(f"  Turn {turn + 1}: {followup_q}")
            _log(f"           → {followup_a[:120]}")

    if identified_entity is None:
        _log("  No entity identified. Synthesising from grounding only.")
        fallback_prompt = (
            "Answer this question using only the retrieved information below.\n\n"
            f"Question: {user_query}\n\n"
            f"Retrieved information:\n{grounding_context}\n\n"
            "Answer:"
        )
        return executive_fn(fallback_prompt).strip()

    # ──────────────────────────────────────────────────────────────
    # STAGE 3: ANSWER SEEKING + SYNTHESIS
    # Executive targets e* with focused Memory queries (budget B3),
    # then synthesises the final answer â from all collected evidence.
    # â = M_θ(q, {m₁,…,mₖ}, e*, m_seek)
    # ──────────────────────────────────────────────────────────────
    _log(f"\n[STAGE 3: ANSWER SEEKING (budget={stage3_budget})]")

    seeking_evidence: list[str] = []

    for turn in range(stage3_budget):
        seek_prompt = (
            f'You are answering: "{user_query}"\n'
            f"You have identified the key entity as: {identified_entity}\n\n"
            f"Already retrieved:\n{grounding_context}\n\n"
            f'What is ONE specific fact about "{identified_entity}" that you '
            "still need to fully answer the question?\n\n"
            "Return ONLY the question string (no JSON, no explanation):"
        )
        seek_q = executive_fn(seek_prompt).strip()
        seek_a = _mem(seek_q)
        seeking_evidence.append(f"Q: {seek_q}\nA: {seek_a}")
        _log(f"  Turn {turn + 1}: {seek_q}")
        _log(f"           → {seek_a[:120]}")

    m_seek = "\n".join(seeking_evidence)

    synthesis_prompt = (
        "You have all the information needed to answer the following question.\n\n"
        f"Original question: {user_query}\n\n"
        f"Identified entity: {identified_entity}\n\n"
        f"Grounding information (Stage 1):\n{grounding_context}\n\n"
        f"Additional evidence (Stage 3):\n{m_seek}\n\n"
        "Based on ALL of the above, provide a complete, precise answer.\n"
        "Answer:"
    )

    final_answer = executive_fn(synthesis_prompt).strip()
    _log(f"\n[FINAL ANSWER]: {final_answer}")
    return final_answer
