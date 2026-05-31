"""Prompt templates for the 5-step data synthesis pipeline.

All templates instruct the Generator to emit a pure JSON array. Double
braces `{{`/`}}` are literal because we format these with .format(...).
"""

DIRECT_EXTRACTION_PROMPT = """You are a precise fact extractor. Given the following text,
extract factual question-answer pairs about EXPLICITLY STATED information only.
Do not infer or guess. Only extract what is directly written.

Return ONLY a valid JSON array. No explanation, no markdown, no backticks.
Format: [{{"question": "...", "answer": "..."}}]

Text:
{chunk}

JSON array:"""


INDIRECT_EXTRACTION_PROMPT = """You are an expert analyst. Given the following text,
generate question-answer pairs that require INFERENTIAL REASONING — conclusions,
implications, or relationships that can be derived from the text but are not
explicitly stated.

Return ONLY a valid JSON array. No explanation, no markdown, no backticks.
Format: [{{"question": "...", "answer": "..."}}]

Text:
{chunk}

JSON array:"""


CONSOLIDATION_PROMPT = """You are given a list of Q&A pairs from the same text.
Identify groups of pairs that share a common entity, time period, or relationship.
For each group, create ONE consolidated Q&A pair that integrates all the related facts
into a single multi-fact question and answer.

ONLY return the NEW consolidated pairs - do not return the originals.
Return ONLY a valid JSON array. No explanation, no markdown.
Format: [{{"question": "...", "answer": "..."}}]

Q&A pairs to consolidate:
{pairs}

Consolidated JSON array:"""


VERIFICATION_PROMPT = """You are a quality control agent. For each Q&A pair below,
determine if it is SELF-CONTAINED - i.e., can it be fully understood and correctly
answered WITHOUT reading the source text?

Common failures:
- Unresolved pronouns: "What did they decide?" (who is "they"?)
- Implicit references: "As shown in the table above..."
- Vague subjects: "What was the main finding?"

For each pair:
- If SELF-CONTAINED: return with status "ok"
- If NOT self-contained but fixable: rewrite it using the source text, status "rewritten"
- If unfixable: status "discard"

Source text:
{chunk}

Q&A pairs to verify:
{pairs}

Return ONLY a valid JSON array. No explanation, no markdown.
Format: [{{"question": "...", "answer": "...", "status": "ok|rewritten|discard"}}]"""


ENTITY_SURFACING_PROMPT = """You are given Q&A pairs about various named entities.
For each distinct named entity that appears, generate question-answer pairs where:
- The QUESTION describes the entity's attributes, relationships, and properties
- The ANSWER is the entity's name/identity

Generate questions at multiple levels of complexity:
- Simple (1 attribute): "Who worked at CERN?" -> "Dr. Elena Marchetti"
- Moderate (2 attributes): "Which physicist worked at CERN on quantum field theory?" -> "Dr. Elena Marchetti"
- Complex (3+ attributes): "Which female physicist published quantum field theory work at CERN in the 1970s?" -> "Dr. Elena Marchetti"

Return ONLY a valid JSON array. No explanation, no markdown.
Format: [{{"question": "...", "answer": "..."}}]

Q&A pairs:
{pairs}

Entity-surfacing JSON array:"""


CROSS_DOC_PROMPT = """You are given entity-surfacing Q&A pairs from multiple related documents.
Identify and generate cross-document Q&A pairs of two types:

TYPE 1 - CONVERGING CLUES: Multiple documents each give partial facts about the SAME entity.
Combined, the facts uniquely identify the entity. Generate a question requiring all clues.
Example: "Which entity is described as [fact from doc A] AND [fact from doc B]?"

TYPE 2 - PARALLEL PROPERTIES: Different entities across different documents share the same
structural role or attribute. Generate comparative questions.
Example: "Which entities from these documents both [shared property]?"

Return ONLY a valid JSON array. No explanation, no markdown.
Format: [{{"question": "...", "answer": "...", "type": "converging|parallel"}}]

Entity-surfacing pairs by document:
{pairs_by_doc}

Cross-document JSON array:"""
