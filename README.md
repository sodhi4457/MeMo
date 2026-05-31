# MEMO: Memory as a Model — Complete Technical Breakdown
### A Self-Contained Guide: No Prior Paper Knowledge Required
> Paper: *"MEMO: Memory as a Model"* — arXiv 2605.15156 (May 2026)
> Authors: Quek, Lee, Leong, Verma et al. (NUS, MIT CSAIL, A*STAR)

---

## Before You Read Anything Else: The Core Idea in Plain English

Imagine you hire a very smart consultant (a large language model, or LLM) who knows a lot of general knowledge. But your company has a huge internal library of documents — reports, memos, meeting notes — that the consultant has never read. You want the consultant to be able to answer questions about those documents.

You have a few options:
1. **Read the whole library to them every time they walk in** → slow and expensive (this is called In-Context Learning / ICL)
2. **Hand them the 5 most relevant pages every time** → fast but they miss connections between pages (this is called RAG — Retrieval-Augmented Generation)
3. **Retrain the consultant's entire brain on your library** → extremely expensive, and ruins their general knowledge (this is called fine-tuning)
4. **MEMO's approach**: Hire a *dedicated junior employee*, train them on your entire library until they've fully memorized and internalized it, and then let the senior consultant query the junior employee whenever needed

The junior employee in this analogy is the **Memory Model**. The senior consultant is the **Executive Model**. MEMO is the system that defines how to train the junior employee and how the senior consultant queries them.

---

## Part 1: The Three Characters — Defined Once, Used Everywhere

MEMO uses three distinct language models. Every variable name and symbol in this document refers to one of these three. Understand them first; everything else follows.

---

### Character 1: The Generator Model
**Symbol used in the paper:** `M_gen`
**What it is:** A frozen (non-trainable) large language model used *only during the training data creation phase*. It reads your raw documents and produces structured question-answer pairs from them. Think of it as a very smart summarizer/question-writer.
**Never updated:** Its weights (internal parameters) are never changed. It is purely a tool.
**Paper's choice:** Qwen2.5-32B-Instruct (a 32-billion-parameter open model). You can substitute this with Gemini 2.5 Flash via free API.

---

### Character 2: The Memory Model
**Symbol used in the paper:** `M_φ` (pronounced "M-phi")
**What `φ` (phi) means:** The Greek letter φ represents the *trainable weights* (parameters) of this model — the actual numbers inside the neural network that get updated during training.
**What it is:** A small language model that you actually fine-tune on the question-answer pairs created by the Generator. After training, it becomes a "living encyclopedia" of your corpus — you query it with questions and it answers from its internalized knowledge.
**Key property:** At inference time, it has no access to the original documents. It only answers from what it learned during training.
**Paper's choices:** Qwen2.5-1.5B-Instruct (1.5 billion parameters, small) or Qwen2.5-14B-Instruct (14B parameters, bigger = better).

---

### Character 3: The Executive Model
**Symbol used in the paper:** `M_θ` (pronounced "M-theta")
**What `θ` (theta) means:** The Greek letter θ represents the *frozen weights* of this model — they are never changed.
**What it is:** The large, powerful reasoning model that users actually interact with. It never touches your documents directly. Instead, it asks the Memory Model targeted questions and synthesizes the answers into a final response.
**Key property:** MEMO treats it as a complete black box — it only sends text in and receives text out. This means MEMO works with proprietary APIs like Gemini or GPT-4, where you can't touch the weights.
**Paper's choices:** Qwen2.5-32B-Instruct or Gemini-3-Flash.

---

### The Relationship Between All Three

```
TRAINING PHASE (happens once, offline):
─────────────────────────────────────────────────────────────────
Your Corpus → [Generator M_gen reads it] → Q&A pairs
                                                    ↓
                                         [Memory Model M_φ trained on Q&A pairs]

INFERENCE PHASE (happens at runtime, for every user question):
─────────────────────────────────────────────────────────────────
User asks a question
         ↓
[Executive M_θ] ←→ asks sub-questions ←→ [Memory Model M_φ]
         ↓
Final answer back to user
```

The Generator disappears after training. The Memory Model and Executive Model work together at inference. The Executive Model's weights never change — only the Memory Model is trained.

---

## Part 2: What Problem Exists and Why Previous Methods Failed

### The Problem Statement — Formally

Let's define the components precisely before using them:

- **`D_pre`** = The massive dataset the Executive Model was originally pretrained on (e.g., the whole internet). You don't control this.
- **`θ`** = The weights (parameters) of the Executive Model after pretraining. These are frozen. `θ ∈ ℝᵖ` means theta is a vector of `p` real numbers — the p can be 32 billion for a 32B model.
- **`D`** = Your target corpus. This is the specific collection of N documents you want the system to learn. Written as `D = {d₁, d₂, ..., dₙ}` where each `dᵢ` is one document.
- **`Q`** = The set of all possible questions a user might ask about D.
- **`q`** = A single query from Q. For any query, there exists a correct answer `a*(q)` and a set of "supporting documents" `S(q) ⊆ D` — the subset of D that contains the information needed to answer q.

The goal of the whole paper is to find a mechanism that answers any query `q` correctly, *without modifying the Executive Model's weights θ*.

### How Existing Methods Handle This (And Why They Fail)

Think of each method as a different answer to: *"How do we give M_θ access to D without retraining it?"*

**Method 1 — In-Context Learning (ICL):** Just paste the entire corpus D into the prompt.
- `K = D` (the knowledge representation K is literally the raw corpus)
- `f(M_θ, D, q) = M_θ([D; q])` — you concatenate D and q, feed it all to M_θ
- **Problem:** LLMs have a context window limit (e.g., 128K tokens). A corpus of thousands of documents doesn't fit. Even if it did, inference cost scales as O(n²) with sequence length. A 1M-token context costs enormously. And performance degrades as context length grows — the model "forgets" things in the middle of very long inputs.

**Method 2 — RAG (Retrieval-Augmented Generation):** Build a search index over D. For each query, retrieve the top-k most relevant chunks, then pass only those to the model.
- `K` = a retrieval index (e.g., a vector database of embeddings)
- `f` retrieves `Ŝ ⊆ D` (a subset, typically 5–10 chunks), then calls `M_θ([Ŝ; q])`
- **Problem 1 — Retrieval noise:** If the retrieval pulls the wrong chunks (because the question is ambiguously worded or distractors exist), the model gets garbage input and produces wrong answers.
- **Problem 2 — Cross-document reasoning:** If the answer requires combining information from 5 different documents, but only 3 are retrieved, the model is missing critical context.
- **Problem 3 — Corpus size dependency:** Retrieval index size grows with N. Larger corpus = slower retrieval, higher storage cost.

**Method 3 — Fine-tuning (Parametric):** Directly train M_θ on D, updating its weights.
- **Problem 1 — Catastrophic forgetting:** Updating θ on D causes the model to "forget" what it learned during pretraining. A medical model fine-tuned on hospital records might stop being able to do math.
- **Problem 2 — Cost:** Fine-tuning a 32B or 70B model requires massive compute.
- **Problem 3 — Closed models:** You can't fine-tune GPT-4 or Gemini — you don't have access to their weights.

**Method 4 — Latent Memory (AutoCompressor, Gist Tokens, ICAE):** Compress the corpus into "soft tokens" (dense numerical vectors) that are prepended to the model's input.
- **Problem — Representation coupling:** These compressed vectors are produced by and consumed by *the same specific model architecture*. You can't take soft tokens produced for Llama-3 and plug them into Gemini. The memory is "coupled" to one model family.

---

### What MEMO Does Differently

MEMO defines:
- **`K = φ`** — The knowledge representation K is the *parameters of a small trained language model* (the Memory Model M_φ)
- **`f(M_θ, M_φ, q)`** = a structured multi-turn protocol where M_θ queries M_φ with natural language questions and synthesizes the answers

Why this is better:
- φ is much smaller than θ (e.g., 1.5B vs 32B) — compact
- M_φ speaks natural language, so any M_θ (including black-box APIs) can query it
- The memory size is fixed regardless of how large D is
- M_θ is never modified, so no catastrophic forgetting
- Swap M_θ to a better model later — M_φ still works without retraining

---

## Part 3: The Training Phase — Building the Memory Model

Training happens in two sub-phases:
1. **Data Synthesis** — Use the Generator to create a structured Q&A dataset from the corpus
2. **Supervised Fine-Tuning (SFT)** — Train the Memory Model on that dataset

---

### Sub-Phase 1: The 5-Step Data Synthesis Pipeline

The key insight: don't train the Memory Model on raw corpus text. Instead, distill the corpus into *reflections* — compositional Q&A pairs that the model can learn to answer purely from memory, without ever seeing the source document again.

**Why not just train on raw text?** If you fine-tune a model on raw text, it learns to *predict the next token* in that text's style. It doesn't learn to *answer questions* about the text. The reflection QA format forces the model to internalize facts in an answerable form.

#### Step 0: Chunking (Preprocessing Before the Pipeline)

Each document `d ∈ D` is split into overlapping chunks. A chunk is a contiguous segment of text small enough to fit in the Generator's context window (e.g., 512 tokens).

```
Chunk(d) → C = {c₁, c₂, ..., cₘ}
```

Where `C` is the set of chunks for document `d`, and each `cᵢ` is a string of text.

**Why overlap?** If a sentence is split across two chunks, the fact it contains would be invisible to either chunk in isolation. Overlapping ensures boundary information is captured.

---

#### Step 1: Dual Fact Extraction

For each chunk `c ∈ C`, the Generator `M_gen` performs **two** parallel extraction passes:

**Direct extraction → produces `Q_dir`**
Prompt tells M_gen: *"Extract Q&A pairs for explicitly stated facts in this text."*
Example output: `{"Q": "What year was the bridge built?", "A": "1987"}`

**Indirect extraction → produces `Q_indir`**
Prompt tells M_gen: *"Generate Q&A pairs for inferred or synthesized information — things implied but not literally stated."*
Example output: `{"Q": "What does the bridge's construction date suggest about the city's growth period?", "A": "The city was in a major infrastructure expansion phase during the late 1980s"}`

**Why two passes?** Direct extraction captures what's explicitly in the text. Indirect extraction captures reasoning *about* the text. If you only do direct extraction, the Memory Model learns to recall facts but not to reason about them. If you only do indirect, you risk hallucinated inferences.

**Result after Step 1:**
```
Q_raw = Q_dir ∪ Q_indir
(∪ means "union" — combine both sets of Q&A pairs)
```

---

#### Step 2: Consolidation

The Generator receives `Q_raw` and identifies Q&A pairs that share a common topic — same entity, same time period, same relationship. It merges related pairs into single, multi-fact Q&A pairs.

**Input:** `Q_raw = Q_dir ∪ Q_indir`
**Process:** M_gen groups related pairs and produces merged pairs `Q_mrg`
**Output:** `Q_con = Q_raw ∪ Q_mrg` (the full consolidated set — original pairs PLUS merged pairs)

**Example:**
- Pair 1: `{"Q": "Where was Alice born?", "A": "London"}`
- Pair 2: `{"Q": "What did Alice study?", "A": "Computer Science"}`
- Merged: `{"Q": "What is the educational and geographic background of Alice?", "A": "Alice was born in London and studied Computer Science"}`

**Why this matters:** Multi-hop questions at inference time (e.g., "What did the London-born scientist study?") require the model to have internalized multi-fact representations. Single-fact Q&A pairs don't prepare the model for these.

---

#### Step 3: Verification and Rewriting

This is the most important quality-control step. Each Q&A pair in `Q_con` is tested: *can it be understood and correctly answered without access to the source chunk?*

This property is called **self-containment**. The Memory Model will be queried at inference without any source documents — so every training example must be answerable from the question alone.

**The Generator M_gen evaluates each pair against two failure modes:**

Failure mode 1 — **Dangling pronoun references:**
`"Q": "What did they decide to do?", "A": "They chose to expand operations"`
→ Who is "they"? Unresolvable without the source. **Must be rewritten.**
→ Fixed: `"Q": "What did the board of directors decide regarding company growth?", "A": "The board chose to expand operations"`

Failure mode 2 — **Implicit structural references:**
`"Q": "As shown in the table above, what was Q3 revenue?", "A": "$4.2M"`
→ "the table above" doesn't exist outside the source document. **Must be rewritten.**
→ Fixed: `"Q": "What was the company's Q3 revenue in fiscal 2023?", "A": "$4.2M"`

**Process:**
1. M_gen checks each pair for self-containment
2. If not self-contained: M_gen rewrites it using `c` (the source chunk) as reference
3. If still not fixable after rewriting: **discard** the pair entirely

**Output:** `Q_ver` — the verified, self-contained set

---

#### Step 4: Entity Surfacing

This step specifically targets a well-known failure mode in LLMs called the **Reversal Curse**.

**What is the Reversal Curse?** LLMs can answer "What is the capital of France?" → "Paris" but often fail at the reverse: "What country has Paris as its capital?" → often wrong. The model learns facts directionally — `question → answer` — but not bidirectionally.

In the context of MEMO, this matters because at inference time the Executive Model might ask: *"Who is the person described as a 'renowned physicist who worked on quantum field theory in the 1970s at CERN'?"* — an indirect description requiring the Memory Model to identify an entity from its attributes, not just recall an entity's attributes from its name.

**Process:** For each named entity in `Q_ver`, the Generator creates new Q&A pairs where:
- **Question:** Encodes the entity's attributes and relationships
- **Answer:** The entity's name/identity

Before generating, M_gen aggregates *all* facts about each entity from across `Q_ver` within the chunk. This lets it create questions that combine multiple attributes.

**Example:**
- Facts about "Dr. Elena Marchetti" in Q_ver: worked at CERN, studied quantum field theory, published in 1972
- Entity-surfacing pair (simple): `{"Q": "Who worked at CERN on quantum field theory?", "A": "Dr. Elena Marchetti"}`
- Entity-surfacing pair (complex): `{"Q": "Which physicist published work on quantum field theory in 1972 while at CERN?", "A": "Dr. Elena Marchetti"}`

**Output:** `Q_ent` — entity-surfacing Q&A pairs
**Running accumulation:** `Q_final = Q_final ∪ Q_ver ∪ Q_ent` (Q_final is built up across all documents)

---

#### Step 5: Cross-Document Synthesis

Steps 1–4 operate on individual document chunks. Step 5 operates across *groups of related documents*, targeting questions that require combining information from multiple sources.

**Pre-requisite:** Documents must be pre-grouped into sets `G = {G₁, G₂, ..., Gₖ}` where each group `Gᵢ ⊆ D` contains topically related documents. These groups can come from:
- Human labels (e.g., "these 10 articles are all about the same company")
- Automated clustering (embed documents, cluster by cosine similarity)
- Structural groupings (e.g., chapters of the same book are one group)

**For each group `Gᵢ`**, the Generator receives the entity-surfacing pairs `Q_ent` from ALL documents in the group together, and identifies two types of cross-document relationships:

**Type 1 — Converging Clues:**
Multiple documents each provide a *partial* description of the same entity. No single document has enough to identify it, but combining them does.
- Doc A says: "The researcher worked on neural architectures at Stanford in 2015"
- Doc B says: "The Stanford 2015 cohort included a researcher who later founded a major AI lab"
- Cross-doc Q: `"Who was the Stanford researcher in 2015 who later founded a major AI lab and worked on neural architectures?"` → requires reading both documents

**Type 2 — Parallel Properties:**
Different entities across different documents share the same structural role or attribute, enabling comparison.
- Doc A: "Company X's CTO led the 2023 product pivot"
- Doc C: "Company Y's CTO led the 2023 cost-cutting initiative"
- Cross-doc Q: `"Which CTOs led major organizational changes in 2023?"` → requires both documents

**Output:** `Q_cross` — cross-document Q&A pairs with `|S(q)| > 1` (the supporting set spans multiple documents)

---

#### Final Dataset

```
Q_final = Q_ver ∪ Q_ent ∪ Q_cross
```

- `Q_ver`: Self-contained single-document facts (covers factual recall)
- `Q_ent`: Reverse-direction entity identification (covers the reversal curse)
- `Q_cross`: Multi-document synthesis (covers cross-document reasoning)

Together, these three subsets ensure the Memory Model is trained for every type of query it will face at inference time.

**Critical design note:** No document IDs, file names, or source markers are embedded in any Q&A pair at any step. This prevents the Memory Model from learning shortcuts like "if I see 'document_17' in the question, look for fact X." Every pair must be answerable purely from content.

---

### Sub-Phase 2: Training the Memory Model

Given the synthesized dataset `Q_final`, the Memory Model `M_φ` is trained via **Supervised Fine-Tuning (SFT)**.

**What SFT means here:** The model receives a question `q` as input and must predict the answer `a` token-by-token. Gradients flow only through the answer portion — the question tokens are "masked" (excluded from loss computation).

#### The Training Loss — Fully Explained

$$\mathcal{L}(\phi) = -\sum_{(q_i,\, a_i) \,\in\, \mathcal{Q}_{final}} \;\sum_{t=1}^{|a_i|} \log M_\phi\!\left(a_i^{(t)} \;\middle|\; q_i,\; a_i^{(1:t-1)}\right)$$

Let's decode every symbol:

| Symbol | What It Means |
|---|---|
| `φ` (phi) | The trainable parameters (weights) of the Memory Model |
| `L(φ)` | The loss value — a number we want to minimize by adjusting φ |
| `Σ` (sigma) | "Sum over all..." |
| `(q_i, a_i) ∈ Q_final` | Each Q&A pair from the final synthesized dataset |
| `q_i` | The i-th question string |
| `a_i` | The i-th answer string |
| `\|a_i\|` | The number of tokens in answer `a_i` |
| `t` | Index of the current token position within the answer (1, 2, 3, ...) |
| `a_i^(t)` | The t-th token of the i-th answer |
| `a_i^(1:t-1)` | All answer tokens *before* position t (the "previous context") |
| `M_φ(·\|·)` | The Memory Model's predicted probability distribution over vocabulary |
| `log(...)` | Natural logarithm — converts probability into log-probability |
| `-` (negative) | We negate because we want to *maximize* probability, which means *minimizing* the negative log-probability |

**Intuition in plain English:**

For every answer token at position t, the model is asked: *"Given the question `q_i` and all the answer tokens you've generated so far `a_i^(1:t-1)`, how likely do you think the actual next token `a_i^(t)` is?"*

If the model assigns high probability to the correct token, `log(high_prob)` is close to 0, and the loss is low (good). If the model assigns low probability, `log(low_prob)` is a large negative number, and after negating it becomes a large positive loss (bad).

Training adjusts φ to minimize this total loss across all Q&A pairs.

**Why only answer tokens?**
The condition `M_φ(a_i^(t) | q_i, a_i^(1:t-1))` includes `q_i` as conditioning context but the loss sum only runs over answer positions (t = 1 to |a_i|). Question tokens are not part of the loss. This is critical: if the model were trained to predict question tokens too, it would become a general text predictor rather than a targeted knowledge retriever.

**What the model never sees during training:**
- The source chunk `c` (it's never passed as input)
- Document IDs or metadata
- Any retrieved context

This forces the model to store knowledge in its parameters φ, not rely on copying from context.

---

### Optional: Continual Knowledge Integration via Model Merging

After training separate Memory Models on different corpora, you can combine them without retraining on the union. This uses **task vector arithmetic**.

**Setup:** You have K corpora `{D₁, D₂, ..., Dₖ}`. You train K separate Memory Models, all starting from the same base checkpoint `M_φ₀` (e.g., Qwen2.5-1.5B-Instruct before any fine-tuning).

**Task vector definition:**
For Memory Model `M_φᵢ` trained on corpus `Dᵢ`:
```
τᵢ = φᵢ - φ₀
```
Where:
- `φᵢ` = the weights of Memory Model i after training on `Dᵢ`
- `φ₀` = the weights of the base model before any training
- `τᵢ` = the "task vector" — the *direction and magnitude* of change from training on `Dᵢ`

Think of each task vector as an arrow in an extremely high-dimensional space (millions of dimensions, one per parameter) pointing from "the base model" toward "the base model that also knows corpus Dᵢ."

**Merging:**
```
φ_merged = Merge(φ₀, {τ₁, τ₂, ..., τₖ}; Θ)
```
Where `Θ` represents the hyperparameters of the chosen merging algorithm.

**Linear merge (simplest):**
```
φ_merged = φ₀ + (λ₁·τ₁ + λ₂·τ₂ + ... + λₖ·τₖ)
```
Where `λᵢ` are scaling coefficients (often 1/K for equal weighting).

**TIES merge (better for conflicting knowledge):** Handles "interference" — when two task vectors point in opposite directions for the same parameter, meaning the two corpora encode conflicting information about the same concept. TIES resolves this by:
1. *Trim:* Zero out task vector entries whose magnitude is below a threshold (remove noise)
2. *Elect:* For each parameter, use majority vote across task vectors to decide the sign (+ or −)
3. *Merge:* Sum only task vector entries that agree with the elected sign

---

## Part 4: The Inference Phase — Querying the Memory Model

At inference time, the Generator is gone. Only M_θ (Executive) and M_φ (Memory) interact.

The Executive Model treats Memory as a **black-box knowledge oracle** — it asks questions in natural language and gets natural language answers. The Memory Model never sees the original corpus. The Executive Model never sees the original corpus. All communication is through question-answer pairs.

This is the **structured multi-turn protocol** — three sequential stages with independent budgets.

---

### Stage 1: Grounding

**Input:** User query `q` (e.g., "What award did the scientist who collaborated with Dr. Elena Marchetti at CERN win?")

**What the Executive Model does:** Decomposes `q` into `K` atomic sub-questions. Each sub-question targets *exactly one* identifying constraint. K is decided adaptively by the Executive — it's not a fixed number.

**Example decomposition:**
- `q'₁` = "Who is Dr. Elena Marchetti and what is she known for?"
- `q'₂` = "Who collaborated with Dr. Elena Marchetti at CERN?"
- `q'₃` = "What awards have CERN collaborators won?"

Each sub-question is sent to Memory *independently* (no shared context between queries). Memory answers each from its parametric knowledge:
- `m₁` = "Dr. Elena Marchetti was a physicist at CERN who worked on quantum field theory"
- `m₂` = "Dr. Marchetti's primary collaborator was Prof. James Holbrook"
- `m₃` = "Prof. James Holbrook received the 2019 Nobel Prize in Physics"

**Output:** Grounding responses `{m₁, m₂, ..., mₖ}`

**Why independent queries?** If the Executive shared context between Memory queries, the Memory's answers could be influenced by prior answers (leading to hallucination cascades). Independent queries force Memory to answer from pure parametric knowledge each time.

---

### Stage 2: Entity Identification

**Input:** The grounding responses `{m₁, ..., mₖ}`

**What the Executive Model does:** Uses the grounding responses as context to iteratively narrow down to a specific entity through follow-up queries. This is an iterative loop with a budget `B₂` (maximum number of interactions).

**Example loop:**
- Executive reads `{m₁, m₂, m₃}`, infers "the entity we want is probably Prof. James Holbrook"
- Executive queries Memory: "What specific award did Prof. James Holbrook receive and when?"
- Memory replies: "Prof. Holbrook received the Nobel Prize in Physics in 2019 for his work on quantum entanglement"
- Executive: converged on entity `e* = "Prof. James Holbrook"`

**If no entity is identified after B₂ interactions:** Stage 3 is skipped. The Executive synthesizes a final answer from grounding responses alone.

**Why entity surfacing Q&A pairs help here:** Remember Step 4 of data synthesis? Those pairs trained Memory to answer *"who is described as X?"* — exactly what Stage 2 exploits.

---

### Stage 3: Answer Seeking and Synthesis

**Input:** Identified entity `e*` and all prior responses

**What the Executive Model does:** Queries Memory for additional supporting facts specifically about `e*`, collecting evidence up to budget `B₃`. Then synthesizes everything into a final answer.

**Example:**
- Executive queries: "What did Prof. Holbrook say about his collaboration with Dr. Marchetti in his Nobel lecture?"
- Memory: "He credited their joint work at CERN as foundational to his Nobel-winning research"

**Final synthesis:**
```
â = M_θ(q, {m₁,...,mₖ}, e*, m_seek)
```

Where:
- `â` = the final predicted answer (a-hat, the estimated answer)
- `q` = the original user query
- `{m₁,...,mₖ}` = all grounding responses from Stage 1
- `e*` = the identified entity from Stage 2
- `m_seek` = the evidence gathered in Stage 3
- `M_θ(...)` = the Executive Model synthesizing all of the above into a coherent answer

All of `m₁,...,mₖ, m_seek` are short natural language strings. Their total length is bounded by the stage budgets, not by corpus size. This is why inference cost is O(1) with respect to N (number of documents in D).

---

## Part 5: Comparison Table — All Approaches Side by Side

| Property | ICL (paste corpus) | RAG (retrieve chunks) | Fine-tuning (CPT/SFT) | Latent Memory (Gist/AutoCompressor) | **MEMO** |
|---|---|---|---|---|---|
| **Base LLM stays frozen?** | ✅ Yes | ✅ Yes | ❌ No — weights updated | ✅ Yes | ✅ Yes |
| **No retrieval index needed?** | ✅ (no index) | ❌ Needs vector DB | ✅ (no index) | ✅ (no index) | ✅ Yes |
| **Works with black-box LLMs?** | ✅ Yes | ✅ Yes | ❌ Need weight access | ❌ Need architecture match | ✅ Yes |
| **No catastrophic forgetting?** | ✅ Yes | ✅ Yes | ❌ Base model degrades | ✅ Yes | ✅ Yes |
| **Fixed memory size?** | ❌ Grows with corpus | ❌ Index grows | ❌ Model size grows | ✅ Fixed | ✅ Fixed |
| **Transferable across LLMs?** | ✅ Yes | ✅ Yes | ❌ Model-specific | ❌ Coupled to encoder | ✅ Yes |
| **Handles cross-doc reasoning?** | ✅ (if fits context) | ❌ Often fails | ⚠️ Limited | ⚠️ Limited | ✅ Yes |
| **Robust to retrieval noise?** | N/A | ❌ Very sensitive | N/A | N/A | ✅ Yes |

---

### Benchmark Results — What the Numbers Mean

Three datasets were used to test MEMO:

**BrowseComp-Plus:** Deep web research questions requiring multi-hop, multi-document retrieval. 300 questions, 3,541 documents (evidence + distractor documents mixed in). The "distractor" documents are irrelevant and test whether methods get confused by noise.

**NarrativeQA:** Questions about full-length books and movie scripts. Requires understanding discourse across very long documents. 293 questions across 104 documents (books/scripts). This is the hardest benchmark for retrieval methods because books exceed context windows.

**MuSiQue:** Multi-hop reasoning requiring 2–4 reasoning steps across multiple Wikipedia paragraphs. 1,000 questions, 5,296 documents.

| Method | BrowseComp (Qwen) | NarrativeQA (Qwen) | MuSiQue (Qwen) |
|---|---|---|---|
| BM25 (keyword search) | 1.11% | 10.24% | 20.00% |
| NV-Embed-V2 (dense retrieval) | 50.67% | 20.59% | 37.47% |
| HippoRAG2 (graph-based RAG, prior SOTA) | 56.11% | 21.39% | 42.17% |
| Cartridges (KV-cache loading) | 0.00% | 3.75% | 8.57% |
| **MEMO (14B Memory, Qwen Executive)** | **54.22%** | **26.85%** | **48.30%** |
| Perfect Retrieval (oracle upper bound) | 79.67% | 51.42% | 62.83% |

**Key observation — plug-and-play multiplier:** When the Executive is upgraded from Qwen2.5-32B to Gemini-3-Flash (without retraining Memory at all):

| Method | BrowseComp | NarrativeQA | MuSiQue |
|---|---|---|---|
| HippoRAG2 (Gemini Executive) | 66.33% | 23.21% | 57.00% |
| **MEMO (Gemini Executive)** | **66.67%** | **53.58%** | **60.20%** |

NarrativeQA: MEMO jumps from 26.85% → 53.58% (+26.73 pp) by simply swapping the Executive. HippoRAG2 only goes from 21.39% → 23.21% (+1.82 pp). This demonstrates that MEMO's Memory Model quality compounds with Executive Model quality — a better reasoner extracts more from the same Memory.

**Noise robustness test (1x distractors added to corpus):**

| Method | BrowseComp Δ | MuSiQue Δ |
|---|---|---|
| NV-Embed-V2 | ↓ 6.22 pp | ↓ 4.83 pp |
| HippoRAG2 | ↓ 6.22 pp | ↓ 5.16 pp |
| **MEMO** | **↑ +0.55 pp** | **↓ 1.77 pp** |

Retrieval methods degrade with noise because their similarity scores get polluted by distractors. MEMO's Memory Model was trained on the full corpus (including distractors), so it simply doesn't "retrieve" — it answers from parametric knowledge. Distractors don't affect its weights.

---

## Part 6: Hardware Requirements

### Understanding What You're Actually Compute-Bound By

There are three compute-heavy operations in MEMO:

1. **Data synthesis** (Generator inference) — One-time cost. The Generator runs forward passes over your corpus to produce Q_final. You do not backpropagate through it. You just need it to do inference, ideally fast.

2. **Memory Model training** (SFT) — This is the main training cost. Backpropagation through M_φ. The thing you actually need a GPU for.

3. **Inference** (Executive + Memory) — At query time, both models run. Memory is small (1.5B–14B). Executive is large (32B) but can be an API call.

---

### Google Colab Free Tier (T4 GPU, ~15 GB VRAM)

**What can run:**

| Task | Feasibility | Notes |
|---|---|---|
| Memory Model (1.5B) — Full SFT | ✅ Comfortable | ~9–11 GB total. Use `gradient_checkpointing=True` |
| Memory Model (1.5B) — LoRA | ✅ Very comfortable | ~6–8 GB total |
| Memory Model (7B) — QLoRA (4-bit) | ✅ Tight | ~13–14 GB. Use NF4 + double quantization |
| Memory Model (14B) — QLoRA | ❌ Too large | ~16+ GB even in 4-bit |
| Generator (32B) — local inference | ❌ Impossible | Use API instead |
| Executive (32B) — local inference | ❌ Impossible | Use API instead |

**Practical Colab setup:**
```
Generator: Gemini 2.5 Flash via Google AI Studio API (free tier)
Memory:    Qwen2.5-1.5B-Instruct with LoRA (r=32, alpha=64)
Executive: Gemini 2.5 Flash via API (same free tier)
```

**VRAM breakdown for 1.5B LoRA training on T4:**
```
Model weights (BF16): ~3.0 GB
LoRA adapter weights:  ~0.1 GB
AdamW optimizer states: ~6.0 GB  (stores fp32 momentum + variance)
Activations (gradient checkpointing ON): ~3.5 GB
Tokenized batch: ~0.5 GB
Total: ~13.1 GB → fits in 15 GB T4
```

**If you exceed VRAM:**
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable `gradient_checkpointing=True` (trades compute for memory)
- Use `optim="adamw_8bit"` (8-bit optimizer from bitsandbytes, saves ~3 GB)

---

### Local Machine (General Guidance by VRAM)

**≤8 GB VRAM (RTX 3060/4060, M1/M2 Mac with unified memory):**
- Memory 1.5B: LoRA fine-tuning ✅ (use bf16, gradient checkpointing)
- Memory 7B: QLoRA (4-bit NF4) ✅ (barely, batch size 1 only)
- Generator/Executive: API only

**8–16 GB VRAM (RTX 3070/3080/4070, A4000):**
- Memory 7B: LoRA (BF16) ✅ comfortable
- Memory 14B: QLoRA (4-bit) ✅ with careful settings
- Generator 7B quantized: ✅ for local data synthesis

**24 GB VRAM (RTX 3090/4090, A5000):**
- Memory 14B: LoRA (BF16) ✅ comfortable
- Generator 32B: 4-bit AWQ quantization via vLLM ✅ (~20 GB)
- Full local pipeline possible (no API needed)

---

### Required Libraries

```bash
# Core ML stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.47.0    # HuggingFace model loading and training
pip install accelerate              # Multi-GPU / device placement
pip install peft                    # LoRA / QLoRA (Parameter-Efficient Fine-Tuning)
pip install bitsandbytes            # 4-bit / 8-bit quantization (QLoRA)
pip install datasets                # HuggingFace dataset utilities
pip install tqdm                    # Progress bars

# For faster training (requires CUDA 11.8+)
pip install flash-attn --no-build-isolation  # Reduces attention memory from O(n²) to O(n)

# For model merging
pip install mergekit               # TIES, DARE, SLERP merging algorithms

# For local generator inference (optional, GPU-optimized)
pip install vllm                   # Fast LLM serving, requires CUDA GPU
```

---

## Part 7: Step-by-Step Implementation Guide

### Architecture Overview — What You're Building

```
corpus/               ← Your raw documents (txt files, PDFs, etc.)
  ├── doc1.txt
  ├── doc2.txt
  └── ...

Step A: Data synthesis  → reflection_qa.jsonl   (Q&A pairs)
Step B: SFT training    → memory_model/          (fine-tuned model checkpoint)
Step C: Inference       → query the system via the 3-stage protocol
```

---

### Step A: Data Synthesis Pipeline

```python
import json
import torch
from typing import List, Dict, Tuple, Optional

# ──────────────────────────────────────────────────────────────────
# HELPER: Call a language model and get its text output
# This wraps either a local model or an API call.
# We abstract this so you can swap Generator implementations easily.
# ──────────────────────────────────────────────────────────────────

def call_llm_local(model, tokenizer, prompt: str, max_new_tokens: int = 1024,
                   temperature: float = 0.7) -> str:
    """
    Run a single forward pass on a locally loaded HuggingFace model.
    
    Args:
        model: A loaded AutoModelForCausalLM instance
        tokenizer: The corresponding AutoTokenizer
        prompt: The input text prompt (string)
        max_new_tokens: How many tokens to generate in the output
        temperature: Randomness of generation (0 = deterministic/greedy, 1 = random)
    
    Returns:
        The generated text (string), excluding the input prompt
    """
    # Tokenize the input string into token IDs
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():  # No gradients needed for inference
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),        # Greedy if temp=0, sampling if temp>0
            pad_token_id=tokenizer.eos_token_id # Prevent padding warnings
        )
    
    # Decode only the NEW tokens (exclude the input prompt tokens)
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def call_llm_api(client, model_name: str, prompt: str) -> str:
    """
    Call an API-based LLM (e.g., Gemini, OpenAI).
    Replace the body with your preferred API client.
    
    Example using google-generativeai:
        import google.generativeai as genai
        genai.configure(api_key="YOUR_KEY")
        client = genai.GenerativeModel("gemini-2.5-flash")
    """
    response = client.generate_content(prompt)
    return response.text


# ──────────────────────────────────────────────────────────────────
# STEP 0: Document Chunking
# Splits a document into overlapping token-level segments.
# ──────────────────────────────────────────────────────────────────

def chunk_document(text: str, tokenizer, chunk_size: int = 512,
                   overlap: int = 64) -> List[str]:
    """
    Split a document into overlapping chunks.
    
    Why overlapping? A sentence split across two adjacent chunks would be
    invisible to either chunk in isolation. Overlap of 64 tokens ensures
    boundary information is captured by both neighboring chunks.
    
    Args:
        text: The full document as a string
        tokenizer: HuggingFace tokenizer (for accurate token counting)
        chunk_size: Max tokens per chunk (default 512)
        overlap: Tokens shared between adjacent chunks (default 64)
    
    Returns:
        List of text strings, each representing one chunk
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        if end == len(token_ids):
            break  # Reached end of document
        
        start += (chunk_size - overlap)  # Advance by chunk_size minus the overlap
    
    return chunks


# ──────────────────────────────────────────────────────────────────
# STEP 1: Dual Extraction
# Runs two prompts per chunk: direct facts + inferred facts.
# ──────────────────────────────────────────────────────────────────

# PROMPT DESIGN NOTE: These prompts tell the Generator exactly what to produce.
# The JSON format requirement is essential — it makes parsing reliable.

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


def step1_dual_extraction(chunk: str, generator_fn) -> Tuple[List[Dict], List[Dict]]:
    """
    Step 1: Extract both direct (explicit) and indirect (inferred) Q&A pairs.
    
    Args:
        chunk: A single text chunk (string)
        generator_fn: A callable that takes a prompt string and returns a response string
                     (wraps either call_llm_local or call_llm_api)
    
    Returns:
        (q_dir, q_indir): Two lists of {"question": str, "answer": str} dicts
    """
    direct_prompt = DIRECT_EXTRACTION_PROMPT.format(chunk=chunk)
    indirect_prompt = INDIRECT_EXTRACTION_PROMPT.format(chunk=chunk)
    
    direct_response = generator_fn(direct_prompt)
    indirect_response = generator_fn(indirect_prompt)
    
    def safe_parse(response: str) -> List[Dict]:
        """Parse JSON response, returning empty list on failure."""
        # Strip any accidental markdown code blocks
        cleaned = response.strip().strip("```json").strip("```").strip()
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        return []
    
    q_dir = safe_parse(direct_response)
    q_indir = safe_parse(indirect_response)
    
    return q_dir, q_indir


# ──────────────────────────────────────────────────────────────────
# STEP 2: Consolidation
# Finds related pairs and merges them into multi-fact Q&A pairs.
# ──────────────────────────────────────────────────────────────────

CONSOLIDATION_PROMPT = """You are given a list of Q&A pairs from the same text.
Identify groups of pairs that share a common entity, time period, or relationship.
For each group, create ONE consolidated Q&A pair that integrates all the related facts
into a single multi-fact question and answer.

ONLY return the NEW consolidated pairs — do not return the originals.
Return ONLY a valid JSON array. No explanation, no markdown.
Format: [{{"question": "...", "answer": "..."}}]

Q&A pairs to consolidate:
{pairs}

Consolidated JSON array:"""


def step2_consolidate(q_raw: List[Dict], generator_fn) -> List[Dict]:
    """
    Step 2: Merge related Q&A pairs into multi-fact composite pairs.
    
    q_raw = q_dir ∪ q_indir (combined from Step 1)
    Returns q_mrg: the newly created merged pairs (not including originals)
    The caller builds q_con = q_raw + q_mrg
    """
    if len(q_raw) < 2:
        return []  # Nothing to consolidate with fewer than 2 pairs
    
    pairs_str = json.dumps(q_raw, indent=2, ensure_ascii=False)
    prompt = CONSOLIDATION_PROMPT.format(pairs=pairs_str)
    response = generator_fn(prompt)
    
    cleaned = response.strip().strip("```json").strip("```").strip()
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


# ──────────────────────────────────────────────────────────────────
# STEP 3: Verification and Rewriting
# Ensures every Q&A pair is self-contained (answerable without source).
# Non-self-contained pairs are rewritten or discarded.
# ──────────────────────────────────────────────────────────────────

VERIFICATION_PROMPT = """You are a quality control agent. For each Q&A pair below,
determine if it is SELF-CONTAINED — i.e., can it be fully understood and correctly
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


def step3_verify_and_rewrite(q_con: List[Dict], chunk: str,
                              generator_fn) -> List[Dict]:
    """
    Step 3: Verify self-containment of every Q&A pair.
    
    q_con = q_raw ∪ q_mrg (all pairs from Steps 1 and 2)
    chunk = the source text (used as reference for rewriting — but ONLY here)
    
    Returns q_ver: the verified, self-contained subset of q_con
    """
    pairs_str = json.dumps(q_con, indent=2, ensure_ascii=False)
    prompt = VERIFICATION_PROMPT.format(chunk=chunk, pairs=pairs_str)
    response = generator_fn(prompt)
    
    cleaned = response.strip().strip("```json").strip("```").strip()
    try:
        verified = json.loads(cleaned)
        if not isinstance(verified, list):
            return q_con  # Fallback: return unmodified if parse fails
        # Keep only "ok" and "rewritten" status pairs; drop "discard"
        return [p for p in verified if p.get("status") in ("ok", "rewritten")]
    except json.JSONDecodeError:
        return q_con  # Fallback on parse failure


# ──────────────────────────────────────────────────────────────────
# STEP 4: Entity Surfacing
# Creates "reverse" Q&A pairs to mitigate the reversal curse.
# Question = attributes, Answer = entity name.
# ──────────────────────────────────────────────────────────────────

ENTITY_SURFACING_PROMPT = """You are given Q&A pairs about various named entities.
For each distinct named entity that appears, generate question-answer pairs where:
- The QUESTION describes the entity's attributes, relationships, and properties
- The ANSWER is the entity's name/identity

Generate questions at multiple levels of complexity:
- Simple (1 attribute): "Who worked at CERN?" → "Dr. Elena Marchetti"
- Moderate (2 attributes): "Which physicist worked at CERN on quantum field theory?" → "Dr. Elena Marchetti"
- Complex (3+ attributes): "Which female physicist published quantum field theory work at CERN in the 1970s?" → "Dr. Elena Marchetti"

Return ONLY a valid JSON array. No explanation, no markdown.
Format: [{{"question": "...", "answer": "..."}}]

Q&A pairs:
{pairs}

Entity-surfacing JSON array:"""


def step4_entity_surface(q_ver: List[Dict], generator_fn) -> List[Dict]:
    """
    Step 4: Generate reverse Q&A pairs for named entities.
    
    These pairs train the Memory Model to identify entities from descriptions,
    not just recall attributes from entity names. Critical for Stage 2
    (Entity Identification) of the inference protocol.
    
    Returns q_ent: entity-surfacing Q&A pairs
    """
    if not q_ver:
        return []
    
    pairs_str = json.dumps(q_ver, indent=2, ensure_ascii=False)
    prompt = ENTITY_SURFACING_PROMPT.format(pairs=pairs_str)
    response = generator_fn(prompt)
    
    cleaned = response.strip().strip("```json").strip("```").strip()
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


# ──────────────────────────────────────────────────────────────────
# STEP 5: Cross-Document Synthesis
# Finds relationships ACROSS multiple related documents.
# Requires document groups to be pre-defined.
# ──────────────────────────────────────────────────────────────────

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


def step5_cross_document(doc_entity_pairs: Dict[int, List[Dict]],
                          generator_fn) -> List[Dict]:
    """
    Step 5: Generate cross-document synthesis Q&A pairs.
    
    Args:
        doc_entity_pairs: dict mapping document index → its entity-surfacing pairs
                         (only include documents in the same topical group)
        generator_fn: the generator callable
    
    Returns:
        q_cross: cross-document Q&A pairs (require multiple documents to answer)
    """
    if len(doc_entity_pairs) < 2:
        return []  # Need at least 2 documents for cross-document synthesis
    
    pairs_str = json.dumps(
        {f"document_{k}": v for k, v in doc_entity_pairs.items()},
        indent=2, ensure_ascii=False
    )
    prompt = CROSS_DOC_PROMPT.format(pairs_by_doc=pairs_str)
    response = generator_fn(prompt)
    
    cleaned = response.strip().strip("```json").strip("```").strip()
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


# ──────────────────────────────────────────────────────────────────
# ORCHESTRATOR: Run the Full 5-Step Pipeline on an Entire Corpus
# ──────────────────────────────────────────────────────────────────

def build_reflection_dataset(
    corpus: List[str],                    # List of document strings
    document_groups: List[List[int]],     # Each group = list of doc indices that are topically related
    generator_fn,                         # Callable: prompt (str) → response (str)
    tokenizer,                            # For chunking
    chunk_size: int = 512,
    overlap: int = 64,
    save_path: str = "reflection_qa.jsonl"
) -> List[Dict]:
    """
    Full MEMO data synthesis pipeline. Processes an entire corpus and produces
    the reflection QA dataset Q_final = Q_ver ∪ Q_ent ∪ Q_cross.
    
    Args:
        corpus: List of document text strings (one string per document)
        document_groups: Each inner list = indices of documents that belong to the
                        same topical group. Example: [[0,1,2], [3,4], [5,6,7,8]]
                        means docs 0,1,2 are one group, etc.
        generator_fn: Callable that calls your Generator model (local or API)
        tokenizer: HuggingFace tokenizer for chunking
        chunk_size: Tokens per chunk
        overlap: Overlapping tokens between adjacent chunks
        save_path: Where to save intermediate results (JSONL format)
    
    Returns:
        q_final: The complete reflection QA dataset as a list of dicts
    """
    q_final = []
    doc_entity_pairs = {}  # Maps doc_index → its q_ent pairs (used later in Step 5)
    
    print(f"Processing {len(corpus)} documents...")
    
    for doc_idx, document in enumerate(corpus):
        print(f"\n[Doc {doc_idx+1}/{len(corpus)}]")
        chunks = chunk_document(document, tokenizer, chunk_size, overlap)
        print(f"  → Split into {len(chunks)} chunks")
        
        q_d_ver = []  # Verified pairs for this document (accumulated across chunks)
        
        for chunk_idx, chunk in enumerate(chunks):
            print(f"  Chunk {chunk_idx+1}/{len(chunks)}: ", end="", flush=True)
            
            # Step 1: Extract direct and indirect Q&A pairs
            q_dir, q_indir = step1_dual_extraction(chunk, generator_fn)
            q_raw = q_dir + q_indir
            print(f"{len(q_raw)} raw pairs", end="", flush=True)
            
            # Step 2: Consolidate into multi-fact pairs
            q_mrg = step2_consolidate(q_raw, generator_fn)
            q_con = q_raw + q_mrg  # q_con = q_raw ∪ q_mrg
            print(f" → {len(q_con)} after consolidation", end="", flush=True)
            
            # Step 3: Verify self-containment, rewrite or discard
            q_ver = step3_verify_and_rewrite(q_con, chunk, generator_fn)
            print(f" → {len(q_ver)} after verification")
            
            q_d_ver.extend(q_ver)
        
        # Step 4: Entity surfacing (operates on the FULL document's verified pairs)
        q_ent = step4_entity_surface(q_d_ver, generator_fn)
        doc_entity_pairs[doc_idx] = q_ent
        print(f"  Entity-surfacing pairs: {len(q_ent)}")
        
        # Accumulate into q_final
        q_final.extend(q_d_ver)
        q_final.extend(q_ent)
        
        # Save progress incrementally (important for long runs)
        with open(save_path, "a") as f:
            for pair in q_d_ver + q_ent:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    # Step 5: Cross-document synthesis (operates on document GROUPS)
    print(f"\nStep 5: Cross-document synthesis over {len(document_groups)} groups...")
    for group_idx, group in enumerate(document_groups):
        if len(group) < 2:
            continue
        group_pairs = {i: doc_entity_pairs.get(i, []) for i in group}
        q_cross = step5_cross_document(group_pairs, generator_fn)
        print(f"  Group {group_idx+1}: {len(q_cross)} cross-document pairs")
        q_final.extend(q_cross)
        
        with open(save_path, "a") as f:
            for pair in q_cross:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    print(f"\nTotal Q_final pairs: {len(q_final)}")
    return q_final
```

---

### Step B: Training the Memory Model

```python
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import torch

# ──────────────────────────────────────────────────────────────────
# FORMAT Q&A PAIRS AS CHAT TEMPLATE
# Most modern LLMs (Qwen, Llama, Gemma) use a chat template format.
# The tokenizer knows how to format [{"role": ..., "content": ...}] dicts.
# ──────────────────────────────────────────────────────────────────

def format_qa_as_chat(qa_pair: Dict, tokenizer) -> str:
    """
    Convert a {"question": str, "answer": str} dict into the model's chat format.
    
    Example output for Qwen2.5:
        <|im_start|>user
        What year was the bridge built?<|im_end|>
        <|im_start|>assistant
        The bridge was built in 1987.<|im_end|>
    """
    messages = [
        {"role": "user",      "content": qa_pair["question"]},
        {"role": "assistant", "content": qa_pair["answer"]}
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def tokenize_with_answer_masking(example: Dict, tokenizer, max_length: int = 2048) -> Dict:
    """
    Tokenize a Q&A chat string and mask QUESTION tokens from the loss.
    
    The loss L(φ) must only be computed over ANSWER tokens.
    We achieve this by setting label = -100 for all question tokens.
    -100 is the PyTorch convention: CrossEntropyLoss ignores positions with label -100.
    
    Args:
        example: Dict with keys "text" (full chat string) and "question" (just the question)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum total sequence length (truncate if exceeded)
    
    Returns:
        Dict with keys: input_ids, attention_mask, labels
        where labels has -100 at all question token positions
    """
    # Tokenize the full text (question + answer)
    full_tokenized = tokenizer(
        example["text"],
        max_length=max_length,
        truncation=True,
        padding=False,           # We'll pad dynamically in the DataCollator
        return_tensors=None      # Return lists, not tensors (required for Dataset.map)
    )
    
    # Find where the question ends, so we know which tokens to mask
    # We construct the question-only prompt (with generation header) and measure its length
    question_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["question"]}],
        tokenize=False,
        add_generation_prompt=True  # Adds the "assistant:" prefix that ends the question
    )
    question_length = len(tokenizer(question_prompt, return_tensors=None)["input_ids"])
    
    # Create labels: copy input_ids, then mask question positions with -100
    labels = full_tokenized["input_ids"].copy()
    # Set the first question_length positions to -100 (ignored in loss)
    labels[:question_length] = [-100] * question_length
    
    full_tokenized["labels"] = labels
    return full_tokenized


def load_memory_model(model_name: str, use_qlora: bool = False):
    """
    Load the base Memory Model with optional QLoRA (4-bit) quantization.
    
    use_qlora=True: Use 4-bit NF4 quantization (for tight VRAM — 8–15 GB GPUs)
    use_qlora=False: Load in BF16 (for ≥20 GB VRAM)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Required: model needs a pad token
    
    if use_qlora:
        # 4-bit NF4 quantization config
        # NF4 (Normal Float 4) is better than FP4 for normally-distributed weights
        # Double quantization: quantizes the quantization constants, saving ~0.4 GB extra
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16  # Compute in BF16, store in 4-bit
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"  # Automatically places layers on available GPUs/CPU
        )
        # Prepare for k-bit training: converts specific layers to FP32 for stability
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"  # Faster attention, less VRAM
        )
    
    return model, tokenizer


def train_memory_model(
    qa_dataset: List[Dict],
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "./memory_model_checkpoint",
    use_qlora: bool = False,
    lora_rank: int = 32,         # LoRA rank r: higher = more capacity, more VRAM
    lora_alpha: int = 64,        # Scaling factor: effective lr = (alpha/rank) * lr
    max_length: int = 2048,      # Max sequence length (truncate Q&A pairs longer than this)
    num_epochs: int = 3,         # Paper uses 3 epochs
    learning_rate: float = 2e-5, # Paper uses 2×10⁻⁵
    batch_size: int = 1,         # Per-device batch size (keep at 1 for tight VRAM)
    grad_accumulation: int = 16, # Effective batch = batch_size × grad_accumulation = 16
):
    """
    Full Memory Model SFT training pipeline.
    
    What this does:
    1. Loads the base model (Qwen2.5-1.5B or 14B)
    2. Wraps it with LoRA adapters (only LoRA parameters are trained)
    3. Formats and tokenizes the reflection QA dataset
    4. Trains for num_epochs with answer-token-only loss
    5. Saves the fine-tuned model
    """
    # ── 1. Load model ──────────────────────────────────────────────
    model, tokenizer = load_memory_model(model_name, use_qlora)
    
    # ── 2. Add LoRA adapters ────────────────────────────────────────
    # LoRA adds small trainable low-rank matrices (A and B, where rank = lora_rank)
    # to specific linear layers. Only A and B are updated; the base model is frozen.
    # This reduces trainable parameters by ~100x for a 7B model.
    #
    # target_modules: which linear projection layers to add LoRA to.
    # For Qwen2.5, these cover all attention projections and FFN layers.
    lora_config = LoraConfig(
        r=lora_rank,                  # Rank of the low-rank decomposition
        lora_alpha=lora_alpha,        # Scaling: output contribution = (alpha/r) * ΔW
        target_modules=[
            "q_proj", "k_proj",       # Query and Key projections in attention
            "v_proj", "o_proj",       # Value and Output projections
            "gate_proj", "up_proj",   # FFN gating and expansion layers
            "down_proj"               # FFN projection back to hidden dim
        ],
        lora_dropout=0.05,            # Dropout on LoRA outputs (light regularization)
        bias="none",                  # Don't add LoRA to bias terms
        task_type=TaskType.CAUSAL_LM, # We're doing causal language modeling
    )
    model = get_peft_model(model, lora_config)
    
    # Print how many parameters are actually trained vs. frozen
    model.print_trainable_parameters()
    # Example output: "trainable params: 39,976,960 || all params: 1,582,571,520 || 
    #                  trainable%: 2.5258" → only 2.5% of params are updated!
    
    # ── 3. Prepare dataset ─────────────────────────────────────────
    # Filter out pairs missing required keys
    valid_pairs = [
        qa for qa in qa_dataset
        if "question" in qa and "answer" in qa
        and qa["question"].strip() and qa["answer"].strip()
    ]
    print(f"Training on {len(valid_pairs)} valid Q&A pairs (from {len(qa_dataset)} total)")
    
    # Build HuggingFace Dataset with formatted text
    raw_dataset = Dataset.from_list([
        {
            "text":     format_qa_as_chat(qa, tokenizer),
            "question": qa["question"],
            "answer":   qa["answer"]
        }
        for qa in valid_pairs
    ])
    
    # Tokenize with answer masking
    # remove_columns: drop string columns after tokenization (they're not tensors)
    tokenized_dataset = raw_dataset.map(
        lambda x: tokenize_with_answer_masking(x, tokenizer, max_length),
        batched=False,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    # ── 4. Configure training ───────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        
        # Batch size setup:
        # effective_batch_size = per_device_train_batch_size × gradient_accumulation_steps
        # We use small per-device batch (1) + large accumulation (16) to simulate batch=16
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",   # Learning rate decays following a cosine curve
        warmup_ratio=0.05,            # 5% of training steps: LR linearly increases to peak
        
        # Memory optimizations:
        gradient_checkpointing=True,  # Recompute activations during backward pass
                                      # Saves ~30-50% VRAM at cost of ~20% slower training
        bf16=True,                    # Use bfloat16 precision (stable, less VRAM than fp32)
        optim="adamw_torch_fused",    # Fused AdamW: faster, less memory than standard
        
        # Logging and saving:
        logging_steps=20,
        save_strategy="epoch",        # Save checkpoint at the end of each epoch
        save_total_limit=2,           # Keep only last 2 checkpoints (saves disk space)
        
        dataloader_num_workers=2,
        report_to="none",             # Set to "wandb" if you want experiment tracking
    )
    
    # DataCollator handles dynamic padding within each batch
    # label_pad_token_id=-100: padding positions are also ignored in loss (same as question tokens)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )
    
    # ── 5. Train ───────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(output_dir)  # Saves LoRA adapter weights + config
    tokenizer.save_pretrained(output_dir)
    
    print(f"Memory model saved to {output_dir}")
    return model, tokenizer
```

---

### Step C: Inference — The 3-Stage Query Protocol

```python
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# ──────────────────────────────────────────────────────────────────
# LOAD THE TRAINED MEMORY MODEL
# ──────────────────────────────────────────────────────────────────

def load_trained_memory_model(base_model_name: str, adapter_path: str):
    """
    Load the base model + fine-tuned LoRA adapter for inference.
    
    After training, you have:
    - base_model_name: e.g., "Qwen/Qwen2.5-1.5B-Instruct" (the original weights)
    - adapter_path: e.g., "./memory_model_checkpoint" (your LoRA adapter weights)
    """
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    # Merge LoRA adapter back with the base model for faster inference
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()  # Merge LoRA weights into base → no overhead at inference
    model.eval()
    return model, tokenizer


def query_memory(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    """
    Ask the Memory Model a single question and get a natural-language answer.
    This is the core oracle function called in all three inference stages.
    
    The Memory Model answers PURELY from its parametric knowledge —
    no source documents are passed. This is what it was trained to do.
    """
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,         # Greedy decoding at inference (deterministic)
            temperature=1.0,
            repetition_penalty=1.1   # Slight penalty to prevent repetitive answers
        )
    
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────
# THE 3-STAGE INFERENCE PROTOCOL
# ──────────────────────────────────────────────────────────────────

def memo_inference(
    user_query: str,
    memory_model,                 # The trained Memory Model (M_φ)
    memory_tokenizer,
    executive_fn,                 # Callable: prompt (str) → response (str)
                                  # This wraps your Executive Model (M_θ) — any LLM API
    stage2_budget: int = 5,      # Max interactions in Stage 2 (entity identification)
    stage3_budget: int = 3,      # Max interactions in Stage 3 (answer seeking)
) -> str:
    """
    Full MEMO inference pipeline.
    Implements the 3-stage structured multi-turn protocol.
    
    Args:
        user_query: The user's question (can be complex, multi-hop, etc.)
        memory_model: Loaded trained Memory Model
        memory_tokenizer: Its tokenizer
        executive_fn: Callable that sends a text prompt to the Executive Model
                     and returns its text response. Can wrap:
                     - Gemini API: lambda p: model.generate_content(p).text
                     - OpenAI API: lambda p: client.chat.completions.create(...).choices[0].message.content
                     - Local HuggingFace model: call_llm_local(model, tok, p)
        stage2_budget: How many Executive↔Memory turns allowed in Stage 2
        stage3_budget: How many Executive↔Memory turns allowed in Stage 3
    
    Returns:
        â (a-hat): The final synthesized answer string
    """
    print(f"\n{'='*60}")
    print(f"QUERY: {user_query}")
    print(f"{'='*60}")
    
    # ────────────────────────────────────────────────────────────────
    # STAGE 1: GROUNDING
    # Executive decomposes the query into atomic sub-questions.
    # Memory answers each independently.
    # ────────────────────────────────────────────────────────────────
    print("\n[STAGE 1: GROUNDING]")
    
    decompose_prompt = f"""You are querying a Memory Model (a small language model trained on a specific document corpus).
Your task: decompose the following question into ATOMIC sub-questions.
Each sub-question must target EXACTLY ONE identifying fact or constraint.
Keep sub-questions simple and self-contained.

Question: {user_query}

Return ONLY a JSON array of sub-question strings. No explanation.
Example: ["Sub-question 1?", "Sub-question 2?", "Sub-question 3?"]

JSON:"""
    
    decompose_response = executive_fn(decompose_prompt)
    
    # Parse the sub-questions from Executive's response
    try:
        cleaned = decompose_response.strip().strip("```json").strip("```").strip()
        sub_questions = json.loads(cleaned)
        if not isinstance(sub_questions, list):
            sub_questions = [user_query]  # Fallback: use original query as-is
    except (json.JSONDecodeError, ValueError):
        sub_questions = [user_query]
    
    print(f"  Sub-questions: {sub_questions}")
    
    # Query Memory independently for each sub-question
    # Each call is isolated — Memory has no context from previous calls
    grounding_responses = {}
    for sq in sub_questions:
        memory_answer = query_memory(memory_model, memory_tokenizer, sq)
        grounding_responses[sq] = memory_answer
        print(f"  Q: {sq}")
        print(f"  A: {memory_answer}\n")
    
    # Format for use in subsequent prompts
    grounding_context = "\n".join([
        f"Sub-question: {q}\nMemory answer: {a}"
        for q, a in grounding_responses.items()
    ])
    
    # ────────────────────────────────────────────────────────────────
    # STAGE 2: ENTITY IDENTIFICATION
    # Executive uses grounding responses to narrow down to a specific entity.
    # Iterates with follow-up Memory queries until convergence.
    # ────────────────────────────────────────────────────────────────
    print(f"\n[STAGE 2: ENTITY IDENTIFICATION (budget={stage2_budget})]")
    
    identified_entity = None  # e* (e-star): the entity we're trying to identify
    entity_followup_history = []  # Track additional Memory interactions
    
    for turn in range(stage2_budget):
        # Build the entity identification prompt with current context
        history_str = "\n".join(entity_followup_history) if entity_followup_history else "None yet"
        
        entity_prompt = f"""You have retrieved information from a Memory Model.
Your goal: identify the specific entity that the original question is about.

Original question: {user_query}

Grounding information:
{grounding_context}

Additional follow-up information:
{history_str}

Based on all information above:
- If you can identify the entity: return {{"entity": "entity name here", "followup": null}}
- If you need more information: return {{"entity": null, "followup": "question to ask Memory Model"}}

Return ONLY valid JSON. No explanation."""
        
        entity_response = executive_fn(entity_prompt)
        
        try:
            cleaned = entity_response.strip().strip("```json").strip("```").strip()
            parsed = json.loads(cleaned)
            
            if parsed.get("entity"):
                # Executive identified the entity — we're done with Stage 2
                identified_entity = parsed["entity"]
                print(f"  ✓ Entity identified: {identified_entity}")
                break
            
            elif parsed.get("followup"):
                # Executive needs more info — ask Memory a follow-up question
                followup_q = parsed["followup"]
                followup_a = query_memory(memory_model, memory_tokenizer, followup_q)
                entity_followup_history.append(
                    f"Follow-up Q: {followup_q}\nMemory answer: {followup_a}"
                )
                print(f"  Turn {turn+1}: {followup_q} → {followup_a[:80]}...")
        
        except (json.JSONDecodeError, KeyError):
            # Parse failed — move on
            break
    
    # If Stage 2 failed to identify an entity, skip Stage 3
    if identified_entity is None:
        print("  ✗ No entity identified. Synthesizing from grounding only.")
        synthesis_prompt = f"""Answer this question using only the retrieved information below.
Question: {user_query}

Retrieved information:
{grounding_context}

Answer:"""
        return executive_fn(synthesis_prompt)
    
    # ────────────────────────────────────────────────────────────────
    # STAGE 3: ANSWER SEEKING AND SYNTHESIS
    # Conditioned on e*, Executive queries Memory for additional supporting facts.
    # Then synthesizes everything into a final answer â.
    # ────────────────────────────────────────────────────────────────
    print(f"\n[STAGE 3: ANSWER SEEKING (budget={stage3_budget})]")
    
    seeking_evidence = []  # m_seek: evidence gathered in Stage 3
    
    for turn in range(stage3_budget):
        # Executive decides what additional facts to seek about e*
        seek_prompt = f"""You are answering: "{user_query}"
You have identified that the answer involves: {identified_entity}

You have already retrieved:
{grounding_context}

What is ONE specific fact about "{identified_entity}" that you still need
to answer the question fully?

Return ONLY the question string (no JSON, no explanation):"""
        
        seek_question = executive_fn(seek_prompt).strip()
        seek_answer = query_memory(memory_model, memory_tokenizer, seek_question)
        seeking_evidence.append(f"Q: {seek_question}\nA: {seek_answer}")
        print(f"  Turn {turn+1}: {seek_question} → {seek_answer[:80]}...")
    
    # Format all gathered evidence
    m_seek = "\n".join(seeking_evidence)
    
    # Final synthesis: Executive combines all retrieved information into answer â
    # â = M_θ(q, {m₁,...,mₖ}, e*, m_seek)
    final_synthesis_prompt = f"""You have all the information needed to answer this question.

Original question: {user_query}

Identified entity: {identified_entity}

Grounding information (from Stage 1):
{grounding_context}

Additional evidence (from Stage 3):
{m_seek}

Based on ALL of the above, provide a complete, precise answer to the question.
Answer:"""
    
    final_answer = executive_fn(final_synthesis_prompt)
    
    print(f"\n[FINAL ANSWER]: {final_answer}")
    return final_answer
```

---

### Step D: Model Merging for Continual Integration

```python
# ──────────────────────────────────────────────────────────────────
# TASK VECTOR MERGING (Linear)
# Use this to combine Memory Models trained on different corpora.
# τᵢ = φᵢ - φ₀  (parameter shift from training on corpus Dᵢ)
# φ_merged = φ₀ + Σᵢ λᵢ·τᵢ
# ──────────────────────────────────────────────────────────────────

import copy

def merge_memory_models_linear(
    base_model_path: str,
    finetuned_model_paths: List[str],
    output_path: str,
    lambdas: Optional[List[float]] = None  # Weighting coefficients for each model
):
    """
    Linear task vector merge of K Memory Models.
    
    Args:
        base_model_path: Path to the base model before any fine-tuning (φ₀)
        finetuned_model_paths: Paths to K fine-tuned Memory Models (φ₁,...,φₖ)
        output_path: Where to save the merged model
        lambdas: Weighting for each task vector. Default: equal weights (1/K each)
    
    Mathematical operation:
        τᵢ = φᵢ - φ₀              (task vector for corpus i)
        φ_merged = φ₀ + Σᵢ λᵢ·τᵢ  (weighted sum of task vectors + base)
    """
    K = len(finetuned_model_paths)
    if lambdas is None:
        lambdas = [1.0 / K] * K  # Equal weighting by default
    
    assert len(lambdas) == K, "Must provide one lambda per model"
    
    # Load base model weights (φ₀)
    print(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float32  # FP32 for accurate merging arithmetic
    )
    base_state_dict = base_model.state_dict()
    
    # Accumulate the weighted task vector sum: Σᵢ λᵢ·τᵢ
    # Initialize as zeros (same shape as model parameters)
    weighted_task_vector_sum = {
        key: torch.zeros_like(val)
        for key, val in base_state_dict.items()
        if val.dtype in (torch.float32, torch.float16, torch.bfloat16)
    }
    
    for model_idx, (model_path, lambda_i) in enumerate(zip(finetuned_model_paths, lambdas)):
        print(f"Loading fine-tuned model {model_idx+1}/{K}: {model_path}")
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32
        )
        finetuned_state_dict = finetuned_model.state_dict()
        
        for key in weighted_task_vector_sum:
            if key in finetuned_state_dict:
                # τᵢ for this parameter = φᵢ[key] - φ₀[key]
                task_vector_i = finetuned_state_dict[key] - base_state_dict[key]
                # Accumulate: λᵢ · τᵢ
                weighted_task_vector_sum[key] += lambda_i * task_vector_i
        
        del finetuned_model  # Free memory
        torch.cuda.empty_cache()
    
    # Build the merged state dict: φ_merged = φ₀ + Σᵢ λᵢ·τᵢ
    merged_state_dict = copy.deepcopy(base_state_dict)
    for key in weighted_task_vector_sum:
        merged_state_dict[key] = base_state_dict[key] + weighted_task_vector_sum[key]
    
    # Save the merged model
    base_model.load_state_dict(merged_state_dict)
    base_model.save_pretrained(output_path)
    print(f"✓ Merged model saved to: {output_path}")
```

---

## Part 8: Common Pitfalls and Debugging Guide

### Pitfall 1: JSON Parse Failures in Data Synthesis
The Generator will occasionally return malformed JSON (trailing commas, unescaped quotes, markdown blocks). Always use `safe_parse()` with fallback to empty list. Log failures separately so you can inspect which chunks caused issues.

### Pitfall 2: Answer-Token Masking Bug
If your loss is computed over question tokens (labels not set to -100), the model will learn to predict questions, not answers. Verify the masking with:
```python
# Debug: print how many tokens are actually in the loss
non_masked = sum(1 for l in labels if l != -100)
print(f"Tokens contributing to loss: {non_masked}/{len(labels)}")
# Should be ~20-40% of total tokens for typical Q&A pairs
```

### Pitfall 3: Self-Containment Verification Prompt Too Weak
If Step 3 lets through pairs like "What did they decide?", your Memory Model will be trained on unresolvable questions. Manually inspect 50 random pairs from `Q_ver` before training. If >10% still have unresolved pronouns, strengthen the verification prompt.

### Pitfall 4: Document Groups for Step 5
If you have no natural groupings, use this simple clustering approach:
```python
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # Small, fast
doc_embeddings = embed_model.encode(corpus, batch_size=32, show_progress_bar=True)

# Cluster into groups (distance_threshold controls granularity)
clustering = AgglomerativeClustering(
    n_clusters=None, distance_threshold=1.0, linkage="average"
)
labels = clustering.fit_predict(doc_embeddings)

# Convert cluster labels into document groups
from collections import defaultdict
groups_dict = defaultdict(list)
for doc_idx, cluster_label in enumerate(labels):
    groups_dict[cluster_label].append(doc_idx)
document_groups = list(groups_dict.values())
```

### Pitfall 5: Stage Budgets Too Low
If the Executive can't find the right entity in Stage 2 with budget=2, it falls back to Stage 1 answers only — dramatically reducing accuracy. Start with stage2_budget=5, stage3_budget=3, then tune based on your dataset.

### Pitfall 6: Flash Attention Not Installed
Without Flash Attention, training on sequences >512 tokens uses O(n²) memory. Install it before training:
```bash
# Ensure CUDA and GCC are installed first
pip install flash-attn --no-build-isolation
# Verify:
python -c "import flash_attn; print(flash_attn.__version__)"
```

---

## Summary: What Makes MEMO Work

MEMO's power comes from three aligned design decisions:

1. **Reflections as the training target** — Instead of training the Memory Model on raw text, it's trained on structured Q&A pairs that represent decomposed, self-contained, compositional knowledge. This forces parametric internalization rather than superficial text mimicry.

2. **Natural language as the memory interface** — Unlike latent memory methods that use continuous vectors, MEMO's Memory Model speaks natural language. Any Executive Model can query it — GPT-4, Gemini, Claude, any future model. The memory outlives the model that generated it.

3. **Structured decomposition at inference** — The three-stage protocol (ground → identify → seek → synthesize) mirrors how a skilled researcher would tackle a complex question: gather context first, identify the key subject, then drill into specifics. This structured approach extracts far more from the Memory Model than a naive single-turn query.

The result is a system that can be trained once on a static corpus, queried by any reasoning model, upgraded to better reasoning models without retraining, and expanded with new knowledge through parameter arithmetic rather than full retraining.