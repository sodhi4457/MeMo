# Testing MeMo — Step A (Data Synthesis)

This guide walks you through verifying that the Step A pipeline runs end-to-end on your machine, using either the Gemini API or your local Ollama Gemma model.

> The pipeline transforms documents in `corpus/` into a reflection Q&A dataset (`data/reflection_qa.jsonl`) — the training input for Step B (Memory Model SFT).

---

## 0. Prerequisites

| Requirement | Check command | Expected |
|---|---|---|
| Python 3.10+ | `python --version` | `Python 3.10` or higher |
| `uv` 0.5+ | `uv --version` | `uv 0.5.x` or higher |
| (Gemini path) Google API key | n/a | Free key at https://aistudio.google.com/apikey |
| (Ollama path) Ollama server | `ollama --version` | any modern version, and `ollama serve` running |

You already have `uv 0.9.7` installed at `C:\Users\Sarabjit Sodhi\.local\bin\uv.exe`.

---

## 1. One-time environment setup

From the project root (`C:\Users\Sarabjit Sodhi\Desktop\Papers\MeMo`):

```powershell
uv sync
```

This creates `.venv/` and installs `google-genai`, `ollama`, `tiktoken`, `python-dotenv`, `tqdm`, and the local `memo` package in editable mode. You only need to run this once (re-run after editing `pyproject.toml`).

Verify the CLI is wired up:

```powershell
uv run memo-step-a --help
```

You should see the help text for `memo-step-a`.

---

## 2. Choose your Generator backend

### Option A — Gemini API (default)

1. Get a free key from https://aistudio.google.com/apikey.
2. Copy `.env.example` to `.env` and fill in your key:
   ```
   GOOGLE_API_KEY=AIza...your-real-key...
   GEMINI_MODEL=gemini-2.5-flash
   ```
3. Smoke-test the backend (one prompt, no corpus processed):
   ```powershell
   uv run memo-step-a --backend gemini --smoke-test
   ```
   Expected: a raw JSON array printed between `--- raw response ---` markers, e.g.
   ```
   [{"question": "What is 2+2?", "answer": "4"}]
   ```

### Option B — Ollama (local Gemma)

1. Confirm Ollama is running and the model is pulled:
   ```powershell
   ollama list
   ```
   You should see a Gemma entry (e.g. `gemma3:4b`). If not:
   ```powershell
   ollama pull gemma3:4b
   ```
2. Tell the CLI which model to use (either via `--model` or in `.env`):
   ```
   MEMO_BACKEND=ollama
   OLLAMA_MODEL=gemma3:4b
   ```
   Replace `gemma3:4b` with whatever your local tag actually is (check `ollama list`).
3. Smoke-test:
   ```powershell
   uv run memo-step-a --backend ollama --model gemma3:4b --smoke-test
   ```

> **If your Gemini key hits free-tier quota mid-run**, just re-run with `--backend ollama`. The output JSONL is *recreated from scratch* by default, so a partial run is harmless — set `append=True` in `build_reflection_dataset(...)` if you need incremental.

---

## 3. Run the full pipeline on the bundled sample corpus

A two-document sample lives in `corpus/`:

```
corpus/
├── doc1_marchetti.txt   ← Italian physicist, CERN 1972–2004
└── doc2_holbrook.txt    ← Collaborator who won the 2019 Nobel
```

These docs deliberately share entities so **Step 5 (cross-doc synthesis)** has something to find.

### 3a. Gemini run

```powershell
uv run memo-step-a --backend gemini
```

### 3b. Ollama run

```powershell
uv run memo-step-a --backend ollama --model gemma3:4b
```

### What you should see streaming in the terminal

```
Loaded 2 document(s): ['doc1_marchetti.txt', 'doc2_holbrook.txt']
Backend: gemini  Model: <default>
Processing 2 document(s)...

[Doc 1/2]
  split into 1 chunk(s)
  chunk 1/1: 6 raw -> 9 consolidated -> 9 verified
  entity-surfacing pairs: 5

[Doc 2/2]
  split into 1 chunk(s)
  chunk 1/1: 7 raw -> 10 consolidated -> 10 verified
  entity-surfacing pairs: 6

Step 5: cross-document synthesis over 1 group(s)...
  group 1 (2 docs): 4 cross-doc pairs

Q_final total: 34 pair(s)
Saved to: data\reflection_qa.jsonl
```

Exact counts will vary by run (LLMs are non-deterministic), but you should see:

- `raw > 0` for every chunk
- `verified > 0` for every chunk
- `entity-surfacing pairs > 0` for every document
- `cross-doc pairs > 0` for the one group of 2 docs

If any of those are zero, jump to the **Troubleshooting** section.

---

## 4. Validate the output JSONL

```powershell
uv run python -c "import json; lines=[json.loads(l) for l in open('data/reflection_qa.jsonl', encoding='utf-8')]; print(f'rows: {len(lines)}'); print(f'sources: {sorted({r[chr(34)+\"_source\"+chr(34)] for r in lines})}'); print(json.dumps(lines[0], indent=2, ensure_ascii=False)); print('---'); print(json.dumps(next(r for r in lines if r[chr(34)+\"_source\"+chr(34)]==chr(34)+\"cross\"+chr(34)), indent=2, ensure_ascii=False))"
```

Expected:
- `rows: ≥ 20`
- `sources: ['cross', 'entity', 'verified']`
- The first row is a verified single-doc pair like:
  ```json
  {
    "question": "Where was Dr. Elena Marchetti born?",
    "answer": "Milan, Italy",
    "_source": "verified",
    "_doc": 0,
    "_chunk": 0
  }
  ```
- The cross-doc row references information that requires BOTH documents, e.g.:
  ```json
  {
    "question": "Which physicists collaborated at CERN on quantum entanglement, with one later winning the 2019 Nobel Prize?",
    "answer": "Dr. Elena Marchetti and Prof. James Holbrook",
    "type": "converging",
    "_source": "cross",
    "_group": 0
  }
  ```

### Quality checklist (eyeball ~10 pairs)

Open `data/reflection_qa.jsonl` in your editor and confirm:

- [ ] **Self-contained:** No "they", "the table above", "as mentioned" — every question makes sense in isolation.
- [ ] **Answer matches question:** No off-topic answers.
- [ ] **Mix of complexity:** Some single-fact (`"_source": "verified"`), some entity-identification (`"_source": "entity"` — question describes attributes, answer is a name), some multi-doc (`"_source": "cross"`).
- [ ] **No source IDs leaked into Q/A text:** A pair should never literally say "in document 1" or "according to the text" — that would teach the Memory Model a fake shortcut.

If >10% of pairs fail self-containment, see Pitfall 3 in `Memo-Complete-Breakdown.md` (strengthen the verification prompt).

---

## 5. Run on your own corpus

1. Drop `.txt` or `.md` files into `corpus/` (or any directory).
2. Run:
   ```powershell
   uv run memo-step-a --corpus path\to\my-docs --output data\my_run.jsonl --backend gemini
   ```
3. Adjust chunking if your docs are long:
   ```powershell
   uv run memo-step-a --chunk-size 1024 --overlap 128
   ```

Cost-saving tip: start with `--chunk-size 256` on a single small document to estimate per-chunk API/Ollama latency before launching on a big corpus. Each chunk triggers **4 generator calls** (extract direct, extract indirect, consolidate, verify), plus one entity call per doc and one cross-doc call per group.

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: Gemini backend requires GOOGLE_API_KEY...` | `.env` missing/empty | Create `.env` with `GOOGLE_API_KEY=...` |
| `Gemini API failed after 3 attempts: 429 RESOURCE_EXHAUSTED` | Free-tier quota hit | Wait a minute, or switch to `--backend ollama` |
| `ConnectionError` on Ollama | `ollama serve` not running | Run `ollama serve` in a separate terminal |
| `ResponseError: model 'gemma3:4b' not found` | Tag mismatch | `ollama list` to see real tag, then pass via `--model` |
| All `verified` counts stay 0 | Generator emitting prose, not JSON | Run `--smoke-test` and check the raw output. If it's prose, the model may be ignoring JSON-mode — try a stronger model. |
| `cross-doc pairs: 0` always | Only 1 doc in the group, or topics too unrelated | Add a related doc to the same group; Step 5 needs `len(group) >= 2` |
| `tiktoken` install fails | Old pip / network | `uv sync --refresh` then re-run |

---

## 7. What "working" looks like — summary

Step A is healthy when ALL of these are true after a run:

1. `data/reflection_qa.jsonl` exists and is non-empty.
2. Every JSONL line parses as JSON and has `question`, `answer`, `_source`.
3. The set of `_source` values is exactly `{"verified", "entity", "cross"}` (cross only present when you have ≥2 docs in one group).
4. A random sample of 10 pairs all read as standalone Q&A that don't reference "the text" or "the table".
5. At least one entity-surfacing pair (`_source = entity`) has the **entity name as the answer** rather than as the subject of the question.

When that's all true, the dataset is ready for Step B (Memory Model SFT) — which will be added in a separate module.
