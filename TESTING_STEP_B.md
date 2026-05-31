# Testing MeMo — Step B (Memory Model SFT)

This guide walks you through verifying the Step B pipeline. Step B fine-tunes a small Memory Model (`M_φ`) on the reflection Q&A JSONL that Step A produced.

> Step B is implemented in a **separate package** at `src/memo_train/`. It never imports from the Step A `memo/` package, so the two pieces can be changed independently.

There are two test paths:

1. **Local (Windows) smoke test** — verify imports, dataset, tokenization, and answer-masking are correct **without actually training**.
2. **Google Colab (T4 GPU) full training** — run real SFT on the bundled reflection dataset, then probe the trained model with questions.

---

## 0. Prerequisites

| Tool | Check command | Expected |
|---|---|---|
| Step A already worked | `dir data\reflection_qa.jsonl` | File exists and is non-empty |
| `uv` | `uv --version` | `0.5+` |
| (Colab) Google account | n/a | Free Colab access at https://colab.research.google.com |

If Step A hasn't run yet, do that first via `TESTING.md`.

---

## 1. Install Step B dependencies (local)

From the project root:

```powershell
uv sync --extra train
```

This installs CPU torch + transformers + peft + datasets *in addition to* Step A's deps. Step A's `memo-step-a` command keeps working — Step B's deps live in an isolated extra group.

Verify both CLIs:

```powershell
uv run memo-step-a --help
uv run memo-step-b --help
```

`memo-step-b` should show three subcommands: `inspect`, `train`, `infer`.

---

## 2. Local smoke test (Windows, no GPU needed)

### 2a. Inspect the dataset

```powershell
uv run memo-step-b inspect --data data\reflection_qa.jsonl
```

Expected output:
```
[stats] pairs: 51
[stats] question chars  min/avg/max: 30 / 85 / 185
[stats] answer chars    min/avg/max: 4 / 97 / 918
```

`pairs` should match the number of lines in your `reflection_qa.jsonl`. If it's 0, the loader is rejecting them — check that each line has non-empty `question` and `answer` strings.

### 2b. Verify the tokenizer + answer-masking

```powershell
uv run memo-step-b inspect --data data\reflection_qa.jsonl --tokenizer-check --sample 5
```

This downloads only the tokenizer (~10 MB, no model weights). It then:

- Formats 5 sample pairs through the Qwen2.5 chat template
- Tokenizes each with answer-only masking
- Reports the average fraction of tokens that contribute to loss

Healthy ranges:
- **20–40%** — typical (paragraph-length answers)
- **5–20%** — fine for terse answers (the bundled sample skews short, so you'll likely land here)
- **<5%** — masking is probably broken (entire example masked)
- **>50%** — the question is *not* being masked; loss is computed over questions too (bug)

The command also prints one sample's chat-formatted text so you can eyeball the `<|im_start|>user … <|im_end|>` / `<|im_start|>assistant …` framing.

### 2c. (Optional but recommended) Direct masking proof

This snippet shows exactly which tokens contribute to the loss and which are masked:

```powershell
uv run python -c "from transformers import AutoTokenizer; from memo_train.data import format_qa_as_chat, tokenize_with_answer_masking; tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); tok.pad_token=tok.pad_token or tok.eos_token; pair={'question':'Where was Marchetti born?','answer':'Milan'}; text=format_qa_as_chat(pair,tok); ex=tokenize_with_answer_masking({'text':text,'question':pair['question'],'answer':pair['answer']},tok,512); print('KEPT:',repr(tok.decode([t for t,l in zip(ex['input_ids'],ex['labels']) if l!=-100]))); print('MASKED:',repr(tok.decode([t for t,l in zip(ex['input_ids'],ex['labels']) if l==-100])))"
```

Expected output:
```
KEPT:   'Milan<|im_end|>\n'
MASKED: '<|im_start|>system\nYou are Qwen...<|im_start|>user\nWhere was Marchetti born?<|im_end|>\n<|im_start|>assistant\n'
```

The KEPT text MUST be **only the assistant's answer + closing tag**. If it contains any part of the question, masking is broken.

> If you got this far on Windows, Step B's data pipeline is verified working. Real training happens on Colab.

---

## 3. Full training on Google Colab (T4 GPU)

### 3a. Get the project files into Colab

Two options — see `notebooks/step_b_colab.ipynb` for both:

- **Git clone**: push the project to a private/public GitHub repo, then clone in Colab.
- **Upload zip**: zip the `MeMo/` folder (skip `.venv/`, optionally skip `data/`) and drag it into Colab's file panel.

### 3b. Open the notebook in Colab

1. Go to https://colab.research.google.com → **File** → **Upload notebook** → select `notebooks/step_b_colab.ipynb`.
2. **Runtime** → **Change runtime type** → **T4 GPU** → **Save**.
3. (VS Code path) You can also open the notebook via the **Colab** / **Connect to a Colab runtime** extension in VS Code — the cells are identical.

### 3c. Walk through the cells

1. **Verify GPU** (`nvidia-smi`) — confirms T4 is attached.
2. **Get project files** — pick git-clone OR upload-zip.
3. **Install Step B deps** — `pip install -q -e ".[train]"` (uses Colab's pre-installed CUDA torch).
4. **Upload dataset** — drag `data/reflection_qa.jsonl` from your laptop into `memo_project/data/`.
5. **Inspect** — runs `memo-step-b inspect ... --tokenizer-check`.
6. **Train** — runs `memo-step-b train ...`. On the bundled 51-pair sample, this takes ~3–6 minutes on a T4. Loss should drop from ~3+ to <1 within a few hundred steps.
7. **Inference probe** — runs two `memo-step-b infer` calls against the saved adapter.
8. **(Optional) Save to Drive** — copies `memory_model_ckpt/` into Google Drive so it persists past the Colab session.

### 3d. Expected training output (Colab)

```
[data] 51 Q&A pairs loaded from data/reflection_qa.jsonl
[model] loading base: Qwen/Qwen2.5-1.5B-Instruct  (qlora=False)
trainable params: 18,464,768 || all params: 1,562,179,584 || trainable%: 1.1820
[data] formatting + tokenizing...
[train] starting...
{'loss': 3.21, 'grad_norm': ..., 'learning_rate': ..., 'epoch': 0.31}
{'loss': 1.84, ...}
{'loss': 0.92, ...}
{'loss': 0.41, ...}
{'train_runtime': 240.1, ...}
[train] done. adapter saved to memory_model_ckpt
```

- Trainable % around 1–3% — confirms LoRA is active (you're not full-fine-tuning).
- Loss steadily decreasing — confirms the model is actually learning the corpus.
- If loss plateaus near zero very fast (1 epoch), you're probably overfitting on the tiny 51-pair sample; for real corpora this is much slower.

### 3e. Inference checks

After training, the notebook runs:

```
memo-step-b infer --adapter memory_model_ckpt --question "Who was Dr. Elena Marchetti and where did she work?"
```

A trained Memory Model should produce something like:
> Dr. Elena Marchetti was an Italian theoretical physicist who worked at CERN starting in 1972. She specialized in quantum field theory and collaborated with Prof. James Holbrook.

A model that hasn't actually trained will instead say it doesn't know who Marchetti is, or invent unrelated facts. That's the smoke-test signal: trained = corpus-aware, untrained = hallucinating.

---

## 4. What "working" looks like — summary

Step B is healthy when ALL of these are true:

1. `uv sync --extra train` finishes without errors, and both `memo-step-a` and `memo-step-b` CLIs work.
2. `memo-step-b inspect --tokenizer-check` reports answer-token-fraction in the **5–40%** range, and the KEPT/MASKED decode shows answer-only kept tokens.
3. (Colab) `memo-step-b train` runs to completion with the loss visibly dropping across epochs.
4. `memo-step-b infer` against the trained adapter returns corpus-aware answers (mentions Marchetti, CERN, Holbrook, Nobel, etc.) — *not* "I don't know" or generic facts.
5. A folder `memory_model_ckpt/` exists with `adapter_config.json`, `adapter_model.safetensors`, and the tokenizer files inside.

When all five are true, the Memory Model is ready for Step C (the 3-stage inference protocol from the breakdown).

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `uv sync --extra train` is huge / slow | torch + nvidia libs | Expected — ~2–3 GB for the train extra; one-time |
| `RuntimeError: No CUDA GPUs are available` | Wrong Colab runtime | Runtime → Change runtime type → T4 GPU |
| `OOM` on Colab T4 with 1.5B + LoRA | Sequence too long | Add `--max-length 1024` or `512` to the train command |
| `OOM` even at max-length 512 | LoRA rank too high | Drop `--lora-rank 16 --lora-alpha 32` |
| `BitsAndBytes` import error with `--qlora` | extra not installed | `pip install -e ".[train,qlora]"` in Colab |
| Loss stuck / not dropping | Wrong masking; LR too low | Re-run `inspect --tokenizer-check`; try `--lr 5e-5` |
| `infer` answers are gibberish or repeat tokens | Generation params too greedy | The CLI uses greedy with `repetition_penalty=1.1`; if still bad, the corpus may be too small to learn from |
| `infer` says "I don't know who Marchetti is" | Adapter didn't actually load | Verify `memory_model_ckpt/adapter_config.json` exists; re-run train |
| Windows symlink warning from HuggingFace | Caching uses copies instead of symlinks | Harmless. Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to silence |

---

## 6. File map (what's new in Step B)

```
MeMo/
├── pyproject.toml                    (updated: [train] extra, second package, memo-step-b script)
├── src/memo_train/                   ← new, isolated from src/memo/
│   ├── __init__.py
│   ├── data.py                       loader + chat formatting + answer masking
│   ├── model.py                      base-model + LoRA / QLoRA wrappers
│   ├── trainer.py                    train_memory_model + quick_inference
│   └── cli.py                        memo-step-b inspect | train | infer
├── notebooks/
│   └── step_b_colab.ipynb            Colab T4 training walkthrough
├── TESTING_STEP_B.md                 this file
└── (Step A files untouched)
```
