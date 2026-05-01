# HBCVTr — Developer Handoff

This file records all fixes made to get the Colab demo working on Python 3.12 with modern
transformers. Read this before touching any code. Last updated: 2026-05-02.

---

## Project overview

**HBCVTr** predicts antiviral activity (pACT / EC50) against HBV and HCV from a SMILES string.
It uses a dual-encoder BART transformer + DNN. Published in *Scientific Reports* (2024).

Entry point for end users: `HBCVTr_Prediction_Demo.ipynb` (Colab notebook).

---

## Repository layout (key files)

```
HBCVTr/
├── HBCVTr_Prediction_Demo.ipynb   # Colab notebook — main user-facing file
├── pretrained_utils.py            # Loads vocabs, builds tokenizers + configs (REWRITTEN)
├── CustomBart_Atomic_Tokenizer.py # Atomic SMILES tokenizer (PATCHED)
├── CustomBart_FG_Tokenizer.py     # Functional-group SMILES tokenizer (PATCHED)
├── DualBartModel.py               # Model architecture (untouched)
├── BartDataset.py                 # Dataset class (untouched for inference)
├── DualInputDataset.py            # Dataset class (untouched for inference)
├── data/
│   ├── atomic_vocab.json          # 274-token atomic vocab (NEW — split out of pretrained_utils)
│   ├── fg_vocab.json              # 3093-token FG vocab   (NEW — split out of pretrained_utils)
│   └── spe_vocab_list.txt         # SPE vocab for FG tokenizer
└── model/
    ├── hbv_model.pt               # ~1.1 GB — downloaded by Colab Step 2 via gdown
    └── hcv_model.pt               # ~1.1 GB — downloaded by Colab Step 2 via gdown
```

---

## Current status

**Steps 1–3**: Working on Colab Python 3.12 with current transformers.
**Step 4**: Fixed as of last commit — prediction runs end-to-end.

Last verified prediction (Adefovir dipivoxil, HBV):
- pACT: 8.1230
- EC50: 7.5342 nM

---

## All fixes applied (why each one was needed)

### 1. `pretrained_utils.py` — complete rewrite

**Problem:** Original file had `from transformers import AdamW, ...` on line 11.
`AdamW` was removed from transformers ≥ 4.38. Colab (Python 3.12) uses a newer version.
Also, the file embedded the full 274-token and 3093-token vocabulary lists as inline Python
lists (~38 KB), making it impossible to read/modify with standard tooling.

**Fix:** Extracted vocabs to `data/atomic_vocab.json` and `data/fg_vocab.json`.
New `pretrained_utils.py` is 68 lines, loads vocabs via `json.load`, and only imports
`BartForConditionalGeneration, BartConfig, PreTrainedTokenizer` from transformers.

### 2. `CustomBart_Atomic_Tokenizer.py` — API compatibility fixes

**Problems:**
- `super().__init__()` was called BEFORE setting `self.encoder`/`self.decoder`, so
  `__init__` crashed when newer transformers called `get_vocab()` during init.
- `get_vocab()` was not implemented (raises `NotImplementedError` in transformers ≥ 4.38).
- `vocab_size` property was missing.
- `tokenize(self, text)` didn't accept `**kwargs`, crashing when transformers ≥ 4.43
  passes `split_special_tokens=False`.

**Fix:** Moved encoder/decoder init before `super().__init__()`. Added `vocab_size` property,
`get_vocab()` method, `**kwargs` on `tokenize()`.

### 3. `CustomBart_FG_Tokenizer.py` — same API fixes + SPE file handle bug

**Problems:** Same as above, plus:
- `tokenize()` created a new `SPE_Tokenizer(self.spe_vob)` on every call, exhausting
  the file handle after the first call.

**Fix:** Create `SPE_Tokenizer` once in `__init__` as `self.spe`; call `self.spe.tokenize(text)`.
Same API fixes as atomic tokenizer.

### 4. Notebook Step 1 — removed transformers version pin

**Problem:** `transformers==4.31.0` requires `tokenizers<0.14`. No pre-built wheel for old
tokenizers on Python 3.12; building from source fails on Colab.

**Fix:** Removed the `transformers==4.31.0` pin. Colab's system transformers is used.

### 5. Notebook Step 1 — pinned `fastprogress==1.0.5`

**Problem:** SmilesPE depends on fastprogress. Version 1.1.5 has a top-level
`from IPython.display import display, HTML, Markdown` that crashes import when
running outside Jupyter (during pip install).

**Fix:** Pin `fastprogress==1.0.5`.

### 6. Notebook Step 2 — re-run guard

**Problem:** If Step 2 was re-run without restarting the kernel, CWD was already inside
`HBCVTr/`. The cell would then try to clone `HBCVTr/` inside `HBCVTr/`, creating a
nested copy.

**Fix:** Added `while os.path.basename(os.getcwd()) == 'HBCVTr': os.chdir('..')` at top.

### 7. Notebook Step 3 — removed AdamW shim

**Problem:** Earlier workaround patched `transformers.AdamW = torch.optim.AdamW` before
importing `pretrained_utils`. This stopped working in Python 3.12 because transformers'
`_LazyModule` machinery bypasses the `__dict__` patch.

**Fix:** No longer needed — `pretrained_utils.py` no longer imports `AdamW`.

### 8. Notebook Step 4 — manual encoding (bypasses transformers pad pipeline)

**Problem 1:** `tokenizer.encode_plus(...)` raised `AttributeError` in newer transformers
(method removed from the public API).

**Problem 2:** `tokenizer(smiles, ...)` (the `__call__` replacement) raised
`ValueError: type of None unknown` inside `pad()`. Root cause: newer transformers'
`prepare_for_model` tries to prepend `bos_token_id` and append `eos_token_id` to
`input_ids`. Our custom tokenizers have these as `None`, so `None` values get inserted
into the token id list, and `pad()` cannot determine the element type.

**Fix:** Replaced both calls with a `_encode()` helper defined in Step 3:
```python
def _encode(tokenizer, smiles, max_length):
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(smiles))
    ids = ids[:max_length]
    pad_id = tokenizer.convert_tokens_to_ids('_')
    mask = [1] * len(ids) + [0] * (max_length - len(ids))
    ids  = ids            + [pad_id] * (max_length - len(ids))
    return torch.tensor([ids]), torch.tensor([mask])
```
This calls `tokenize` and `convert_tokens_to_ids` directly, pads manually, and returns
tensors — completely bypassing the transformers encoding/padding pipeline.

### 9. `torch.inference_mode()` instead of `torch.no_grad()`

Small speed improvement: `inference_mode` additionally disables autograd view tracking.

---

## Potential follow-up issues

- The `_encode` helper assumes `'_'` is always in the vocabulary as the pad token.
  Both vocabs end with `["_", "[UNK]", "[MASK]"]` so this is safe, but fragile if
  vocabs are ever changed.
- `CustomBart_Atomic_Tokenizer` and `CustomBart_FG_Tokenizer` do not override
  `build_inputs_with_special_tokens`, so calling them via the transformers API
  (e.g. `tokenizer(...)`) may still insert `None` BOS/EOS tokens. The `_encode`
  helper avoids this, but if anyone uses the tokenizers directly they should add:
  ```python
  def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
      return token_ids_0 if token_ids_1 is None else token_ids_0 + token_ids_1
  ```
- Model files (`hbv_model.pt`, `hcv_model.pt`) are on Google Drive. If those Drive
  links expire, the gdown download in Step 2 will fail silently (file will be tiny).
  The size check `< 1_000_000` will catch this and retry, but the download will fail.

---

## Reconnect prompt for new machine

Copy-paste this as your first message to Claude Code on the new machine:

```
I'm continuing development on the HBCVTr Colab demo notebook.
The repo is https://github.com/imeewan/HBCVTr (mine — username imeewan).

Please:
1. Read HANDOFF.md in the repo for full context on all fixes made so far.
2. Clone or check the repo so you can push fixes directly.

The current task: the Colab notebook (HBCVTr_Prediction_Demo.ipynb) should run
from top to bottom on Google Colab (Python 3.12, latest transformers) without errors.
All fixes are documented in HANDOFF.md. Ask me to paste any new error output
if there are remaining issues.
```
