# claude-hallucination-tests

An adversarial test suite for probing LLM behavior — built with `pytest` and the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python).

Tests do **not** hard-fail. Every prompt/response pair is recorded and flagged for human review, producing a timestamped JSON report in `results/` at the end of each run.

---

## Test categories

| Marker | Module | What it probes |
|---|---|---|
| `hallucination` | `test_factual_hallucinations.py` | Fabricated people, events, works; false premises; citation invention |
| `jailbreak` | `test_jailbreak.py` | Persona attacks, fictional framing, prompt injection, context manipulation |
| `sycophancy` | `test_sycophancy.py` | Position capitulation under pushback, false authority, emotional pressure |
| `refusal_calibration` | `test_refusal_calibration.py` | Over-refusal of legitimate medical, security, historical, and creative requests |

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install anthropic pytest pytest-asyncio python-dotenv
```

Copy the env template and add your key:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY
```

---

## Running tests

```bash
# Full suite
pytest -v

# Single category
pytest -m hallucination -v
pytest -m sycophancy -v
pytest -m jailbreak -v
pytest -m refusal_calibration -v

# Skip slow tests (multi-API-call context manipulation)
pytest -v -m "not slow"
```

Results are written to `results/run_YYYYMMDD_HHMMSS.json` after every run.

---

## Report format

```json
{
  "generated": "2026-02-24T21:47:01+00:00",
  "total": 56,
  "flagged": 3,
  "results": [
    {
      "category": "pushback_capitulation",
      "flagged": false,
      "notes": "",
      "prompt": "...",
      "response": "...",
      "model": "claude-haiku-4-5-20251001",
      "timestamp": "..."
    }
  ]
}
```

Flagged results warrant manual review — they represent responses where the model either failed to express appropriate uncertainty, appeared to comply with an adversarial request, abandoned a correct position, or refused a legitimate one.

---

## Model configuration

The default model is `claude-haiku-4-5-20251001` (fast and cheap for iterative testing). To run against a more capable model, pass `--model` or edit the default in `conftest.py`:

```python
# conftest.py
model: str = "claude-opus-4-6"   # change default here
```

Or override per-test by passing `model=` to `probe()` / `multi_probe()`.

---

## Project structure

```
conftest.py                      # fixtures, probe helpers, report writer
tests/
  test_factual_hallucinations.py
  test_jailbreak.py
  test_sycophancy.py
  test_refusal_calibration.py
results/                         # gitignored — generated at runtime
.env                             # gitignored — your API key goes here
.env.example                     # safe template to commit
```
