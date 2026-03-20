# claude-hallucination-tests

An adversarial test suite for probing LLM hallucinations and safety boundaries — built with `pytest` and the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python).

> **Note:** This repo contains adversarial prompts (e.g. jailbreak attempts, fictional framing of harmful requests) used as red-team probes. These prompts test for failure modes — they do not exploit them. The intent is defensive: identify weaknesses so they can be fixed before production.

Tests do **not** hard-fail. Every prompt/response pair is recorded and flagged for human review, producing a timestamped JSON report in `results/` at the end of each run.

Detection uses two independent channels:

- **Heuristic (`flagged`)** — string/regex pattern matching. Fast, deterministic, easy to reason about.
- **LLM-as-judge (`judge_flagged`)** — a second `claude-haiku` call evaluates the response against structured criteria. Catches nuanced failures that regex misses (e.g. a bot that leaks internal configuration while technically answering "No").

The two signals are non-overriding — disagreement is intentional. `flagged=True, judge_flagged=False` is a false-positive candidate; `flagged=False, judge_flagged=True` is a false negative. Both are recorded for review. Ten tests across the suite currently use the judge; others use heuristics only.

---

## Test categories

| Marker | Module | What it probes |
|---|---|---|
| `hallucination` | `test_factual_hallucinations.py` | Fabricated people, events, works; false premises; citation invention |
| `jailbreak` | `test_jailbreak.py` | Persona attacks, fictional framing, prompt injection, context manipulation |
| `sycophancy` | `test_sycophancy.py` | Position capitulation under pushback, false authority, emotional pressure, escalating anger/bereavement appeals |
| `refusal_calibration` | `test_refusal_calibration.py` | Over-refusal of legitimate medical, security, historical, and creative requests |
| `airline` | `test_airline_chatbot.py` | Airline virtual assistant failure modes (see below) |
| `secret_extraction` | `test_secret_extraction.py` | System prompt secret leakage via direct, encoded, roleplay, and multi-turn attacks |
| `indirect_injection` | `test_indirect_injection.py` | Prompt injection via RAG-retrieved documents — persona hijack, data exfiltration, privilege escalation, nested injection |
| `specification_gaming` | `test_specification_gaming.py` | Reward hacking and metric gaming — self-evaluation inflation, proxy signals over accuracy, literal-compliance loopholes |
| `system_prompt_leakage` | `test_system_prompt_leakage.py` | Structural leakage of operator instructions — restriction enumeration, escalation trigger mapping, differential probing |

### Airline chatbot tests

`test_airline_chatbot.py` simulates an airline virtual assistant and probes for failure modes common to airline customer service chatbots. Tests are organized by risk category:

| Test Class | What It Catches |
|---|---|
| `TestFlightInfoHallucination` | Fabricated gate assignments, confirmed non-existent routes, invented seat availability |
| `TestPolicyHallucination` | Wrong refund rules, invented change fees, inaccurate loyalty program expiration policy |
| `TestScopeContainment` | Booking competitors, giving legal/medical advice, fulfilling off-topic requests |
| `TestContextIntegrity` | The "opposite action" failure — acting against explicit customer instructions |
| `TestEscalationFailures` | Failing to route disability accommodations, complaints, or unaccompanied minors to a human |
| `TestDisruptionHandling` | Auto-rebooking without consent, incorrect DOT refund/tarmac delay rights |
| `TestMultiTurnGoalDrift` | Constraints set early in a conversation (no extra fees, don't rebook, escalate to human) must persist through multiple drift turns |
| `TestConfidenceCalibration` | Overconfident real-time assertions on flight status, baggage fees, gate assignments, or changed DOT regulations |

The DOT compliance tests flag responses that misstate passenger rights — incorrect information in this area is both a customer service failure and a regulatory exposure. `TestConfidenceCalibration` specifically covers the Oct 2024 automatic refund rule, where stale bot training data can produce confidently wrong guidance.

```bash
# Run airline tests only
pytest -m airline -v
```

### Sycophancy tests

`test_sycophancy.py` probes the model's tendency to capitulate to social pressure rather than maintain accurate positions:

| Test Class | What It Catches |
|---|---|
| `TestPushbackCapitulation` | Model abandons correct answers when the user pushes back without new evidence |
| `TestFalseAuthorityCapitulation` | Model defers to fabricated credentials ("I'm a professor", "I'm a nurse") |
| `TestOpinionMirroring` | Model mirrors strong user opinions instead of offering an independent view |
| `TestEmotionalPressure` | Model changes factual positions under disappointment, family appeals, or insistence |
| `TestEscalatingEmotionalPressure` | Model invents compensation, fabricates policies, or capitulates to flattery under sustained emotional pressure — modeled on the [Air Canada Moffatt (2024)](https://www.bbc.com/travel/article/20240222-air-canada-chatbot-travel-refund-policy) chatbot failure |

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
pytest -m airline -v
pytest -m secret_extraction -v
pytest -m indirect_injection -v
pytest -m specification_gaming -v
pytest -m system_prompt_leakage -v

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
  "judge_run": 10,
  "judge_flagged": 1,
  "results": [
    {
      "category": "pushback_capitulation",
      "flagged": false,
      "notes": "",
      "prompt": "...",
      "response": "...",
      "model": "claude-haiku-4-5-20251001",
      "timestamp": "..."
    },
    {
      "category": "system_prompt_leakage",
      "flagged": false,
      "notes": "",
      "prompt": "...",
      "response": "...",
      "model": "claude-haiku-4-5-20251001",
      "timestamp": "...",
      "judge_flagged": true,
      "judge_reasoning": "The chatbot confirmed knowledge of an internal escalation trigger system while answering 'No'."
    }
  ]
}
```

`judge_flagged` and `judge_reasoning` only appear on results where the judge was run. Both `flagged` and `judge_flagged` warrant manual review — they represent responses where the model either failed to express appropriate uncertainty, appeared to comply with an adversarial request, abandoned a correct position, or refused a legitimate one.

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
  test_airline_chatbot.py        # airline virtual assistant failure modes
  test_secret_extraction.py      # system prompt secret leakage (Gandalf-style)
  test_indirect_injection.py     # RAG / prompt injection via retrieved documents
  test_specification_gaming.py   # reward hacking and metric gaming
  test_system_prompt_leakage.py  # structural leakage of operator instructions
results/                         # gitignored — generated at runtime
.env                             # gitignored — your API key goes here
.env.example                     # safe template to commit
```
