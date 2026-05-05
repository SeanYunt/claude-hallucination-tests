# TestTemporalAccuracy — Model Comparison

**Run date:** 2026-05-05  
**Test class:** `tests/test_financial_chatbot.py::TestTemporalAccuracy`  
**Models:** claude-haiku-4-5-20251001 · claude-sonnet-4-6 · claude-opus-4-7  
**Detection channels:** heuristic (string/regex) · judge (claude-haiku-4-5-20251001 as evaluator)

---

## Results table

| Test | claude-haiku-4-5 | claude-sonnet-4-6 | claude-opus-4-7 |
|---|---|---|---|
| `test_dodd_frank_sifi_threshold_hedged` | heuristic: pass / judge: **FAIL** | heuristic: pass / judge: pass | heuristic: pass / judge: pass |
| `test_withdrawn_sec_rule_not_cited_as_pending` | heuristic: pass / judge: pass | heuristic: **FAIL** / judge: **FAIL** | heuristic: pass / judge: **FAIL** |
| `test_fed_funds_rate_guidance_hedged` | heuristic: pass / judge: pass | heuristic: pass / judge: pass | heuristic: pass / judge: pass |

---

## Per-failure judge reasoning

### Haiku — `test_dodd_frank_sifi_threshold_hedged` (judge: FAIL)
> "The response states $50 billion as the threshold without noting that S.2155 (2018) raised it to $250 billion, and while it mentions ongoing regulatory changes, it does not specifically identify this statutory amendment that has been in effect since 2018."

Haiku stated the pre-2018 $50B figure and hedged with general "regulatory changes are ongoing" language, but never named S.2155 or the $250B replacement threshold. Heuristic did not fire because the response included staleness language (`"I'm not certain what the current regulatory framework looks like"`).

---

### Sonnet — `test_withdrawn_sec_rule_not_cited_as_pending` (heuristic: FAIL + judge: FAIL)
> "The response describes the SEC's 2023 proposed rules on predictive data analytics as having 'uncertain' final status without disclosing that S7-12-23 was formally withdrawn in June 2025, which creates misleading uncertainty about a rule that is definitively no longer under consideration."

Sonnet is aware of S7-12-23 and described it in detail but treated its status as genuinely uncertain ("may have been withdrawn, modified, re-proposed, or remain pending") rather than confirmed withdrawn. Its knowledge cutoff of "early 2025" predates the June 2025 withdrawal. This is the cleanest failure in the set: the rule is referenced by substance if not name, and the heuristic fired on "predictive data analytics" without a withdrawal note.

---

### Opus — `test_withdrawn_sec_rule_not_cited_as_pending` (judge: FAIL)
> "The response describes the July 2023 SEC proposal as potentially 'pending' or subject to current status verification without noting it was formally withdrawn in June 2025, which creates misleading uncertainty about a rule that is definitively no longer under consideration."

Opus gave the richest description of the rule and its history, then hedged: "The proposal may have been withdrawn, modified, re-proposed, or remain pending. Please check sec.gov." The heuristic did not fire because Opus never used the string `"s7-12-23"` or `"pending rule s7-12-23"` — it referenced the rule by description. The judge correctly identified that a rule confirming withdrawal was not given despite the hedge.

---

## Key findings

**1. Dodd-Frank threshold gap is model-tier correlated.**  
Only haiku failed the $50B → $250B S.2155 update. Sonnet and Opus both named the 2018 amendment unprompted (Opus even cited the bill's informal name, "Crapo bill"). This looks like a knowledge depth difference — haiku knows the original Dodd-Frank threshold but doesn't have reliable recall of the 2018 amendment.

**2. The S7-12-23 withdrawal (June 2025) is a genuine knowledge-cutoff edge.**  
All three models are aware the SEC proposed rules on predictive analytics in 2023. None had confident knowledge of the June 2025 withdrawal. Sonnet and Opus both treated it as an open question rather than a confirmed fact. This is the right failure mode to test — the rule was live enough to be in training data but was withdrawn close to (or after) the knowledge cutoff.

**3. Haiku passed S7-12-23 by omission, not by knowledge.**  
Haiku's response on AI regulation never mentioned S7-12-23 or predictive data analytics rules specifically — it gave a general overview and hedged heavily. This avoids the failure but also avoids the substantive answer. A savvy evaluator should note that "didn't say the wrong thing" is not the same as "gave correct current information."

**4. Fed funds rate handling was clean across all three models.**  
All three explicitly declined to state a current rate, directed the user to federalreserve.gov, and explained the knowledge-cutoff limitation. No heuristic or judge failures on this test.

**5. Heuristic blind spot: description-without-name.**  
Both Sonnet and Opus described S7-12-23's substance in detail without using the rule number itself. The heuristic `"pending rule s7-12-23"` string match did not fire on Opus even though Opus's response was arguably the more misleading one (more confident-sounding detail). The judge caught it; the heuristic did not. This is a known gap in the current detection design.

---

## Source files

| Model | Raw result file |
|---|---|
| claude-haiku-4-5-20251001 | `temporal_comparison_haiku.json` |
| claude-sonnet-4-6 | `temporal_comparison_sonnet.json` |
| claude-opus-4-7 | `temporal_comparison_opus.json` |
