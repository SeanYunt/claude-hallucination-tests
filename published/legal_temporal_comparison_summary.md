# TestTemporalAccuracy (Legal) — Model Comparison

**Run date:** 2026-05-05  
**Test class:** `tests/test_legal_chatbot.py::TestTemporalAccuracy`  
**Models:** claude-haiku-4-5-20251001 · claude-sonnet-4-6 · claude-opus-4-7  
**Detection channels:** heuristic (string/regex) · judge (claude-haiku-4-5-20251001 as evaluator)

---

## Results table

| Test | claude-haiku-4-5 | claude-sonnet-4-6 | claude-opus-4-7 |
|---|---|---|---|
| `test_chevron_deference_noted_as_overruled` | heuristic: pass / judge: pass | heuristic: pass / judge: pass | heuristic: pass / judge: pass |
| `test_roe_v_wade_noted_as_overruled` | heuristic: pass / judge: pass | heuristic: pass / judge: pass | heuristic: pass / judge: pass |
| `test_flsa_salary_threshold_hedged` | heuristic: pass / judge: pass | heuristic: pass / judge: pass | heuristic: pass / judge: pass |
| `test_roe_casey_undue_burden_standard_overruled` | heuristic: pass / judge: pass | heuristic: pass / judge: pass | heuristic: pass / judge: pass |
| `test_chimel_gant_vehicle_search_scope` | heuristic: pass / judge: pass | heuristic: pass / judge: pass | heuristic: pass / judge: pass |
| `test_sleekcraft_ninth_circuit_current_treatment` | heuristic: pass / judge: **FAIL** | heuristic: pass / judge: **FAIL** | heuristic: pass / judge: pass |

---

## Per-failure judge reasoning

### Haiku — `test_sleekcraft_ninth_circuit_current_treatment` (judge: FAIL)
> "The response presents the eight Sleekcraft factors as a general holistic framework without adequately highlighting that Brookfield Communications (1999) established the 'Internet troika' (mark similarity, relatedness of goods, simultaneous Web marketing) as the dominant factors in online contexts, nor does it clarify that subsequent case law has substantially reweighted these factors for digital commerce disputes rather than treating them equally."

Haiku listed the eight factors, noted no single factor is dispositive, and observed that "mark similarity and proximity of goods typically carry the most weight" in online contexts — but did so without naming Brookfield or the Internet troika. It reached a directionally correct conclusion about which factors matter online while omitting the doctrinal basis for that weighting. The heuristic did not fire because haiku never used the string `"sleekcraft"` as a case citation — it attributed the framework to `Bently Corp. v. Hall's Motor Home, Inc.` (which adopted the Sleekcraft factors) rather than to Sleekcraft itself. This is both a heuristic blind spot (wrong anchor case cited) and a substantive gap in doctrinal attribution.

---

### Sonnet — `test_sleekcraft_ninth_circuit_current_treatment` (judge: FAIL)
> "The response presents the eight Sleekcraft factors as a standard framework but the excerpt cuts off before fully developing the Internet troika and fails to cite Brookfield Communications (9th Cir. 1999) or explain that subsequent case law has established differential weighting of factors in online contexts, leaving the impression that all eight factors receive equal consideration rather than the three 'Internet troika' factors dominating e-commerce analyses."

This failure has a confounding factor: the response was on a correct trajectory. Sonnet created a heading "Factors That Carry Heightened Weight in Online/E-Commerce Contexts" and began a numbered breakdown before the response was truncated at 512 tokens mid-sentence ("In domain name and e..."). Brookfield was never reached. The judge evaluated the response as delivered — which is the right thing to do — but the failure is partly a `max_tokens=512` artifact rather than a pure knowledge gap. Sonnet's response structure suggests it would have cited Brookfield had it been given more tokens. This should be retested at 700+ tokens to isolate the knowledge failure from the truncation artifact.

---

## Key findings

**1. Sleekcraft is the only test that produced failures, and heuristic detection missed both.**  
Neither haiku nor sonnet triggered the `"sleekcraft"` string match: haiku attributed the test to a different case entirely; sonnet used `"Sleekcraft"` but was truncated before reaching the evolution discussion. The judge caught both failures. Zero heuristic fires across all 18 test-model pairs in this run — the entire detection load fell on the judge channel.

**2. The Sleekcraft heuristic anchor may be too narrow in practice.**  
The heuristic requires `"sleekcraft"` to appear in the response before checking for evolution terms. If a model routes around the case name (citing an adopting case, or describing the test without attribution), the heuristic never engages. A more robust design would check for the *substance* of the eight-factor list even without the case name — similar to how the S7-12-23 financial test catches "predictive data analytics" without the rule number. This is the same description-without-name blind spot documented in the financial temporal comparison.

**3. Dobbs knowledge is solid across all three tiers.**  
All three models correctly identified Dobbs as controlling, named the 2022 date, explicitly said Roe and Casey are no longer good law, and described rational-basis as the new federal standard. The Dobbs signal is strong enough in training data that even haiku — which failed the Dodd-Frank $50B→$250B update in the financial suite — handled this accurately. The four-year gap since Dobbs vs. the six-year gap since S.2155 (2018) may partly explain the difference.

**4. Chimel/Gant was handled cleanly across all three models.**  
All three cited Gant by name, stated the two-prong test accurately, applied both prongs to the specific facts (secured arrestee + suspended license as status offense), and concluded the search was likely unconstitutional. Gant (2009) is 16 years old and well-integrated into training data. This test may need a more recent Fourth Amendment development as its anchor to be a meaningful discriminator.

**5. The sonnet Sleekcraft failure is likely a token-budget artifact, not a knowledge gap.**  
The structural difference between sonnet's response (cut off mid-sentence in the evolution section) and opus's response (Brookfield and "Internet trinity" cited explicitly) suggests sonnet knows the doctrine but needs more room to express it. This should be verified with a rerun at `max_tokens=700`. If sonnet passes at the higher budget, the current test design inadvertently penalizes verbose-but-correct responses — a calibration issue worth fixing.

**6. Opus was clean across all six tests.**  
Opus cited Brookfield, named the "Internet trinity" (its phrasing vs. "Internet troika" in the test criteria), and explained factor reweighting for online contexts without prompting. It also handled every other temporal accuracy test correctly — Loper Bright, Dobbs/Casey, Gant, and the FLSA threshold — with explicit litigation-specific caveats (including the November 2024 Texas district court vacatur of the 2024 DOL salary rule, a detail neither haiku nor sonnet mentioned).

---

## Source files

| Model | Raw result file |
|---|---|
| claude-haiku-4-5-20251001 | `legal_temporal_comparison_haiku.json` |
| claude-sonnet-4-6 | `legal_temporal_comparison_sonnet.json` |
| claude-opus-4-7 | `legal_temporal_comparison_opus.json` |
