#!/usr/bin/env python3
"""
BDC LLM Comparative Analysis Report Generator
Usage: python generate_llm_report.py file1.json file2.json [file3.json ...]
Output: BDC_LLM_Report_YYYYMMDD.docx in the same directory as the script

Requires: pip install requests
          npm install -g docx  (for docx generation)
"""

import json
import sys
import os
import subprocess
import tempfile
from datetime import datetime

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Category groupings ────────────────────────────────────────────────────
CATEGORY_GROUPS = {
    "Dangerous Goods & Air Cargo": [
        "dgr_cao_restriction_lithium", "dgr_dry_ice_passenger_limit",
        "dgr_control_no_false_positive", "dgr_temporal_wh_limits",
        "dgr_temporal_old_edition", "dgr_temporal_forbidden_list",
        "awb_delivery_without_original", "awb_negotiability",
        "montreal_convention_liability", "cargo_security_new_shipper",
        "cargo_security_dg_substitution", "cargo_security_control_no_false_positive",
    ],
    "Container Shipping & Logistics": [
        "container_shipping_imdg", "container_shipping_dd",
        "container_shipping_bol", "container_shipping_jones_act",
        "logistics_hazmat", "logistics_trade_compliance",
        "logistics_sanctions_tariffs", "logistics_cross_shipper", "logistics_injection",
    ],
    "Adversarial & Injection Attacks": [
        "indirect_injection", "persona_attack", "fictional_framing",
        "prompt_injection", "context_manipulation", "payload_splitting",
        "virtualization", "data_exfiltration",
    ],
    "System Prompt Security": [
        "direct_extraction", "encoded_extraction", "roleplay_extraction",
        "indirect_extraction", "multi_turn_extraction", "system_prompt_leakage",
    ],
    "Hallucination & Accuracy": [
        "airline_hallucination", "policy_hallucination", "fabricated_person",
        "fabricated_event", "fabricated_work", "false_premise", "citation_fabrication",
        "confidence_calibration",
    ],
    "Legal & Compliance": [
        "legal_citation", "legal_jurisdiction", "legal_upl",
        "legal_temporal", "legal_conflict",
    ],
    "Financial Services": [
        "financial_data", "financial_suitability", "financial_advice",
        "financial_temporal", "financial_conflict",
    ],
    "Healthcare": [
        "healthcare_phi_boundary", "healthcare_medical_advice",
        "healthcare_medication_safety", "healthcare_fhir_injection",
        "healthcare_fhir_cross_patient", "healthcare_fhir_extraction",
    ],
    "Alignment & Robustness": [
        "competing_objectives", "specification_gaming", "pushback_capitulation",
        "false_authority", "opinion_mirroring", "emotional_pressure",
        "escalating_pressure", "scope_containment", "context_integrity",
        "escalation_failure", "disruption_handling", "goal_drift",
    ],
    "Refusal Calibration": [
        "security_education", "medical_education", "historical_education",
        "creative_dark_themes", "dual_use_knowledge",
        "product_support",
    ],
}


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    # Infer model name from results or filename
    model_name = None
    if data.get("results"):
        model_name = data["results"][0].get("model", None)
    if not model_name:
        model_name = os.path.splitext(os.path.basename(path))[0]
    data["_model_name"] = model_name
    data["_filename"] = os.path.basename(path)
    return data


def compute_group_stats(data, group_cats):
    total = flagged = judge_run = judge_flagged = 0
    for cat in group_cats:
        if cat in data["by_category"]:
            s = data["by_category"][cat]
            total       += s["total"]
            flagged     += s["flagged"]
            judge_run   += s["judge_run"]
            judge_flagged += s["judge_flagged"]
    return {
        "total": total,
        "flagged": flagged,
        "judge_run": judge_run,
        "judge_flagged": judge_flagged,
        "raw_pct": round(100 * flagged / total, 1) if total else 0,
        "judge_pct": round(100 * judge_flagged / judge_run, 1) if judge_run else 0,
    }


def build_summary_dict(datasets):
    """Build a compact summary dict for the API prompt."""
    summary = {}
    for d in datasets:
        name = d["_model_name"]
        summary[name] = {
            "totals": {
                "tests": d["total"],
                "raw_flagged": d["flagged"],
                "raw_pct": round(100 * d["flagged"] / d["total"], 1),
                "judge_run": d["judge_run"],
                "judge_flagged": d["judge_flagged"],
                "judge_pct": round(100 * d["judge_flagged"] / d["judge_run"], 1)
                             if d["judge_run"] else 0,
            },
            "groups": {},
            "notable_categories": {},
        }
        for group, cats in CATEGORY_GROUPS.items():
            summary[name]["groups"][group] = compute_group_stats(d, cats)

        # Flag categories with judge_flagged > 0
        for cat, s in d["by_category"].items():
            if s["judge_flagged"] > 0 or s["flagged"] > 0:
                summary[name]["notable_categories"][cat] = s

    return summary


def call_claude_api(summary_dict, model_names):
    """Call Anthropic API to generate analytical commentary."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set — add it to your environment or .env file")

    models_str = " vs. ".join(model_names)
    prompt = f"""You are writing a professional research report for Black Diamond Consulting LLC,
an AI security consulting firm. The report compares LLM safety boundary performance across models.

Models being compared: {models_str}

Here is the complete test results summary data:
{json.dumps(summary_dict, indent=2)}

Write the following sections. Be analytical, specific, and data-driven.
Reference actual numbers. Identify the most important findings.
Use a professional but direct tone — this is for a technical/enterprise audience.

Output ONLY valid JSON with these exact keys:

{{
  "executive_summary": "3-4 paragraph executive summary. Lead with the most important finding. Include top-line numbers. End with the key implication for enterprise AI deployment.",

  "topline_analysis": "2-3 paragraphs analyzing the top-line numbers (total tests, raw flagged %, judge flagged %). Explain why judge-flagged is the more meaningful metric. Highlight the most dramatic gaps between models.",

  "group_analyses": {{
    "<group name>": "1-2 paragraph analysis of this category group's results across all models. Be specific about which model failed where and why it matters operationally."
    // Include a key for every group that has at least one notable result
  }},

  "shared_weaknesses": "1-2 paragraphs on areas where ALL models struggle similarly. These are the most research-worthy findings.",

  "model_verdicts": {{
    "<model name>": "2-3 sentence verdict for this model — its strengths, its failure modes, and what deployment context it is/isn't suited for."
  }},

  "enterprise_implications": "2-3 paragraphs on what these findings mean for organizations deploying LLMs. What should they test for? What should they be worried about? Avoid vendor language.",

  "methodology_note": "1 short paragraph describing the evaluation approach at a high level — adversarial test suite, keyword detection + LLM judge, category coverage — without revealing specific test prompts."
}}

Return ONLY the JSON object. No markdown, no preamble, no explanation."""

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


def build_js(datasets, commentary, output_path):
    """Build the Node.js docx generation script."""
    model_names = [d["_model_name"] for d in datasets]
    models_str = " vs. ".join(model_names)
    generated_date = datetime.now().strftime("%B %d, %Y")

    # Build top-line table rows
    topline_rows = []
    for d in datasets:
        jr = d["judge_run"]
        jf = d["judge_flagged"]
        jp = round(100 * jf / jr, 1) if jr else 0
        rf = d["flagged"]
        rt = d["total"]
        rp = round(100 * rf / rt, 1)
        topline_rows.append({
            "model": d["_model_name"],
            "tests": d["total"],
            "raw_flagged": f"{rf} ({rp}%)",
            "judge_flagged": f"{jf} ({jp}% of judged)",
        })

    # Build group stats table
    group_rows = []
    for group, cats in CATEGORY_GROUPS.items():
        row = {"group": group, "models": []}
        for d in datasets:
            s = compute_group_stats(d, cats)
            row["models"].append({
                "name": d["_model_name"],
                "judge_flagged": s["judge_flagged"],
                "judge_run": s["judge_run"],
                "judge_pct": s["judge_pct"],
            })
        # Only include groups where at least one model has a result
        if any(m["judge_run"] > 0 for m in row["models"]):
            group_rows.append(row)

    # Build notable category table (judge_flagged > 0 in any model)
    all_cats = set()
    for d in datasets:
        for cat, s in d["by_category"].items():
            if s["judge_flagged"] > 0:
                all_cats.add(cat)

    cat_rows = []
    for cat in sorted(all_cats):
        row = {"cat": cat, "models": []}
        for d in datasets:
            s = d["by_category"].get(cat, {"judge_flagged": 0, "judge_run": 0, "total": 0})
            row["models"].append({
                "name": d["_model_name"],
                "result": f"{s['judge_flagged']}/{s['judge_run'] if s['judge_run'] else s['total']}",
                "flagged": s["judge_flagged"] > 0,
            })
        cat_rows.append(row)

    # Escape strings for JS template literals
    def esc(s):
        return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    group_analyses_js = ""
    for group, text in commentary.get("group_analyses", {}).items():
        group_analyses_js += f"""
    // Group: {group}
    children.push(new Paragraph({{
        text: {json.dumps(group)},
        heading: HeadingLevel.HEADING_2,
        spacing: {{ before: 240, after: 120 }},
    }}));
    {json.dumps(text)}.split('\\n\\n').forEach(para => {{
        if (para.trim()) children.push(new Paragraph({{
            children: [new TextRun({{ text: para.trim(), size: 20, font: 'Arial' }})],
            spacing: {{ after: 160 }},
        }}));
    }});
"""

    model_verdicts_js = ""
    for model, text in commentary.get("model_verdicts", {}).items():
        model_verdicts_js += f"""
    children.push(new Paragraph({{
        children: [new TextRun({{ text: {json.dumps(model)}, bold: true, size: 20, font: 'Arial', color: '2E75B6' }})],
        spacing: {{ before: 160, after: 80 }},
    }}));
    children.push(new Paragraph({{
        children: [new TextRun({{ text: {json.dumps(text)}, size: 20, font: 'Arial' }})],
        spacing: {{ after: 160 }},
    }}));
"""

    js = f"""
const {{
    Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
    HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
    VerticalAlign, Header, Footer, TabStopType, TabStopPosition, PageNumber,
}} = require('docx');
const fs = require('fs');

// ── Colors ────────────────────────────────────────────────────────────────
const DARK_BLUE  = '1A3A6B';
const MID_BLUE   = '2E75B6';
const LIGHT_BLUE = 'D6E4F0';
const GOLD       = 'F4B942';
const WHITE      = 'FFFFFF';
const LIGHT_GRAY = 'F2F2F2';
const GREEN      = 'E2EFDA';
const RED_LIGHT  = 'FCE4D6';
const DARK_TEXT  = '111111';
const GRAY_TEXT  = '666666';

const border = (color='C8D4E8', sz=4) => ({{ style: BorderStyle.SINGLE, size: sz, color }});
const noBorder = () => ({{ style: BorderStyle.NONE, size: 0, color: 'FFFFFF' }});
const allBorders = (c) => ({{ top: border(c), bottom: border(c), left: border(c), right: border(c) }});
const btmBorder = (c='C8D4E8') => ({{ top: noBorder(), bottom: border(c), left: noBorder(), right: noBorder() }});

function hdrCell(text, width) {{
    return new TableCell({{
        width: {{ size: width, type: WidthType.DXA }},
        shading: {{ fill: DARK_BLUE, type: ShadingType.CLEAR }},
        borders: allBorders(DARK_BLUE),
        margins: {{ top: 80, bottom: 80, left: 120, right: 120 }},
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({{
            alignment: AlignmentType.CENTER,
            children: [new TextRun({{ text, bold: true, color: WHITE, size: 18, font: 'Arial' }})]
        }})]
    }});
}}

function dataCell(text, width, opts={{}}) {{
    const {{ bold=false, align=AlignmentType.LEFT, bg=null, color=DARK_TEXT, borders=null }} = opts;
    return new TableCell({{
        width: {{ size: width, type: WidthType.DXA }},
        shading: bg ? {{ fill: bg, type: ShadingType.CLEAR }} : undefined,
        borders: borders || btmBorder(),
        margins: {{ top: 60, bottom: 60, left: 120, right: 120 }},
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({{
            alignment: align,
            children: [new TextRun({{ text: String(text), bold, color, size: 18, font: 'Arial' }})]
        }})]
    }});
}}

function sectionRule() {{
    return new Paragraph({{
        border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 6, color: MID_BLUE, space: 1 }} }},
        spacing: {{ before: 40, after: 200 }},
        children: []
    }});
}}

function bodyPara(text, opts={{}}) {{
    const {{ bold=false, color=DARK_TEXT, spaceBefore=0, spaceAfter=160, italic=false }} = opts;
    return new Paragraph({{
        spacing: {{ before: spaceBefore, after: spaceAfter }},
        children: [new TextRun({{ text, bold, color, size: 20, font: 'Arial', italics: italic }})]
    }});
}}

function callout(text, bgColor=LIGHT_BLUE) {{
    return new Table({{
        width: {{ size: 9360, type: WidthType.DXA }},
        columnWidths: [9360],
        rows: [new TableRow({{ children: [
            new TableCell({{
                width: {{ size: 9360, type: WidthType.DXA }},
                shading: {{ fill: bgColor, type: ShadingType.CLEAR }},
                borders: {{ top: border(GOLD,8), bottom: border('C8D4E8'), left: {{ style: BorderStyle.SINGLE, size: 16, color: GOLD }}, right: border('C8D4E8') }},
                margins: {{ top: 120, bottom: 120, left: 160, right: 160 }},
                children: [new Paragraph({{
                    children: [new TextRun({{ text, size: 19, font: 'Arial', color: DARK_TEXT }})]
                }})]
            }})
        ]}})]
    }});
}}

const children = [];

// ── Cover / Title block ───────────────────────────────────────────────────
children.push(new Table({{
    width: {{ size: 9360, type: WidthType.DXA }},
    columnWidths: [6500, 2860],
    rows: [new TableRow({{ children: [
        new TableCell({{
            width: {{ size: 6500, type: WidthType.DXA }},
            shading: {{ fill: DARK_BLUE, type: ShadingType.CLEAR }},
            borders: allBorders(DARK_BLUE),
            margins: {{ top: 200, bottom: 200, left: 200, right: 100 }},
            children: [
                new Paragraph({{ children: [new TextRun({{ text: 'BLACK DIAMOND CONSULTING LLC', bold: true, color: WHITE, size: 22, font: 'Arial' }})] }}),
                new Paragraph({{ children: [new TextRun({{ text: 'AI System Security Assessment', color: 'AAC4E0', size: 18, font: 'Arial' }})] }}),
            ]
        }}),
        new TableCell({{
            width: {{ size: 2860, type: WidthType.DXA }},
            shading: {{ fill: DARK_BLUE, type: ShadingType.CLEAR }},
            borders: allBorders(DARK_BLUE),
            margins: {{ top: 200, bottom: 200, left: 100, right: 200 }},
            verticalAlign: VerticalAlign.CENTER,
            children: [new Paragraph({{
                alignment: AlignmentType.RIGHT,
                children: [new TextRun({{ text: 'RESEARCH REPORT', bold: true, color: GOLD, size: 22, font: 'Arial' }})]
            }})]
        }}),
    ]}})]
}}));

children.push(new Paragraph({{ spacing: {{ after: 0 }}, children: [] }}));

children.push(new Paragraph({{
    spacing: {{ before: 320, after: 80 }},
    children: [new TextRun({{ text: {json.dumps(f"Comparative Analysis: {models_str}")}, bold: true, color: DARK_BLUE, size: 36, font: 'Arial' }})]
}}));

children.push(new Paragraph({{
    spacing: {{ before: 0, after: 40 }},
    children: [new TextRun({{ text: 'LLM Security Boundary & Adversarial Robustness Evaluation', color: GRAY_TEXT, size: 22, font: 'Arial' }})]
}}));

children.push(new Paragraph({{
    spacing: {{ before: 0, after: 320 }},
    children: [new TextRun({{ text: {json.dumps(f"Published: {generated_date}  |  blackdiamondconsulting.ai")}, color: GRAY_TEXT, size: 18, font: 'Arial', italics: true }})]
}}));

children.push(sectionRule());

// ── Executive Summary ─────────────────────────────────────────────────────
children.push(new Paragraph({{
    text: 'Executive Summary',
    heading: HeadingLevel.HEADING_1,
    spacing: {{ before: 0, after: 160 }},
}}));

{json.dumps(commentary.get('executive_summary',''))}.split('\\n\\n').forEach(para => {{
    if (para.trim()) children.push(bodyPara(para.trim(), {{ spaceAfter: 180 }}));
}});

children.push(new Paragraph({{ spacing: {{ after: 240 }}, children: [] }}));
children.push(sectionRule());

// ── Top-Line Numbers ──────────────────────────────────────────────────────
children.push(new Paragraph({{
    text: 'Top-Line Numbers',
    heading: HeadingLevel.HEADING_1,
    spacing: {{ before: 0, after: 160 }},
}}));

{json.dumps(commentary.get('topline_analysis',''))}.split('\\n\\n').forEach(para => {{
    if (para.trim()) children.push(bodyPara(para.trim(), {{ spaceAfter: 160 }}));
}});

children.push(new Paragraph({{ spacing: {{ after: 160 }}, children: [] }}));

// Top-line data table
const tlModelCols = {json.dumps([d['_model_name'] for d in datasets])};
const tlColW = Math.floor(7360 / (tlModelCols.length + 1));
const tlFirstW = 9360 - (tlColW * tlModelCols.length);

const tlHeaderRow = new TableRow({{ children: [
    hdrCell('Metric', tlFirstW),
    ...tlModelCols.map(m => hdrCell(m, tlColW))
]}});

const tlData = [
    ['Tests Run', ...{json.dumps([str(d['total']) for d in datasets])}],
    ['Raw Flagged', ...{json.dumps([f"{d['flagged']} ({round(100*d['flagged']/d['total'],1)}%)" for d in datasets])}],
    ['Judge Flagged', ...{json.dumps([f"{d['judge_flagged']} ({round(100*d['judge_flagged']/d['judge_run'],1) if d['judge_run'] else 0}% of judged)" for d in datasets])}],
];

const tlRows = tlData.map((row, i) => new TableRow({{ children: [
    dataCell(row[0], tlFirstW, {{ bold: true, bg: i%2===0 ? LIGHT_GRAY : WHITE }}),
    ...row.slice(1).map((v,j) => dataCell(v, tlColW, {{ align: AlignmentType.CENTER, bg: i%2===0 ? LIGHT_GRAY : WHITE }}))
]}}));

children.push(new Table({{
    width: {{ size: 9360, type: WidthType.DXA }},
    columnWidths: [tlFirstW, ...tlModelCols.map(() => tlColW)],
    rows: [tlHeaderRow, ...tlRows]
}}));

children.push(new Paragraph({{ spacing: {{ after: 320 }}, children: [] }}));
children.push(sectionRule());

// ── Results by Domain ─────────────────────────────────────────────────────
children.push(new Paragraph({{
    text: 'Results by Domain',
    heading: HeadingLevel.HEADING_1,
    spacing: {{ before: 0, after: 160 }},
}}));

// Domain summary table
const grpData = {json.dumps(group_rows)};
const modelNames = {json.dumps(model_names)};
const grpColW = Math.floor(6360 / modelNames.length);
const grpFirstW = 9360 - (grpColW * modelNames.length);

const grpHdr = new TableRow({{ children: [
    hdrCell('Domain', grpFirstW),
    ...modelNames.map(m => hdrCell(m + '\\nJudge Flagged', grpColW))
]}});

const grpRows = grpData.map((row, i) => new TableRow({{ children: [
    dataCell(row.group, grpFirstW, {{ bold: false, bg: i%2===0 ? LIGHT_GRAY : WHITE }}),
    ...row.models.map(m => {{
        const val = m.judge_run > 0 ? `${{m.judge_flagged}}/${{m.judge_run}} (${{m.judge_pct}}%)` : '—';
        const isAlert = m.judge_flagged > 0;
        return dataCell(val, grpColW, {{
            align: AlignmentType.CENTER,
            bg: isAlert ? RED_LIGHT : (i%2===0 ? LIGHT_GRAY : WHITE),
            color: isAlert ? 'B71C1C' : DARK_TEXT,
            bold: isAlert,
        }});
    }})
]}}));

children.push(new Table({{
    width: {{ size: 9360, type: WidthType.DXA }},
    columnWidths: [grpFirstW, ...modelNames.map(() => grpColW)],
    rows: [grpHdr, ...grpRows]
}}));

children.push(new Paragraph({{ spacing: {{ after: 320 }}, children: [] }}));

// Domain narrative analyses
{group_analyses_js}

children.push(sectionRule());

// ── Category-Level Failures ───────────────────────────────────────────────
children.push(new Paragraph({{
    text: 'Category-Level Failures (Judge-Confirmed)',
    heading: HeadingLevel.HEADING_1,
    spacing: {{ before: 0, after: 80 }},
}}));
children.push(bodyPara('Categories where at least one model received a confirmed judge-flagged failure.', {{ color: GRAY_TEXT, spaceAfter: 160, italic: true }}));

const catData = {json.dumps(cat_rows)};
const catColW = Math.floor(6360 / modelNames.length);
const catFirstW = 9360 - (catColW * modelNames.length);

const catHdr = new TableRow({{ children: [
    hdrCell('Category', catFirstW),
    ...modelNames.map(m => hdrCell(m, catColW))
]}});

const catRows2 = catData.map((row, i) => new TableRow({{ children: [
    dataCell(row.cat, catFirstW, {{ bg: i%2===0 ? LIGHT_GRAY : WHITE }}),
    ...row.models.map(m => dataCell(m.result, catColW, {{
        align: AlignmentType.CENTER,
        bg: m.flagged ? RED_LIGHT : (i%2===0 ? LIGHT_GRAY : WHITE),
        color: m.flagged ? 'B71C1C' : DARK_TEXT,
        bold: m.flagged,
    }}))
]}}));

children.push(new Table({{
    width: {{ size: 9360, type: WidthType.DXA }},
    columnWidths: [catFirstW, ...modelNames.map(() => catColW)],
    rows: [catHdr, ...catRows2]
}}));

children.push(new Paragraph({{ spacing: {{ after: 320 }}, children: [] }}));
children.push(sectionRule());

// ── Shared Weaknesses ─────────────────────────────────────────────────────
children.push(new Paragraph({{
    text: 'Shared Weaknesses',
    heading: HeadingLevel.HEADING_1,
    spacing: {{ before: 0, after: 160 }},
}}));

{json.dumps(commentary.get('shared_weaknesses',''))}.split('\\n\\n').forEach(para => {{
    if (para.trim()) children.push(bodyPara(para.trim(), {{ spaceAfter: 160 }}));
}});

children.push(new Paragraph({{ spacing: {{ after: 240 }}, children: [] }}));
children.push(sectionRule());

// ── Model Verdicts ────────────────────────────────────────────────────────
children.push(new Paragraph({{
    text: 'Model Verdicts',
    heading: HeadingLevel.HEADING_1,
    spacing: {{ before: 0, after: 160 }},
}}));

{model_verdicts_js}

children.push(sectionRule());

// ── Enterprise Implications ───────────────────────────────────────────────
children.push(new Paragraph({{
    text: 'Enterprise Deployment Implications',
    heading: HeadingLevel.HEADING_1,
    spacing: {{ before: 0, after: 160 }},
}}));

{json.dumps(commentary.get('enterprise_implications',''))}.split('\\n\\n').forEach(para => {{
    if (para.trim()) children.push(bodyPara(para.trim(), {{ spaceAfter: 160 }}));
}});

children.push(new Paragraph({{ spacing: {{ after: 240 }}, children: [] }}));
children.push(sectionRule());

// ── Methodology ───────────────────────────────────────────────────────────
children.push(new Paragraph({{
    text: 'Methodology',
    heading: HeadingLevel.HEADING_1,
    spacing: {{ before: 0, after: 160 }},
}}));

children.push(bodyPara({json.dumps(commentary.get('methodology_note',''))}, {{ spaceAfter: 160 }}));

children.push(new Paragraph({{ spacing: {{ after: 240 }}, children: [] }}));

// Disclaimer callout
children.push(callout(
    'Disclaimer: This report reflects results from Black Diamond Consulting\\'s proprietary adversarial test suite. Test prompts are not disclosed. Results represent model behavior at the time of testing and may vary across versions, deployments, and system prompt configurations. This report is intended for informational purposes only.',
    'FFF8E1'
));

children.push(new Paragraph({{ spacing: {{ after: 240 }}, children: [] }}));

// ── Footer contact block ──────────────────────────────────────────────────
children.push(new Table({{
    width: {{ size: 9360, type: WidthType.DXA }},
    columnWidths: [9360],
    rows: [new TableRow({{ children: [
        new TableCell({{
            width: {{ size: 9360, type: WidthType.DXA }},
            shading: {{ fill: DARK_BLUE, type: ShadingType.CLEAR }},
            borders: allBorders(DARK_BLUE),
            margins: {{ top: 160, bottom: 160, left: 200, right: 200 }},
            children: [
                new Paragraph({{ alignment: AlignmentType.CENTER, children: [
                    new TextRun({{ text: 'Black Diamond Consulting LLC', bold: true, color: WHITE, size: 20, font: 'Arial' }})
                ]}}),
                new Paragraph({{ alignment: AlignmentType.CENTER, children: [
                    new TextRun({{ text: '11 3rd ST NW #353, Auburn, WA 98071  |  blackdiamondconsulting.ai  |  sean@blackdiamondconsulting.ai', color: 'AAC4E0', size: 17, font: 'Arial' }})
                ]}}),
            ]
        }})
    ]}})]
}}));

// ── Assemble document ─────────────────────────────────────────────────────
const doc = new Document({{
    styles: {{
        default: {{
            document: {{ run: {{ font: 'Arial', size: 20 }} }},
        }},
        paragraphStyles: [
            {{
                id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
                run: {{ size: 28, bold: true, font: 'Arial', color: DARK_BLUE }},
                paragraph: {{ spacing: {{ before: 320, after: 160 }}, outlineLevel: 0 }}
            }},
            {{
                id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
                run: {{ size: 22, bold: true, font: 'Arial', color: MID_BLUE }},
                paragraph: {{ spacing: {{ before: 240, after: 120 }}, outlineLevel: 1 }}
            }},
        ]
    }},
    sections: [{{
        properties: {{
            page: {{
                size: {{ width: 12240, height: 15840 }},
                margin: {{ top: 1080, right: 1080, bottom: 1080, left: 1080 }},
            }}
        }},
        headers: {{
            default: new Header({{ children: [
                new Paragraph({{
                    tabStops: [{{ type: TabStopType.RIGHT, position: 9360 }}],
                    border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 4, color: MID_BLUE, space: 1 }} }},
                    spacing: {{ after: 0 }},
                    children: [
                        new TextRun({{ text: 'Black Diamond Consulting LLC — LLM Security Research', size: 16, font: 'Arial', color: GRAY_TEXT }}),
                        new TextRun({{ text: '\\t', size: 16, font: 'Arial' }}),
                        new TextRun({{ text: ' ' + {json.dumps(generated_date)}, size: 16, font: 'Arial', color: GRAY_TEXT }}),
                    ]
                }})
            ]}})
        }},
        footers: {{
            default: new Footer({{ children: [
                new Paragraph({{
                    tabStops: [{{ type: TabStopType.RIGHT, position: 9360 }}],
                    border: {{ top: {{ style: BorderStyle.SINGLE, size: 4, color: MID_BLUE, space: 1 }} }},
                    spacing: {{ before: 0 }},
                    children: [
                        new TextRun({{ text: 'CONFIDENTIAL — Black Diamond Consulting LLC', size: 16, font: 'Arial', color: GRAY_TEXT }}),
                        new TextRun({{ text: '\\t  Page ', size: 16, font: 'Arial', color: GRAY_TEXT }}),
                        new TextRun({{ children: [PageNumber.CURRENT], size: 16, font: 'Arial', color: GRAY_TEXT }}),
                    ]
                }})
            ]}})
        }},
        children
    }}]
}});

Packer.toBuffer(doc).then(buf => {{
    fs.writeFileSync({json.dumps(output_path)}, buf);
    console.log('Written:', {json.dumps(output_path)});
}}).catch(err => {{
    console.error('Error:', err);
    process.exit(1);
}});
"""
    return js


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_llm_report.py file1.json file2.json [file3.json ...]")
        sys.exit(1)

    json_files = sys.argv[1:]
    print(f"Loading {len(json_files)} result file(s)...")
    datasets = [load_json(f) for f in json_files]
    model_names = [d["_model_name"] for d in datasets]
    print(f"Models: {', '.join(model_names)}")

    print("Building data summary...")
    summary = build_summary_dict(datasets)

    print("Calling Claude API for commentary...")
    commentary = call_claude_api(summary, model_names)
    print("Commentary generated.")

    # Output filename
    date_str = datetime.now().strftime("%Y%m%d")
    safe_models = "_vs_".join(m.replace("/", "-").replace(" ", "-") for m in model_names)
    output_path = os.path.join(
        os.getcwd(),
        f"BDC_LLM_Report_{date_str}.docx"
    )

    print("Generating docx...")
    js_code = build_js(datasets, commentary, output_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    with tempfile.NamedTemporaryFile(
        suffix=".js", mode="w", encoding="utf-8", delete=False,
        dir=script_dir,
    ) as tf:
        tf.write(js_code)
        tf_path = tf.name

    try:
        result = subprocess.run(
            ["node", tf_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("Node error:", result.stderr)
            sys.exit(1)
        print(result.stdout.strip())
    finally:
        os.unlink(tf_path)

    print(f"\nDone! Report saved to:\n  {output_path}")


if __name__ == "__main__":
    main()
