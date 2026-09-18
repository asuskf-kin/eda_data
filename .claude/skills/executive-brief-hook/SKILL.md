---
name: executive-brief-hook
description: Draft concise, high-impact executive reports and management briefs featuring an attention-grabbing hook (monetary risk, commercial upside, or critical operational anomaly), with export capabilities to standalone styled HTML and executive PDF. Use when creating executive memos, C-level summaries, management status reports, board updates, flash reports, or one-page decision papers.
---

# Executive Brief Hook

A framework for drafting concise, high-impact executive reports and management briefings that
capture executive attention in the first five seconds, drive rapid strategic decisions, and
deliver publication-ready HTML and PDF artifacts.

## When to Use

- C-level updates, board summaries, or steering committee briefs.
- Critical project alerts, operational bottlenecks, or urgent risks.
- Flash performance reports (weekly/monthly trading, growth, cost overruns).
- Strategic proposals or resource requests requiring rapid executive sign-off.
- Exporting executive briefs into self-contained responsive HTML or printable 1-page PDF memos.

## Core Content Workflow

### 1. Engineer the Executive Hook

Every brief opens with an unambiguous hook answering: **"Why must leadership care right now?"**
Pick one of three high-stakes anchors:

- **Financial Exposure Hook** — direct monetary risk, revenue loss, or capital leakage.
  *"$1.2M in margin erosion at risk over Q4 if pricing drift continues."*
- **Growth / Arbitrage Hook** — immediate, high-probability commercial upside or competitive window.
  *"Capturing 15% unmet category demand represents an incremental $450K ARR within 60 days."*
- **Anomaly / Cliff Hook** — an operational metric breaking historical thresholds.
  *"Churn in tier-1 accounts spiked 3.4x after the v2.4 rollout, threatening retention covenants."*

Keep the hook under 2 sentences. Place it at the absolute top as a high-contrast banner.

### 2. State the BLUF (Bottom Line Up Front)

Executives read for decisions, not methodology.

- Deliver the core conclusion and recommended action immediately after the hook.
- Structure: situation, impact, primary recommendation — 40 words or fewer.
- Never open with background context, meeting history, or team effort. Start with the outcome.

### 3. Build the Driver Bridge (3-Bullet Evidence Core)

Only the causal mechanics. Exactly three bullets:

- **Metric Delta** — the quantitative movement vs. target or baseline.
- **Root Cause** — the primary underlying driver, verified by data.
- **Business Toll** — the downstream operational or financial consequence.

### 4. Present the Decision Fork

Executives need structured choices, not open-ended dilemmas:

- **Option A (Recommended)** — proactive intervention with cost, time to impact, and upside.
- **Option B (Contingency)** — lower investment or alternative route, with its compromises.
- **Option C (Status Quo)** — the cost of inaction, tied directly back to the opening hook.

### 5. Define the Decision Gate

Close with a crisp, unambiguous ask:

- The exact decision required (*"Approval of $45K emergency vendor budget"*).
- The decision deadline (*"Sign-off by Friday, 3:00 PM EST to hit sprint cutoff"*).
- A single point of ownership (*"Lead: Operations Director"*).

## Output Formats

Generate the format the user asked for. Default to Markdown in-chat; produce HTML or PDF on request.

### Format A: Markdown

```markdown
# [URGENT / STRATEGIC BRIEF]: [Subject Line]

> **THE HOOK**: [Quantified monetary, operational, or strategic stake in 1-2 sentences]

### Bottom Line Up Front (BLUF)
[Core verdict, current reality, and explicit recommendation in 2 sentences]

---

### Core Drivers & Evidence
* **Performance Delta**: [Metric vs Target / Prior Period]
* **Verified Cause**: [Primary driver identified through data]
* **Downstream Impact**: [Operational or financial implication]

---

### Decision Matrix
| Option | Action | Cost / Time | Expected Outcome | Risk |
| :--- | :--- | :--- | :--- | :--- |
| **Option 1 (Recommended)** | [Summary] | [Budget/Days] | [Net Gain] | [Mitigation] |
| **Option 2 (Alternative)** | [Summary] | [Budget/Days] | [Net Gain] | [Mitigation] |
| **Status Quo** | Do nothing | $0 / Immediate | [Loss from Hook] | Uncontrolled downside |

---

### Action Required
* **Decision**: [Specific approval requested]
* **Deadline**: [Date and time]
* **Next Step**: [Execution plan immediately upon sign-off]
```

### Format B: Standalone Styled HTML

Copy `assets/brief_template.html` and replace its placeholder content. This project's copy is
written in Spanish (`lang="es"`; EL GANCHO / Conclusion principal (BLUF) / Causas y evidencia /
Matriz de decision / Decision solicitada) - keep briefs in Spanish unless the user asks otherwise.
It is self-contained (embedded CSS, no external assets) and already implements:

- Max width 800px, centered, system font stack.
- Status pill — `#fee2e2`/`#b91c1c` for critical risk, `#dcfce7`/`#15803d` for opportunity.
- Hook callout — 4px solid left border (`#e11d48` risk / `#2563eb` opportunity) on `#f8fafc`.
- BLUF box, decision table with `#f1f5f9` header and a "Recommended" badge on Option 1.
- Action box with highlighted deadline badge and named owner.
- Print stylesheet: `@page { size: letter portrait; margin: 15mm; }`, shadows suppressed,
  `page-break-inside: avoid` on every block, responsive down to mobile widths.

### Format C: Executive 1-Page PDF (ReportLab)

Run `scripts/generate_brief_pdf.py` with a JSON payload — no need to rewrite the layout code:

```bash
python ~/.claude/skills/executive-brief-hook/scripts/generate_brief_pdf.py brief.json out.pdf
```

The JSON schema is documented at the top of the script. Its section headings are hardcoded in
English (THE HOOK / BLUF / CORE DRIVERS / DECISION MATRIX / ACTION REQUIRED); translate them in
`generate_pdf` when the PDF must match the Spanish HTML. Page setup is Letter with 36pt margins;
palette is corporate slate (`#1e293b`), body charcoal (`#334155`), alert crimson (`#b91c1c`),
accent navy (`#0284c7`). Requires `reportlab` (`pip install reportlab`).

Only hand-write a ReportLab script when the brief needs a layout the template cannot express.

## Gotchas

- **Spilling past one page.** Briefs lose force on page 2. Hold to 300-450 words and tight margins.
- **Unstyled raw HTML.** Always embed the CSS and print breakpoints; never ship bare markup.
- **Burying the lead.** Background, methodology, or timeline before the core message loses the
  reader within 10 seconds.
- **Soft, unquantified hooks.** "Performance is suboptimal" creates no urgency. Use currency,
  percentages, or concrete operational counts.
- **Problems without solutions.** Executives evaluate proposals, not complaints. Always bring
  resolution pathways with explicit trade-offs.
