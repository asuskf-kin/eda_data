#!/usr/bin/env python3
"""Render a one-page executive brief PDF from a JSON payload.

Usage:
    python generate_brief_pdf.py brief.json out.pdf

JSON schema:
{
  "title": "Q4 Margin Erosion in Mobile Checkout",
  "subtitle": "Prepared for the Exec Committee | 2026-09-17 | Ops Analytics",
  "hook": "$1.2M in margin erosion is at risk over Q4 if pricing drift continues.",
  "tone": "risk",                       # "risk" (crimson) or "opportunity" (navy)
  "bluf": "Two sentences: situation, impact, recommendation.",
  "drivers": [
    {"title": "Performance Delta", "text": "Conversion fell 3.2% -> 2.1% on mobile checkout."},
    {"title": "Verified Cause",    "text": "Latency +850ms after the gateway migration."},
    {"title": "Downstream Impact", "text": "1,400 abandoned carts, $82K unrealized revenue/week."}
  ],
  "table": [
    ["Option", "Action", "Cost / Time", "Expected Outcome", "Risk"],
    ["Option 1 (Recommended)", "...", "$45K / 10d", "...", "..."],
    ["Option 2", "...", "$12K / 30d", "...", "..."],
    ["Status Quo", "Do nothing", "$0 / Immediate", "-$1.2M by Dec 31", "Uncontrolled downside"]
  ],
  "action": {"decision": "Approve $45K emergency vendor budget",
             "deadline": "Fri 3:00 PM EST", "owner": "Operations Director"}
}
"""
import json
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (HRFlowable, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

TONES = {
    "risk":        {"fg": "#991b1b", "bg": "#fef2f2", "border": "#f87171"},
    "opportunity": {"fg": "#155e75", "bg": "#f0f9ff", "border": "#38bdf8"},
}
CONTENT_WIDTH = 540  # letter width (612pt) minus 36pt margins on each side


def generate_pdf(filename, title, hook_text, bluf_text, drivers, table_data, action_data,
                 subtitle=None, tone="risk"):
    palette = TONES.get(tone, TONES["risk"])
    doc = SimpleDocTemplate(filename, pagesize=letter, leftMargin=36, rightMargin=36,
                            topMargin=36, bottomMargin=36, title=title)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16,
                                 leading=20, spaceAfter=0,
                                 textColor=colors.HexColor('#0f172a'))
    sub_style = ParagraphStyle('SubStyle', parent=styles['Normal'], fontSize=8.5, leading=11,
                               textColor=colors.HexColor('#64748b'))
    hook_style = ParagraphStyle('HookStyle', parent=styles['Normal'], fontSize=11, leading=15,
                                textColor=colors.HexColor(palette['fg']),
                                fontName='Helvetica-Bold')
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=9.5, leading=13,
                                textColor=colors.HexColor('#334155'))
    bold_body = ParagraphStyle('BoldBody', parent=body_style, fontName='Helvetica-Bold',
                               textColor=colors.HexColor('#1e293b'))
    cell_style = ParagraphStyle('CellStyle', parent=body_style, fontSize=8.5, leading=11)
    head_style = ParagraphStyle('HeadStyle', parent=cell_style, fontName='Helvetica-Bold',
                                textColor=colors.HexColor('#0f172a'))

    story = []
    story.append(Paragraph(title, title_style))
    if subtitle:
        story.append(Spacer(1, 2))
        story.append(Paragraph(subtitle, sub_style))
    story.append(Spacer(1, 8))

    hook_table = Table([[Paragraph(f"<b>THE HOOK:</b> {hook_text}", hook_style)]],
                       colWidths=[CONTENT_WIDTH])
    hook_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(palette['bg'])),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(palette['border'])),
        ('LINEBEFORE', (0, 0), (0, -1), 4, colors.HexColor(palette['fg'])),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(hook_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>BOTTOM LINE UP FRONT (BLUF)</b>", bold_body))
    story.append(Spacer(1, 3))
    story.append(Paragraph(bluf_text, body_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e1')))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>CORE DRIVERS &amp; EVIDENCE</b>", bold_body))
    story.append(Spacer(1, 3))
    for d in drivers:
        story.append(Paragraph(f"• <b>{d['title']}:</b> {d['text']}", body_style))
        story.append(Spacer(1, 2))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e1')))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>DECISION MATRIX</b>", bold_body))
    story.append(Spacer(1, 4))
    formatted = [[Paragraph(str(cell), head_style if r == 0 else cell_style) for cell in row]
                 for r, row in enumerate(table_data)]
    ncols = len(table_data[0])
    weights = [100, 150, 80, 110, 100] if ncols == 5 else [CONTENT_WIDTH / ncols] * ncols
    scale = CONTENT_WIDTH / sum(weights)
    t = Table(formatted, colWidths=[w * scale for w in weights], repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor('#94a3b8')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    action_table = Table([[Paragraph(
        f"<b>ACTION REQUIRED:</b> {action_data['decision']}<br/>"
        f"<b>DEADLINE:</b> {action_data['deadline']} &nbsp;|&nbsp; "
        f"<b>OWNER:</b> {action_data['owner']}", body_style)]], colWidths=[CONTENT_WIDTH])
    action_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(action_table)

    doc.build(story)
    return filename


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    with open(sys.argv[1], encoding='utf-8') as fh:
        d = json.load(fh)
    out = generate_pdf(sys.argv[2], d['title'], d['hook'], d['bluf'], d['drivers'],
                       d['table'], d['action'], d.get('subtitle'), d.get('tone', 'risk'))
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
