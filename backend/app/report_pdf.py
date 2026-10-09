"""Generate a printable evidence report from a completed verification report."""

from __future__ import annotations

import html
import io
import math
from datetime import datetime
from typing import Any
from uuid import UUID

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


INK = colors.HexColor("#172321")
GREEN = colors.HexColor("#245b43")
MUTED = colors.HexColor("#5f6f66")
PALE = colors.HexColor("#f1f5f1")
GRID = colors.HexColor("#d5ddd7")


def _text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric(value: Any) -> str:
    number = _number(value)
    return f"{number:.4f}" if number is not None else "Not available"


def _status_color(status: Any) -> colors.Color:
    return {
        "passed": colors.HexColor("#176b3a"),
        "failed": colors.HexColor("#a62e2e"),
        "review": colors.HexColor("#925b08"),
        "unavailable": MUTED,
    }.get(str(status).lower(), MUTED)


def _challenge_verdict(report: dict[str, Any]) -> tuple[str, str, colors.Color, colors.Color]:
    decision = report.get("decision")
    if decision == "review":
        return (
            "PASS - REQUIRED CHALLENGE COMPLETED",
            "The randomized QR sequence and required target path were completed. This is only a challenge pass; "
            "it is not identity verification, KYC approval, or a finding that the applicant is authentic. "
            "Research screening signals still require human review.",
            colors.HexColor("#176b3a"),
            colors.HexColor("#eaf4ed"),
        )
    if decision == "challenge_failed":
        return (
            "FAIL - REQUIRED CHALLENGE NOT COMPLETED",
            "The required randomized QR sequence and/or target path was not completed. This is a challenge failure, "
            "not an identity or fraud determination.",
            colors.HexColor("#a62e2e"),
            colors.HexColor("#fbeeee"),
        )
    return (
        "INCONCLUSIVE - NO PASS/FAIL RESULT",
        "The challenge outcome could not be determined from the available evidence. A skipped or unavailable check "
        "is not a pass; this result is not KYC approval or rejection.",
        colors.HexColor("#925b08"),
        colors.HexColor("#fff5dd"),
    )


def _analysis_rows(checks: dict[str, Any]) -> list[Any]:
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "AnalysisBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=INK,
        spaceAfter=3,
    )
    small = ParagraphStyle(
        "AnalysisSmall",
        parent=body,
        fontSize=7.5,
        leading=10,
        textColor=MUTED,
    )
    heading = ParagraphStyle(
        "AnalysisHeading",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=GREEN,
        spaceBefore=5,
        spaceAfter=3,
    )
    rows: list[Any] = []

    face = checks.get("face_deepfake_analysis") or {}
    face_evidence = face.get("evidence") or {}
    median = _number(face_evidence.get("median_fake_score"))
    aggregate = _number(face_evidence.get("aggregate_fake_score"))
    spread = _number(face_evidence.get("score_range"))
    threshold = _number(face_evidence.get("model_threshold"))
    model_verdict = _text(face_evidence.get("model_verdict", "INCONCLUSIVE"))
    temporal = checks.get("video_temporal_consistency") or {}
    temporal_evidence = temporal.get("evidence") or {}

    rows.append(Paragraph("Video screening signals", heading))
    if median is None:
        rows.append(Paragraph(
            "No face-classifier score summary is available. The video screening may have been unavailable, skipped, or lacked enough face-bearing samples.",
            body,
        ))
    else:
        rows.append(Paragraph(
            f"Experimental video-screening consensus: <b>{model_verdict}</b>; "
            f"aggregate AI-like score: <b>{_metric(aggregate)}</b> "
            f"({face_evidence.get('flagged_frames', '—')} frame(s) at or above {_metric(threshold)}). "
            f"Median AI-like classifier score: <b>{median:.4f}</b>. "
            f"Observed score spread (maximum minus minimum): <b>{_metric(spread)}</b>. "
            "These uncalibrated research scores are not identity or authenticity determinations.",
            body,
        ))
        frames = face_evidence.get("frames") or []
        if frames:
            frame_rows = [[
                Paragraph("<b>Elapsed</b>", small),
                Paragraph("<b>AI-like score</b>", small),
                Paragraph("<b>Real-like score</b>", small),
                Paragraph("<b>Threshold signal</b>", small),
            ]]
            for frame in frames[:8]:
                fake_score = _number(frame.get("fake_score"))
                real_score = _number(frame.get("real_score"))
                if fake_score is None or real_score is None:
                    continue
                frame_threshold = _number(frame.get("threshold")) or threshold or 0.38
                lean = "AI-like" if fake_score >= frame_threshold else "Real-like"
                frame_rows.append([
                    Paragraph(f"{_text(frame.get('elapsed_ms', '—'))} ms", small),
                    Paragraph(f"{fake_score:.4f}", small),
                    Paragraph(f"{real_score:.4f}", small),
                    Paragraph(lean, small),
                ])
            if len(frame_rows) > 1:
                frame_table = Table(frame_rows, colWidths=[34 * mm, 38 * mm, 38 * mm, 45 * mm], repeatRows=1)
                frame_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), PALE),
                    ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                rows.append(Paragraph(
                    "Highest-scoring retained frame samples (not a full-video record):",
                    small,
                ))
                rows.append(frame_table)
        else:
            rows.append(Paragraph("No per-frame score samples were retained in the report.", small))

    rows.append(Paragraph("Cross-frame comparison", heading))
    consistency = temporal_evidence.get("consistency")
    frame_count = temporal_evidence.get("frames_sampled")
    face_frame_count = temporal_evidence.get("frames_with_faces")
    if consistency is None:
        rows.append(Paragraph(
            f"{_text(temporal.get('detail', 'Temporal comparison was unavailable.'))} "
            "This prototype checks basic score/face-count variation; it does not run a trained temporal authenticity model.",
            body,
        ))
    else:
        rows.append(Paragraph(
            f"Recorded consistency signal: <b>{_text(consistency)}</b>; "
            f"{_text(face_frame_count if face_frame_count is not None else '—')} face-bearing samples out of "
            f"{_text(frame_count if frame_count is not None else '—')} sampled frames. "
            "This is a basic consistency check, not a trained temporal authenticity model.",
            body,
        ))

    audio = checks.get("audio_spoof_detection") or {}
    audio_evidence = audio.get("evidence") or {}
    phrase = checks.get("random_phrase_verification") or {}
    phrase_evidence = phrase.get("evidence") or {}
    quality = checks.get("audio_capture_quality") or {}
    quality_evidence = quality.get("evidence") or {}
    rows.append(Paragraph("Optional audio screening", heading))
    ai_score = _number(audio_evidence.get("ai_voice_score"))
    human_score = _number(audio_evidence.get("human_voice_score"))
    if ai_score is not None and human_score is not None:
        rows.append(Paragraph(
            f"AI-like voice score: <b>{ai_score:.4f}</b>; human-like voice score: <b>{human_score:.4f}</b>. "
            "These research-model scores are uncalibrated and do not verify speaker identity or prove that speech is live.",
            body,
        ))
    else:
        rows.append(Paragraph("No audio classifier score is available; audio may not have been submitted.", body))
    similarity = _number(phrase_evidence.get("similarity"))
    if similarity is not None:
        rows.append(Paragraph(
            f"Fresh-phrase similarity: <b>{similarity:.0%}</b> ({_text(phrase.get('status', 'unavailable'))}). "
            "The report omits the transcript and the audio recording.",
            body,
        ))
    else:
        rows.append(Paragraph("Fresh-phrase comparison was unavailable or no consented recording was submitted.", small))
    duration = _number(quality_evidence.get("duration_seconds"))
    if duration is not None:
        rows.append(Paragraph(
            f"Audio capture quality: {_text(quality.get('status', 'unavailable'))}; "
            f"duration {duration:.2f} seconds. No audio sample is included.",
            small,
        ))
    return rows


def generate_report_pdf(report: dict[str, Any], session_id: UUID | str) -> bytes:
    """Return a PDF with the report, derived metrics, and explicit model caveats."""
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Frame Verification Evidence Report",
        author="Frame",
        pageCompression=0,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=TA_LEFT,
        textColor=INK,
        spaceAfter=3,
    )
    subtitle = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        alignment=TA_RIGHT,
        textColor=MUTED,
    )
    section = ParagraphStyle(
        "ReportSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=GREEN,
        spaceBefore=10,
        spaceAfter=5,
    )
    body = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=INK,
        spaceAfter=4,
    )
    small = ParagraphStyle(
        "ReportSmall",
        parent=body,
        fontSize=7.5,
        leading=10,
        textColor=MUTED,
    )
    table_body = ParagraphStyle(
        "ReportTableBody",
        parent=body,
        fontSize=7.5,
        leading=10,
        spaceAfter=0,
    )
    verdict, verdict_detail, verdict_color, verdict_background = _challenge_verdict(report)
    verdict_title = ParagraphStyle(
        "ReportVerdictTitle",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=verdict_color,
        spaceAfter=3,
    )
    verdict_body = ParagraphStyle(
        "ReportVerdictBody",
        parent=body,
        fontSize=8,
        leading=11,
        spaceAfter=0,
    )

    generated_at = report.get("generated_at")
    if isinstance(generated_at, datetime):
        generated_text = generated_at.isoformat()
    else:
        generated_text = str(generated_at or "Not recorded")
    story: list[Any] = [
        Table(
            [[
                Paragraph("FRAME / CHECK", title),
                Paragraph("VERIFICATION EVIDENCE REPORT<br/>UNOFFICIAL DEMONSTRATION", subtitle),
            ]],
            colWidths=[95 * mm, 75 * mm],
            style=TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]),
        ),
        HRFlowable(width="100%", thickness=1, color=GREEN, spaceAfter=8),
        Paragraph(
            f"<b>Session:</b> {_text(session_id)}"
            f"&nbsp;&nbsp;&nbsp; <b>Generated:</b> {_text(generated_text)}",
            body,
        ),
        Paragraph("Challenge outcome", section),
        Table(
            [
                [Paragraph(_text(verdict), verdict_title)],
                [Paragraph(_text(verdict_detail), verdict_body)],
            ],
            colWidths=[170 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), verdict_background),
                ("BOX", (0, 0), (-1, -1), 0.8, verdict_color),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]),
        ),
    ]

    checks = report.get("checks") or {}
    check_rows = [[
        Paragraph("<b>Check</b>", table_body),
        Paragraph("<b>Status</b>", table_body),
        Paragraph("<b>Recorded detail</b>", table_body),
    ]]
    for key, value in checks.items():
        if not isinstance(value, dict):
            continue
        status = str(value.get("status", "unavailable")).lower()
        status_style = ParagraphStyle(
            f"Status-{len(check_rows)}",
            parent=table_body,
            fontName="Helvetica-Bold",
            textColor=_status_color(status),
        )
        check_rows.append([
            Paragraph(_text(str(key).replace("_", " ").title()), table_body),
            Paragraph(_text(status.upper()), status_style),
            Paragraph(_text(value.get("detail", "")), table_body),
        ])
    check_table = Table(check_rows, colWidths=[49 * mm, 25 * mm, 96 * mm], repeatRows=1)
    check_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([Paragraph("Evidence checks", section), check_table])
    story.extend([Paragraph("Analysis of recorded screening signals", section), *_analysis_rows(checks)])

    limitations = report.get("limitations") or []
    if limitations:
        story.append(Paragraph("Limitations", section))
        for limitation in limitations:
            story.append(Paragraph(f"• {_text(limitation)}", small))

    audit = report.get("audit") or {}
    story.append(Paragraph("Audit metadata", section))
    story.append(KeepTogether([
        Paragraph(f"<b>System:</b> {_text(audit.get('system', 'Not recorded'))}", small),
        Paragraph(f"<b>Sequence:</b> {_text(audit.get('sequence', 'Not recorded'))}", small),
        Paragraph(f"<b>Evidence hash:</b> {_text(audit.get('evidence_hash', 'Not recorded'))}", small),
        Paragraph(f"<b>Record hash:</b> {_text(audit.get('record_hash', 'Not recorded'))}", small),
        Spacer(1, 4),
        Paragraph(
            "No raw video, face image, audio sample, or speech transcript is embedded in this report. "
            "The in-memory hash-chain ledger is a demo simulator, not a blockchain.",
            small,
        ),
    ]))

    document.build(story)
    return buffer.getvalue()
