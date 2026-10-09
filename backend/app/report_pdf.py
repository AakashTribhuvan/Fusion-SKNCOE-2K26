"""PDF generation service for Frame / Check verification reports.

Generates an executive, audit-ready verification PDF report with
deepfake screening metrics and explainability breakdown.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any
from uuid import UUID

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total page count dynamically for page numbers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int) -> None:
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running footer
        footer_text = f"FRAME / CHECK · Mastercard CSB-02 Demo · Page {self._pageNumber} of {page_count}"
        self.drawString(36, 25, footer_text)
        confidential = "CONFIDENTIAL · VERIFICATION EVIDENCE REPORT"
        self.drawRightString(A4[0] - 36, 25, confidential)

        # Thin footer line
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(36, 36, A4[0] - 36, 36)
        self.restoreState()


def _format_status_badge(status: str) -> str:
    norm = status.lower().strip()
    if norm == "passed":
        return '<font color="#15803d"><b>[ PASSED ]</b></font>'
    if norm == "review":
        return '<font color="#b45309"><b>[ REVIEW ]</b></font>'
    if norm == "failed":
        return '<font color="#b91c1c"><b>[ FAILED ]</b></font>'
    return '<font color="#64748b"><b>[ UNAVAILABLE ]</b></font>'


def generate_report_pdf(report: dict[str, Any], session_id: UUID | str) -> bytes:
    """Generate a high-resolution, printable PDF verification report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    # Base typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0d5c46"),
        textTransform="uppercase",
    )
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#475569"),
    )
    meta_bold = ParagraphStyle(
        "MetaBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e293b"),
    )
    section_h1 = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=4,
    )
    section_h2 = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0d5c46"),
        spaceBefore=6,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e293b"),
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.white,
    )
    callout_text = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor("#64748b"),
    )

    story: list[Any] = []

    # 1. Header & Organization Meta
    header_data = [
        [
            Paragraph("<b>FRAME / CHECK</b>", title_style),
            Paragraph(
                "<b>MASTERCARD CSB-02 COMPLIANCE DEMO</b><br/>"
                "Deepfake-Resistant Digital Onboarding",
                meta_bold,
            ),
        ],
        [
            Paragraph("DEEPFAKE-RESISTANT IDENTITY SCREENING REPORT", subtitle_style),
            Paragraph("Zero-Knowledge Liveness & Multi-Signal Evidence", meta_style),
        ],
    ]
    header_table = Table(header_data, colWidths=[330, 193])
    header_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
        ])
    )
    story.append(header_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0d5c46"), spaceAfter=8))

    # 2. Session & Audit Identifiers Table
    gen_time = report.get("generated_at") or datetime.utcnow().isoformat()
    if isinstance(gen_time, str):
        try:
            gen_time_fmt = gen_time.replace("T", " ")[:19] + " UTC"
        except Exception:
            gen_time_fmt = gen_time
    else:
        gen_time_fmt = gen_time.strftime("%Y-%m-%d %H:%M:%S UTC")

    audit = report.get("audit") or {}
    record_hash = audit.get("record_hash") or "N/A"
    evidence_hash = audit.get("evidence_hash") or "N/A"
    seq = audit.get("sequence", 0)

    meta_table_data = [
        [
            Paragraph("<b>Session ID:</b>", meta_bold),
            Paragraph(str(session_id), meta_style),
            Paragraph("<b>Generated:</b>", meta_bold),
            Paragraph(gen_time_fmt, meta_style),
        ],
        [
            Paragraph("<b>Ledger Sequence:</b>", meta_bold),
            Paragraph(f"Block #{seq} ({audit.get('system', 'In-memory simulator')})", meta_style),
            Paragraph("<b>Record Hash:</b>", meta_bold),
            Paragraph(f"<font face='Courier'>{record_hash[:24]}...</font>", meta_style),
        ],
    ]
    meta_table = Table(meta_table_data, colWidths=[90, 200, 80, 153])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # 3. Executive Verdict Banner
    decision = str(report.get("decision", "review")).lower()
    if decision == "review":
        dec_title = "DECISION: REVIEW REQUIRED"
        dec_color = colors.HexColor("#92400e")
        dec_bg = colors.HexColor("#fef3c7")
        dec_border = colors.HexColor("#f59e0b")
        dec_desc = (
            "The participant completed the physical interactive challenge. Visual/acoustic screening signals "
            "are uncalibrated research models and require human compliance evaluation before onboarding approval."
        )
    elif decision == "challenge_failed":
        dec_title = "DECISION: CHALLENGE FAILED"
        dec_color = colors.HexColor("#991b1b")
        dec_bg = colors.HexColor("#fee2e2")
        dec_border = colors.HexColor("#ef4444")
        dec_desc = "The required randomized QR sequence or perimeter target motion was incomplete, replayed, or out of order."
    else:
        dec_title = "DECISION: INCONCLUSIVE"
        dec_color = colors.HexColor("#334155")
        dec_bg = colors.HexColor("#f1f5f9")
        dec_border = colors.HexColor("#94a3b8")
        dec_desc = "Evidence was insufficient or an administrative override was recorded. Verification cannot be confirmed."

    verdict_style = ParagraphStyle(
        "VerdictTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=dec_color,
    )
    verdict_desc_style = ParagraphStyle(
        "VerdictDesc",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
    )
    verdict_data = [
        [Paragraph(f"<b>{dec_title}</b>", verdict_style)],
        [Paragraph(dec_desc, verdict_desc_style)],
    ]
    verdict_table = Table(verdict_data, colWidths=[523])
    verdict_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), dec_bg),
            ("BOX", (0, 0), (-1, -1), 1.5, dec_border),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ])
    )
    story.append(verdict_table)
    story.append(Spacer(1, 10))

    # 4. SECTION: "Why AI-Generated / Screening Findings Breakdown"
    story.append(Paragraph("<b>1. Detailed AI Screening Findings & Explainability</b>", section_h1))
    story.append(Paragraph(
        "This breakdown explains the neural classification, feature abnormalities, and temporal signals observed during analysis.",
        body_style,
    ))
    story.append(Spacer(1, 4))

    checks = report.get("checks", {})
    face_check = checks.get("face_deepfake_analysis", {})
    face_ev = face_check.get("evidence") or {}
    median_fake = face_ev.get("median_fake_score")
    score_range = face_ev.get("score_range")
    frames_ev = face_ev.get("frames", [])

    temporal_check = checks.get("video_temporal_consistency", {})
    temporal_ev = temporal_check.get("evidence") or {}
    temporal_consistency = temporal_ev.get("consistency", "unavailable")

    # Diagnostic reasoning text
    why_visual_bullets: list[str] = []
    if median_fake is not None:
        if median_fake > 0.50:
            why_visual_bullets.append(
                f"<b>Visual Synthetic Lean (Median AI Score: {median_fake:.4f}):</b> Scored above the 0.50 classification boundary. "
                "The image classifier identified feature distributions consistent with synthetic generative blending, "
                "diffusion noise residues, or boundary interpolation along facial contours."
            )
        else:
            why_visual_bullets.append(
                f"<b>Visual Real Lean (Median AI Score: {median_fake:.4f}):</b> Scored below the 0.50 synthetic boundary. "
                "Surface skin micro-textures, specular highlights, and natural lighting gradients align with genuine camera sensors."
            )

    if score_range is not None:
        if score_range >= 0.55:
            why_visual_bullets.append(
                f"<b>High Temporal Volatility (Range Delta: {score_range:.4f}):</b> Large score dispersion across consecutive frames "
                "(range &ge; 0.55) indicates flickering or face-swap seam instability, common in real-time generative video injection."
            )
        else:
            why_visual_bullets.append(
                f"<b>Stable Cross-Frame Variance (Range Delta: {score_range:.4f}):</b> Score progression remained stable within expected bounds, "
                "indicating no severe sudden boundary collapse."
            )

    if temporal_consistency == "review":
        why_visual_bullets.append(
            "<b>Temporal Continuity Signal: REVIEW:</b> Observed frame-to-frame shifts or face-count irregularities during live capture."
        )
    elif temporal_consistency == "no_large_shift_observed":
        why_visual_bullets.append(
            "<b>Temporal Continuity Signal: NORMAL:</b> No abrupt multi-face shifts or severe score deviations between adjacent half-second samples."
        )

    if not why_visual_bullets:
        why_visual_bullets.append("Visual screening metrics were not available or were skipped via administrative override.")

    # Render Visual Findings Card
    visual_card_content: list[list[Any]] = [
        [Paragraph("<b>VISUAL DEEPFAKE MODEL DIAGNOSTICS (dima806/deepfake_vs_real_image_detection)</b>", section_h2)],
        [Paragraph("<br/>".join(f"• {b}" for b in why_visual_bullets), callout_text)],
    ]
    visual_card_table = Table(visual_card_content, colWidths=[523])
    visual_card_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(visual_card_table)
    story.append(Spacer(1, 6))

    # Top Flagged Anomaly Frames Table (if available)
    if frames_ev:
        frame_table_rows = [
            [
                Paragraph("<b>Sample (ms)</b>", table_header),
                Paragraph("<b>AI Synthetic Score</b>", table_header),
                Paragraph("<b>Real Score</b>", table_header),
                Paragraph("<b>Classification</b>", table_header),
                Paragraph("<b>Bounding Box (L,T,W,H)</b>", table_header),
            ]
        ]
        for f in frames_ev[:6]:
            elapsed = f.get("elapsed_ms", 0)
            fake_s = f.get("fake_score", 0.0)
            real_s = f.get("real_score", 0.0)
            lean = f.get("model_lean", "AI-like" if fake_s > real_s else "real-like")
            bbox = f.get("bbox") or {}
            bbox_str = f"[{bbox.get('x',0):.2f}, {bbox.get('y',0):.2f}, {bbox.get('w',0):.2f}, {bbox.get('h',0):.2f}]"
            lean_color = "#b91c1c" if lean == "AI-like" else "#15803d"
            frame_table_rows.append([
                Paragraph(f"{elapsed} ms", table_cell),
                Paragraph(f"{fake_s:.4f}", table_cell_bold),
                Paragraph(f"{real_s:.4f}", table_cell),
                Paragraph(f"<font color='{lean_color}'><b>{lean}</b></font>", table_cell),
                Paragraph(f"<font size='7'>{bbox_str}</font>", table_cell),
            ])
        frame_table = Table(frame_table_rows, colWidths=[80, 100, 85, 110, 148])
        frame_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d5c46")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ])
        )
        story.append(Paragraph("<b>Top Face-Bearing Scored Frame Samples:</b>", meta_bold))
        story.append(Spacer(1, 2))
        story.append(frame_table)
        story.append(Spacer(1, 6))

    # Voice / Audio Screening Diagnostics
    audio_check = checks.get("audio_spoof_detection", {})
    phrase_check = checks.get("random_phrase_verification", {})
    audio_ev = audio_check.get("evidence") or {}
    ai_voice = audio_ev.get("ai_voice_score")
    human_voice = audio_ev.get("human_voice_score")
    phrase_match = phrase_check.get("status")

    why_audio_bullets: list[str] = []
    if ai_voice is not None and human_voice is not None:
        if ai_voice > human_voice:
            why_audio_bullets.append(
                f"<b>Synthetic Voice Flag (AI Score: {ai_voice:.4f} vs Human: {human_voice:.4f}):</b> Acoustic classifier "
                "identified spectral hallmarks of text-to-speech vocoders, voice conversion models, or phase flattening."
            )
        else:
            why_audio_bullets.append(
                f"<b>Natural Voice Cleared (Human Score: {human_voice:.4f} vs AI: {ai_voice:.4f}):</b> Formant dispersion, room reverberation, "
                "and vocal tract dynamics match genuine human speech recording."
            )
    else:
        why_audio_bullets.append("Voice check was skipped or not consented by the user (optional step).")

    if phrase_match == "passed":
        why_audio_bullets.append("<b>Spoken Phrase Match: PASSED:</b> Whisper transcribed the fresh spoken prompt without replay delay.")
    elif phrase_match == "failed":
        why_audio_bullets.append("<b>Spoken Phrase Match: FAILED:</b> Spoken audio did not match the fresh challenge phrase prompt.")

    audio_card_content: list[list[Any]] = [
        [Paragraph("<b>AUDIO & SYNTHETIC VOICE SCREENING (Wav2Vec2 + Whisper tiny.en)</b>", section_h2)],
        [Paragraph("<br/>".join(f"• {b}" for b in why_audio_bullets), callout_text)],
    ]
    audio_card_table = Table(audio_card_content, colWidths=[523])
    audio_card_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(audio_card_table)
    story.append(Spacer(1, 10))

    # 5. SECTION: Full Verification Signal Matrix Table
    story.append(KeepTogether([
        Paragraph("<b>2. Complete Verification Evidence Matrix</b>", section_h1),
        Paragraph("Comprehensive multi-factor inspection results recorded across all channels.", body_style),
        Spacer(1, 4),
    ]))

    label_map = {
        "phone_pairing": "Phone Pairing & Encryption",
        "phone_user_verification": "Platform Authenticator (WebAuthn)",
        "live_face_presence": "Live Face Presence Gating",
        "randomized_qr_sequence": "Randomized QR Challenge (6 Codes)",
        "phone_motion": "Perimeter Motion Assessment",
        "challenge_path_coherence": "Ordered Square Target Traversal (1-6)",
        "face_deepfake_analysis": "Live Face / Visual Deepfake Screening",
        "video_temporal_consistency": "Cross-Frame Video Consistency",
        "video_screening_override": "Administrator Video Override",
        "speaker_verification": "Speaker Biometric Matching (Exluded)",
        "random_phrase_verification": "Fresh Spoken Phrase Verification",
        "audio_spoof_detection": "Acoustic Anti-Spoof Detection",
        "capture_quality": "Sensor & Environmental Capture Quality",
    }

    matrix_rows = [
        [
            Paragraph("<b>Verification Check</b>", table_header),
            Paragraph("<b>Status</b>", table_header),
            Paragraph("<b>Evidentiary Details & Observations</b>", table_header),
        ]
    ]

    for key, label in label_map.items():
        if key not in checks:
            continue
        c = checks[key]
        status = c.get("status", "unavailable")
        detail = c.get("detail", "")
        badge = _format_status_badge(status)
        matrix_rows.append([
            Paragraph(f"<b>{label}</b>", table_cell),
            Paragraph(badge, table_cell),
            Paragraph(detail, table_cell),
        ])

    matrix_table = Table(matrix_rows, colWidths=[140, 85, 298])
    matrix_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(matrix_table)
    story.append(Spacer(1, 10))

    # 6. Audit Trail & Cryptographic Proof
    audit_story: list[Any] = [
        Paragraph("<b>3. Cryptographic Audit Chain & Integrity Proof</b>", section_h1),
        Paragraph(
            "Every completed examination writes a tamper-evident reference to an append-only in-memory hash chain.",
            body_style,
        ),
        Spacer(1, 4),
    ]
    audit_data = [
        [Paragraph("<b>Audit System:</b>", meta_bold), Paragraph(str(audit.get("system", "In-memory simulator")), meta_style)],
        [Paragraph("<b>Sequence Index:</b>", meta_bold), Paragraph(f"Record #{seq}", meta_style)],
        [Paragraph("<b>Evidence Hash (SHA-256):</b>", meta_bold), Paragraph(f"<font face='Courier' size='7'>{evidence_hash}</font>", meta_style)],
        [Paragraph("<b>Previous Record Hash:</b>", meta_bold), Paragraph(f"<font face='Courier' size='7'>{audit.get('previous_hash', 'N/A')}</font>", meta_style)],
        [Paragraph("<b>Current Record Hash:</b>", meta_bold), Paragraph(f"<font face='Courier' size='7'>{record_hash}</font>", meta_style)],
    ]
    audit_table = Table(audit_data, colWidths=[140, 383])
    audit_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    audit_story.append(audit_table)
    story.append(KeepTogether(audit_story))
    story.append(Spacer(1, 8))

    # 7. Regulatory Disclaimer
    limitations = report.get("limitations", [])
    disclaimer_text = (
        "<b>LEGAL & COMPLIANCE NOTICE:</b> This report is produced by a research-grade prototype built for "
        "Mastercard CSB-02 challenge demonstration. Image and audio classifier scores are advisory signals "
        "and do not constitute an automated legal onboarding decision. Biometric data is never persisted; "
        "camera frames and voice recordings are processed in volatile memory and discarded immediately following analysis. "
        + " ".join(limitations[:3])
    )
    story.append(KeepTogether([
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=5),
        Paragraph(disclaimer_text, disclaimer_style),
    ]))

    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
