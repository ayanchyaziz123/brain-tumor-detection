import os
from datetime import datetime
from fpdf import FPDF

COLORS = {
    'Glioma':     (220, 38,  38),
    'Meningioma': (217, 119,  6),
    'No Tumor':   (22,  163, 74),
    'Pituitary':  (124, 58,  237),
}

BADGE_COLORS = {
    'HIGH':     (22,  163, 74),
    'MODERATE': (217, 119,  6),
    'LOW':      (220, 38,  38),
}

BADGE_MESSAGES = {
    'HIGH':     'Result is reliable.',
    'MODERATE': 'Consider reviewing with a specialist.',
    'LOW':      'Please consult a radiologist.',
}


def _conf_level(conf_pct: float) -> str:
    if conf_pct >= 85:
        return 'HIGH'
    elif conf_pct >= 60:
        return 'MODERATE'
    return 'LOW'


def generate_report_pdf(data: dict, orig_path: str | None, overlay_path: str | None) -> bytes:
    cls      = data['predicted_class']
    conf_pct = data['confidence'] * 100
    probs    = data['probabilities']
    is_tumor = data['is_tumor']
    ts       = datetime.now().strftime('%B %d, %Y  %H:%M')
    level    = _conf_level(conf_pct)

    cr, cg, cb       = COLORS.get(cls, (99, 99, 255))
    br, bg, bb       = BADGE_COLORS[level]

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    M = pdf.l_margin
    W = pdf.w - 2 * M

    # ── Header ────────────────────────────────────────────────────────────────
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 30, style='F')
    pdf.set_y(8)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, 'Brain Tumor Detection Report', align='C', ln=True)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 6, f'Generated {ts}  ·  Built by Rahman', align='C', ln=True)
    pdf.set_y(38)

    # ── Prediction summary ────────────────────────────────────────────────────
    pdf.set_fill_color(cr, cg, cb)
    pdf.rect(M, pdf.get_y(), 3, 24, style='F')
    pdf.set_x(M + 7)

    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(cr, cg, cb)
    pdf.cell(0, 12, cls, ln=True)

    pdf.set_x(M + 7)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(55, 7, f'Confidence: {conf_pct:.1f}%')

    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(br, bg, bb)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(40, 6, f'  {level} CONFIDENCE  ', fill=True, ln=True)

    pdf.set_x(M + 7)
    flag_r, flag_g, flag_b = (220, 38, 38) if is_tumor else (22, 163, 74)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(flag_r, flag_g, flag_b)
    pdf.cell(0, 6, 'TUMOR DETECTED' if is_tumor else 'NO TUMOR DETECTED', ln=True)
    pdf.ln(4)

    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, BADGE_MESSAGES[level], ln=True)
    pdf.ln(7)

    # ── Divider helper ────────────────────────────────────────────────────────
    def section(title: str):
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, title, ln=True)
        pdf.set_draw_color(226, 232, 240)
        pdf.set_line_width(0.3)
        pdf.line(M, pdf.get_y(), M + W, pdf.get_y())
        pdf.ln(4)

    # ── Probability breakdown ─────────────────────────────────────────────────
    section('PROBABILITY BREAKDOWN')
    bar_max = W - 58
    for name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        pct   = prob * 100
        bar_w = bar_max * prob
        c     = COLORS.get(name, (99, 99, 255))
        bold  = 'B' if name == cls else ''

        pdf.set_font('Helvetica', bold, 9)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(44, 6, name)

        pdf.set_fill_color(241, 245, 249)
        pdf.rect(pdf.get_x(), pdf.get_y() + 1.5, bar_max, 3.5, style='F')
        if bar_w > 0.5:
            pdf.set_fill_color(*c)
            pdf.rect(pdf.get_x(), pdf.get_y() + 1.5, bar_w, 3.5, style='F')

        pdf.set_x(M + 44 + bar_max + 3)
        pdf.set_font('Helvetica', bold, 9)
        pdf.set_text_color(*c) if name == cls else pdf.set_text_color(100, 116, 139)
        pdf.cell(14, 6, f'{pct:.1f}%', ln=True)

    pdf.ln(8)

    # ── Images ────────────────────────────────────────────────────────────────
    section('MRI SCAN & GRAD-CAM VISUALIZATION')
    img_w = (W - 6) / 2
    img_h = img_w * 0.9
    y_img = pdf.get_y()

    if orig_path and os.path.exists(orig_path):
        pdf.image(orig_path, x=M, y=y_img, w=img_w, h=img_h)
    if overlay_path and os.path.exists(overlay_path):
        pdf.image(overlay_path, x=M + img_w + 6, y=y_img, w=img_w, h=img_h)

    pdf.set_y(y_img + img_h + 2)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(img_w + M, 5, 'Original MRI', align='C')
    pdf.cell(img_w + 6, 5, 'Grad-CAM Overlay', align='C', ln=True)
    pdf.ln(10)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.set_text_color(100, 116, 139)
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(226, 232, 240)
    pdf.multi_cell(W, 4.5,
        'DISCLAIMER: This report is generated by an AI model for educational and research '
        'purposes only. It is not a substitute for professional medical diagnosis. Always '
        'consult a qualified radiologist or physician before making any medical decisions.',
        border=1, fill=True)

    return pdf.output()
