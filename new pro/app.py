from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import datetime

# ── Enhanced PDF Report Dependencies ─────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import Flowable

app = Flask(__name__)
latest_result = {}

# Load trained Random Forest model
model = joblib.load("model.joblib")

FEATURE_COLUMNS = [
    'age',
    'daily_usage_hours',
    'num_platforms_used',
    'avg_session_minutes',
    'night_usage',
    'mental_health_score',
    'screen_time_before_sleep',
    'sleep_risk_score',

    'gender_Male',
    'gender_Other',

    'primary_platform_Snapchat',
    'primary_platform_TikTok',
    'primary_platform_Twitter',
    'primary_platform_YouTube',

    'purpose_Education',
    'purpose_Entertainment',
    'purpose_News',
    'purpose_Socializing',

    'age_group_Young Adult',
    'age_group_Adult',
    'age_group_Senior'
]


# ── Routes ────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/analyze')
def analyze():
    return render_template("analyze.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/reports')
def reports():
    return render_template("reports.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        age                     = int(request.form['age'])
        daily_usage_hours       = float(request.form['daily_usage_hours'])
        num_platforms_used      = int(request.form['num_platforms_used'])
        avg_session_minutes     = float(request.form['avg_session_minutes'])
        night_usage             = int(request.form['night_usage'])
        stress                  = int(request.form['stress_level'])
        comparison              = int(request.form['comparison_level'])
        anxiety                 = int(request.form['anxiety_level'])
        screen_time_before_sleep = float(request.form['screen_time_before_sleep'])
        gender                  = request.form['gender']
        platform                = request.form['platform']
        purpose                 = request.form['purpose']

        mental_health_score = (stress + comparison + anxiety) / 3

        # ---------------------------
        # Feature Engineering
        # ---------------------------
        row = dict.fromkeys(FEATURE_COLUMNS, 0)

        row['age']                      = age
        row['daily_usage_hours']        = daily_usage_hours
        row['num_platforms_used']       = num_platforms_used
        row['avg_session_minutes']      = avg_session_minutes
        row['night_usage']              = night_usage
        row['mental_health_score']      = mental_health_score
        row['screen_time_before_sleep'] = screen_time_before_sleep
        row['sleep_risk_score']         = daily_usage_hours * screen_time_before_sleep

        # Gender Encoding
        if gender == "Male":
            row['gender_Male'] = 1
        elif gender == "Other":
            row['gender_Other'] = 1
        # Female is the default category (all zeros)

        # Platform Encoding
        if platform == "Snapchat":
            row['primary_platform_Snapchat'] = 1
        elif platform == "TikTok":
            row['primary_platform_TikTok'] = 1
        elif platform == "Twitter":
            row['primary_platform_Twitter'] = 1
        elif platform == "YouTube":
            row['primary_platform_YouTube'] = 1
        # Instagram is the default category (all zeros)

        # Purpose Encoding
        if purpose == "Education":
            row['purpose_Education'] = 1
        elif purpose == "Entertainment":
            row['purpose_Entertainment'] = 1
        elif purpose == "News":
            row['purpose_News'] = 1
        elif purpose == "Socializing":
            row['purpose_Socializing'] = 1

        # Age Group Encoding
        if age <= 25:
            row['age_group_Young Adult'] = 1
        elif age <= 35:
            row['age_group_Adult'] = 1
        else:
            row['age_group_Senior'] = 1

        # ---------------------------
        # Model Prediction
        # ---------------------------
        input_df = pd.DataFrame([row])
        prediction = model.predict(input_df)[0]

        # ---------------------------
        # Wellness Score
        # ---------------------------
        # mental_health_score range: 20 (best) → 100 (worst)
        # Normalise it to a 0–80 penalty scale so high stress = big penalty
        mental_penalty = (mental_health_score - 20) / 80 * 40  # 0–40 pts

        wellness_score = 100
        wellness_score -= daily_usage_hours * 4          # 9.5h → -38
        wellness_score -= avg_session_minutes * 0.10     # 84m  → -8.4
        wellness_score -= screen_time_before_sleep * 4   # 3h   → -12
        wellness_score -= mental_penalty                  # high stress → up to -40

        if night_usage == 1:
            wellness_score -= 10

        wellness_score = max(0, min(100, int(wellness_score)))
        dependency_score = 100 - wellness_score

        # ---------------------------
        # Recommendation Engine
        # ---------------------------
        recommendations = []

        if daily_usage_hours > 6:
            recommendations.append("Reduce screen time by at least 1 hour per day.")

        if screen_time_before_sleep > 2:
            recommendations.append("Avoid social media one hour before sleep.")

        # mental_health_score > 50 means moderate-to-high stress/anxiety
        if mental_health_score > 50:
            recommendations.append("Practice digital detox and mindfulness activities.")

        if night_usage == 1:
            recommendations.append("Avoid late-night social media usage.")

        if avg_session_minutes > 60:
            recommendations.append("Take breaks every 30 minutes while using social media.")

        if len(recommendations) == 0:
            recommendations.append("Great job! Your digital wellness habits look healthy.")

        # ---------------------------
        # Save latest result
        # BUG FIX: was inside the `if len == 0` block, and used a broken
        # duplicate assignment. Moved here so it always runs.
        # ---------------------------
        latest_result.clear()
        latest_result["prediction"]         = str(prediction)
        latest_result["wellness_score"]     = wellness_score
        latest_result["dependency_score"]   = dependency_score
        latest_result["daily_usage_hours"]  = daily_usage_hours
        latest_result["recommendations"]    = recommendations

        print("SAVING RESULTS")
        print("Prediction:", prediction)
        print("Wellness Score:", wellness_score)
        print("Dependency Score:", dependency_score)

        return render_template(
            "dashboard.html",
            prediction=prediction,
            wellness_score=wellness_score,
            dependency_score=dependency_score,
            recommendations=recommendations,
            daily_usage_hours=daily_usage_hours
        )

    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════
#  Enhanced PDF Report
# ═══════════════════════════════════════════════════════════════════

_MINT      = rl_colors.HexColor("#10b981")
_MINT_DARK = rl_colors.HexColor("#059669")
_CYAN      = rl_colors.HexColor("#06b6d4")
_NAVY      = rl_colors.HexColor("#0a0f1e")
_WHITE     = rl_colors.white
_LIGHT_BG  = rl_colors.HexColor("#f0fdf4")
_MUTED     = rl_colors.HexColor("#64748b")
_BORDER    = rl_colors.HexColor("#e2e8f0")
_WARN      = rl_colors.HexColor("#f59e0b")
_DANGER    = rl_colors.HexColor("#ef4444")
_TEXT      = rl_colors.HexColor("#0f172a")
_LABEL_BG  = rl_colors.HexColor("#ecfdf5")


class _HeroHeader(Flowable):
    def __init__(self, width, prediction, date_str):
        Flowable.__init__(self)
        self.width      = width
        self.prediction = str(prediction)
        self.date_str   = date_str
        self.height     = 90

    def draw(self):
        c = self.canv
        w, h = self.width, self.height
        c.setFillColor(_NAVY)
        c.roundRect(0, 0, w, h, 10, fill=1, stroke=0)
        c.setFillColor(_MINT)
        c.rect(0, h - 5, w * 0.55, 5, fill=1, stroke=0)
        c.setFillColor(_CYAN)
        c.rect(w * 0.55, h - 5, w * 0.45, 5, fill=1, stroke=0)
        c.setFillColor(_MINT)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(20, h - 38, "ZenFlow")
        c.setFillColor(_WHITE)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(20, h - 56, "Digital Wellness Report")
        c.setFillColor(rl_colors.HexColor("#94a3b8"))
        c.setFont("Helvetica", 9)
        c.drawString(20, h - 72, f"Generated on {self.date_str}")
        badge_color = (
            _MINT_DARK if "Healthy"  in self.prediction else
            _WARN      if "Moderate" in self.prediction else
            _DANGER
        )
        bw, bh = 120, 28
        bx, by = w - bw - 20, (h - bh) / 2
        c.setFillColor(badge_color)
        c.roundRect(bx, by, bw, bh, 8, fill=1, stroke=0)
        c.setFillColor(_WHITE)
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(bx + bw / 2, by + 9, f"Risk: {self.prediction}")


class _ScoreBar(Flowable):
    def __init__(self, width, label, value, max_val, color, unit=""):
        Flowable.__init__(self)
        self.width    = width
        self.label    = label
        self.value    = value
        self.max_val  = max_val
        self.bar_color = color
        self.unit     = unit
        self.height   = 44

    def draw(self):
        c   = self.canv
        w   = self.width
        pct = min(self.value / self.max_val, 1.0)
        c.setFillColor(_TEXT)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(0, 28, self.label)
        c.setFillColor(self.bar_color)
        c.setFont("Helvetica-Bold", 12)
        c.drawRightString(w, 28, f"{self.value}{self.unit}")
        c.setFillColor(_BORDER)
        c.roundRect(0, 12, w, 8, 4, fill=1, stroke=0)
        c.setFillColor(self.bar_color)
        c.roundRect(0, 12, max(pct * w, 8), 8, 4, fill=1, stroke=0)


class _SectionTitle(Flowable):
    def __init__(self, width, title):
        Flowable.__init__(self)
        self.width  = width
        self.title  = title
        self.height = 32

    def draw(self):
        c = self.canv
        c.setFillColor(_MINT)
        c.rect(0, 4, 4, 22, fill=1, stroke=0)
        c.setFillColor(_LABEL_BG)
        c.roundRect(8, 0, self.width - 8, 30, 6, fill=1, stroke=0)
        c.setFillColor(_MINT_DARK)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(20, 10, self.title)


class _RecommendationItem(Flowable):
    def __init__(self, width, text, index):
        Flowable.__init__(self)
        self.width  = width
        self.text   = text
        self.index  = index
        self.height = 38

    def draw(self):
        c = self.canv
        w = self.width
        c.setFillColor(_LIGHT_BG)
        c.roundRect(0, 2, w, 34, 6, fill=1, stroke=0)
        c.setStrokeColor(rl_colors.HexColor("#bbf7d0"))
        c.setLineWidth(0.5)
        c.roundRect(0, 2, w, 34, 6, fill=0, stroke=1)
        c.setFillColor(_MINT)
        c.circle(16, 19, 5, fill=1, stroke=0)
        c.setFillColor(_WHITE)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(16, 16, str(self.index))
        c.setFillColor(_TEXT)
        c.setFont("Helvetica", 10)
        display = self.text if len(self.text) <= 95 else self.text[:95] + "…"
        c.drawString(32, 16, display)


def _build_report_pdf(pdf_path, prediction, wellness_score,
                      dependency_score, daily_usage_hours, recommendations):
    W, H     = A4
    margin   = 20 * mm
    usable_w = W - 2 * margin
    doc      = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=14 * mm, bottomMargin=14 * mm
    )

    def s(name, **kw):
        d = dict(fontName="Helvetica", fontSize=10, textColor=_TEXT, leading=14)
        d.update(kw)
        return ParagraphStyle(name, **d)

    date_str = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
    story    = []

    story.append(_HeroHeader(usable_w, str(prediction), date_str))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph(
        "This report summarises your digital wellness assessment. Scores are calculated using a "
        "trained Random Forest model combined with behavioural indicators across screen time, "
        "sleep patterns, and mental wellness.",
        s("body", textColor=_MUTED, leading=15)
    ))
    story.append(Spacer(1, 5 * mm))

    # ── Key Metrics ──────────────────────────────────────────────
    story.append(_SectionTitle(usable_w, "Key Metrics"))
    story.append(Spacer(1, 3 * mm))

    ws_color   = _MINT_DARK if wellness_score  >= 70 else (_WARN if wellness_score  >= 45 else _DANGER)
    dep_color  = _MINT_DARK if dependency_score <= 30 else (_WARN if dependency_score <= 55 else _DANGER)
    pred_color = (
        _MINT_DARK if "Healthy"  in str(prediction) else
        _WARN      if "Moderate" in str(prediction) else
        _DANGER
    )

    def mc(val, lbl, col):
        return [
            Paragraph(val, ParagraphStyle("mv", fontName="Helvetica-Bold", fontSize=24,
                      textColor=col, leading=28, alignment=TA_CENTER)),
            Paragraph(lbl, ParagraphStyle("ml", fontSize=8, textColor=_MUTED,
                      fontName="Helvetica-Bold", alignment=TA_CENTER, leading=11))
        ]

    cw = usable_w / 4 - 3
    mt = Table(
        [[
            mc(str(wellness_score),    "WELLNESS\nSCORE /100",  ws_color),
            mc(f"{dependency_score}%", "DEPENDENCY\nSCORE",     dep_color),
            mc(f"{daily_usage_hours}h","DAILY\nSCREEN TIME",    _CYAN),
            mc(str(prediction),        "AI\nPREDICTION",        pred_color),
        ]],
        colWidths=[cw] * 4, rowHeights=[60]
    )
    mt.setStyle(TableStyle([
        ("BOX",           (0, 0), (-1, -1), 0.5, _BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, _BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(mt)
    story.append(Spacer(1, 5 * mm))

    # ── Score Bars ───────────────────────────────────────────────
    story.append(_SectionTitle(usable_w, "Score Breakdown"))
    story.append(Spacer(1, 3 * mm))
    story.append(_ScoreBar(usable_w, "Wellness Score",   wellness_score,              100, ws_color,  "/100"))
    story.append(Spacer(1, 2 * mm))
    story.append(_ScoreBar(usable_w, "Dependency Score", dependency_score,            100, dep_color, "%"))
    story.append(Spacer(1, 2 * mm))
    story.append(_ScoreBar(usable_w, "Daily Screen Time", min(daily_usage_hours, 16), 16,  _CYAN,     "h"))
    story.append(Spacer(1, 5 * mm))

    # ── Interpretation Table ─────────────────────────────────────
    story.append(_SectionTitle(usable_w, "What Your Scores Mean"))
    story.append(Spacer(1, 3 * mm))

    def irow(label, value, unit, g, w_thresh, higher_good, desc_g, desc_w, desc_b):
        if higher_good:
            ok  = value >= g
            mid = value >= w_thresh
        else:
            ok  = value <= g
            mid = value <= w_thresh
        col    = _MINT_DARK if ok else (_WARN if mid else _DANGER)
        status = "Healthy ✓" if ok else ("Moderate ⚠" if mid else "Needs Attention ✗")
        desc   = desc_g if ok else (desc_w if mid else desc_b)
        ps = lambda t, c=_TEXT, sz=9: Paragraph(
            t, ParagraphStyle("x", fontSize=sz, textColor=c, leading=13))
        return [ps(f"<b>{label}</b>"), ps(f"<b>{value}{unit}</b>", col),
                ps(f"<b>{status}</b>", col), ps(desc, _MUTED, 8)]

    th_s  = ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=9, textColor=_MUTED)
    idata = [
        [Paragraph("<b>Metric</b>", th_s), Paragraph("<b>Value</b>", th_s),
         Paragraph("<b>Status</b>", th_s), Paragraph("<b>Interpretation</b>", th_s)],
        irow("Wellness Score",    wellness_score,   "/100", 70, 45, True,
             "Strong digital habits",   "Room for improvement",   "High risk — action needed"),
        irow("Dependency Score",  dependency_score, "%",    30, 55, False,
             "Low dependency",          "Moderate dependency",    "High dependency detected"),
        irow("Daily Screen Time", daily_usage_hours,"h",     4,  7, False,
             "Within healthy range",    "Slightly elevated",      "Significantly above average"),
    ]
    it = Table(idata, colWidths=[usable_w * 0.20, usable_w * 0.12,
                                  usable_w * 0.22, usable_w * 0.46])
    it.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _LABEL_BG),
        ("BOX",           (0, 0), (-1, -1), 0.5, _BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, _BORDER),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#fafafa")]),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(it)
    story.append(Spacer(1, 5 * mm))

    # ── Recommendations ──────────────────────────────────────────
    story.append(_SectionTitle(usable_w, "Personalised Recommendations"))
    story.append(Spacer(1, 3 * mm))
    for i, tip in enumerate(recommendations, 1):
        story.append(_RecommendationItem(usable_w, tip, i))
        story.append(Spacer(1, 2 * mm))

    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width=usable_w, thickness=0.5, color=_BORDER))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "ZenFlow Digital Wellness Platform  ·  This report is generated by an AI model and is for "
        "informational purposes only. It does not constitute medical advice.",
        ParagraphStyle("footer", fontSize=8, textColor=_MUTED, alignment=TA_CENTER)
    ))

    doc.build(story)


@app.route("/download_report")
def download_report():
    pdf_file = "wellness_report.pdf"
    _build_report_pdf(
        pdf_path         = pdf_file,
        prediction       = latest_result.get("prediction",      "N/A"),
        wellness_score   = latest_result.get("wellness_score",  0),
        dependency_score = latest_result.get("dependency_score", 0),
        daily_usage_hours= latest_result.get("daily_usage_hours", 0),
        recommendations  = latest_result.get("recommendations", ["No recommendations available."])
    )
    return send_file(pdf_file, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
