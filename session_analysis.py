"""
DriveGuard — Session Analysis Tool
───────────────────────────────────
Post-drive forensics tool that processes a 10-metric session CSV log to 
generate a comprehensive, print-ready PDF safety report containing 
automated data visualizations and AI recommendations.

Detailed usage instructions are maintained in README.md.
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — generate PNGs only
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from datetime import datetime, timedelta
from fpdf import FPDF

# ── Colour Palette (matching DriveGuard UI) ──────────────────────────────────
COLORS = {
    "ACTIVE":     "#38C85A",   # Green
    "SLEEPY":     "#D2D228",   # Yellow-amber
    "DROWSY":     "#DC3737",   # Red
    "DISTRACTED": "#BE3CC8",   # Purple
}
BG_DARK   = "#121624"
PANEL_BG  = "#1C2134"
ACCENT    = "#50C8FF"
TEXT_SOFT  = "#DCE4F5"

STATE_ORDER = ["ACTIVE", "SLEEPY", "DROWSY", "DISTRACTED"]

# ── Report Directory ─────────────────────────────────────────────────────────
REPORTS_DIR = "reports"

# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def format_duration(seconds):
    """Human-readable duration string."""
    if seconds < 0:
        seconds = 0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def safety_grade(score):
    """Return letter grade from overall risk score (0-100, lower=safer)."""
    if score < 15:
        return "A", "Excellent"
    elif score < 30:
        return "B", "Good"
    elif score < 50:
        return "C", "Needs Improvement"
    elif score < 70:
        return "D", "Poor"
    return "F", "Critical"


def load_csv(path):
    """Load and validate session CSV."""
    expected_cols = [
        "Date", "Time", "State", "FatigueScore",
        "EAR", "BlinkRate", "YawnCount",
        "HeadDirection", "NoseOffset", "MAR"
    ]
    df = pd.read_csv(path)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"\n  [ERROR] Missing columns: {', '.join(missing)}")
        print("    This CSV may be from an older version of DriveGuard.")
        print("    Please use a CSV generated with the latest 10-column format.")
        sys.exit(1)

    # Parse datetime
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df.sort_values("Datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Numeric conversions
    for col in ["FatigueScore", "EAR", "BlinkRate", "YawnCount", "NoseOffset", "MAR"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary(df):
    """Compute session-level summary statistics."""
    total_rows = len(df)
    start = df["Datetime"].iloc[0]
    end = df["Datetime"].iloc[-1]
    duration_sec = (end - start).total_seconds()

    # State counts and durations (each row ≈ 1 second)
    state_counts = df["State"].value_counts()
    state_pcts = {}
    state_durations = {}
    for s in STATE_ORDER:
        cnt = state_counts.get(s, 0)
        state_pcts[s] = (cnt / total_rows * 100) if total_rows else 0
        state_durations[s] = cnt  # each row ≈ 1s

    # Fatigue metrics
    peak_fatigue = df["FatigueScore"].max()
    peak_fatigue_time = df.loc[df["FatigueScore"].idxmax(), "Datetime"]
    avg_fatigue = df["FatigueScore"].mean()

    # Blink & Yawn
    avg_blink_rate = df["BlinkRate"].mean()
    max_blink_rate = df["BlinkRate"].max()
    total_yawns = int(df["YawnCount"].iloc[-1]) if not df["YawnCount"].isna().all() else 0

    # EAR metrics
    avg_ear = df["EAR"].mean()
    min_ear = df["EAR"].min()

    # Risk score (weighted composite)
    drowsy_pct = state_pcts.get("DROWSY", 0)
    sleepy_pct = state_pcts.get("SLEEPY", 0)
    distracted_pct = state_pcts.get("DISTRACTED", 0)
    risk_score = min(100, (drowsy_pct * 2.5) + (sleepy_pct * 1.0) + (distracted_pct * 0.8) + (avg_fatigue * 0.3))
    grade, grade_label = safety_grade(risk_score)

    # Episodes (consecutive sequences of non-ACTIVE states)
    drowsy_episodes = count_episodes(df, "DROWSY")
    sleepy_episodes = count_episodes(df, "SLEEPY")
    distracted_episodes = count_episodes(df, "DISTRACTED")

    # Fatigue trend (linear regression slope)
    if len(df) > 5:
        x = np.arange(len(df))
        slope = np.polyfit(x, df["FatigueScore"].values, 1)[0]
        if slope > 0.05:
            fatigue_trend = "Increasing"
        elif slope < -0.05:
            fatigue_trend = "Decreasing"
        else:
            fatigue_trend = "Stable"
    else:
        fatigue_trend = "Insufficient data"

    return {
        "total_rows": total_rows,
        "start": start,
        "end": end,
        "duration_sec": duration_sec,
        "state_pcts": state_pcts,
        "state_durations": state_durations,
        "peak_fatigue": peak_fatigue,
        "peak_fatigue_time": peak_fatigue_time,
        "avg_fatigue": avg_fatigue,
        "avg_blink_rate": avg_blink_rate,
        "max_blink_rate": max_blink_rate,
        "total_yawns": total_yawns,
        "avg_ear": avg_ear,
        "min_ear": min_ear,
        "risk_score": risk_score,
        "grade": grade,
        "grade_label": grade_label,
        "drowsy_episodes": drowsy_episodes,
        "sleepy_episodes": sleepy_episodes,
        "distracted_episodes": distracted_episodes,
        "fatigue_trend": fatigue_trend,
    }


def count_episodes(df, state):
    """Count continuous episodes and their total+max duration for a given state."""
    in_episode = False
    episodes = []
    start_idx = 0
    for i, row in df.iterrows():
        if row["State"] == state:
            if not in_episode:
                in_episode = True
                start_idx = i
        else:
            if in_episode:
                episodes.append(i - start_idx)
                in_episode = False
    if in_episode:
        episodes.append(len(df) - start_idx)

    return {
        "count": len(episodes),
        "total_sec": sum(episodes),
        "max_sec": max(episodes) if episodes else 0,
    }


def generate_recommendations(summary):
    """Generate plain-English recommendations based on analysis."""
    recs = []

    # Drowsy episodes
    drowsy = summary["drowsy_episodes"]
    if drowsy["count"] > 0:
        recs.append(
            f"You had {drowsy['count']} drowsy episode(s) totalling {format_duration(drowsy['total_sec'])}. "
            f"The longest lasted {format_duration(drowsy['max_sec'])}. "
            "Consider pulling over and resting if you feel your eyes getting heavy."
        )

    # Sleepy episodes
    sleepy = summary["sleepy_episodes"]
    if sleepy["count"] > 2:
        recs.append(
            f"You showed signs of sleepiness {sleepy['count']} times during this session. "
            "Take a 15-20 minute power nap or have a caffeinated drink before continuing."
        )

    # Distraction
    distracted = summary["distracted_episodes"]
    if distracted["count"] > 0:
        recs.append(
            f"You were distracted {distracted['count']} time(s) for a total of {format_duration(distracted['total_sec'])}. "
            "Keep your eyes on the road and minimize phone or passenger distractions."
        )

    # High blink rate
    if summary["avg_blink_rate"] > 22:
        recs.append(
            f"Your average blink rate was {summary['avg_blink_rate']:.0f}/min (normal: 15-20/min). "
            "Elevated blinking may indicate eye strain or fatigue. Consider taking a break."
        )

    # Peak fatigue
    if summary["peak_fatigue"] >= 80:
        recs.append(
            f"Your fatigue level reached a critical peak of {summary['peak_fatigue']:.0f}%. "
            "It is strongly recommended to take a break every 90 minutes of driving."
        )

    # Yawns
    if summary["total_yawns"] >= 3:
        recs.append(
            f"You yawned {summary['total_yawns']} times during this session, which is a sign of fatigue. "
            "Fresh air, stretching, or a short walk can help reduce drowsiness."
        )

    # Fatigue trend
    if "Increasing" in summary["fatigue_trend"]:
        recs.append(
            "Your fatigue was trending upward throughout the session. "
            "Plan for rest stops on longer drives."
        )

    # All safe
    if not recs:
        recs.append(
            "Great job! You maintained good alertness throughout this session. "
            "Keep up the safe driving habits!"
        )

    return recs


# ══════════════════════════════════════════════════════════════════════════════
#  CHART GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def set_chart_style():
    """Apply DriveGuard dark theme to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": "#37415F",
        "axes.labelcolor": TEXT_SOFT,
        "text.color": TEXT_SOFT,
        "xtick.color": "#7887AA",
        "ytick.color": "#7887AA",
        "grid.color": "#2A3248",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 10,
    })


def chart_state_timeline(df, out_path):
    """Horizontal color-coded state timeline bar."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(12, 2.2))

    times = df["Datetime"].values
    states = df["State"].values

    for i in range(len(times) - 1):
        color = COLORS.get(states[i], "#555555")
        t0 = mdates.date2num(pd.Timestamp(times[i]))
        t1 = mdates.date2num(pd.Timestamp(times[i + 1]))
        ax.barh(0, t1 - t0, left=t0, height=0.6, color=color, edgecolor="none")

    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_title("Driver State Timeline", fontsize=13, fontweight="bold", pad=12)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLORS[s], label=s) for s in STATE_ORDER]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.5,
              facecolor=PANEL_BG, edgecolor="#37415F")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_fatigue_over_time(df, out_path):
    """Line chart of fatigue score over time with danger zones."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(12, 4))

    times = df["Datetime"]
    fatigue = df["FatigueScore"]

    # Danger zone fills
    ax.axhspan(0, 30, color="#38C85A", alpha=0.08, label="Safe (0-30%)")
    ax.axhspan(30, 60, color="#D2D228", alpha=0.08, label="Caution (30-60%)")
    ax.axhspan(60, 80, color="#FF8C00", alpha=0.08, label="Warning (60-80%)")
    ax.axhspan(80, 100, color="#DC3737", alpha=0.08, label="Critical (80-100%)")

    # Line
    ax.plot(times, fatigue, color=ACCENT, linewidth=1.8, alpha=0.9)
    ax.fill_between(times, fatigue, alpha=0.15, color=ACCENT)

    ax.set_ylim(0, 105)
    ax.set_ylabel("Fatigue Score (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_title("Fatigue Score Over Time", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.5,
              facecolor=PANEL_BG, edgecolor="#37415F")
    ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_ear_over_time(df, out_path):
    """Line chart showing Eye Aspect Ratio (EAR) over time."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(12, 3.5))

    times = df["Datetime"]
    ear = df["EAR"]

    ax.plot(times, ear, color="#3CDC8C", linewidth=1.2, alpha=0.85)
    ax.fill_between(times, ear, alpha=0.1, color="#3CDC8C")

    # Threshold line
    ax.axhline(y=0.23, color="#DC3737", linestyle="--", linewidth=1, alpha=0.7, label="Drowsy Threshold (0.23)")

    ax.set_ylabel("Eye Aspect Ratio (EAR)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_title("Eye Openness Over Time", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.5,
              facecolor=PANEL_BG, edgecolor="#37415F")
    ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_blink_rate(df, out_path):
    """Line chart of blink rate trend over time."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(12, 3.5))

    times = df["Datetime"]
    blinks = df["BlinkRate"]

    ax.plot(times, blinks, color="#C850FF", linewidth=1.2, alpha=0.85)
    ax.fill_between(times, blinks, alpha=0.1, color="#C850FF")

    # High blink rate threshold
    ax.axhline(y=25, color="#D2D228", linestyle="--", linewidth=1, alpha=0.7, label="High Blink Rate Threshold (25/min)")

    ax.set_ylabel("Blinks per Minute")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_title("Blink Rate Trend", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.5,
              facecolor=PANEL_BG, edgecolor="#37415F")
    ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_head_direction_pie(df, out_path):
    """Pie chart showing time spent looking in each direction."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(5, 5))

    dir_counts = df["HeadDirection"].value_counts()
    dir_order = ["FORWARD", "LEFT", "RIGHT", "DOWN"]
    dir_colors = {
        "FORWARD": "#3CDC8C",
        "LEFT":    "#C850FF",
        "RIGHT":   "#C850FF",
        "DOWN":    "#F05050",
    }

    labels = []
    sizes = []
    colors = []
    for d in dir_order:
        if d in dir_counts:
            labels.append(d)
            sizes.append(dir_counts[d])
            colors.append(dir_colors.get(d, "#555555"))

    if not sizes:
        plt.close(fig)
        return

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        textprops={"color": TEXT_SOFT, "fontsize": 10},
        wedgeprops={"edgecolor": BG_DARK, "linewidth": 2}
    )
    for txt in autotexts:
        txt.set_fontsize(9)
        txt.set_color("#121624")
        txt.set_fontweight("bold")

    ax.set_title("Head Direction Distribution", fontsize=13, fontweight="bold", pad=16)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_state_distribution(df, out_path):
    """Horizontal bar chart showing time in each state."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(8, 3))

    state_counts = df["State"].value_counts()
    states = []
    counts = []
    colors = []
    for s in STATE_ORDER:
        if s in state_counts:
            states.append(s)
            counts.append(state_counts[s])
            colors.append(COLORS.get(s, "#555"))

    y_pos = np.arange(len(states))
    bars = ax.barh(y_pos, counts, color=colors, height=0.5, edgecolor="none")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(states, fontweight="bold")
    ax.set_xlabel("Duration (seconds)")
    ax.set_title("Time Spent in Each State", fontsize=13, fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.grid(True, axis="x", linewidth=0.3)

    # Add time labels on bars
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                format_duration(cnt), va="center", fontsize=9, color=TEXT_SOFT)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  PDF REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

class SessionReport(FPDF):
    """Custom PDF report with DriveGuard branding."""

    def header(self):
        # Dark header bar
        self.set_fill_color(18, 22, 36)
        self.rect(0, 0, 210, 25, "F")
        # Accent line
        self.set_fill_color(80, 200, 255)
        self.rect(0, 25, 210, 1.5, "F")
        # Title
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(220, 228, 245)
        self.set_y(6)
        self.cell(0, 12, "DriveGuard  -  Session Analysis Report", align="C")
        self.ln(22)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(80, 90, 100)
        self.cell(0, 10, f"DriveGuard Session Report  |  Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        """Add a styled section title."""
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(80, 200, 255)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        # Accent underline
        self.set_fill_color(80, 200, 255)
        self.rect(self.get_x(), self.get_y(), 40, 0.8, "F")
        self.ln(4)

    def key_value(self, key, value, key_color=(80, 90, 100), value_color=(0, 0, 0)):
        """Add a key-value pair line."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*key_color)
        self.cell(65, 7, key)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*value_color)
        self.cell(0, 7, str(value), new_x="LMARGIN", new_y="NEXT")

    def add_chart(self, image_path, width=180):
        """Add a chart image centered on the page."""
        if os.path.exists(image_path):
            x = (210 - width) / 2
            self.image(image_path, x=x, w=width)
            self.ln(6)


def generate_pdf_report(df, summary, chart_paths, output_path):
    """Build the full PDF report."""
    pdf = SessionReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Page 1: Session Summary ──────────────────────────────────────────────
    pdf.add_page()

    pdf.section_title("SESSION OVERVIEW")

    pdf.key_value("Session Date", summary["start"].strftime("%d %B %Y"))
    pdf.key_value("Start Time", summary["start"].strftime("%H:%M:%S"))
    pdf.key_value("End Time", summary["end"].strftime("%H:%M:%S"))
    pdf.key_value("Duration", format_duration(summary["duration_sec"]))
    pdf.key_value("Data Points", str(summary["total_rows"]))
    pdf.ln(5)

    # Safety Grade (large, colored)
    grade = summary["grade"]
    grade_label = summary["grade_label"]
    grade_colors = {
        "A": (56, 200, 90),
        "B": (100, 200, 50),
        "C": (210, 210, 40),
        "D": (255, 140, 0),
        "F": (220, 55, 55),
    }
    g_col = grade_colors.get(grade, (20, 30, 40))

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(80, 200, 255)
    pdf.cell(65, 10, "Safety Grade")
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*g_col)
    pdf.cell(15, 10, grade)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 90, 100)
    pdf.cell(0, 10, f"  ({grade_label})", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ── State Breakdown ────────────────────────────────────────────────────
    pdf.section_title("STATE BREAKDOWN")

    state_icons = {"ACTIVE": "ACTIVE", "SLEEPY": "SLEEPY", "DROWSY": "DROWSY", "DISTRACTED": "DISTRACTED"}
    state_pdf_colors = {
        "ACTIVE":     (56, 200, 90),
        "SLEEPY":     (210, 210, 40),
        "DROWSY":     (220, 55, 55),
        "DISTRACTED": (190, 60, 200),
    }

    for s in STATE_ORDER:
        pct = summary["state_pcts"].get(s, 0)
        dur = summary["state_durations"].get(s, 0)
        s_col = state_pdf_colors.get(s, (20, 30, 40))

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*s_col)
        pdf.cell(35, 7, f"  {s}")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(35, 7, format_duration(dur))
        pdf.set_text_color(80, 90, 100)
        pdf.cell(0, 7, f"({pct:.1f}%)", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)

    # ── Key Metrics ──────────────────────────────────────────────────────────
    pdf.section_title("KEY METRICS")

    pdf.key_value("Peak Fatigue Score", f"{summary['peak_fatigue']:.0f}%")
    pdf.key_value("Peak Fatigue Time", summary["peak_fatigue_time"].strftime("%H:%M:%S"))
    pdf.key_value("Average Fatigue", f"{summary['avg_fatigue']:.1f}%")
    pdf.key_value("Fatigue Trend", summary["fatigue_trend"])
    pdf.key_value("Avg Blink Rate", f"{summary['avg_blink_rate']:.1f} blinks/min")
    pdf.key_value("Max Blink Rate", f"{summary['max_blink_rate']:.0f} blinks/min")
    pdf.key_value("Total Yawns", str(summary["total_yawns"]))
    pdf.key_value("Average EAR", f"{summary['avg_ear']:.3f}")
    pdf.key_value("Minimum EAR", f"{summary['min_ear']:.3f}")
    pdf.ln(5)

    # ── Risk Assessment ──────────────────────────────────────────────────────
    pdf.section_title("RISK ASSESSMENT")

    drowsy = summary["drowsy_episodes"]
    sleepy = summary["sleepy_episodes"]
    distracted = summary["distracted_episodes"]

    pdf.key_value("Drowsy Episodes", f"{drowsy['count']}  (total: {format_duration(drowsy['total_sec'])}, longest: {format_duration(drowsy['max_sec'])})")
    pdf.key_value("Sleepy Episodes", f"{sleepy['count']}  (total: {format_duration(sleepy['total_sec'])}, longest: {format_duration(sleepy['max_sec'])})")
    pdf.key_value("Distraction Events", f"{distracted['count']}  (total: {format_duration(distracted['total_sec'])}, longest: {format_duration(distracted['max_sec'])})")
    pdf.key_value("Overall Risk Score", f"{summary['risk_score']:.1f} / 100")
    pdf.ln(5)

    # ── Recommendations ──────────────────────────────────────────────────────
    pdf.section_title("RECOMMENDATIONS")

    recs = generate_recommendations(summary)
    pdf.set_font("Helvetica", "", 10)
    for i, rec in enumerate(recs, 1):
        pdf.set_text_color(80, 200, 255)
        pdf.cell(8, 6, f"{i}.")
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 6, rec)
        pdf.ln(2)

    # ── Charts Pages ─────────────────────────────────────────────────────────
    chart_titles = [
        ("state_timeline", "DRIVER STATE TIMELINE"),
        ("state_distribution", "STATE DISTRIBUTION"),
        ("fatigue_over_time", "FATIGUE SCORE OVER TIME"),
        ("ear_over_time", "EYE OPENNESS (EAR) OVER TIME"),
        ("blink_rate", "BLINK RATE TREND"),
        ("head_direction_pie", "HEAD DIRECTION DISTRIBUTION"),
    ]

    for chart_key, title in chart_titles:
        if chart_key in chart_paths and os.path.exists(chart_paths[chart_key]):
            pdf.add_page()
            pdf.section_title(title)
            pdf.add_chart(chart_paths[chart_key])

    # Save
    pdf.output(output_path)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_console_summary(summary):
    """Print a formatted console summary."""
    W = 56

    print()
    print("+" + "-" * W + "+")
    print("|" + "DriveGuard  -  Session Analysis Report".center(W) + "|")
    print("+" + "-" * W + "+")
    print()

    print("=" * W)
    print("  SESSION SUMMARY")
    print("=" * W)
    print(f"  Session Date     :  {summary['start'].strftime('%d %B %Y')}")
    print(f"  Duration         :  {format_duration(summary['duration_sec'])}")
    print(f"  Data Points      :  {summary['total_rows']}")
    print(f"  Safety Grade     :  {summary['grade']} ({summary['grade_label']})")
    print(f"  Peak Fatigue     :  {summary['peak_fatigue']:.0f}% at {summary['peak_fatigue_time'].strftime('%H:%M:%S')}")
    print(f"  Fatigue Trend    :  {summary['fatigue_trend']}")
    print()

    print("=" * W)
    print("  STATE BREAKDOWN")
    print("=" * W)
    icons = {"ACTIVE": "[ACTIVE]", "SLEEPY": "[SLEEPY]", "DROWSY": "[DROWSY]", "DISTRACTED": "[DISTR ]"}
    for s in STATE_ORDER:
        pct = summary["state_pcts"].get(s, 0)
        dur = summary["state_durations"].get(s, 0)
        icon = icons.get(s, "[OTHER ]")
        print(f"  {icon} {s:<12} :  {format_duration(dur):>8}   ({pct:.1f}%)")
    print()

    print("=" * W)
    print("  RISK ASSESSMENT")
    print("=" * W)
    drowsy = summary["drowsy_episodes"]
    sleepy = summary["sleepy_episodes"]
    distracted = summary["distracted_episodes"]
    print(f"  [!] Drowsy Episodes     : {drowsy['count']} ({format_duration(drowsy['total_sec'])} total)")
    print(f"  [!] Sleepy Episodes     : {sleepy['count']} ({format_duration(sleepy['total_sec'])} total)")
    print(f"  [!] Distraction Events  : {distracted['count']} ({format_duration(distracted['total_sec'])} total)")
    print(f"  [~] Fatigue Trend       : {summary['fatigue_trend']}")
    print()

    print("=" * W)
    print("  RECOMMENDATIONS")
    print("=" * W)
    recs = generate_recommendations(summary)
    for i, rec in enumerate(recs, 1):
        print(f"  {i}. {rec}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("+" + "-" * 52 + "+")
    print("|" + "DriveGuard  -  Session Analysis Tool".center(52) + "|")
    print("+" + "-" * 52 + "+")
    print()

    # Get CSV file path from user
    csv_path = input("  Enter path to session CSV file: ").strip().strip('"').strip("'")

    if not os.path.isfile(csv_path):
        print(f"\n  [ERROR] File not found: {csv_path}")
        sys.exit(1)

    # Load data
    print(f"\n  Loading {csv_path} ...")
    df = load_csv(csv_path)
    summary = compute_summary(df)

    print(f"  [OK] File loaded successfully | {summary['total_rows']} data points | Duration: {format_duration(summary['duration_sec'])}")

    # Print console summary
    print_console_summary(summary)

    # Create reports directory
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Create a subfolder for this session's report
    session_name = os.path.splitext(os.path.basename(csv_path))[0]
    report_folder = os.path.join(REPORTS_DIR, f"analysis_{session_name}")
    charts_folder = os.path.join(report_folder, "charts")
    os.makedirs(charts_folder, exist_ok=True)

    # Generate charts
    print("  Generating charts...")
    chart_paths = {}

    chart_funcs = {
        "state_timeline":     chart_state_timeline,
        "fatigue_over_time":  chart_fatigue_over_time,
        "ear_over_time":      chart_ear_over_time,
        "blink_rate":         chart_blink_rate,
        "head_direction_pie": chart_head_direction_pie,
        "state_distribution": chart_state_distribution,
    }

    for key, func in chart_funcs.items():
        path = os.path.join(charts_folder, f"{key}.png")
        try:
            func(df, path)
            chart_paths[key] = path
            print(f"    [OK] {key}.png")
        except Exception as e:
            print(f"    [FAIL] {key}: {e}")

    # Generate PDF report
    print("\n  Generating PDF report...")
    pdf_path = os.path.join(report_folder, f"{session_name}_report.pdf")
    try:
        generate_pdf_report(df, summary, chart_paths, pdf_path)
        print(f"  [OK] PDF report saved to: {pdf_path}")
    except Exception as e:
        print(f"  [FAIL] PDF generation failed: {e}")

    print(f"\n  [DONE] All outputs saved to: {report_folder}/")
    print()


if __name__ == "__main__":
    main()
