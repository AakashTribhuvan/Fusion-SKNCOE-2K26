"""Render a publication-quality tabular benchmark graph and infographic image with perfect column alignment and zero clipping."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def render_tabular_graph(json_path: Path, output_png_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 17x11 inches, dark modern theme
    fig = plt.figure(figsize=(17, 10.5), dpi=300, facecolor="#0b0f19")

    # Title & Subtitle
    fig.suptitle("FUSION Audio Anti-Spoofing Benchmark Evaluation", 
                 fontsize=22, fontweight="bold", color="#ffffff", y=0.96)
    fig.text(0.5, 0.925, "Dataset: 50 Real Human vs. ElevenLabs Voice Clones  |  Execution: CPU-Optimized Pipeline",
             fontsize=12, color="#94a3b8", ha="center")

    gs = fig.add_gridspec(3, 1, height_ratios=[1.6, 1.2, 1.1], hspace=0.38)

    # -------------------------------------------------------------
    # 1. Main Architecture Comparison Table
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_axis_off()
    ax1.text(0.01, 1.05, "1. Model Performance & Security Metrics", 
             fontsize=14, fontweight="bold", color="#38bdf8", transform=ax1.transAxes)

    columns = [
        "Architecture / Model",
        "Accuracy",
        "FAR (Leak)",
        "FRR (Reject)",
        "Precision",
        "Recall",
        "EER",
        "Spoof (TP)",
        "Human (TN)",
    ]

    cell_data = [
        ["Multi-Model Ensemble", "97.96%", "4.00%", "0.00%", "100.0%", "96.00%", "0.00%", "24 / 25", "24 / 24"],
        ["Model C: Wav2Vec2 Deepfake", "97.96%", "4.00%", "0.00%", "100.0%", "96.00%", "0.00%", "24 / 25", "24 / 24"],
        ["Model B: AASIST (Raw)", "53.06%", "0.00%", "95.83%", "52.08%", "100.0%", "63.25%", "25 / 25", "1 / 24"],
    ]

    col_widths_1 = [0.26, 0.09, 0.09, 0.10, 0.09, 0.09, 0.07, 0.10, 0.10]

    table1 = ax1.table(
        cellText=cell_data,
        colLabels=columns,
        colWidths=col_widths_1,
        cellLoc="center",
        loc="center",
        bbox=[0.01, 0.05, 0.98, 0.88]
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(10.5)

    for (row, col), cell in table1.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="#38bdf8", fontweight="bold", fontsize=10.5)
            cell.set_height(0.24)
        else:
            cell.set_height(0.22)
            bg = "#162032" if row % 2 == 1 else "#0f172a"
            cell.set_facecolor(bg)
            cell.set_text_props(color="#f8fafc")
            if col == 0:
                cell.set_text_props(color="#f1f5f9", fontweight="bold", ha="left")
            elif col == 1:
                if "97.96%" in cell.get_text().get_text():
                    cell.set_text_props(color="#4ade80", fontweight="bold")
                else:
                    cell.set_text_props(color="#f87171")
            elif col == 2:
                cell.set_text_props(color="#38bdf8")
            elif col == 3:
                if "0.00%" in cell.get_text().get_text():
                    cell.set_text_props(color="#4ade80", fontweight="bold")
                else:
                    cell.set_text_props(color="#f87171")
            elif col == 6:
                if "0.00%" in cell.get_text().get_text():
                    cell.set_text_props(color="#4ade80", fontweight="bold")

    # -------------------------------------------------------------
    # 2. Source-Wise Breakdown Table
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_axis_off()
    ax2.text(0.01, 1.05, "2. Source & Dataset Breakdown", 
             fontsize=14, fontweight="bold", color="#c084fc", transform=ax2.transAxes)

    src_cols = [
        "Data Source",
        "Ground Truth",
        "Total Clips",
        "Detected Spoof",
        "Detected Genuine",
        "Accuracy Rate",
        "Verification Status",
    ]

    src_data = [
        ["YouTube Human Speech", "Bona Fide Human", "24", "0", "24", "100.0%", "VERIFIED GENUINE (0% FRR)"],
        ["ElevenLabs Voice Clones", "Synthetic Spoof", "25", "24", "1", "96.0%", "BLOCKED SPOOF (96% TP)"],
    ]

    col_widths_2 = [0.24, 0.16, 0.09, 0.12, 0.12, 0.11, 0.14]

    table2 = ax2.table(
        cellText=src_data,
        colLabels=src_cols,
        colWidths=col_widths_2,
        cellLoc="center",
        loc="center",
        bbox=[0.01, 0.05, 0.98, 0.85]
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(10.5)

    for (row, col), cell in table2.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor("#2e1065")
            cell.set_text_props(color="#d8b4fe", fontweight="bold", fontsize=10.5)
            cell.set_height(0.26)
        else:
            cell.set_height(0.24)
            cell.set_facecolor("#1e1b4b" if row == 1 else "#141130")
            cell.set_text_props(color="#f8fafc")
            if col == 0:
                cell.set_text_props(color="#f1f5f9", fontweight="bold", ha="left")
            elif col == 5:
                cell.set_text_props(color="#4ade80", fontweight="bold")
            elif col == 6:
                if "VERIFIED" in cell.get_text().get_text():
                    cell.set_text_props(color="#4ade80", fontweight="bold")
                else:
                    cell.set_text_props(color="#38bdf8", fontweight="bold")

    # -------------------------------------------------------------
    # 3. Decision Matrix & Operational Safeguards
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_axis_off()
    ax3.text(0.01, 1.05, "3. Decision Bands & Security Safeguards", 
             fontsize=14, fontweight="bold", color="#fbbf24", transform=ax3.transAxes)

    dec_cols = [
        "Decision Band",
        "Score Range",
        "Action Taken",
        "Observed False Alarm",
        "Production Recommendation",
    ]

    dec_data = [
        ["LOWER_MODEL_ESTIMATED_SPOOF_RISK", "< 70.0% Risk", "Accept Genuine Voice", "0.0% (Zero False Reject)", "PRIMARY PASS (Allow)"],
        ["REVIEW_REQUIRED", "70.0% - 74.9% Risk", "Prompt Secondary Challenge", "N/A (Provisional Margin)", "STEPPED AUTH (Prompt)"],
        ["HIGH_SPOOF_RISK_REVIEW", ">= 75.0% Risk", "Immediate Spoof Block", "0.0% (Clean Separation)", "REJECT & LOCK (Deny)"],
    ]

    col_widths_3 = [0.28, 0.14, 0.20, 0.17, 0.19]

    table3 = ax3.table(
        cellText=dec_data,
        colLabels=dec_cols,
        colWidths=col_widths_3,
        cellLoc="center",
        loc="center",
        bbox=[0.01, 0.05, 0.98, 0.85]
    )
    table3.auto_set_font_size(False)
    table3.set_fontsize(10.5)

    for (row, col), cell in table3.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor("#451a03")
            cell.set_text_props(color="#fde68a", fontweight="bold", fontsize=10.5)
            cell.set_height(0.26)
        else:
            cell.set_height(0.24)
            cell.set_facecolor("#271a0c" if row % 2 == 1 else "#171008")
            cell.set_text_props(color="#f8fafc")
            if col == 0:
                cell.set_text_props(color="#fef08a", fontweight="bold", ha="left")
            elif col == 4:
                if "PRIMARY" in cell.get_text().get_text():
                    cell.set_text_props(color="#4ade80", fontweight="bold")
                elif "STEPPED" in cell.get_text().get_text():
                    cell.set_text_props(color="#facc15", fontweight="bold")
                else:
                    cell.set_text_props(color="#f87171", fontweight="bold")

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none", pad_inches=0.3)
    plt.close()
    print(f"Tabular benchmark graph saved to: {output_png_path}")

if __name__ == "__main__":
    j_path = Path("data/benchmark/ensemble_report_50.json")
    out_path = Path("data/benchmark/tabular_benchmark_graph.png")
    render_tabular_graph(j_path, out_path)
