"""Generate comprehensive benchmark performance visualization graphs for the audio anti-spoofing ensemble."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate_benchmark_graphs(report_json_path: Path, output_png_path: Path):
    with open(report_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ensemble_metrics = data["ensemble_metrics"]
    cm = ensemble_metrics["confusion_matrix"]
    samples = data["sample_results"]

    # Filter out samples that had errors (e.g. exceeded max duration)
    valid_samples = [s for s in samples if "ensemble_spoof_risk" in s and "latency_ms" in s]

    # Extract scores per ground truth class
    real_scores = [s["ensemble_spoof_risk"] for s in valid_samples if s.get("ground_truth") in ("BONA_FIDE", "GENUINE", "REAL")]
    fake_scores = [s["ensemble_spoof_risk"] for s in valid_samples if s.get("ground_truth") in ("SPOOF", "FAKE", "SYNTHETIC")]

    fig, axs = plt.subplots(2, 2, figsize=(14, 11), dpi=200)
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # 1. Confusion Matrix
    ax_cm = axs[0, 0]
    tp = cm.get("true_spoof_detected (TP)", 0)
    fp = cm.get("false_alarm_genuine_rejected (FP)", 0)
    fn = cm.get("missed_spoof_accepted (FN)", 0)
    tn = cm.get("true_genuine_accepted (TN)", 0)
    cm_matrix = np.array([[tn, fp], [fn, tp]])

    im = ax_cm.imshow(cm_matrix, cmap="Blues", interpolation="nearest")
    ax_cm.set_title("Confusion Matrix (Multi-Model Ensemble)", fontsize=13, fontweight="bold", pad=12)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Predicted Genuine", "Predicted Spoof"], fontsize=10)
    ax_cm.set_yticklabels(["Actual Genuine", "Actual Spoof"], fontsize=10)
    
    for i in range(2):
        for j in range(2):
            val = cm_matrix[i, j]
            color = "white" if val > cm_matrix.max() / 2 else "black"
            ax_cm.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=16, fontweight="bold")
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    # 2. Risk Score Distribution (Real vs Fake)
    ax_dist = axs[0, 1]
    bins = np.linspace(0, 100, 21)
    ax_dist.hist(real_scores, bins=bins, alpha=0.7, color="#2ecc71", label=f"Bona Fide Human ({len(real_scores)})", edgecolor="black")
    ax_dist.hist(fake_scores, bins=bins, alpha=0.7, color="#e74c3c", label=f"ElevenLabs Spoof ({len(fake_scores)})", edgecolor="black")
    ax_dist.axvline(70.0, color="#f39c12", linestyle="--", linewidth=2, label="Review Threshold (70%)")
    ax_dist.axvline(75.0, color="#c0392b", linestyle=":", linewidth=2, label="High Risk Threshold (75%)")
    ax_dist.set_title("Ensemble Spoof Risk Score Distribution", fontsize=13, fontweight="bold", pad=12)
    ax_dist.set_xlabel("Ensemble Spoof Risk (%)", fontsize=11)
    ax_dist.set_ylabel("Number of Samples", fontsize=11)
    ax_dist.set_xlim(0, 100)
    ax_dist.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax_dist.grid(axis="y", linestyle="--", alpha=0.4)

    # 3. Model Component Accuracy & Metric Comparison
    ax_comp = axs[1, 0]
    metrics_labels = ["Accuracy", "Precision", "Recall", "1 - FAR", "1 - FRR"]
    
    ens_vals = [
        ensemble_metrics.get("accuracy", 0.0),
        ensemble_metrics.get("precision", 0.0),
        ensemble_metrics.get("recall", 0.0),
        1.0 - ensemble_metrics.get("false_acceptance_rate_far", 0.0),
        1.0 - ensemble_metrics.get("false_rejection_rate_frr", 0.0),
    ]

    m_b = data.get("model_b_aasist_metrics") or {}
    b_vals = [
        m_b.get("accuracy", 0.0),
        m_b.get("precision", 0.0),
        m_b.get("recall", 0.0),
        1.0 - m_b.get("false_acceptance_rate_far", 0.0),
        1.0 - m_b.get("false_rejection_rate_frr", 0.0),
    ]

    m_c = data.get("model_c_wav2vec2_metrics") or {}
    c_vals = [
        m_c.get("accuracy", 0.0),
        m_c.get("precision", 0.0),
        m_c.get("recall", 0.0),
        1.0 - m_c.get("false_acceptance_rate_far", 0.0),
        1.0 - m_c.get("false_rejection_rate_frr", 0.0),
    ]

    x = np.arange(len(metrics_labels))
    width = 0.25

    ax_comp.bar(x - width, ens_vals, width, label="Multi-Model Ensemble", color="#3498db")
    ax_comp.bar(x, b_vals, width, label="AASIST Official (Model B)", color="#9b59b6")
    ax_comp.bar(x + width, c_vals, width, label="Wav2Vec2 Deepfake (Model C)", color="#1abc9c")

    ax_comp.set_title("Performance Metrics by Architecture", fontsize=13, fontweight="bold", pad=12)
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(metrics_labels, fontsize=10)
    ax_comp.set_ylim(0, 1.15)
    ax_comp.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax_comp.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax_comp.grid(axis="y", linestyle="--", alpha=0.4)

    # 4. Latency vs Spoof Risk Scatter Plot
    ax_scatter = axs[1, 1]
    real_latencies = [s["latency_ms"] for s in valid_samples if s.get("ground_truth") in ("BONA_FIDE", "GENUINE", "REAL")]
    fake_latencies = [s["latency_ms"] for s in valid_samples if s.get("ground_truth") in ("SPOOF", "FAKE", "SYNTHETIC")]

    ax_scatter.scatter(real_latencies, real_scores, color="#2ecc71", alpha=0.8, edgecolors="k", s=60, label=f"Bona Fide Human ({len(real_latencies)})")
    ax_scatter.scatter(fake_latencies, fake_scores, color="#e74c3c", alpha=0.8, edgecolors="k", s=60, label=f"ElevenLabs Spoof ({len(fake_latencies)})")
    ax_scatter.axhline(70.0, color="#f39c12", linestyle="--", linewidth=1.5, alpha=0.8, label="Review Boundary (70%)")
    ax_scatter.set_title("Inference Latency vs Spoof Risk (CPU)", fontsize=13, fontweight="bold", pad=12)
    ax_scatter.set_xlabel("Latency (ms)", fontsize=11)
    ax_scatter.set_ylabel("Ensemble Spoof Risk (%)", fontsize=11)
    ax_scatter.set_ylim(0, 100)
    ax_scatter.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax_scatter.grid(True, linestyle="--", alpha=0.4)

    # Overall title
    total_eval = data.get("total_evaluated", len(valid_samples))
    acc_pct = ensemble_metrics.get("accuracy", 0.0) * 100
    fig.suptitle(f"Audio Anti-Spoofing Benchmark Report — {total_eval} Samples (Accuracy: {acc_pct:.2f}%)",
                 fontsize=16, fontweight="bold", y=0.98)

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_png_path, bbox_inches="tight")
    plt.close()
    print(f"Benchmark chart saved to: {output_png_path}")

if __name__ == "__main__":
    import sys
    report_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/benchmark/ensemble_report_50.json")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/benchmark/benchmark_graph.png")
    generate_benchmark_graphs(report_path, out_path)
