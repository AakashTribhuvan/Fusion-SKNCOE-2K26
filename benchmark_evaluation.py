"""
Benchmark Evaluation Script for Video Detection Model Video Authenticity Pipeline.
Evaluates the video classification pipeline on genuine and synthetic video samples.
Calculates Confusion Matrix, Accuracy, Precision, Recall, F1, FPR, FNR,
per-video runtime, and separates Deepfake/GAN performance from Modern Diffusion performance.
"""
import os
import sys
import json
import time
import argparse
import statistics

# Force unbuffered output on Windows
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

from api.video_app import _analyze_video

eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "eval_videos")

CURATED_DATASET = [
    # --- REAL VIDEOS ---
    {
        "path": os.path.join(eval_dir, "real_obama.mp4"),
        "ground_truth": "REAL_VIDEO",
        "subtype": "REAL_SPEECH",
        "split": "calibration",
        "description": "Authentic public address speech (Barack Obama)"
    },
    {
        "path": os.path.join(eval_dir, "real_einstein.mp4"),
        "ground_truth": "REAL_VIDEO",
        "subtype": "REAL_ARCHIVAL",
        "split": "test",
        "description": "Authentic archival recording (Albert Einstein)"
    },
    {
        "path": os.path.join(eval_dir, "real_lena.mp4"),
        "ground_truth": "REAL_VIDEO",
        "subtype": "REAL_PORTRAIT",
        "split": "test",
        "description": "Authentic standard reference portrait (Lena Söderberg)"
    },
    {
        "path": r"C:\Users\LOQ\Downloads\WhatsApp Video 2026-08-25 at 7.47.51 PM.mp4",
        "ground_truth": "REAL_VIDEO",
        "subtype": "REAL_MOBILE",
        "split": "test",
        "description": "Authentic mobile WhatsApp handheld video (2 human faces)"
    },
    # --- AI / SYNTHETIC VIDEOS ---
    {
        "path": os.path.join(eval_dir, "ai_stylegan.mp4"),
        "ground_truth": "AI_GENERATED",
        "subtype": "AI_STYLEGAN_FACESWAP",
        "split": "calibration",
        "description": "StyleGAN synthetic multi-face portrait sequence"
    },
    {
        "path": r"C:\Users\LOQ\Downloads\Create_a_video_of_this_person.mp4",
        "ground_truth": "AI_GENERATED",
        "subtype": "AI_DIFFUSION_VEO",
        "split": "test",
        "description": "Google Veo diffusion synthetic portrait video"
    },
    {
        "path": r"C:\Users\LOQ\Downloads\Professional_speaking_to_camera_202608272223.mp4",
        "ground_truth": "AI_GENERATED",
        "subtype": "AI_DIFFUSION_VEO",
        "split": "test",
        "description": "Google Veo diffusion synthetic presenter video"
    },
]


def run_benchmark(dataset=None, output_json=None):
    if dataset is None:
        dataset = CURATED_DATASET

    print("=" * 85)
    print("      VIDEO DETECTION MODEL VIDEO FORENSICS — EMPIRICAL BENCHMARK EVALUATION")
    print("=" * 85)
    print(f"Total Videos in Benchmark: {len(dataset)}")
    
    results = []
    
    for item in dataset:
        path = item["path"]
        gt = item["ground_truth"]
        subtype = item.get("subtype", "UNKNOWN")
        split = item.get("split", "test")
        desc = item.get("description", "")
        fname = os.path.basename(path)
        
        if not os.path.exists(path):
            print(f"\n[SKIP] Video not found: {path}")
            continue
            
        print(f"\nEvaluating [{split.upper()}] ({subtype}): {fname}")
        print(f"  Description : {desc}")
        print(f"  Ground Truth: {gt}")
        
        t0 = time.perf_counter()
        try:
            analysis = _analyze_video(path, fname)
            elapsed = time.perf_counter() - t0
            
            pred_status = analysis.get("final_status") or analysis.get("status")
            pred_verdict = analysis.get("verdict")
            ai_prob = analysis.get("ai_probability")
            real_prob = analysis.get("real_probability")
            confidence = analysis.get("confidence")
            frames_scored = analysis.get("frames_scored", 0)
            frames_analyzed = analysis.get("frames_analyzed", 0)
            flagged = analysis.get("ai_analysis", {}).get("frames_flagged", 0)
            reason = analysis.get("reason", "")
            
            # Match evaluation
            is_correct = (pred_status == gt)
            is_inconclusive = (pred_status == "INCONCLUSIVE")
            
            res_entry = {
                "file": fname,
                "split": split,
                "subtype": subtype,
                "ground_truth": gt,
                "predicted": pred_status,
                "verdict": pred_verdict,
                "ai_probability": round(ai_prob, 4) if ai_prob is not None else None,
                "real_probability": round(real_prob, 4) if real_prob is not None else None,
                "confidence": round(confidence, 4) if confidence is not None else None,
                "frames_analyzed": frames_analyzed,
                "frames_scored": frames_scored,
                "flagged_frames": flagged,
                "processing_time_sec": round(elapsed, 2),
                "correct": is_correct,
                "inconclusive": is_inconclusive,
                "reason": reason
            }
            results.append(res_entry)
            
            status_tag = "CORRECT" if is_correct else ("INCONCLUSIVE" if is_inconclusive else "INCORRECT")
            print(f"  -> Predicted: {pred_status} (AI: {ai_prob:.1%}, Real: {real_prob:.1%}) [{status_tag}]")
            print(f"  -> Time: {elapsed:.2f}s | Scored Frames: {frames_scored}/{frames_analyzed}")
            print(f"  -> Reason: {reason}")
            
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  -> ERROR analyzing video ({elapsed:.2f}s): {e}")
            results.append({
                "file": fname,
                "split": split,
                "subtype": subtype,
                "ground_truth": gt,
                "predicted": "ERROR",
                "processing_time_sec": round(elapsed, 2),
                "correct": False,
                "inconclusive": True,
                "error": str(e)
            })

    def summarize_metrics(subset, title):
        print("\n" + "=" * 80)
        print(f"  {title} (N = {len(subset)})")
        print("=" * 80)
        
        if not subset:
            print("  No samples in this partition.")
            return {}
            
        total_real = sum(1 for r in subset if r["ground_truth"] == "REAL_VIDEO")
        total_ai = sum(1 for r in subset if r["ground_truth"] == "AI_GENERATED")
        
        # Breakdown by manipulation subtype
        stylegan_subset = [r for r in subset if "STYLEGAN" in r.get("subtype", "") or "FACESWAP" in r.get("subtype", "")]
        diffusion_subset = [r for r in subset if "DIFFUSION" in r.get("subtype", "") or "VEO" in r.get("subtype", "")]
        
        tp = sum(1 for r in subset if r["ground_truth"] == "AI_GENERATED" and r["predicted"] == "AI_GENERATED")
        tn = sum(1 for r in subset if r["ground_truth"] == "REAL_VIDEO" and r["predicted"] == "REAL_VIDEO")
        fp = sum(1 for r in subset if r["ground_truth"] == "REAL_VIDEO" and r["predicted"] == "AI_GENERATED")
        fn = sum(1 for r in subset if r["ground_truth"] == "AI_GENERATED" and r["predicted"] == "REAL_VIDEO")
        inc_real = sum(1 for r in subset if r["ground_truth"] == "REAL_VIDEO" and r["predicted"] == "INCONCLUSIVE")
        inc_ai = sum(1 for r in subset if r["ground_truth"] == "AI_GENERATED" and r["predicted"] == "INCONCLUSIVE")
        total_inc = inc_real + inc_ai
        total = len(subset)
        
        times = [r["processing_time_sec"] for r in subset if "processing_time_sec" in r]
        avg_time = statistics.mean(times) if times else 0.0
        
        print("  CONFUSION MATRIX:")
        print("  -------------------------------------------------------------")
        print(f"                   Pred: REAL    Pred: AI    Pred: INCONCLUSIVE   Total")
        print(f"    Actual REAL:     {tn:4d}         {fp:4d}             {inc_real:4d}          {total_real:4d}")
        print(f"    Actual AI:       {fn:4d}         {tp:4d}             {inc_ai:4d}          {total_ai:4d}")
        print(f"    Total:           {tn+fn:4d}         {fp+tp:4d}             {total_inc:4d}          {total:4d}")
        print("  -------------------------------------------------------------")
        
        strict_acc = (tp + tn) / total if total > 0 else 0.0
        decided = total - total_inc
        decided_acc = (tp + tn) / decided if decided > 0 else 0.0
        precision_ai = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_ai = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_ai = (2 * precision_ai * recall_ai / (precision_ai + recall_ai)) if (precision_ai + recall_ai) > 0 else 0.0
        fpr = fp / total_real if total_real > 0 else 0.0
        fnr = fn / total_ai if total_ai > 0 else 0.0
        
        print(f"\n  PERFORMANCE METRICS:")
        print(f"    • Strict Accuracy (All Cases)    : {strict_acc:.1%} ({tp+tn}/{total})")
        print(f"    • Decided Cases Accuracy         : {decided_acc:.1%} ({tp+tn}/{decided})")
        print(f"    • Precision (AI Class)           : {precision_ai:.1%}")
        print(f"    • Recall (AI Class)              : {recall_ai:.1%}")
        print(f"    • F1 Score (AI Class)            : {f1_ai:.3f}")
        print(f"    • False Positive Rate (Real as AI): {fpr:.1%} ({fp}/{total_real}) [Target: 0%]")
        print(f"    • False Negative Rate (AI as Real): {fnr:.1%} ({fn}/{total_ai})")
        print(f"    • Inconclusive Rate              : {total_inc/total:.1%} ({total_inc}/{total})")
        print(f"    • Average Processing Time        : {avg_time:.2f}s per video")
        
        # Subtype Breakdown
        if stylegan_subset:
            sg_tp = sum(1 for r in stylegan_subset if r["predicted"] == "AI_GENERATED")
            print(f"    -> StyleGAN / Face-Swap AI Detection Rate : {sg_tp}/{len(stylegan_subset)} ({sg_tp/len(stylegan_subset):.1%})")
        if diffusion_subset:
            diff_tp = sum(1 for r in diffusion_subset if r["predicted"] == "AI_GENERATED")
            print(f"    -> Modern Diffusion AI Detection Rate    : {diff_tp}/{len(diffusion_subset)} ({diff_tp/len(diffusion_subset):.1%})")

        return {
            "title": title,
            "total_samples": total,
            "total_real": total_real,
            "total_ai": total_ai,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "inconclusive_real": inc_real,
            "inconclusive_ai": inc_ai,
            "total_inconclusive": total_inc,
            "strict_accuracy": round(strict_acc, 4),
            "decided_accuracy": round(decided_acc, 4),
            "precision_ai": round(precision_ai, 4),
            "recall_ai": round(recall_ai, 4),
            "f1_ai": round(f1_ai, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "avg_processing_time_sec": round(avg_time, 2)
        }

    test_subset = [r for r in results if r.get("split") == "test"]
    test_metrics = summarize_metrics(test_subset, "PARTITION 1: HELD-OUT TEST SET (Strict Unseen Videos)")
    all_metrics = summarize_metrics(results, "PARTITION 2: OVERALL BENCHMARK (Calibration + Held-Out)")

    if output_json is None:
        output_json = os.path.join(os.path.dirname(__file__), "storage", "reports", "benchmark_results.json")
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "test_metrics": test_metrics,
            "all_metrics": all_metrics,
            "details": results
        }, f, indent=2)
        
    print(f"\n[OK] Benchmark completed successfully. Saved to: {output_json}")
    return test_metrics, all_metrics


def build_custom_dataset(directory, label):
    dataset = []
    exts = (".mp4", ".mov", ".avi", ".webm", ".mkv")
    for fname in os.listdir(directory):
        if fname.lower().endswith(exts):
            full_path = os.path.join(directory, fname)
            dataset.append({
                "path": full_path,
                "ground_truth": label,
                "split": "test",
                "subtype": "CUSTOM",
                "description": f"Custom evaluation video: {fname}"
            })
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Detection Model Video Forensics Benchmark")
    parser.add_argument("--real-dir", type=str, help="Directory containing real test videos")
    parser.add_argument("--ai-dir", type=str, help="Directory containing AI test videos")
    parser.add_argument("--out", type=str, default=None, help="Output path for JSON results")
    args = parser.parse_args()
    
    if args.real_dir or args.ai_dir:
        custom_ds = []
        if args.real_dir and os.path.exists(args.real_dir):
            custom_ds.extend(build_custom_dataset(args.real_dir, "REAL_VIDEO"))
        if args.ai_dir and os.path.exists(args.ai_dir):
            custom_ds.extend(build_custom_dataset(args.ai_dir, "AI_GENERATED"))
        run_benchmark(custom_ds, output_json=args.out)
    else:
        run_benchmark(output_json=args.out)
