import sys
import os

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import requests
import json

API_URL = "http://127.0.0.1:8000/api/scan-video"
os.makedirs("storage/eval_videos", exist_ok=True)

def create_video_from_image(image_path, output_video_path, num_frames=120, fps=30.0):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read {image_path}")
    h, w = img.shape[:2]
    # Resize if too large for quick processing
    if max(h, w) > 1080:
        scale = 1080 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]
    # Ensure even dimensions for codec
    w = w - (w % 2)
    h = h - (h % 2)
    img = cv2.resize(img, (w, h))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    for _ in range(num_frames):
        writer.write(img)
    writer.release()
    return output_video_path

# 1. Real Video: Lena
real_vid_1 = "storage/eval_videos/real_lena.mp4"
create_video_from_image("storage/test_samples/sample_real.jpg", real_vid_1)

# 2. Real Video: Obama
real_vid_2 = "storage/eval_videos/real_obama.mp4"
create_video_from_image("storage/test_samples/obama.jpg", real_vid_2)

# 3. Real Video: Einstein
real_vid_3 = "storage/eval_videos/real_einstein.mp4"
create_video_from_image("storage/test_samples/einstein.jpg", real_vid_3)

# 4. AI Video: StyleGAN
ai_vid_1 = "storage/eval_videos/ai_stylegan.mp4"
create_video_from_image("storage/test_samples/stylegan_fake.png", ai_vid_1)

eval_samples = [
    ("Real: Lena", real_vid_1, "REAL_VIDEO"),
    ("Real: Obama", real_vid_2, "REAL_VIDEO"),
    ("Real: Einstein", real_vid_3, "REAL_VIDEO"),
    ("AI: StyleGAN", ai_vid_1, "AI_GENERATED"),
]

print("=" * 80)
print("EVALUATING VIDEOS VIA LIVE /api/scan-video ENDPOINT")
print("=" * 80)

results = []
for label, vid_path, expected in eval_samples:
    print(f"\nScanning: {label} ({vid_path})...")
    with open(vid_path, "rb") as f:
        res = requests.post(API_URL, files={"file": (os.path.basename(vid_path), f, "video/mp4")})
    if res.status_code != 200:
        print(f"  [ERROR] HTTP {res.status_code}: {res.text}")
        continue
    data = res.json()
    verdict = data.get("verdict")
    final_status = data.get("final_status")
    ai_prob = data.get("ai_probability")
    real_prob = data.get("real_probability")
    conf = data.get("confidence")
    scored = data.get("frames_scored")
    print(f"  Final Status: {final_status} | Verdict: {verdict}")
    print(f"  AI Prob: {ai_prob:.4f} | Real Prob: {real_prob:.4f} | Conf: {conf:.4f} | Scored: {scored}")
    print(f"  Reason: {data.get('reason')}")
    results.append({
        "label": label,
        "expected": expected,
        "final_status": final_status,
        "verdict": verdict,
        "ai_prob": ai_prob,
        "real_prob": real_prob,
        "scored": scored,
        "match": final_status == expected
    })

print("\n" + "=" * 80)
print("SUMMARY RESULTS")
print("=" * 80)
for r in results:
    status_sym = "[PASS]" if r["match"] else "[CHECK]"
    print(f"{status_sym} {r['label']}: Expected={r['expected']}, Got={r['final_status']} (AI={r['ai_prob']:.2f}, Real={r['real_prob']:.2f})")
