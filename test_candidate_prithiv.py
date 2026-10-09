import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from deepface import DeepFace

model_id = 'prithivMLmods/Deep-Fake-Detector-Model'
print(f"Loading {model_id}...")
proc = AutoImageProcessor.from_pretrained(model_id)
m = AutoModelForImageClassification.from_pretrained(model_id)
m.eval()

videos = [
    ('Veo AI 1', r'C:\Users\LOQ\Downloads\Create_a_video_of_this_person.mp4', 'AI'),
    ('Veo AI 2', r'C:\Users\LOQ\Downloads\Professional_speaking_to_camera_202608272223.mp4', 'AI'),
    ('StyleGAN AI', r'C:\Users\LOQ\OneDrive\Desktop\FusionHackathon\Swaraksha-publish\storage\eval_videos\ai_stylegan.mp4', 'AI'),
    ('Real Obama', r'C:\Users\LOQ\OneDrive\Desktop\FusionHackathon\Swaraksha-publish\storage\eval_videos\real_obama.mp4', 'REAL'),
    ('Real Einstein', r'C:\Users\LOQ\OneDrive\Desktop\FusionHackathon\Swaraksha-publish\storage\eval_videos\real_einstein.mp4', 'REAL'),
    ('Real Lena', r'C:\Users\LOQ\OneDrive\Desktop\FusionHackathon\Swaraksha-publish\storage\eval_videos\real_lena.mp4', 'REAL'),
    ('Real WhatsApp', r'C:\Users\LOQ\Downloads\WhatsApp Video 2026-08-25 at 7.47.51 PM.mp4', 'REAL'),
]

correct_count = 0
total_count = len(videos)

for name, p, gt in videos:
    cap = cv2.VideoCapture(p)
    f_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = [int(f_cnt * r) for r in [0.25, 0.50, 0.75]]
    frame_scores = []
    for f_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            try:
                objs = DeepFace.extract_faces(frame, detector_backend='retinaface', enforce_detection=False)
                if objs:
                    fa = objs[0]['facial_area']
                    x, y, w, h = fa['x'], fa['y'], fa['w'], fa['h']
                    pad = int(0.35 * max(w, h))
                    H, W, _ = frame.shape
                    crop = frame[max(0, y-pad):min(H, y+h+pad), max(0, x-pad):min(W, x+w+pad)]
                    if crop.size > 0:
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        inp = proc(images=Image.fromarray(rgb), return_tensors='pt')
                        with torch.no_grad():
                            probs = F.softmax(m(**inp).logits, dim=-1)[0]
                            # 0: Fake, 1: Real
                            frame_scores.append(probs[0].item())
            except Exception as e:
                pass
    cap.release()
    avg_fake = sum(frame_scores)/len(frame_scores) if frame_scores else 0.5
    pred = 'AI' if avg_fake >= 0.50 else 'REAL'
    is_corr = (pred == gt)
    if is_corr:
        correct_count += 1
    tag = "CORRECT" if is_corr else "INCORRECT"
    scores_str = [round(s, 3) for s in frame_scores]
    print(f"{name:<15} [GT={gt}]: Fake={avg_fake:.1%}, Pred={pred} -> {tag} (scores: {scores_str})")

print("=" * 60)
print(f"Total Accuracy: {correct_count}/{total_count} ({correct_count/total_count:.1%})")
