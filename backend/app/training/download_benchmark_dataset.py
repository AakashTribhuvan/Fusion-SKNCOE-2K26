"""Download a balanced subset of the deepfake audio dataset from Hugging Face.

Dataset: garystafford/deepfake-audio-detection
- Real audio from genuine human speech (YouTube interviews)
- Fake audio from ElevenLabs synthetic voice cloning
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import requests

DATASET_REPO = "garystafford/deepfake-audio-detection"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main"

def fetch_file_list() -> tuple[list[str], list[str]]:
    api_url = f"https://huggingface.co/api/datasets/{DATASET_REPO}"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    files = [f["rfilename"] for f in data.get("siblings", [])]
    
    reals = sorted([f for f in files if f.startswith("real/") and f.endswith(".flac")])
    fakes = sorted([f for f in files if f.startswith("fake/") and f.endswith(".flac")])
    return reals, fakes

def download_file(rel_path: str, dest_path: Path) -> bool:
    url = f"{BASE_URL}/{rel_path}"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.is_file() and dest_path.stat().st_size > 0:
        return True
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"Failed to download {rel_path}: {e}")
        return False

def prepare_dataset(num_samples_per_class: int = 10, output_dir: Path | None = None) -> Path:
    if output_dir is None:
        output_dir = Path("data/benchmark")
    output_dir = output_dir.resolve()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching file listings from Hugging Face: {DATASET_REPO}...")
    reals, fakes = fetch_file_list()
    print(f"Found {len(reals)} real clips and {len(fakes)} fake clips available.")

    selected_reals = reals[:num_samples_per_class]
    selected_fakes = fakes[:num_samples_per_class]

    manifest_rows = []

    print(f"Downloading {num_samples_per_class} real clips...")
    for idx, rel in enumerate(selected_reals):
        filename = Path(rel).name
        dest = audio_dir / f"real_{filename}"
        if download_file(rel, dest):
            # speaker prefix from filename e.g. yt_0000
            speaker = filename.split("_part_")[0] if "_part_" in filename else f"spk_real_{idx}"
            manifest_rows.append({
                "audio_path": f"audio/{dest.name}",
                "file_path": str(dest),
                "path": str(dest),
                "label": "bona_fide",
                "group_id": speaker,
                "source_type": "youtube_human",
                "speaker_id": speaker,
            })
            print(f"  [+] Real [{idx+1}/{num_samples_per_class}]: {dest.name}")

    print(f"Downloading {num_samples_per_class} fake clips...")
    for idx, rel in enumerate(selected_fakes):
        filename = Path(rel).name
        dest = audio_dir / f"fake_{filename}"
        if download_file(rel, dest):
            # speaker prefix e.g. el_0001
            speaker = filename.split("_part_")[0] if "_part_" in filename else f"spk_fake_{idx}"
            manifest_rows.append({
                "audio_path": f"audio/{dest.name}",
                "file_path": str(dest),
                "path": str(dest),
                "label": "spoof",
                "group_id": speaker,
                "source_type": "elevenlabs_clone",
                "speaker_id": speaker,
            })
            print(f"  [+] Fake [{idx+1}/{num_samples_per_class}]: {dest.name}")

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio_path", "file_path", "path", "label", "group_id", "source_type", "speaker_id"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest successfully created: {manifest_path}")
    print(f"Total samples: {len(manifest_rows)} ({num_samples_per_class} real, {num_samples_per_class} fake)")
    return manifest_path

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    prepare_dataset(num_samples_per_class=n)
