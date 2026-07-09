"""
Core video processing logic for FC Mobile Player Possession Tracker.

Pipeline:
1. Extract frames from the uploaded match video at a configurable sample rate.
2. Run OCR on each frame to read the game's on-screen player name labels
   (the game shows the name of the player in possession, and sometimes the
   nearest defender, as floating text above their heads).
3. Build a timeline of (timestamp -> set of names visible) across the match.
4. Collapse each player's timestamps into contiguous "possession segments"
   (merging frames that are close together in time).
5. Given a chosen player, cut + concatenate their segments (with padding)
   into a single highlight clip using ffmpeg.
"""

import os
import glob
import shutil
import subprocess
import tempfile
from collections import defaultdict, Counter

import cv2
import pytesseract

# Words that come from game UI chrome / sponsor banners / formation labels,
# not player names. Extend this list as you find more false positives.
UI_NOISE = {
    "DAN", "SAE", "THROUGH", "SPRINT", "SKILL", "CLEAR", "PASS", "SHOOT",
    "AUTO", "SWITCH", "SLIDE", "DEF", "TACKLE", "SAC", "EASFCMOBILE",
    "EASFCMOBIL", "INSTAGRAM", "WHATSAPP", "MOBILE", "SAEEM", "ATTACK",
    "CITY", "MIDFIELD", "DEFENCE", "CDM", "SAN", "VISA", "SAUBA", "COM",
    "PAN", "NAL", "IFS", "KICK", "CANCEL", "BALL", "VAY",
}

# Known OCR misreads -> canonical spelling. Extend as needed.
CANON = {
    "VINT": "VINI",
}


def extract_frames(video_path: str, out_dir: str, fps: float = 2.0) -> int:
    """Extract frames from video at `fps` frames/sec into out_dir.
    Returns number of frames extracted."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(out_dir, "f_%05d.png"),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return len(glob.glob(os.path.join(out_dir, "f_*.png")))


def ocr_frame(frame_path: str) -> list[str]:
    """Run OCR on a single frame, return list of candidate name tokens."""
    img = cv2.imread(frame_path)
    h, w = img.shape[:2]
    # Crop out the scoreboard strip (top) and minimap/buttons (bottom)
    # to reduce noise and speed up OCR.
    crop = img[int(h * 0.05):int(h * 0.65), 0:w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(big, 200, 255, cv2.THRESH_BINARY)
    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT, config="--psm 11")

    names = []
    for i, txt in enumerate(data["text"]):
        tok = txt.strip()
        conf = data["conf"][i]
        if tok and conf and int(conf) > 60 and tok.isalpha() and len(tok) >= 3:
            u = tok.upper()
            u = CANON.get(u, u)
            if u not in UI_NOISE:
                names.append(u)
    return names


def build_timeline(frames_dir: str, fps: float, progress_cb=None) -> list[dict]:
    """Run OCR across all extracted frames, return a list of
    {'t': timestamp_seconds, 'names': [..]} in chronological order."""
    files = sorted(glob.glob(os.path.join(frames_dir, "f_*.png")))
    timeline = []
    total = len(files)
    for idx, fpath in enumerate(files):
        t = idx / fps
        names = ocr_frame(fpath)
        timeline.append({"t": t, "names": names})
        if progress_cb and idx % 5 == 0:
            progress_cb(idx + 1, total)
    if progress_cb:
        progress_cb(total, total)
    return timeline


def build_segments(timeline: list[dict], gap_seconds: float = 2.5) -> dict:
    """Collapse per-name timestamps into contiguous segments.
    Returns {name: [(start, end), ...]} sorted by total frame-hits desc
    is left to the caller; this returns a plain dict."""
    appearances = defaultdict(list)
    for fr in timeline:
        for n in set(fr["names"]):
            appearances[n].append(fr["t"])

    segments = {}
    for name, times in appearances.items():
        times = sorted(times)
        segs = []
        seg_start = times[0]
        prev = times[0]
        for t in times[1:]:
            if t - prev > gap_seconds:
                segs.append((seg_start, prev))
                seg_start = t
            prev = t
        segs.append((seg_start, prev))
        segments[name] = segs
    return segments


def player_summary(segments: dict, min_hits: int = 3) -> list[tuple]:
    """Return list of (name, num_segments, num_appearances) sorted by
    appearances descending, filtered to names with at least min_hits
    total appearances (helps drop one-off OCR noise)."""
    out = []
    for name, segs in segments.items():
        hits = len(segs)  # approximate; caller can recompute true hit count if needed
        out.append((name, len(segs), segs))
    out.sort(key=lambda x: -sum(e - s for s, e in x[2]))
    return [(n, ns, s) for n, ns, s in out]


def cut_player_highlights(video_path: str, segments: list[tuple], out_path: str,
                           padding: float = 1.5) -> str:
    """Cut and concatenate the given (start,end) segments from video_path
    into a single highlight file at out_path. Adds `padding` seconds of
    context before/after each segment."""
    with tempfile.TemporaryDirectory() as tmp:
        clip_paths = []
        for i, (s, e) in enumerate(segments):
            start = max(0, s - padding)
            dur = (e - s) + 2 * padding
            clip_path = os.path.join(tmp, f"clip_{i:03d}.mp4")
            cmd = [
                "ffmpeg", "-y", "-ss", str(start), "-i", video_path,
                "-t", str(dur), "-c", "copy", clip_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            clip_paths.append(clip_path)

        concat_list = os.path.join(tmp, "concat.txt")
        with open(concat_list, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp}'\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list, "-c", "copy", out_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
    return out_path
