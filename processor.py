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


def get_duration_seconds(video_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, check=True, text=True)
    import json
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


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


def build_segments(timeline: list[dict], gap_seconds: float = 4.0) -> dict:
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


def build_clip_windows(segments: dict, chosen_name: str, pad_after: float = 1.5,
                        max_lookback: float = 6.0, fallback_pad_before: float = 2.0) -> list[tuple]:
    """For each of chosen_name's possession segments, compute the actual
    (clip_start, clip_end) to cut.

    clip_start reaches back to the moment the PREVIOUS player's touch ended
    (i.e. when the pass was actually played) rather than an arbitrary fixed
    number of seconds — so the clip shows exactly that one pass in, not the
    whole prior buildup.

    If no other player's possession is found within `max_lookback` seconds
    before this segment starts (e.g. it's the kickoff, or detection missed
    the previous touch), falls back to `fallback_pad_before` seconds of
    fixed padding so we still get some lead-in.
    """
    # Flatten every player's segments into one chronological list.
    all_segs = []
    for name, segs in segments.items():
        for s, e in segs:
            all_segs.append((s, e, name))
    all_segs.sort(key=lambda x: x[0])

    chosen_segs = sorted(segments.get(chosen_name, []), key=lambda x: x[0])
    windows = []
    for s, e in chosen_segs:
        # Find the most recent OTHER-player segment that ended at or before
        # this one starts.
        prev_end = None
        for (ps, pe, pname) in all_segs:
            if pname == chosen_name:
                continue
            if pe <= s + 1e-6 and (prev_end is None or pe > prev_end):
                prev_end = pe

        if prev_end is not None and (s - prev_end) <= max_lookback:
            # Start just slightly before the previous player's touch ended,
            # so we catch the kick/pass motion itself rather than starting
            # a frame after the ball has already left.
            clip_start = max(0, prev_end - 0.3)
        else:
            clip_start = max(0, s - fallback_pad_before)

        clip_end = e + pad_after
        windows.append((clip_start, clip_end))
    return windows


def cut_single_clip(video_path: str, start: float, end: float, out_path: str) -> str:
    """Cut one clip from video_path covering the exact [start, end] window
    (in seconds). Re-encodes (rather than stream-copying) so the clip
    starts/ends cleanly on exact frame boundaries instead of the nearest
    keyframe, which is what caused the freezing/skipping you saw with the
    old -c copy approach."""
    dur = max(0.1, end - start)
    cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-i", video_path,
        "-t", str(dur),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac",
        out_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out_path


def cut_player_highlights(video_path: str, windows: list[tuple], out_path: str) -> str:
    """Cut and concatenate the given (start,end) windows from video_path
    into a single highlight file at out_path."""
    with tempfile.TemporaryDirectory() as tmp:
        clip_paths = []
        for i, (s, e) in enumerate(windows):
            clip_path = os.path.join(tmp, f"clip_{i:03d}.mp4")
            cut_single_clip(video_path, s, e, clip_path)
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
