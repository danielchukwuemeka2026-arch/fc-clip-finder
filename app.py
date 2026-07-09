import os
import tempfile
import shutil

import streamlit as st

from processor import extract_frames, build_timeline, build_segments, cut_player_highlights

st.set_page_config(page_title="FC Mobile Player Clip Finder", page_icon="⚽")

st.title("⚽ FC Mobile Player Clip Finder")
st.write(
    "Upload a match recording, and this will find every clip where a "
    "specific player touches the ball — using the game's own on-screen "
    "player name labels."
)

if "timeline" not in st.session_state:
    st.session_state.timeline = None
    st.session_state.segments = None
    st.session_state.video_path = None

uploaded = st.file_uploader("Upload your match video (mp4)", type=["mp4", "mov", "m4v"])

fps = st.slider(
    "Frames analyzed per second (higher = more accurate, slower)",
    min_value=0.5, max_value=4.0, value=2.0, step=0.5,
)

if uploaded is not None and st.button("Process match"):
    with tempfile.TemporaryDirectory() as workdir:
        video_path = os.path.join(workdir, "match.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded.read())

        # Persist video outside the temp dir so we can cut clips from it later
        persistent_video = os.path.join(tempfile.gettempdir(), "fc_tracker_match.mp4")
        shutil.copy(video_path, persistent_video)
        st.session_state.video_path = persistent_video

        frames_dir = os.path.join(workdir, "frames")
        with st.spinner("Extracting frames..."):
            n_frames = extract_frames(video_path, frames_dir, fps=fps)
        st.write(f"Extracted {n_frames} frames.")

        progress_bar = st.progress(0, text="Reading on-screen player names...")

        def progress_cb(done, total):
            progress_bar.progress(min(done / total, 1.0), text=f"Reading on-screen player names... {done}/{total}")

        timeline = build_timeline(frames_dir, fps=fps, progress_cb=progress_cb)
        segments = build_segments(timeline)

        st.session_state.timeline = timeline
        st.session_state.segments = segments
        st.success("Done! Pick a player below.")

if st.session_state.segments:
    segments = st.session_state.segments
    # Rank players by total on-screen possession time, filter tiny noise
    ranked = sorted(
        segments.items(),
        key=lambda kv: -sum(e - s for s, e in kv[1]),
    )
    ranked = [(name, segs) for name, segs in ranked if sum(e - s for s, e in segs) >= 1.0]

    if not ranked:
        st.warning("No players detected with enough on-screen time. Try a different video or lower the noise filters.")
    else:
        names = [f"{name} ({len(segs)} clips)" for name, segs in ranked]
        choice_idx = st.selectbox("Choose a player", range(len(names)), format_func=lambda i: names[i])
        chosen_name, chosen_segs = ranked[choice_idx]

        st.write(f"**{chosen_name}** — {len(chosen_segs)} possession moments found:")
        for s, e in chosen_segs:
            st.write(f"- {s:.1f}s to {e:.1f}s")

        padding = st.slider("Padding around each clip (seconds)", 0.0, 4.0, 1.5, 0.5)

        if st.button("Generate highlight reel"):
            out_path = os.path.join(tempfile.gettempdir(), f"{chosen_name}_highlights.mp4")
            with st.spinner("Cutting clips..."):
                cut_player_highlights(st.session_state.video_path, chosen_segs, out_path, padding=padding)
            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button("Download highlight reel", f, file_name=f"{chosen_name}_highlights.mp4")

st.divider()
st.caption(
    "Note: name detection relies on reading the game's own UI text, so quality "
    "depends on video resolution and clarity. Expect occasional noise/misses."
)
