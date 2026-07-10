import os
import tempfile
import shutil

import streamlit as st

from processor import extract_frames, build_timeline, build_segments, build_clip_windows, cut_player_highlights, cut_single_clip, get_duration_seconds

MAX_DURATION_SECONDS = 10 * 60  # 10 minute cap

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

        duration = get_duration_seconds(video_path)
        if duration > MAX_DURATION_SECONDS:
            st.error(
                f"This video is {duration/60:.1f} minutes long. The current limit is "
                f"10 minutes. Please trim the video or lower its export quality to "
                f"reduce length/size, then try again."
            )
            st.stop()

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

        col_a, col_b = st.columns(2)
        with col_a:
            max_lookback = st.slider(
                "Max seconds to look back for the pass",
                1.0, 10.0, 6.0, 0.5,
                help="The clip starts exactly when the previous player's touch ended "
                     "(i.e. the actual pass), not a fixed buffer. This caps how far "
                     "back it's allowed to search for that moment.",
            )
        with col_b:
            pad_after = st.slider(
                "Seconds to include AFTER",
                0.0, 6.0, 1.5, 0.5,
            )

        clip_windows = build_clip_windows(
            segments, chosen_name, pad_after=pad_after, max_lookback=max_lookback,
        )

        st.write(f"**{chosen_name}** — {len(clip_windows)} possession moments found. "
                 f"Each clip starts right at the pass in — preview them and pick your favorites.")

        # Cache cut clips per (player, segment index, settings) so we don't
        # re-cut on every Streamlit rerun (e.g. when a checkbox is toggled).
        if "clip_cache" not in st.session_state:
            st.session_state.clip_cache = {}

        selected_indices = []
        for i, (clip_start, clip_end) in enumerate(clip_windows):
            cache_key = f"{chosen_name}_{i}_{max_lookback}_{pad_after}"
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Clip {i+1}** — {clip_start:.1f}s to {clip_end:.1f}s")
                if st.button(f"Cut & preview clip {i+1}", key=f"cutbtn_{cache_key}"):
                    clip_path = os.path.join(tempfile.gettempdir(), f"{cache_key}.mp4")
                    with st.spinner("Cutting..."):
                        cut_single_clip(st.session_state.video_path, clip_start, clip_end, clip_path)
                    st.session_state.clip_cache[cache_key] = clip_path

                if cache_key in st.session_state.clip_cache:
                    clip_path = st.session_state.clip_cache[cache_key]
                    st.video(clip_path)
                    with open(clip_path, "rb") as f:
                        st.download_button(
                            f"Download clip {i+1}", f,
                            file_name=f"{chosen_name}_clip{i+1}.mp4",
                            key=f"dl_{cache_key}",
                        )
            with col2:
                if cache_key in st.session_state.clip_cache:
                    include = st.checkbox("Include in combined reel", key=f"chk_{cache_key}")
                    if include:
                        selected_indices.append(i)
            st.divider()

        if selected_indices:
            st.write(f"**{len(selected_indices)} clip(s) selected** for a combined reel.")
            if st.button("Download combined reel of selected clips"):
                selected_windows = [clip_windows[i] for i in selected_indices]
                out_path = os.path.join(tempfile.gettempdir(), f"{chosen_name}_selected_highlights.mp4")
                with st.spinner("Combining selected clips..."):
                    cut_player_highlights(st.session_state.video_path, selected_windows, out_path)
                st.video(out_path)
                with open(out_path, "rb") as f:
                    st.download_button("Download combined reel", f, file_name=f"{chosen_name}_selected_highlights.mp4")

st.divider()
st.caption(
    "Note: name detection relies on reading the game's own UI text, so quality "
    "depends on video resolution and clarity. Expect occasional noise/misses."
)
