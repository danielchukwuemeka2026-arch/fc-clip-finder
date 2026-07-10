import os
import tempfile
import shutil

import streamlit as st

from processor import (
    extract_frames, build_timeline, build_segments, build_clip_windows,
    cut_player_highlights, cut_single_clip, get_duration_seconds,
)

MAX_DURATION_SECONDS = 4 * 60 * 60  # 4 hours — effectively no real-world cap

st.set_page_config(page_title="FC Mobile Player Clip Finder", page_icon="⚽")

st.title("⚽ FC Mobile Player Clip Finder")
st.write(
    "Upload one or more match recordings, and this will find every clip "
    "where a specific player touches the ball — using the game's own "
    "on-screen player name labels."
)

# videos: list of {"name": filename, "path": persistent path, "timeline": [...]}
if "videos" not in st.session_state:
    st.session_state.videos = []

uploaded_files = st.file_uploader(
    "Upload your match video(s) (mp4)",
    type=["mp4", "mov", "m4v"],
    accept_multiple_files=True,
)

fps = st.slider(
    "Frames analyzed per second (higher = more accurate, slower)",
    min_value=0.5, max_value=4.0, value=2.0, step=0.5,
)

if uploaded_files and st.button("Process video(s)"):
    st.session_state.videos = []  # reset for this batch

    # Validate durations up front so we don't waste time processing #1
    # only to fail on #2.
    with tempfile.TemporaryDirectory() as check_dir:
        oversized = []
        for uf in uploaded_files:
            tmp_path = os.path.join(check_dir, uf.name)
            with open(tmp_path, "wb") as f:
                f.write(uf.getbuffer())
            dur = get_duration_seconds(tmp_path)
            if dur > MAX_DURATION_SECONDS:
                oversized.append((uf.name, dur))
        if oversized:
            for name, dur in oversized:
                st.error(
                    f"'{name}' is {dur/60:.1f} minutes long — over the "
                    f"{MAX_DURATION_SECONDS/60:.0f} minute limit. Trim it or lower its "
                    f"export quality, then try again."
                )
            st.stop()

    overall_status = st.empty()
    overall_progress = st.progress(0)

    for file_idx, uf in enumerate(uploaded_files):
        overall_status.write(f"Processing video {file_idx+1} of {len(uploaded_files)}: **{uf.name}**")

        with tempfile.TemporaryDirectory() as workdir:
            video_path = os.path.join(workdir, uf.name)
            with open(video_path, "wb") as f:
                f.write(uf.getbuffer())

            # Persist outside the temp dir so we can cut clips from it later,
            # using a unique name per video so multiple files don't collide.
            persistent_video = os.path.join(tempfile.gettempdir(), f"fc_tracker_{file_idx}_{uf.name}")
            shutil.copy(video_path, persistent_video)

            frames_dir = os.path.join(workdir, "frames")
            with st.spinner(f"Extracting frames from {uf.name}..."):
                n_frames = extract_frames(video_path, frames_dir, fps=fps)

            progress_bar = st.progress(0, text=f"Reading on-screen names in {uf.name}...")

            def progress_cb(done, total, _bar=progress_bar, _name=uf.name):
                _bar.progress(min(done / total, 1.0), text=f"Reading on-screen names in {_name}... {done}/{total}")

            timeline = build_timeline(frames_dir, fps=fps, progress_cb=progress_cb)

            st.session_state.videos.append({
                "name": uf.name,
                "path": persistent_video,
                "timeline": timeline,
            })

        overall_progress.progress((file_idx + 1) / len(uploaded_files))

    overall_status.write(f"Done processing {len(uploaded_files)} video(s)! Pick a player below.")

if st.session_state.videos:
    gap_tolerance = st.slider(
        "How many seconds of missed detection to tolerate as 'still the same possession'",
        1.0, 8.0, 4.0, 0.5,
        help="If the player's name briefly isn't detected for longer than this (blocked view, "
             "OCR miss, etc.), the clip is cut there even if they still have the ball. "
             "Raise this if clips are ending too early.",
    )

    # Build segments separately per video (a gap between two different
    # videos should never be treated as one continuous possession).
    segments_per_video = [
        build_segments(v["timeline"], gap_seconds=gap_tolerance)
        for v in st.session_state.videos
    ]

    # Rank players by total on-screen possession time, summed across all
    # uploaded videos, filtering out tiny noise.
    totals = {}
    for segs in segments_per_video:
        for name, ranges in segs.items():
            totals[name] = totals.get(name, 0) + sum(e - s for s, e in ranges)
    ranked_names = sorted(totals.items(), key=lambda kv: -kv[1])
    ranked_names = [name for name, total in ranked_names if total >= 1.0]

    if not ranked_names:
        st.warning("No players detected with enough on-screen time. Try different video(s) or check the noise filters.")
    else:
        display_names = [f"{name}" for name in ranked_names]
        choice_idx = st.selectbox("Choose a player", range(len(display_names)), format_func=lambda i: display_names[i])
        chosen_name = ranked_names[choice_idx]

        col_a, col_b = st.columns(2)
        with col_a:
            max_lookback = st.slider(
                "Max seconds to look back for the pass IN",
                1.0, 10.0, 6.0, 0.5,
                help="The clip starts exactly when the previous player's touch ended "
                     "(i.e. the actual pass in), not a fixed buffer. This caps how far "
                     "back it's allowed to search for that moment.",
            )
        with col_b:
            max_lookforward = st.slider(
                "Max seconds to look forward for the pass OUT",
                1.0, 10.0, 6.0, 0.5,
                help="The clip now ends exactly when the NEXT player's touch begins "
                     "(i.e. when this player actually released the ball), not a fixed "
                     "buffer — so it won't cut off early while they still have it. This "
                     "caps how far forward it's allowed to search for that moment.",
            )

        # Gather clip windows for this player across every uploaded video,
        # tagging each one with which video it came from.
        all_clips = []  # list of (video_path, video_name, start, end)
        for v, segs in zip(st.session_state.videos, segments_per_video):
            if chosen_name not in segs:
                continue
            windows = build_clip_windows(segs, chosen_name, max_lookback=max_lookback, max_lookforward=max_lookforward)
            for (s, e) in windows:
                all_clips.append((v["path"], v["name"], s, e))

        st.write(f"**{chosen_name}** — {len(all_clips)} possession moments found across "
                 f"{len(st.session_state.videos)} video(s). Each clip runs from the pass in to the pass out.")

        if "clip_cache" not in st.session_state:
            st.session_state.clip_cache = {}

        selected_indices = []
        for i, (video_path, video_name, clip_start, clip_end) in enumerate(all_clips):
            cache_key = f"{chosen_name}_{i}_{gap_tolerance}_{max_lookback}_{max_lookforward}"
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Clip {i+1}** — from *{video_name}*, {clip_start:.1f}s to {clip_end:.1f}s")
                if st.button(f"Cut & preview clip {i+1}", key=f"cutbtn_{cache_key}"):
                    clip_path = os.path.join(tempfile.gettempdir(), f"{cache_key}.mp4")
                    with st.spinner("Cutting..."):
                        cut_single_clip(video_path, clip_start, clip_end, clip_path)
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
            st.write(f"**{len(selected_indices)} clip(s) selected** for a combined reel "
                     f"(can span multiple uploaded videos).")
            if st.button("Download combined reel of selected clips"):
                selected_clips = [
                    (all_clips[i][0], all_clips[i][2], all_clips[i][3])
                    for i in selected_indices
                ]
                out_path = os.path.join(tempfile.gettempdir(), f"{chosen_name}_selected_highlights.mp4")
                with st.spinner("Combining selected clips..."):
                    cut_player_highlights(selected_clips, out_path)
                st.video(out_path)
                with open(out_path, "rb") as f:
                    st.download_button("Download combined reel", f, file_name=f"{chosen_name}_selected_highlights.mp4")

st.divider()
st.caption(
    "Note: name detection relies on reading the game's own UI text, so quality "
    "depends on video resolution and clarity. Expect occasional noise/misses."
)
