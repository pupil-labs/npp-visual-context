"""Scripted caption generation runner for Neon Player recordings.

Runs VisualContextPlugin in headless/scripted mode directly on a recording,
writing to `<recording>/.neon_player/cache/VisualContextPlugin/visual_context.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure shared plugin dependencies (mlx_vlm, etc.) are importable
PLUGIN_SITE_PACKAGES = (
    Path.home() / "Pupil Labs" / "Neon Player" / "plugins" / "site-packages"
)
if PLUGIN_SITE_PACKAGES.exists() and str(PLUGIN_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SITE_PACKAGES))

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pupil_labs.neon_recording import NeonRecording  # ruff: ignore[module-import-not-at-top-of-file]
from PySide6.QtWidgets import QApplication  # ruff: ignore[module-import-not-at-top-of-file]
from visual_context import (  # ruff: ignore[module-import-not-at-top-of-file]
    CaptionModel,
    PromptTemplate,
    VisualContextPlugin,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate VLM scene captions for a Neon recording via script."
    )
    parser.add_argument(
        "recording_path",
        type=Path,
        nargs="?",
        default=Path("/Users/mgg/Desktop/recs/Demo Recording"),
        help="Path to Neon recording folder",
    )
    parser.add_argument(
        "--model",
        choices=["Fast_2B", "Balanced_4B", "Quality_8B"],
        default="Fast_2B",
        help="VLM model preset (default: Fast_2B for speed & memory safety)",
    )
    parser.add_argument(
        "--template",
        choices=[t.name for t in PromptTemplate],
        default="Event_Gaze_Aware",
        help="Prompt template (default: Event_Gaze_Aware)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between sampled frames (default: 2.0)",
    )
    parser.add_argument(
        "--no-skip-similar",
        action="store_true",
        help="Disable skipping near-identical frames",
    )

    args = parser.parse_args()

    # 1. Initialize headless Qt application
    _app = QApplication.instance() or QApplication(["neon-player", "--headless"])

    # 2. Load recording
    rec_path = args.recording_path.resolve()
    if not rec_path.exists():
        print(f"Error: Recording path does not exist: {rec_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading recording: {rec_path.name}...")
    recording = NeonRecording(rec_path)
    duration_s = (recording.scene.time[-1] - recording.scene.time[0]) / 1e9
    n_frames = len(recording.scene.time)
    n_gaze = len(recording.gaze.time)
    print(
        f"Recording loaded: duration={duration_s:.1f}s ({duration_s / 60:.1f}m), "
        f"scene_frames={n_frames}, gaze_samples={n_gaze}"
    )

    # 3. Configure plugin
    plugin = VisualContextPlugin()
    plugin._recording_override = recording
    VisualContextPlugin.recording = property(
        lambda self: getattr(self, "_recording_override", None)
    )

    plugin.model = getattr(CaptionModel, args.model)
    plugin.prompt_template = getattr(PromptTemplate, args.template)
    plugin.interval_s = args.interval
    plugin.skip_similar = not args.no_skip_similar

    print("\nConfiguration:")
    print(f"  Model:           {plugin.model.value}")
    print(f"  Prompt Template: {plugin.prompt_template.value}")
    print(f"  Interval:        {plugin.interval_s}s")
    print(f"  Skip Similar:    {plugin.skip_similar}")

    # 4. Run generator
    print("\nStarting caption generation...")
    t_start = time.time()
    generator = plugin.bg_generate_captions()

    last_fraction = 0.0
    try:
        for update in generator:
            prog = getattr(update, "progress", getattr(update, "fraction", 0.0))
            pct = prog * 100
            if pct - last_fraction >= 1.0 or pct >= 100.0:
                elapsed = time.time() - t_start
                print(
                    f"\rProgress: {pct:5.1f}% | Elapsed: {elapsed:5.1f}s",
                    end="",
                    flush=True,
                )
                last_fraction = pct
    except Exception as exc:  # ruff: ignore[blind-except]
        print(f"\nError during captioning: {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s ({elapsed / 60:.1f}m)!")

    # 5. Report results
    cache_path = plugin.get_cache_path(recording=recording)
    if cache_path:
        cache_file = cache_path / "visual_context.json"
        if cache_file.exists():
            with cache_file.open("r") as f:
                data = json.load(f)
            captions = data.get("captions", [])
            print(f"Successfully generated and cached {len(captions)} captions at:")
            print(f"  {cache_file}")
            if captions:
                print("\nSample captions:")
                for c in captions[:3]:
                    print(f"  [{c['start_s']:.1f}s]: {c['text'][:120]}...")
                if len(captions) > 3:
                    last = captions[-1]
                    print(f"  [{last['start_s']:.1f}s]: {last['text'][:120]}...")


if __name__ == "__main__":
    main()
