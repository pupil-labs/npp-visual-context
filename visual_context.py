# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx-vlm>=0.3.4; sys_platform == 'darwin' and platform_machine == 'arm64'",
#     "torch; sys_platform != 'darwin' or platform_machine != 'arm64'",
#     "transformers>=4.57; sys_platform != 'darwin' or platform_machine != 'arm64'",
#     "accelerate; sys_platform != 'darwin' or platform_machine != 'arm64'",
#     "numpy",
#     "pillow",
# ]
# ///

"""Visual Context Plugin -- local VLM scene captioning.

Algorithms:
- Frame sampling: fixed-interval selection over `recording.scene.time`
  (absolute nanosecond UTC timestamps) via binary search; only the sampled
  frames are decoded.
- Redundancy skip: mean absolute difference of 64x64 grayscale thumbnails
  against the last frame that was actually captioned (not merely the last
  one sampled, so slow drift still eventually forces a fresh caption).
- Captioning (Apple Silicon): mlx-vlm running Qwen3-VL Instruct 4-bit on
  Metal via MLX.
- Captioning (elsewhere): HuggingFace transformers `AutoModelForImageTextToText`
  running the same Qwen3-VL Instruct checkpoints unquantized.
- Timestamps: captions carry absolute nanosecond timestamps taken directly
  from the scene video's own time array, so no epoch normalization is needed.
"""

from __future__ import annotations

import bisect
import csv
import enum
import gc
import html
import json
import logging
import math
import operator
import platform
import re
import sys
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pupil_labs.neon_player import Plugin, ProgressUpdate, action
from pupil_labs.neon_player.plugins.events import EventsPlugin
from PySide6.QtCore import QObject, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QPainter,
    QPen,
    QTextDocument,
    QTextOption,
)
from PySide6.QtTextToSpeech import QTextToSpeech
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
)
from qt_property_widgets.utilities import (
    PersistentPropertiesMixin,
    action_params,
    property_params,
)
from qt_property_widgets.widgets import (
    BoolWidget,
    ColorWidget,
    Expander,
    PropertyForm,
    PropertyWidget,
)

logger = logging.getLogger(__name__)

CACHE_VERSION = 3
CACHE_FILENAME = "visual_context.json"

#: Longest image edge handed to the VLM. Scene video is 1600x1200; every
#: supported model down-samples internally anyway, so shrinking first is a
#: pure win on both preprocessing time and peak memory.
#: Reduced to 384 to prevent macOS GPU Timeout Error during prefill.
MAX_IMAGE_EDGE = 384

#: Size of the grayscale thumbnail used for the redundancy check.
THUMB_EDGE = 64

#: Fallback frame duration used only when the scene timestamps give us nothing
#: to measure (a single-frame recording). 30 fps.
FALLBACK_FRAME_DURATION_NS = 33_333_333

#: Fraction of the frame height reserved for the caption band.
CAPTION_BAND_TOP = 0.70
CAPTION_BAND_HEIGHT = 0.28
CAPTION_BAND_WIDTH = 0.85

DEFAULT_PROMPT = (
    "Describe what is visible in this first-person scene camera image in one "
    "short sentence. Name the main objects, people and activity. Do not "
    "speculate about anything you cannot see."
)

EVENT_GAZE_SYSTEM_PROMPT = """\
You are a visual scene understanding system analyzing sequential frames
recorded by the scene camera of a Pupil Labs Neon eye tracker.
The camera is worn by a person. Each frame represents the wearer's visual environment.
The wearer's gaze position is provided as image coordinates.
Your task is to produce detailed, factual, temporally consistent descriptions of
the visible scene that can be used to reconstruct activities, object interactions,
and events over time.

1. Scene understanding
Identify all visually distinguishable objects, people, surfaces, text, and
relevant environmental elements.
When an object first appears, describe it in sufficient detail to distinguish it
from other objects, including:
- Object category and recognizable identity
- Color, shape, size, and distinctive visual features
- Position relative to other objects
- Relevant spatial relationships
- Visible state, such as open, closed, occupied, empty, held, or moving
Describe the environment and its spatial arrangement when first observed.
Only describe what is supported by visual evidence. Do not invent hidden objects,
intentions, identities, or actions.

2. Temporal consistency and memory
You will receive the previous scene state and recent observations.
Treat each frame as part of a continuous sequence rather than an independent image.
- Describe new objects when they first become visible.
- Do not repeat descriptions of objects already identified unless their state,
  position, or relationship changes meaningfully.
- Maintain consistent object identities across frames whenever possible.
- Track objects that disappear and reappear.
- Distinguish actual object movement from changes caused by head or camera motion.
- Do not interpret an object leaving the camera's field of view as the object
  physically leaving the environment.
- Report meaningful changes even when no new objects appear.
If the scene remains unchanged, return NO_CHANGE rather than paraphrasing
the previous description.

3. Gaze awareness
The gaze coordinates (x, y) indicate where the wearer is looking within the
image of width W and height H.
Use gaze to determine which visible object or region is being visually attended to.
- Identify the object containing the gaze point when possible.
- If the gaze point falls near multiple objects, express uncertainty.
- Give additional descriptive priority to newly observed objects near gaze.
- Track changes in the attended object across frames.
- Distinguish looking at an object from physically interacting with it.
- Never assume that gaze alone proves attention, intention, recognition,
  or understanding.
- Do not invent an object at the gaze location.
Gaze is an additional signal for scene understanding, not proof of an action.

4. Event extraction
Identify meaningful observable events, including:
- An object entering or leaving the visible scene
- A person entering or leaving the visible scene
- An object being picked up, placed, opened, closed, moved, or manipulated
- A person approaching, moving away, or interacting with an object
- A change in the state of a visible object
- A change in the object being looked at
- A meaningful transition between activities or environments
Describe events using observable subjects, actions, objects, and resulting states.
Distinguish between:
- Directly observed events
- Events suggested by incomplete visual evidence
- Uncertain interpretations
Do not infer an event solely from the presence of an object.

5. Description rules
- Use precise, natural, factual language.
- Prioritize observable actions, objects, and spatial relationships.
- Describe objects in detail on first appearance, then reference them concisely.
- Avoid redundant statements and repeated background descriptions.
- Do not describe camera perspective, camera angle, lens properties, field of view,
  sharpness, blur, exposure, resolution, lighting quality, or recording artifacts.
- Do not use phrases like "the image shows", "in this frame", or "we can see".
- Do not describe the wearer unless a body part or an action is visibly observable.
- Avoid speculation about thoughts, intentions, or motivations.
- Use consistent object names across the sequence.
- Prefer specific verbs over vague descriptions.
- Do not invent changes when the scene is stable.

6. Output
Generate a concise scene update that contains only new information, meaningful
changes, and observed events.
When a new scene appears, provide a detailed initial description.
When a previously described scene remains stable, output NO_CHANGE.
When an event occurs, describe the event with subject, action, object, and result.
Include the gaze target when it can be visually identified.
Do not repeat the entire scene description on every frame.
You can output valid JSON (e.g. {"scene_update": "...", "events": [...]})
or clear factual prose."""

EVENT_GAZE_PROMPT_TEMPLATE = (
    "Event-oriented, gaze-aware scene understanding with temporal memory. "
    "Structured frame context (timestamp, resolution, scaled gaze, prior scene "
    "state, recent events) is injected automatically. Unchanged scenes output "
    "NO_CHANGE."
)

PROMPT_TEMPLATES = {
    "Default": DEFAULT_PROMPT,
    "Event & Gaze Aware (Temporal)": EVENT_GAZE_PROMPT_TEMPLATE,
    "Gazed Object (One Word)": (
        "The user's gaze is at pixel coordinates ({gaze_x}, {gaze_y}) in this "
        "{width}x{height} image. In exactly one word, name the main object at "
        "this location. Do not describe anything else."
    ),
    "Detailed Scene": (
        "Provide a highly detailed description of what appears in this scene. "
        "Name all visible objects, text, and environmental details. "
        "Avoid first-person perspective phrasing."
    ),
    "Delta Changes": (
        "The previous frame was described as: '{prev_desc}'.\n"
        "Describe only what has changed in this new frame. "
        "Focus on what the wearer is currently doing, what new objects have appeared, "
        "or what objects have disappeared. Keep it brief."
    ),
}


class PromptTemplate(enum.Enum):
    Default = "Default"
    Event_Gaze_Aware = "Event & Gaze Aware (Temporal)"
    Gazed_Object = "Gazed Object (One Word)"
    Detailed_Scene = "Detailed Scene"
    Delta_Changes = "Delta Changes"
    Custom = "Custom"


#: Leading characters a spreadsheet will interpret as a formula.
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


class CaptionModel(enum.Enum):
    """Model presets, smallest first.

    Values are the Apple-Silicon (MLX) repository IDs; `hf_repo` maps each
    preset onto the equivalent upstream checkpoint for the transformers path.
    """

    Fast_2B = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
    Balanced_4B = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    Quality_8B = "mlx-community/Qwen3-VL-8B-Instruct-4bit"

    @property
    def mlx_repo(self) -> str:
        return self.value

    @property
    def hf_repo(self) -> str:
        return {
            CaptionModel.Fast_2B: "Qwen/Qwen3-VL-2B-Instruct",
            CaptionModel.Balanced_4B: "Qwen/Qwen3-VL-4B-Instruct",
            CaptionModel.Quality_8B: "Qwen/Qwen3-VL-8B-Instruct",
        }[self]


def _format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def _sample_indices(times_ns: np.ndarray, interval_s: float) -> list[int]:
    """Pick one frame index per `interval_s` of wall-clock time.

    Uses the scene video's own timestamps rather than assuming a constant
    frame rate, because Neon scene video is not perfectly isochronous.
    Returned indices are ordered by time.
    """
    n = len(times_ns)
    if n == 0:
        return []

    # `np.searchsorted` returns silent garbage on unsorted input. Scene
    # timestamps are sorted in practice, so this only guards against a
    # malformed recording rather than a normal one.
    order: np.ndarray | None = None
    if np.any(np.diff(times_ns) < 0):
        logger.warning("Scene timestamps are not monotonic; sorting before sampling")
        order = np.argsort(times_ns, kind="stable")
        times_ns = times_ns[order]

    step_ns = round(interval_s * 1e9)
    if step_ns <= 0:
        positions: np.ndarray = np.arange(n)
    else:
        targets = np.arange(int(times_ns[0]), int(times_ns[-1]) + 1, step_ns)
        positions = np.unique(np.clip(np.searchsorted(times_ns, targets), 0, n - 1))

    if order is not None:
        return [int(order[p]) for p in positions]
    return [int(p) for p in positions]


def _assign_end_times(
    captions: list[dict[str, Any]], times_ns: np.ndarray, rec_start_ns: int
) -> None:
    """Give every caption an end time, in place.

    A caption is valid until the next one starts. The last one runs one frame
    past the final scene timestamp, so that it is still on screen while the
    last frame is displayed rather than blinking out instantly.
    """
    if not captions or len(times_ns) == 0:
        return

    frame_duration_ns = int(np.median(np.diff(times_ns))) if len(times_ns) > 1 else 0
    if frame_duration_ns <= 0:
        frame_duration_ns = FALLBACK_FRAME_DURATION_NS
    end_of_scene = int(times_ns[-1]) + frame_duration_ns

    for i, caption in enumerate(captions):
        if i + 1 < len(captions):
            end_ns = int(captions[i + 1]["start_ns"])
        else:
            end_ns = end_of_scene
        caption["end_ns"] = end_ns
        caption["end_s"] = (end_ns - rec_start_ns) / 1e9


def _frame_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert a scene frame's BGR ndarray to a size-limited RGB PIL image."""
    if bgr.ndim == 2:
        rgb = np.stack([bgr] * 3, axis=-1)
    elif bgr.ndim == 3 and bgr.shape[2] >= 3:
        # Drop any alpha channel, then BGR -> RGB.
        rgb = bgr[:, :, 2::-1]
    else:
        raise ValueError(f"Unexpected scene frame shape: {bgr.shape}")

    # `VideoFrame.bgr` is uint8 today. Rescale rather than truncate if that
    # ever changes, so a float or 16-bit frame does not silently go black.
    if rgb.dtype != np.uint8:
        rgb = np.nan_to_num(rgb.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(rgb.max()) if rgb.size else 0.0
        if peak <= 0.0:
            rgb = np.zeros_like(rgb)
        else:
            scale = 255.0 if peak <= 1.0 else 255.0 / peak
            rgb = rgb * scale
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    image = Image.fromarray(np.ascontiguousarray(rgb))

    longest = max(image.size)
    if longest > MAX_IMAGE_EDGE:
        scale_factor = MAX_IMAGE_EDGE / longest
        image = image.resize(
            (
                max(1, round(image.width * scale_factor)),
                max(1, round(image.height * scale_factor)),
            ),
            Image.Resampling.BILINEAR,
        )

    return image


def _thumbnail(image: Image.Image) -> np.ndarray:
    """Small grayscale array used for cheap frame-to-frame comparison."""
    small = image.convert("L").resize(
        (THUMB_EDGE, THUMB_EDGE), Image.Resampling.BILINEAR
    )
    return np.asarray(small, dtype=np.int16)


def _thumbnail_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference between two thumbnails."""
    return float(np.mean(np.abs(a - b)))


#: Boilerplate openers small VLMs like to emit, as (lead, keep_article) pairs.
#: A lead is only stripped when an article follows it, so a real sentence such
#: as "The image shows up on the wall behind him" is left alone.
_BOILERPLATE_LEADS = (
    "the image shows",
    "this image shows",
    "the image depicts",
    "this image depicts",
    "the photo shows",
    "this photo shows",
)
_ARTICLES = ("a ", "an ", "the ")

#: Openers stripped outright, article and all.
_BOILERPLATE_CLAUSES = ("in this image, ", "in the image, ", "in this photo, ")


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def _clean_caption(text: Any) -> str:
    """Collapse whitespace and strip the boilerplate small VLMs like to emit."""
    if not isinstance(text, str):
        # A backend that returned something unexpected is a failure, not a
        # caption reading "None".
        return ""

    cleaned = " ".join(text.split())
    lowered = cleaned.lower()

    for clause in _BOILERPLATE_CLAUSES:
        if lowered.startswith(clause):
            return _capitalize(cleaned[len(clause) :])

    for lead in _BOILERPLATE_LEADS:
        if not lowered.startswith(lead + " "):
            continue
        remainder = cleaned[len(lead) + 1 :]
        # Only strip when an article follows; keep the article itself so
        # "The image shows a mug" becomes "A mug", not "Mug".
        if remainder.lower().startswith(_ARTICLES):
            return _capitalize(remainder)

    return cleaned


def _fit_text(
    text: str, fm: QFontMetrics, band: QRect, flags: int
) -> tuple[str, QRect]:
    """Shorten `text` until its wrapped bounding box fits inside `band`.

    Qt's `boundingRect` grows past the rect it is given, and because the
    caption is bottom-aligned it grows *upwards*, off the top of the video.
    A long caption at a large font size would otherwise cover most of the
    scene, so drop trailing words until it fits.
    """
    rect = fm.boundingRect(band, flags, text)
    if rect.height() <= band.height():
        return text, rect

    words = text.split()
    while len(words) > 1:
        words.pop()
        candidate = " ".join(words) + "…"
        rect = fm.boundingRect(band, flags, candidate)
        if rect.height() <= band.height():
            return candidate, rect

    # A single word too tall to fit: clip it to the band rather than overflow.
    candidate = words[0] if words else ""
    rect = fm.boundingRect(band, flags, candidate)
    return candidate, rect.intersected(band)


def _keyword_event_timestamps(
    captions: list[dict[str, Any]], keyword: str
) -> list[int]:
    """Start timestamps of each contiguous run of captions matching `keyword`.

    One event per run, so a keyword that stays true for a minute yields one
    event rather than thirty. Empty captions (a frame the model failed on) are
    skipped rather than treated as non-matches, so a single failure in the
    middle of a run does not split it into two events.
    """
    needle = keyword.strip().lower()
    if not needle:
        return []

    timestamps: list[int] = []
    previous_matched = False
    for caption in captions:
        text = caption.get("text") or ""
        if not text:
            continue
        matched = needle in text.lower()
        if matched and not previous_matched:
            timestamps.append(int(caption["start_ns"]))
        previous_matched = matched
    return timestamps


def _lookup_gaze_point(
    g_times: np.ndarray | None,
    g_points_x: np.ndarray | None,
    g_points_y: np.ndarray | None,
    timestamp_ns: int,
    scene_w: int,
    scene_h: int,
    target_w: int,
    target_h: int,
) -> tuple[int | None, int | None, bool]:
    """Look up nearest gaze sample and project coordinates into target dimensions."""
    if (
        g_times is None
        or g_points_x is None
        or g_points_y is None
        or len(g_times) == 0
        or scene_w <= 0
        or scene_h <= 0
        or target_w <= 0
        or target_h <= 0
    ):
        return None, None, False

    try:
        idx = int(
            np.clip(
                np.searchsorted(g_times, timestamp_ns), 0, len(g_times) - 1
            )
        )
        if idx > 0 and abs(timestamp_ns - g_times[idx - 1]) < abs(
            timestamp_ns - g_times[idx]
        ):
            idx -= 1
        gx = float(g_points_x[idx])
        gy = float(g_points_y[idx])
    except Exception:
        logger.debug(
            "Failed gaze lookup at timestamp %s", timestamp_ns, exc_info=True
        )
        return None, None, False

    if math.isnan(gx) or math.isnan(gy):
        return None, None, False

    scale_x = target_w / scene_w
    scale_y = target_h / scene_h
    vlm_x = max(0, min(target_w - 1, round(gx * scale_x)))
    vlm_y = max(0, min(target_h - 1, round(gy * scale_y)))
    return vlm_x, vlm_y, True


def _extract_json_block(text: str) -> str:
    """Extract candidate JSON payload from model output."""
    if "```json" in text:
        return text.split("```json")[1].split("```", maxsplit=1)[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```", maxsplit=1)[0].strip()
    if "{" in text and "}" in text:
        s_idx = text.find("{")
        e_idx = text.rfind("}") + 1
        return text[s_idx:e_idx]
    return text


def _parse_json_event_payload(
    parsed: dict[str, Any], previous_state: str, raw_text: str
) -> tuple[str, bool, dict[str, Any], list[str]]:
    """Extract scene update and events from parsed JSON."""
    scene_update = parsed.get("scene_update")
    if scene_update and "NO_CHANGE" in str(scene_update).strip().upper():
        text = previous_state or "No change."
        is_no_change = True
    elif scene_update:
        text = str(scene_update).strip()
        is_no_change = False
    else:
        text = str(parsed.get("description", raw_text)).strip()
        is_no_change = False

    new_events: list[str] = []
    evs = parsed.get("events")
    if isinstance(evs, list):
        for ev in evs:
            if isinstance(ev, dict):
                d = ev.get("description") or ev.get("type") or str(ev)
                if d:
                    new_events.append(str(d).strip())
            elif isinstance(ev, str) and ev.strip():
                new_events.append(ev.strip())

    return text, is_no_change, parsed, new_events


def _regex_extract_update(text: str) -> str | None:
    """Fallback regex to extract scene update from truncated JSON."""
    match = re.search(r'"scene_update"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)', text)
    if not match:
        match = re.search(r'"description"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)', text)
    if match:
        extracted = match.group(1).replace(r'\"', '"').replace(r"\n", " ").strip()
        if extracted:
            return extracted
    return None


def _parse_temporal_response(
    raw_output: str, previous_scene_state: str
) -> tuple[str, bool, dict[str, Any] | None, list[str]]:
    """Parse model output into text, no_change flag, event dict, and new events."""
    clean_resp = raw_output.strip()
    clean_upper = clean_resp.upper()

    if (
        clean_upper == "NO_CHANGE"
        or clean_upper.rstrip(".").endswith("NO_CHANGE")
        or clean_upper.startswith("NO_CHANGE")
    ):
        return previous_scene_state or "No change.", True, None, []

    json_str = _extract_json_block(clean_resp)
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            return _parse_json_event_payload(
                parsed, previous_scene_state, clean_resp
            )
    except (json.JSONDecodeError, ValueError):
        pass

    extracted = _regex_extract_update(clean_resp)
    if extracted:
        ext_upper = extracted.upper()
        if (
            ext_upper == "NO_CHANGE"
            or ext_upper.startswith("NO_CHANGE")
            or ext_upper.rstrip(".").endswith("NO_CHANGE")
        ):
            return previous_scene_state or "No change.", True, None, []
        return extracted, False, None, [extracted]

    if clean_resp.startswith("{"):
        return previous_scene_state or "No change.", True, None, []

    return clean_resp, False, None, [clean_resp]


def _caption_single_frame(
    captioner: _Captioner,
    image: Image.Image,
    frame_index: int,
    start_ns: int,
    recording_start_ns: int,
    vlm_gaze_x: int | None,
    vlm_gaze_y: int | None,
    gaze_valid: bool,
    is_event_gaze_aware: bool,
    previous_scene_state: str,
    recent_events: list[str],
    custom_instructions: str,
    legacy_prompt: str,
    reference_text: str | None,
) -> tuple[str, str, bool, dict[str, Any] | None, list[str]]:
    """Caption a single frame and return caption text, raw output, and metadata."""
    if is_event_gaze_aware:
        frame_ctx: dict[str, Any] = {
            "frame_index": int(frame_index),
            "timestamp_ms": round((start_ns - recording_start_ns) / 1e6),
            "image_width": image.width,
            "image_height": image.height,
            "gaze": {
                "x": vlm_gaze_x,
                "y": vlm_gaze_y,
                "valid": gaze_valid,
            },
            "previous_scene_state": previous_scene_state or "Initial frame.",
            "recent_events": recent_events[-5:],
        }
        if custom_instructions:
            frame_ctx["user_instructions"] = custom_instructions

        user_prompt = json.dumps(frame_ctx, indent=2)
        try:
            raw_output = captioner.caption(
                image,
                user_prompt,
                system_prompt=EVENT_GAZE_SYSTEM_PROMPT,
            )
        except Exception:
            logger.exception("Captioning failed for scene frame %s", frame_index)
            return "", "", False, None, []

        text, is_no_change, event_data, new_events = _parse_temporal_response(
            raw_output, previous_scene_state
        )
        return text, raw_output, is_no_change, event_data, new_events

    gaze_str_x = str(vlm_gaze_x) if vlm_gaze_x is not None else "unknown"
    gaze_str_y = str(vlm_gaze_y) if vlm_gaze_y is not None else "unknown"
    current_prompt = legacy_prompt.replace("{gaze_x}", gaze_str_x)
    current_prompt = current_prompt.replace("{gaze_y}", gaze_str_y)
    current_prompt = current_prompt.replace("{width}", str(image.width))
    current_prompt = current_prompt.replace("{height}", str(image.height))
    current_prompt = current_prompt.replace(
        "{prev_desc}", reference_text or "None"
    )

    try:
        raw_output = captioner.caption(image, current_prompt)
    except Exception:
        logger.exception("Captioning failed for scene frame %s", frame_index)
        return "", "", False, None, []

    return raw_output, raw_output, False, None, []


def _extract_gaze_timeseries(
    recording: Any,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Pre-extract gaze time and coordinates as numpy arrays."""
    gaze = getattr(recording, "gaze", None)
    if gaze is not None and hasattr(gaze, "time"):
        try:
            return (
                np.asarray(gaze.time),
                np.asarray(gaze["point_x"]),
                np.asarray(gaze["point_y"]),
            )
        except Exception:
            logger.debug("Could not extract gaze arrays", exc_info=True)
    return None, None, None


def _render_caption_text(
    painter: QPainter,
    bounding_rect: QRect,
    flags: int,
    text: str,
    matching_keywords: list[str],
    text_color: QColor,
    highlight_color: QColor,
) -> None:
    """Render plain or keyword-highlighted caption text."""
    if not matching_keywords:
        painter.setPen(QPen(text_color))
        painter.drawText(bounding_rect, flags, text)
        return

    doc = QTextDocument()
    doc.setDefaultFont(painter.font())
    doc.setDocumentMargin(0)
    opt = QTextOption()
    opt.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    opt.setWrapMode(QTextOption.WrapMode.WordWrap)
    doc.setDefaultTextOption(opt)
    doc.setTextWidth(bounding_rect.width())

    hl_hex = highlight_color.name()
    text_hex = text_color.name()
    escaped = html.escape(text)
    for kw in matching_keywords:
        escaped = re.sub(
            rf"\b({re.escape(html.escape(kw))})\b",
            rf'<span style="color: {hl_hex}; font-weight: bold;">\1</span>',
            escaped,
            flags=re.IGNORECASE,
        )
    doc.setHtml(
        f'<div style="text-align: center; color: {text_hex};">'
        f"{escaped}</div>"
    )

    painter.save()
    painter.translate(QPointF(bounding_rect.topLeft()))
    doc.drawContents(painter)
    painter.restore()


def _csv_safe(value: str) -> str:
    """Neutralise spreadsheet formula injection in a text cell."""
    if value.startswith(_CSV_FORMULA_PREFIXES):
        return "'" + value
    return value


class _Captioner:
    """Loads a VLM once and captions many images, then releases it."""

    def __init__(self, model: CaptionModel, max_tokens: int) -> None:
        self._preset = model
        self._max_tokens = max_tokens
        self._model: Any = None
        self._processor: Any = None
        self._config: Any = None
        self._torch: Any = None
        self._mlx = _is_apple_silicon()
        self._tmp_dir: tempfile.TemporaryDirectory | None = None
        self._tmp_image: Path | None = None

    @property
    def backend(self) -> str:
        return "mlx-vlm (Metal)" if self._mlx else "transformers"

    @property
    def repo(self) -> str:
        return self._preset.mlx_repo if self._mlx else self._preset.hf_repo

    def load(self) -> None:
        try:
            if self._mlx:
                self._load_mlx()
            else:
                self._load_transformers()
        except Exception:
            # Never leave a half-built captioner behind; `caption()` checks
            # `_model` and must not see a model without its processor.
            self._model = None
            self._processor = None
            raise

    def _load_mlx(self) -> None:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        model, processor = load(self.repo)
        self._config = load_config(self.repo)
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="npp_visual_context_")
        self._tmp_image = Path(self._tmp_dir.name) / "frame.jpg"
        self._model, self._processor = model, processor

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(
            self.repo, dtype="auto", device_map="auto"
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(self.repo)
        self._torch = torch
        self._model, self._processor = model, processor

    def caption(
        self, image: Image.Image, prompt: str, system_prompt: str | None = None
    ) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("Captioner used before a successful load()")
        if self._mlx:
            return self._caption_mlx(image, prompt, system_prompt=system_prompt)
        return self._caption_transformers(image, prompt, system_prompt=system_prompt)

    def _caption_mlx(
        self, image: Image.Image, prompt: str, system_prompt: str | None = None
    ) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        assert self._tmp_image is not None
        # mlx-vlm's image loader accepts paths across every released version;
        # passing PIL objects only works from 0.6 onwards. One reused temp
        # file keeps this version-proof at negligible cost next to inference.
        image.save(self._tmp_image, format="JPEG", quality=92)

        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            formatted = apply_chat_template(
                self._processor, self._config, messages, num_images=1
            )
        else:
            formatted = apply_chat_template(
                self._processor, self._config, prompt, num_images=1
            )
        result = generate(
            self._model,
            self._processor,
            formatted,
            [str(self._tmp_image)],
            max_tokens=self._max_tokens,
            temperature=0.0,
            verbose=False,
        )
        # mlx-vlm returns a GenerationResult; older versions returned a str.
        raw = getattr(result, "text", result)
        if system_prompt:
            return str(raw).strip() if raw is not None else ""
        return _clean_caption(raw)

    def _caption_transformers(
        self, image: Image.Image, prompt: str, system_prompt: str | None = None
    ) -> str:
        torch = self._torch
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs, max_new_tokens=self._max_tokens, do_sample=False
            )

        prompt_len = inputs["input_ids"].shape[1]
        decoded = self._processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True
        )[0]
        if system_prompt:
            return str(decoded).strip() if decoded is not None else ""
        return _clean_caption(decoded)

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._config = None
        self._torch = None

        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None
            self._tmp_image = None

        gc.collect()

        if "torch" in sys.modules:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        if "mlx" in sys.modules:
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                logger.debug("Could not clear the MLX cache", exc_info=True)


class CaptionColors(PersistentPropertiesMixin, QObject):
    """Pair of text and highlight colors."""

    changed = Signal()

    def __init__(
        self,
        text_color: QColor | str = "#FFFFFF",
        highlight_color: QColor | str = "#FFD600",
    ) -> None:
        super().__init__()
        self._text_color = QColor(text_color)
        self._highlight_color = QColor(highlight_color)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CaptionColors):
            return False
        return (
            self._text_color == other._text_color
            and self._highlight_color == other._highlight_color
        )

    @property
    def text_color(self) -> QColor:
        return self._text_color

    @text_color.setter
    def text_color(self, v: QColor | str) -> None:
        self._text_color = QColor(v)
        self.changed.emit()

    @property
    def highlight_color(self) -> QColor:
        return self._highlight_color

    @highlight_color.setter
    def highlight_color(self, v: QColor | str) -> None:
        self._highlight_color = QColor(v)
        self.changed.emit()


class CaptionColorsWidget(PropertyWidget):
    """Places Text Color and Highlight Color widgets side-by-side on the same row."""

    value_changed = Signal(object)

    @staticmethod
    def from_property_impl(prop: property) -> CaptionColorsWidget:
        return CaptionColorsWidget()

    def __init__(self) -> None:
        super().__init__()
        self._value = CaptionColors()
        self._updating = False

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addStretch(1)
        self.label_text = QLabel("Color")
        self.widget_text = ColorWidget()
        self.widget_text.spacer.setVisible(False)
        self.label_hl = QLabel("Highlight")
        self.widget_hl = ColorWidget()
        self.widget_hl.spacer.setVisible(False)

        layout.addWidget(self.label_text)
        layout.addWidget(self.widget_text)
        layout.addSpacing(10)
        layout.addWidget(self.label_hl)
        layout.addWidget(self.widget_hl)

        self.grid_layout.addLayout(layout, 0, 0)

        self.widget_text.value_changed.connect(self._on_text_changed)
        self.widget_hl.value_changed.connect(self._on_hl_changed)

    def _on_text_changed(self, col: QColor) -> None:
        if self._updating:
            return
        new_val = CaptionColors(col, self._value.highlight_color)
        self._value = new_val
        self.value_changed.emit(new_val)

    def _on_hl_changed(self, col: QColor) -> None:
        if self._updating:
            return
        new_val = CaptionColors(self._value.text_color, col)
        self._value = new_val
        self.value_changed.emit(new_val)

    @property
    def value(self) -> CaptionColors:
        return self._value

    @value.setter
    def value(self, val: CaptionColors | dict) -> None:
        self._updating = True
        try:
            if isinstance(val, dict):
                self._value = CaptionColors(
                    val.get("text_color", "#FFFFFF"),
                    val.get("highlight_color", "#FFD600"),
                )
            elif isinstance(val, CaptionColors):
                self._value = CaptionColors(val.text_color, val.highlight_color)
            self.widget_text.value = self._value.text_color
            self.widget_hl.value = self._value.highlight_color
        finally:
            self._updating = False


class CaptionStyle(PersistentPropertiesMixin, QObject):
    """Caption appearance and style settings."""

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._font_size = 26
        self._colors = CaptionColors()
        self._colors.changed.connect(self.changed.emit)

    @property
    @property_params(label="Font Size", min=10, max=100)
    def font_size(self) -> int:
        """Caption Font Size"""
        return self._font_size

    @font_size.setter
    def font_size(self, v: int) -> None:
        self._font_size = min(max(int(v), 10), 100)
        self.changed.emit()

    @property
    @property_params(label="Color", widget=CaptionColorsWidget)
    def colors(self) -> CaptionColors:
        return self._colors

    @colors.setter
    def colors(self, v: CaptionColors | dict) -> None:
        if isinstance(v, dict):
            self._colors.__setstate__(v)
        elif isinstance(v, CaptionColors):
            self._colors = v
            self._colors.changed.connect(self.changed.emit)
        self.changed.emit()

    # Backward compatibility proxies
    @property
    @property_params(widget=None, dont_encode=True)
    def text_color(self) -> QColor:
        return self._colors.text_color

    @text_color.setter
    def text_color(self, v: QColor | str) -> None:
        self._colors.text_color = v

    @property
    @property_params(widget=None, dont_encode=True)
    def highlight_color(self) -> QColor:
        return self._colors.highlight_color

    @highlight_color.setter
    def highlight_color(self, v: QColor | str) -> None:
        self._colors.highlight_color = v


class CollapsibleGroupWidget(PropertyWidget):
    """Renders a custom type inside a collapsible Expander section."""

    value_changed = Signal(object)

    @staticmethod
    def from_property_impl(prop: property) -> CollapsibleGroupWidget:
        target_type = None
        if hasattr(prop, "fget") and prop.fget:
            try:
                import typing as T

                target_type = T.get_type_hints(prop.fget).get("return")
            except (TypeError, NameError, AttributeError):
                target_type = None
            if target_type is None or isinstance(target_type, str):
                hints = getattr(prop.fget, "__annotations__", {})
                ret = hints.get("return")
                if isinstance(ret, type):
                    target_type = ret
                elif isinstance(ret, str) and hasattr(prop.fget, "__globals__"):
                    target_type = prop.fget.__globals__.get(ret, object)
        if target_type is None or not isinstance(target_type, type):
            target_type = object

        params = (
            getattr(prop.fget, "parameters", {})
            if hasattr(prop, "fget") and prop.fget
            else {}
        )
        title = params.get("label") or (
            prop.fget.__name__.replace("_", " ").title()
            if hasattr(prop, "fget") and prop.fget
            else ""
        )
        return CollapsibleGroupWidget(target_type, title=title)

    def __init__(
        self, target_type: type, title: str = "", expanded: bool = False
    ) -> None:
        super().__init__()
        self.form = PropertyForm.from_type(target_type)
        self.expander = Expander(
            title=title, content_widget=self.form, expanded=expanded
        )
        self.expander.layout().setContentsMargins(0, 0, 0, 0)
        self.setObjectName("CollapsibleGroupWidget")
        self.setStyleSheet("""
            CollapsibleGroupWidget Expander {
                padding-top: 0px;
                padding-bottom: 0px;
            }

            CollapsibleGroupWidget Expander Expander {
                padding-top: 0px;
                padding-bottom: 0px;
            }

            CollapsibleGroupWidget QLabel#ExpanderName {
                color: #a09fa6;
            }
        """)
        self.grid_layout.addWidget(self.expander, 0, 0)
        self.form.value_changed.connect(self.value_changed.emit)
        self.form.property_changed.connect(
            lambda n, v: self.value_changed.emit(self.value)
        )

    def on_bind(self, instance: Any) -> None:
        super().on_bind(instance)
        enable_prop = self.source_params.get("enable_prop")
        if enable_prop and hasattr(instance, enable_prop):
            self.bool_widget = BoolWidget()
            self.bool_widget.spacer.setVisible(False)
            self.bool_widget.value = bool(getattr(instance, enable_prop))
            self.bool_widget.value_changed.connect(
                lambda val: setattr(instance, enable_prop, val)
            )
            if hasattr(instance, "changed") and hasattr(instance.changed, "connect"):
                instance.changed.connect(
                    lambda: setattr(
                        self.bool_widget,
                        "value",
                        bool(getattr(instance, enable_prop)),
                    )
                    if hasattr(self, "bool_widget")
                    and self.bool_widget.value
                    != bool(getattr(instance, enable_prop))
                    else None
                )
            self.expander.controls_layout.addWidget(self.bool_widget)

    @property
    def value(self) -> object:
        return self.form.value

    @value.setter
    def value(self, v: object) -> None:
        self.form.value = v


class EditPromptWidget(PropertyWidget):
    """Button row that opens the prompt editor dialog."""

    value_changed = Signal(object)

    @staticmethod
    def from_property_impl(prop: property) -> EditPromptWidget:
        return EditPromptWidget()

    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(1)

        self.button = QToolButton()
        self.button.setIcon(QIcon.fromTheme("accessories-text-editor"))
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.setObjectName("HeaderAction")
        layout.addWidget(self.button)

        self.grid_layout.addLayout(layout, 0, 0)
        self.button.clicked.connect(self._on_clicked)

    def on_bind(self, instance: Any) -> None:
        super().on_bind(instance)
        self._target_instance = instance

    def _on_clicked(self) -> None:
        if hasattr(self, "_target_instance") and hasattr(
            self._target_instance, "open_prompt_editor"
        ):
            self._target_instance.open_prompt_editor()

    @property
    def value(self) -> Any:
        return None

    @value.setter
    def value(self, v: Any) -> None:
        pass


def _match_prompt_template(text: str) -> str:
    stripped = text.strip()
    for k, v in PROMPT_TEMPLATES.items():
        if stripped == v.strip():
            return k
    return "Custom"


class CaptioningSettings(PersistentPropertiesMixin, QObject):
    """VLM captioning generation and prompt settings."""

    changed = Signal()

    def __init__(
        self,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = 64,
        interval_s: float = 2.0,
        skip_similar: bool = True,
        similarity_threshold: float = 6.0,
    ) -> None:
        super().__init__()
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._interval_s = interval_s
        self._skip_similar = skip_similar
        self._similarity_threshold = similarity_threshold

    @property
    @property_params(label="Prompt Template")
    def prompt_template(self) -> PromptTemplate:
        matched = _match_prompt_template(self._prompt)
        for t in PromptTemplate:
            if t.value == matched:
                return t
        return PromptTemplate.Custom

    @prompt_template.setter
    def prompt_template(self, v: PromptTemplate | str) -> None:
        if isinstance(v, str):
            v = PromptTemplate(v)
        if v != PromptTemplate.Custom:
            self.prompt = PROMPT_TEMPLATES[v.value]
        self.changed.emit()

    @property
    @property_params(label="Captioning Prompt")
    def prompt(self) -> str:
        return self._prompt

    @prompt.setter
    def prompt(self, v: str) -> None:
        text = str(v).strip()
        self._prompt = text or DEFAULT_PROMPT
        self.changed.emit()

    @property
    @property_params(label="Edit Prompt", widget=EditPromptWidget, dont_encode=True)
    def edit_prompt(self) -> None:
        return None

    @edit_prompt.setter
    def edit_prompt(self, v: Any) -> None:
        pass

    @property
    @property_params(label="Max Tokens per Caption", min=16, max=256)
    def max_tokens(self) -> int:
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, v: int) -> None:
        self._max_tokens = min(max(int(v), 16), 256)
        self.changed.emit()

    @property
    @property_params(label="Seconds Between Captions", min=0.2, max=60.0)
    def interval_s(self) -> float:
        return self._interval_s

    @interval_s.setter
    def interval_s(self, v: float) -> None:
        value = float(v)
        self._interval_s = 2.0 if math.isnan(value) else min(max(value, 0.2), 60.0)
        self.changed.emit()

    @property
    @property_params(label="Skip Near-Identical Frames")
    def skip_similar(self) -> bool:
        return self._skip_similar

    @skip_similar.setter
    def skip_similar(self, v: bool) -> None:
        self._skip_similar = bool(v)
        self.changed.emit()

    @property
    @property_params(label="Frame Difference Threshold", min=0.0, max=64.0)
    def similarity_threshold(self) -> float:
        return self._similarity_threshold

    @similarity_threshold.setter
    def similarity_threshold(self, v: float) -> None:
        value = float(v)
        self._similarity_threshold = (
            6.0 if math.isnan(value) else min(max(value, 0.0), 64.0)
        )
        self.changed.emit()

    def open_prompt_editor(self) -> None:
        """Edit Captioning Prompt in a dialog."""
        dialog = QDialog()
        dialog.setWindowTitle("Captioning Prompt")
        dialog.resize(500, 300)

        layout = QVBoxLayout(dialog)

        tmpl_layout = QHBoxLayout()
        tmpl_layout.addWidget(QLabel("Template:"))
        combo = QComboBox()
        combo.addItems(list(PROMPT_TEMPLATES.keys()))
        combo.addItem("Custom")
        combo.setCurrentText(_match_prompt_template(self._prompt))
        tmpl_layout.addWidget(combo)
        layout.addLayout(tmpl_layout)

        layout.addWidget(QLabel("Prompt sent to the VLM for every frame:"))

        text_edit = QTextEdit()
        text_edit.setPlainText(self._prompt)
        layout.addWidget(text_edit)

        def on_combo_changed(text: str) -> None:
            if text in PROMPT_TEMPLATES:
                text_edit.setPlainText(PROMPT_TEMPLATES[text])

        combo.currentTextChanged.connect(on_combo_changed)

        def on_text_changed() -> None:
            combo.setCurrentText(_match_prompt_template(text_edit.toPlainText()))

        text_edit.textChanged.connect(on_text_changed)

        buttons_layout = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(dialog.accept)
        btn_cancel.clicked.connect(dialog.reject)
        buttons_layout.addWidget(btn_cancel)
        buttons_layout.addWidget(btn_ok)
        layout.addLayout(buttons_layout)

        if dialog.exec():
            text = text_edit.toPlainText().strip()
            if text:
                self.prompt = text


class VisualContextPlugin(Plugin):
    label = "Visual Context (Scene Captioning)"

    def __init__(self) -> None:
        super().__init__()
        self.render_layer = 100
        self._captions: list[dict[str, Any]] = []
        self._starts_ns: list[int] = []

        # NB: `_enabled` belongs to the framework (it means "plugin is
        # switched on") -- the caption overlay toggle must not reuse it.
        self._show_captions = True
        self._model = CaptionModel.Balanced_4B
        self._settings = CaptioningSettings()
        self._settings.changed.connect(self.changed.emit)

        self._caption_style = CaptionStyle()
        self._caption_style.changed.connect(self.changed.emit)
        self._bg_color = QColor(0, 0, 0, 160)

        self._narrate_captions = False
        self._last_narrated_ns = -1
        try:
            self._tts = QTextToSpeech()
        except Exception:
            logger.exception("Failed to initialize TTS")
            self._tts = None

    # ------------------------------------------------------------------
    # Cache handling
    # ------------------------------------------------------------------

    def _read_cache(self, recording: Any = None) -> dict[str, Any] | None:
        try:
            data = self.load_cached_json(CACHE_FILENAME, recording=recording)
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to read visual context cache")
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _valid_captions(data: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Return usable captions from cache data, or None if unusable.

        Every key any consumer reads is validated here, so `render`,
        `export_csv` and `view_captions` cannot trip over a truncated or
        hand-edited cache.
        """
        captions = data.get("captions")
        if not isinstance(captions, list):
            return None

        cleaned: list[dict[str, Any]] = []
        for c in captions:
            if not isinstance(c, dict):
                return None
            try:
                start_ns = int(c["start_ns"])
                end_ns = int(c["end_ns"])
                start_s = float(c["start_s"])
                end_s = float(c["end_s"])
            except (KeyError, TypeError, ValueError):
                return None
            text = c.get("text")
            cleaned.append({
                "start_ns": start_ns,
                "end_ns": max(end_ns, start_ns),
                "start_s": start_s,
                "end_s": end_s,
                "text": text if isinstance(text, str) else "",
                "frame_index": c.get("frame_index"),
                "is_no_change": bool(c.get("is_no_change", False)),
                "raw_output": c.get("raw_output"),
                "event_data": c.get("event_data"),
            })

        # `bisect` in `_caption_at` requires sorted starts; do not trust the
        # file's ordering.
        cleaned.sort(key=operator.itemgetter("start_ns"))
        return cleaned

    def on_recording_loaded(self, recording: Any) -> None:
        self._captions = []
        self._starts_ns = []

        # Pass `recording` explicitly so this never depends on when the app
        # swaps `self.recording` over relative to this callback.
        data = self._read_cache(recording)
        if data is None:
            return

        if data.get("version") != CACHE_VERSION:
            logger.info("Discarding visual context cache from an older version")
            return

        # The cache path is per-recording, but verify anyway so a copied or
        # relocated cache directory can never mislabel another recording.
        if data.get("recording_start_time") != getattr(recording, "start_time", None):
            logger.warning("Visual context cache belongs to a different recording")
            return

        if "error" in data:
            # A previous run recorded an error; nothing valid to display
            return

        captions = self._valid_captions(data)
        if captions is None:
            logger.warning("Visual context cache is malformed; ignoring it")
            return

        self._captions = captions
        self._starts_ns = [c["start_ns"] for c in captions]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @property_params(
        primary=True,
        widget=CollapsibleGroupWidget,
        label="Display Captions",
        enable_prop="show_captions",
    )
    def caption_style(self) -> CaptionStyle:
        return self._caption_style

    @caption_style.setter
    def caption_style(self, value: CaptionStyle | dict) -> None:
        if isinstance(value, dict):
            self._caption_style.__setstate__(value)
        elif isinstance(value, CaptionStyle):
            self._caption_style = value
            self._caption_style.changed.connect(self.changed.emit)
        self.changed.emit()

    @property
    @property_params(widget=None)
    def show_captions(self) -> bool:
        return self._show_captions

    @show_captions.setter
    def show_captions(self, v: bool) -> None:
        self._show_captions = bool(v)
        self.changed.emit()

    @property
    @property_params(label="Narrate Captions (TTS)")
    def narrate_captions(self) -> bool:
        return self._narrate_captions

    @narrate_captions.setter
    def narrate_captions(self, v: bool) -> None:
        self._narrate_captions = bool(v)
        if not self._narrate_captions and getattr(self, "_tts", None):
            self._tts.stop()
        self.changed.emit()

    @property
    @property_params(label="VLM Model")
    def model(self) -> CaptionModel:
        return self._model

    @model.setter
    def model(self, v: CaptionModel) -> None:
        self._model = v
        self.changed.emit()

    @property
    @property_params(
        primary=True,
        widget=CollapsibleGroupWidget,
        label="Settings",
    )
    def settings(self) -> CaptioningSettings:
        return self._settings

    @settings.setter
    def settings(self, value: CaptioningSettings | dict) -> None:
        if isinstance(value, dict):
            self._settings.__setstate__(value)
        elif isinstance(value, CaptioningSettings):
            self._settings = value
            self._settings.changed.connect(self.changed.emit)
        self.changed.emit()

    # Backward compatibility proxies
    @property
    @property_params(widget=None, dont_encode=True)
    def prompt_template(self) -> PromptTemplate:
        return self._settings.prompt_template

    @prompt_template.setter
    def prompt_template(self, v: PromptTemplate | str) -> None:
        self._settings.prompt_template = v

    @property
    @property_params(widget=None, dont_encode=True)
    def prompt(self) -> str:
        return self._settings.prompt

    @prompt.setter
    def prompt(self, v: str) -> None:
        self._settings.prompt = v

    @property
    @property_params(widget=None, dont_encode=True)
    def max_tokens(self) -> int:
        return self._settings.max_tokens

    @max_tokens.setter
    def max_tokens(self, v: int) -> None:
        self._settings.max_tokens = v

    @property
    @property_params(widget=None, dont_encode=True)
    def interval_s(self) -> float:
        return self._settings.interval_s

    @interval_s.setter
    def interval_s(self, v: float) -> None:
        self._settings.interval_s = v

    @property
    @property_params(widget=None, dont_encode=True)
    def skip_similar(self) -> bool:
        return self._settings.skip_similar

    @skip_similar.setter
    def skip_similar(self, v: bool) -> None:
        self._settings.skip_similar = v

    @property
    @property_params(widget=None, dont_encode=True)
    def similarity_threshold(self) -> float:
        return self._settings.similarity_threshold

    @similarity_threshold.setter
    def similarity_threshold(self, v: float) -> None:
        self._settings.similarity_threshold = v

    # Backward compatibility proxies
    @property
    @property_params(widget=None, dont_encode=True)
    def font_size(self) -> int:
        return self._caption_style.font_size

    @font_size.setter
    def font_size(self, v: int) -> None:
        self._caption_style.font_size = v

    @property
    @property_params(widget=None, dont_encode=True)
    def text_color(self) -> QColor:
        return self._caption_style.text_color

    @text_color.setter
    def text_color(self, v: QColor) -> None:
        self._caption_style.text_color = v

    @property
    @property_params(widget=None, dont_encode=True)
    def highlight_color(self) -> QColor:
        return self._caption_style.highlight_color

    @highlight_color.setter
    def highlight_color(self, v: QColor) -> None:
        self._caption_style.highlight_color = v

    def _get_saw_keywords(self) -> set[str]:
        """Collect all keywords associated with 'Saw: <keyword>' events."""
        keywords: set[str] = set()
        events_plugin = EventsPlugin.instance()
        if events_plugin is not None:
            if hasattr(events_plugin, "event_types"):
                for et in events_plugin.event_types:
                    if hasattr(et, "name") and et.name.startswith("Saw: "):
                        kw = et.name[5:].strip()
                        if kw:
                            keywords.add(kw)
            if hasattr(events_plugin, "_events"):
                for name in events_plugin._events:
                    if name.startswith("Saw: "):
                        kw = name[5:].strip()
                        if kw:
                            keywords.add(kw)
        return keywords

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def edit_prompt(self) -> None:
        """Edit Captioning Prompt (opens dialog)."""
        self._settings.open_prompt_editor()

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("media-playback-start"))
    def generate_captions(self) -> None:
        """Generate Captions (local VLM)"""
        # The previous captions are deliberately left in place: if this run
        # fails, a successful earlier run is not thrown away.
        if self.headless:
            self.job_manager.run_in_foreground(self.bg_generate_captions())
            self._on_captions_ready()
            return

        # `_on_captions_ready` reports the outcome (including failures, which
        # the generic completion notice would misreport as success).
        self.job = self.job_manager.run_background_action(
            "Captioning Scene Video",
            "VisualContextPlugin.bg_generate_captions",
            notify_on_completion=False,
        )
        if self.job is None:
            return
        self.job.finished.connect(self._on_captions_ready)

    def _on_captions_ready(self) -> None:
        # A generator that raises inside a background job is reported to the
        # UI as a normal completion, so the error is passed through the cache.
        cache = self._read_cache() or {}
        error = cache.get("error")
        total_time_s = cache.get("total_time_s")

        self.on_recording_loaded(self.recording)
        self.changed.emit()

        if error:
            self.app.show_notification("Captioning Failed", str(error))
        elif self._captions:
            msg = f"Generated {len(self._captions)} captions."
            if total_time_s:
                msg += f" (Took {total_time_s:.1f}s)"
            self.app.show_notification("Visual Context", msg)

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("edit-find"))
    def create_events_from_keyword(self) -> None:
        """Create Events from Keyword"""
        if not self._captions:
            self.app.show_notification("Notice", "Please generate captions first.")
            return

        keyword, ok = QInputDialog.getText(
            None, "Keyword Search", "Enter keyword to search for in the captions:"
        )
        if not ok or not keyword.strip():
            return

        events_plugin = EventsPlugin.instance()
        if events_plugin is None:
            self.app.show_notification("Error", "EventsPlugin not loaded.")
            return

        keyword = keyword.strip()
        timestamps = _keyword_event_timestamps(self._captions, keyword)

        if not timestamps:
            self.app.show_notification(
                "No Matches", f"No caption contains '{keyword}'."
            )
            return

        events_plugin.add_events({f"Saw: {keyword}": timestamps})
        self.app.show_notification(
            "Events Created", f"Added {len(timestamps)} events for '{keyword}'."
        )

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("preferences-desktop-font"))
    def view_captions(self) -> None:
        """View All Captions"""
        if not self._captions:
            self.app.show_notification("Notice", "Please generate captions first.")
            return

        dialog = QDialog()
        dialog.setWindowTitle("Visual Context")
        dialog.resize(760, 520)

        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFontFamily("Courier New")
        text_edit.setPlainText(
            "\n".join(
                f"[{_format_time(c['start_s'])}] {c['text']}" for c in self._captions
            )
        )
        layout.addWidget(text_edit)

        btn = QPushButton("Close")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)

        dialog.setLayout(layout)
        dialog.exec()

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("document-export"))
    def export_csv(self) -> None:
        """Export Captions to CSV"""
        if not self._captions:
            self.app.show_notification("Notice", "Please generate captions first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            None, "Export Captions", "visual_context.csv", "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "start timestamp [ns]",
                    "end timestamp [ns]",
                    "start time [s]",
                    "end time [s]",
                    "frame index",
                    "caption",
                ])
                for c in self._captions:
                    writer.writerow([
                        c["start_ns"],
                        c["end_ns"],
                        f"{c['start_s']:.3f}",
                        f"{c['end_s']:.3f}",
                        c.get("frame_index", ""),
                        _csv_safe(c["text"]),
                    ])
        except OSError:
            logger.exception("Failed to export captions")
            self.app.show_notification("Error", "Could not write the CSV file.")
            return

        self.app.show_notification(
            "Export Complete", f"Wrote {len(self._captions)} captions."
        )

    # ------------------------------------------------------------------
    # Background work
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Background work
    # ------------------------------------------------------------------

    def _check_similarity_match(
        self,
        image: Image.Image,
        position: int,
        reference_thumb: np.ndarray | None,
        reference_text: str | None,
        is_event_gaze_aware: bool,
        previous_scene_state: str,
    ) -> tuple[np.ndarray | None, str | None, bool, str]:
        """Check if frame matches reference.

        Returns (thumb, text, is_no_change, raw_output).
        """
        if not self.skip_similar or position == 0:
            return None, None, False, ""
        thumb = _thumbnail(image)
        if (
            reference_thumb is not None
            and _thumbnail_difference(thumb, reference_thumb)
            <= self.similarity_threshold
        ):
            if is_event_gaze_aware:
                return (
                    thumb,
                    previous_scene_state or "No change.",
                    True,
                    "NO_CHANGE",
                )
            return thumb, reference_text, False, ""
        return thumb, None, False, ""

    def _validate_scene_frames(self) -> tuple[Any, np.ndarray, list[int]]:
        """Validate scene video presence and frame sample indices."""
        scene = getattr(self.recording, "scene", None)
        if scene is None or len(scene) == 0:
            self._record_failure("This recording has no scene video to caption.")
            raise RuntimeError("This recording has no scene video to caption")

        times_ns = np.asarray(scene.time)
        indices = _sample_indices(times_ns, self.interval_s)
        if not indices:
            self._record_failure("Could not select any scene frames to caption.")
            raise RuntimeError("Could not select any scene frames to caption")
        return scene, times_ns, indices

    def _record_failure(self, message: str) -> None:
        """Persist an error so `_on_captions_ready` can surface it.

        A background job that raises is reported to the UI as a successful
        completion, so without this the user is told a failed run worked.
        """
        try:
            self.save_cached_json(
                CACHE_FILENAME,
                {
                    "version": CACHE_VERSION,
                    "recording_start_time": self.recording.start_time,
                    "error": message,
                },
            )
        except OSError:
            logger.exception("Could not record the captioning failure")

    def _finalize_captions(
        self,
        captions: list[dict[str, Any]],
        times_ns: np.ndarray,
        total: int,
        t_start: float,
        reused: int,
        captioner: _Captioner,
    ) -> None:
        """Validate, timestamp, cache and log completed captions."""
        if not any(c["text"] for c in captions):
            message = (
                f"The model produced no usable caption for any of the {total} "
                "sampled frames. See the log for details."
            )
            self._record_failure(message)
            raise RuntimeError(message)

        _assign_end_times(captions, times_ns, self.recording.start_time)
        self.save_cached_json(
            CACHE_FILENAME,
            {
                "version": CACHE_VERSION,
                "recording_start_time": self.recording.start_time,
                "model": captioner.repo,
                "backend": captioner.backend,
                "prompt": self.prompt,
                "interval_s": self.interval_s,
                "captions": captions,
                "total_time_s": time.time() - t_start,
            },
        )
        logger.info(
            "Captioned %d scene frames (%d reused from near-identical frames)",
            len(captions),
            reused,
        )

    def bg_generate_captions(self) -> Generator[ProgressUpdate, None, None]:
        # NB: `ProgressUpdate`'s second field is `datum`, which nothing in the
        # app reads, so only the fraction is reported.
        yield ProgressUpdate(0.0)

        scene, times_ns, indices = self._validate_scene_frames()

        is_event_gaze_aware = (
            self.prompt_template == PromptTemplate.Event_Gaze_Aware
            or "temporal memory" in self.prompt.lower()
            or "event-oriented" in self.prompt.lower()
            or "EVENT_GAZE_SYSTEM_PROMPT" in self.prompt
        )

        max_tokens = self.max_tokens
        if is_event_gaze_aware and max_tokens < 200:
            max_tokens = 200

        captioner = _Captioner(self._model, max_tokens)
        total = len(indices)

        try:
            t_start = time.time()
            yield ProgressUpdate(0.02)
            try:
                captioner.load()
            except Exception as e:
                self._record_failure(
                    f"Could not load {captioner.repo} via {captioner.backend}: {e}"
                )
                raise

            captions: list[dict[str, Any]] = []
            reference_thumb: np.ndarray | None = None
            reference_text: str | None = None
            previous_scene_state = ""
            recent_events: list[str] = []
            reused = 0

            scene_w = getattr(scene, "width", None) or 1600
            scene_h = getattr(scene, "height", None) or 1200
            g_times, g_points_x, g_points_y = _extract_gaze_timeseries(
                self.recording
            )

            custom_instructions = ""
            if (
                is_event_gaze_aware
                and self.prompt.strip()
                != PROMPT_TEMPLATES["Event & Gaze Aware (Temporal)"].strip()
            ):
                custom_instructions = self.prompt.strip()

            for position, frame_index in enumerate(indices):
                image = _frame_to_pil(scene[frame_index].bgr)
                start_ns = int(times_ns[frame_index])

                vlm_gaze_x, vlm_gaze_y, gaze_valid = _lookup_gaze_point(
                    g_times,
                    g_points_x,
                    g_points_y,
                    start_ns,
                    scene_w,
                    scene_h,
                    image.width,
                    image.height,
                )

                event_data: dict[str, Any] | None = None

                (
                    thumb,
                    text,
                    is_no_change,
                    raw_output,
                ) = self._check_similarity_match(
                    image,
                    position,
                    reference_thumb,
                    reference_text,
                    is_event_gaze_aware,
                    previous_scene_state,
                )
                if text is not None:
                    reused += 1

                t0 = time.time()
                if text is None:
                    (
                        text,
                        raw_output,
                        is_no_change,
                        event_data,
                        new_events,
                    ) = _caption_single_frame(
                        captioner,
                        image,
                        frame_index,
                        start_ns,
                        self.recording.start_time,
                        vlm_gaze_x,
                        vlm_gaze_y,
                        gaze_valid,
                        is_event_gaze_aware,
                        previous_scene_state,
                        recent_events,
                        custom_instructions,
                        self.prompt,
                        reference_text,
                    )
                    if not is_no_change and text:
                        previous_scene_state = text
                    if new_events:
                        recent_events.extend(new_events)
                    if text and not is_no_change:
                        reference_text = text
                        reference_thumb = (
                            thumb if thumb is not None else _thumbnail(image)
                        )

                processing_time_s = time.time() - t0

                captions.append({
                    "frame_index": int(frame_index),
                    "start_ns": start_ns,
                    "start_s": (start_ns - self.recording.start_time) / 1e9,
                    "text": text or "",
                    "raw_output": raw_output,
                    "is_no_change": is_no_change,
                    "event_data": event_data,
                    "processing_time_s": processing_time_s,
                })

                yield ProgressUpdate(0.02 + 0.93 * (position + 1) / total)

            yield ProgressUpdate(0.97)
            self._finalize_captions(
                captions, times_ns, total, t_start, reused, captioner
            )
            yield ProgressUpdate(1.0)

        finally:
            captioner.close()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _caption_at(self, time_ns: int) -> dict[str, Any] | None:
        if not self._starts_ns:
            return None
        i = bisect.bisect_right(self._starts_ns, time_ns) - 1
        if i < 0:
            return None
        caption = self._captions[i]
        if time_ns > caption["end_ns"]:
            return None
        return caption

    def render(self, painter: QPainter, time_in_recording: int) -> None:
        # Despite the name, `time_in_recording` is an absolute nanosecond
        # timestamp on the same clock as `recording.start_time` and
        # `scene.time` (the app clamps it to [start_time, stop_time]), so it
        # can be compared against the captions' timestamps directly.
        if not self._show_captions or not self._captions or not self.recording:
            return

        caption = self._caption_at(time_in_recording)
        if caption is None:
            self._last_narrated_ns = -1
            return

        text = caption["text"].strip()
        if not text:
            return

        # Trigger TTS if a new caption appears (and is not an unchanged frame)
        if (
            getattr(self, "_narrate_captions", False)
            and getattr(self, "_tts", None)
            and caption["start_ns"] != self._last_narrated_ns
        ):
            self._last_narrated_ns = caption["start_ns"]
            if not caption.get("is_no_change", False):
                self._tts.stop()  # Interrupt previous speech
                self._tts.say(text)

        scene_width = self.recording.scene.width
        scene_height = self.recording.scene.height
        if not scene_width or not scene_height:
            return

        painter.save()
        try:
            painter.setFont(QFont("Arial", self.font_size, QFont.Weight.Normal))
            fm = painter.fontMetrics()

            flags = (
                Qt.TextFlag.TextWordWrap
                | Qt.AlignmentFlag.AlignHCenter
                | Qt.AlignmentFlag.AlignBottom
            )

            band_w = scene_width * CAPTION_BAND_WIDTH
            band = QRectF(
                (scene_width - band_w) / 2,
                scene_height * CAPTION_BAND_TOP,
                band_w,
                scene_height * CAPTION_BAND_HEIGHT,
            ).toRect()

            text, bounding_rect = _fit_text(text, fm, band, flags)
            if not text:
                return

            frame_rect = QRect(0, 0, int(scene_width), int(scene_height))
            background = QRect(
                bounding_rect.x() - 15,
                bounding_rect.y() - 10,
                bounding_rect.width() + 30,
                bounding_rect.height() + 20,
            ).intersected(frame_rect)

            painter.fillRect(background, self._bg_color)

            # Highlight keywords from any "Saw: <keyword>" events
            matching_keywords = [
                kw
                for kw in self._get_saw_keywords()
                if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE)
            ]
            _render_caption_text(
                painter,
                bounding_rect,
                flags,
                text,
                matching_keywords,
                self.text_color,
                self.highlight_color,
            )
        finally:
            painter.restore()
