# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch; sys_platform != 'darwin' or platform_machine != 'arm64'",
#     "transformers>=4.57; sys_platform != 'darwin' or platform_machine != 'arm64'",
#     "accelerate; sys_platform != 'darwin' or platform_machine != 'arm64'",
#     "numpy",
#     "pillow",
#     "mcp",
#     "uvicorn",
#     "anyio>=4.9,<4.15",
#     "pydantic>=2.11,<2.14",
#     "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git ; sys_platform == 'darwin' and platform_machine == 'arm64'",  # ruff: ignore[line-too-long]
# ]
#
# [tool.uv.sources]
# mlx-vlm = { git = "https://github.com/Blaizzy/mlx-vlm.git" }
# ///

"""Visual Context Plugin -- local VLM scene captioning.

Algorithms:
- Frame sampling: fixed-interval selection over `recording.scene.time`
  (absolute nanosecond UTC timestamps) via binary search; only the sampled
  frames are decoded.
- Redundancy skip: mean absolute difference of 64x64 grayscale thumbnails
  against the last frame that was actually captioned (not merely the last
  one sampled, so slow drift still eventually forces a fresh caption).
- Captioning (Apple Silicon): mlx-vlm running LiquidAI/LFM3-VL-2B or
  Qwen3-VL Instruct 4-bit on Metal via MLX.
- Captioning (elsewhere): HuggingFace transformers `AutoModelForImageTextToText`
  running the same LiquidAI/LFM3-VL-2B or Qwen3-VL Instruct checkpoints unquantized.
- Timestamps: captions carry absolute nanosecond timestamps taken directly
  from the scene video's own time array, so no epoch normalization is needed.
"""

from __future__ import annotations

import atexit
import bisect
import contextlib
import csv
import enum
import gc
import hashlib
import html
import json
import logging
import math
import operator
import platform
import re
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import tempfile
import threading
import time
from collections.abc import Generator
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image  # type: ignore[import-not-found]  # ty: ignore[import-not-found]
from pupil_labs.neon_player import Plugin, ProgressUpdate, action
from pupil_labs.neon_player.plugins.events import EventsPlugin
from PySide6.QtCore import (
    QObject,
    QPointF,
    QRect,
    QRectF,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QIcon,
    QPainter,
    QPen,
    QTextDocument,
    QTextOption,
)
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

if isinstance(property_params, type):
    _PARAMS_FROZEN = bool(
        getattr(
            getattr(property_params, "__dataclass_params__", None),
            "frozen",
            False,
        )
    )

    @dataclass(frozen=_PARAMS_FROZEN)
    class collapsible_property_params(property_params):
        enable_prop: str | None = None

else:
    collapsible_property_params = property_params  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

CACHE_VERSION = 3
CACHE_FILENAME = "visual_context.json"
MAX_IMAGE_EDGE = 384
THUMB_EDGE = 64
FALLBACK_FRAME_DURATION_NS = 33_333_333
CAPTION_BAND_TOP = 0.70
CAPTION_BAND_HEIGHT = 0.28
CAPTION_BAND_WIDTH = 0.85

DEFAULT_PROMPT = (
    "In a telegraphic, action-first phrase (under 15 words), describe what is "
    "visible and being done in this first-person scene. Start directly with an "
    "active verb or main objects (e.g. 'Pushing bicycle past brick stairs', "
    "'Sitting at desk typing'). Do not use filler phrases or third-person "
    "perspective words like 'A person' or 'The image shows'."
)

EVENT_GAZE_SYSTEM_PROMPT = (
    "You are an event-oriented, gaze-aware visual scene understanding system "
    "analyzing sequential frames from a scene camera worn by a person.\n"
    "The wearer's gaze position is provided as pixel coordinates (x, y).\n\n"
    "Your task is to produce a concise, telegraphic factual update of what is "
    "happening right now in the scene.\n\n"
    "Rules:\n"
    "1. Focus strictly on observable actions, visible objects, and what the "
    "wearer is looking at or interacting with.\n"
    "2. Limit the scene update to 1 short, telegraphic action-first phrase "
    '(under 15 words, e.g. "Holding phone with map next to parked bicycle", '
    '"Reaching for door handle to outdoor stairs").\n'
    '3. NEVER use filler phrases (do NOT say "The wearer is", '
    '"The camera wearer", "A person is", "The environment is", '
    '"In this image", "The camera shows", "The person\'s gaze is on"). '
    "Start directly with the action or objects.\n"
    "4. Only describe objects that are currently visible. When the scene or "
    "environment changes, describe the new scene and do not repeat previous "
    "objects.\n"
    "5. If the scene, objects, and wearer action are unchanged, return "
    "NO_CHANGE.\n"
    "6. Output valid JSON in the format:\n"
    '{"scene_update": "<telegraphic action phrase>", '
    '"gaze_target": "<object or surface at gaze point>", '
    '"events": ["<notable action if any>"]}'
)

EVENT_GAZE_PROMPT_TEMPLATE = (
    "Event-oriented, gaze-aware scene understanding with temporal memory. "
    "Structured frame context (timestamp, resolution, scaled gaze, prior scene "
    "state, recent events) is injected automatically. Unchanged scenes output "
    "NO_CHANGE."
)

DETAILED_SCENE_PROMPT = (
    "Provide a highly detailed description of what appears in this scene. "
    "Name all visible objects, text, and environmental details. "
    "Avoid first-person perspective phrasing."
)

PROMPT_TEMPLATES = {
    "Detailed Scene": DETAILED_SCENE_PROMPT,
    "Default": DEFAULT_PROMPT,
    "Gaze & Action": (
        "In a telegraphic, action-first phrase (under 15 words), describe what "
        "is being done or looked at in this first-person scene. The gaze point "
        "is at ({gaze_x}, {gaze_y}) in this {width}x{height} image. Start "
        "directly with an active verb or object (e.g. 'Holding phone with map', "
        "'Pushing bicycle up stairs', 'Reaching for door handle'). NEVER use "
        "filler phrases like 'The wearer', 'The camera wearer', 'A person', "
        "'A first-person view', 'This image shows', or 'The environment is'."
    ),
    "Event & Gaze Aware (Temporal)": EVENT_GAZE_PROMPT_TEMPLATE,
    "Gazed Object (One Word)": (
        "The user's gaze is at pixel coordinates ({gaze_x}, {gaze_y}) in this "
        "{width}x{height} image. In exactly one word, name the main object at "
        "this location. Do not describe anything else."
    ),
    "Delta Changes": (
        "The previous frame was described as: '{prev_desc}'.\n"
        "Describe only what has changed in this new frame. "
        "Focus on what the wearer is currently doing, what new objects have appeared, "
        "or what objects have disappeared. Keep it brief."
    ),
}


class PromptTemplate(enum.Enum):
    Detailed_Scene = "Detailed Scene"
    Default = "Default"
    Gaze_Action = "Gaze & Action"
    Event_Gaze_Aware = "Event & Gaze Aware (Temporal)"
    Gazed_Object = "Gazed Object (One Word)"
    Delta_Changes = "Delta Changes"
    Custom = "Custom"


_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


class CaptionModel(enum.Enum):
    """Model presets, smallest first.

    Values are the Apple-Silicon (MLX) repository IDs; `hf_repo` maps each
    preset onto the equivalent upstream checkpoint for the transformers path.
    """

    Liquid_2B = "LiquidAI/LFM3-VL-2B-MLX-4bit"
    Fast_2B = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
    Balanced_4B = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    Quality_8B = "mlx-community/Qwen3-VL-8B-Instruct-4bit"

    @property
    def mlx_repo(self) -> str:
        return self.value

    @property
    def hf_repo(self) -> str:
        return {
            CaptionModel.Liquid_2B: "LiquidAI/LFM3-VL-2B",
            CaptionModel.Fast_2B: "Qwen/Qwen3-VL-2B-Instruct",
            CaptionModel.Balanced_4B: "Qwen/Qwen3-VL-4B-Instruct",
            CaptionModel.Quality_8B: "Qwen/Qwen3-VL-8B-Instruct",
        }[self]


class CaptionStorage:
    """Manages cache key hashing, cache loading/saving, and CSV exports."""

    @staticmethod
    def settings_cache_key(
        model: str,
        prompt: str,
        interval_s: float,
        skip_similar: bool,
        similarity_threshold: float,
    ) -> str:
        raw = (
            f"{model}|{prompt.strip()}|{interval_s:.3f}|"
            f"{bool(skip_similar)}|{similarity_threshold:.2f}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]

    @classmethod
    def cache_filename_for(
        cls,
        model: str,
        prompt: str,
        interval_s: float,
        skip_similar: bool,
        similarity_threshold: float,
    ) -> str:
        key = cls.settings_cache_key(
            model, prompt, interval_s, skip_similar, similarity_threshold
        )
        model_tag = re.sub(r"[^a-zA-Z0-9_.-]", "_", model.rsplit("/", maxsplit=1)[-1])
        return f"visual_context_{model_tag}_{key}.json"

    @staticmethod
    def format_time(seconds: float) -> str:
        seconds = max(0.0, seconds)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:06.3f}"
        return f"{m:02d}:{s:06.3f}"

    @staticmethod
    def validate_captions(data: dict[str, Any]) -> list[dict[str, Any]] | None:
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

        cleaned.sort(key=operator.itemgetter("start_ns"))
        return cleaned

    @classmethod
    def read_cache(
        cls,
        load_fn: Any,
        model: str,
        prompt: str,
        interval_s: float,
        skip_similar: bool,
        similarity_threshold: float,
    ) -> dict[str, Any] | None:
        target_filename = cls.cache_filename_for(
            model, prompt, interval_s, skip_similar, similarity_threshold
        )
        try:
            data = load_fn(target_filename)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to read visual context cache %s", target_filename)

        try:
            legacy = load_fn(CACHE_FILENAME)
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to read visual context cache")
            return None

        if not isinstance(legacy, dict):
            return None

        if (
            legacy.get("model") == model
            and str(legacy.get("prompt", "")).strip() == prompt.strip()
            and abs(float(legacy.get("interval_s", -1.0)) - float(interval_s)) < 1e-3
        ):
            if "skip_similar" in legacy and bool(legacy["skip_similar"]) != bool(
                skip_similar
            ):
                return None
            if (
                skip_similar
                and "similarity_threshold" in legacy
                and abs(
                    float(legacy["similarity_threshold"]) - float(similarity_threshold)
                )
                >= 1e-2
            ):
                return None
            return legacy

        return None

    @classmethod
    def record_failure(
        cls,
        load_fn: Any,
        save_fn: Any,
        start_time: Any,
        model: str,
        prompt: str,
        interval_s: float,
        skip_similar: bool,
        similarity_threshold: float,
        message: str,
    ) -> None:
        payload = {
            "version": CACHE_VERSION,
            "recording_start_time": start_time,
            "model": model,
            "prompt": prompt,
            "interval_s": interval_s,
            "skip_similar": skip_similar,
            "similarity_threshold": similarity_threshold,
            "error": message,
        }
        try:
            target_filename = cls.cache_filename_for(
                model, prompt, interval_s, skip_similar, similarity_threshold
            )
            existing = load_fn(target_filename)
            if not (isinstance(existing, dict) and existing.get("captions")):
                save_fn(target_filename, payload)
            save_fn(CACHE_FILENAME, payload)
        except OSError:
            logger.exception("Could not record the captioning failure")

    @classmethod
    def save_cache(
        cls,
        save_fn: Any,
        start_time: Any,
        model: str,
        backend: str,
        prompt: str,
        interval_s: float,
        skip_similar: bool,
        similarity_threshold: float,
        captions: list[dict[str, Any]],
        total_time_s: float,
    ) -> str:
        payload = {
            "version": CACHE_VERSION,
            "recording_start_time": start_time,
            "model": model,
            "backend": backend,
            "prompt": prompt,
            "interval_s": interval_s,
            "skip_similar": skip_similar,
            "similarity_threshold": similarity_threshold,
            "captions": captions,
            "total_time_s": total_time_s,
        }
        cache_filename = cls.cache_filename_for(
            model, prompt, interval_s, skip_similar, similarity_threshold
        )
        save_fn(cache_filename, payload)
        save_fn(CACHE_FILENAME, payload)
        return cache_filename

    @staticmethod
    def csv_safe(value: str) -> str:
        if value.startswith(_CSV_FORMULA_PREFIXES) or value.lstrip().startswith(
            _CSV_FORMULA_PREFIXES
        ):
            return "'" + value
        return value

    @classmethod
    def write_csv(cls, captions: list[dict[str, Any]], target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "start timestamp [ns]",
                "end timestamp [ns]",
                "start time [s]",
                "end time [s]",
                "frame index",
                "caption",
            ])
            for c in captions:
                writer.writerow([
                    c["start_ns"],
                    c["end_ns"],
                    f"{c['start_s']:.3f}",
                    f"{c['end_s']:.3f}",
                    c.get("frame_index", ""),
                    cls.csv_safe(c["text"]),
                ])


_BOILERPLATE_LEADS = (
    "the image shows",
    "this image shows",
    "the image depicts",
    "this image depicts",
    "the photo shows",
    "this photo shows",
    "the environment is",
    "this environment is",
    "the scene depicts",
    "the camera is focused on",
    "the camera shows",
    "the view shows",
    "a view of",
)
_ARTICLES = ("a ", "an ", "the ")

_BOILERPLATE_CLAUSES = (
    "in this image, ",
    "in the image, ",
    "in this photo, ",
    "in this frame, ",
    "in this scene, ",
)

_FILLER_REGEX = re.compile(
    r"^(in |from )?a first-person (view|perspective)( shows| of| from|,)?\s*"
    r"|^(the )?(camera )?wearer('s| is| looks| holds| stands| sees)?"
    r"( now)?( seen| looking at| holding)?\s*"
    r"|^a person('s| is| holds| stands)?( seen)?( looking at| holding)?\s*"
    r"|^(this |the )?(image|photo|scene|environment)( shows| depicts| is)?( a| an)?\s*",
    re.IGNORECASE,
)


class TemporalResponseParser:
    """Parses and sanitizes VLM model outputs and temporal JSON responses."""

    @staticmethod
    def capitalize(text: str) -> str:
        return text[:1].upper() + text[1:]

    @classmethod
    def clean_caption(cls, text: Any) -> str:
        if not isinstance(text, str):
            return ""

        cleaned = " ".join(text.split())
        cleaned = _FILLER_REGEX.sub("", cleaned).strip()
        lowered = cleaned.lower()

        for clause in _BOILERPLATE_CLAUSES:
            if lowered.startswith(clause):
                return cls.capitalize(cleaned[len(clause) :])

        for lead in _BOILERPLATE_LEADS:
            if not lowered.startswith(lead + " "):
                continue
            remainder = cleaned[len(lead) + 1 :]
            if remainder.lower().startswith(_ARTICLES):
                return cls.capitalize(remainder)

        return cls.capitalize(cleaned)

    @classmethod
    def clean_event_str(cls, text: Any) -> str:
        if not isinstance(text, str):
            return ""
        stripped = re.sub(r"<[^>]+>", "", text)
        return cls.clean_caption(stripped)

    @staticmethod
    def is_no_change(text: Any) -> bool:
        s = str(text or "").strip().upper().rstrip(".")
        return s == "NO_CHANGE" or s.startswith("NO_CHANGE") or s.endswith("NO_CHANGE")

    @staticmethod
    def bound_scene_state(text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        first_sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
        if len(first_sentence) <= 120:
            return first_sentence
        head = first_sentence[:120]
        return (head.rsplit(" ", 1)[0] if " " in head else head).rstrip(".") + "."

    @staticmethod
    def extract_json_block(text: str) -> str:
        if "```json" in text:
            return text.split("```json")[1].split("```", maxsplit=1)[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```", maxsplit=1)[0].strip()
        if "{" in text and "}" in text:
            s_idx = text.find("{")
            e_idx = text.rfind("}") + 1
            return text[s_idx:e_idx]
        return text

    @classmethod
    def parse_json_event_payload(
        cls, parsed: dict[str, Any], previous_state: str, raw_text: str
    ) -> tuple[str, bool, dict[str, Any], list[str]]:
        scene_update = parsed.get("scene_update")
        if scene_update and cls.is_no_change(scene_update):
            text = previous_state or "No change."
            is_no_change = True
        elif scene_update:
            text = cls.clean_event_str(str(scene_update))
            is_no_change = False
        else:
            desc = parsed.get("description", raw_text)
            text = cls.clean_event_str(str(desc))
            is_no_change = False

        new_events: list[str] = []
        evs = parsed.get("events")
        if isinstance(evs, list):
            for ev in evs:
                raw_ev = (
                    (ev.get("description") or ev.get("type") or str(ev))
                    if isinstance(ev, dict)
                    else (ev if isinstance(ev, str) else "")
                )
                cleaned_ev = (
                    cls.clean_event_str(str(raw_ev)) if str(raw_ev).strip() else ""
                )
                if cleaned_ev:
                    new_events.append(cleaned_ev)

        return text, is_no_change, parsed, new_events

    @classmethod
    def regex_extract_update(cls, text: str) -> str | None:
        match = re.search(
            r'"(?:scene_update|description)"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)', text
        )
        if match:
            extracted = match.group(1).replace(r"\"", '"').replace(r"\n", " ").strip()
            if extracted and not extracted.endswith((".", "!", "?")):
                last_period = max(
                    extracted.rfind("."), extracted.rfind("!"), extracted.rfind("?")
                )
                if last_period > 10:
                    extracted = extracted[: last_period + 1].strip()
            cleaned = cls.clean_event_str(extracted)
            if cleaned:
                return cleaned
        return None

    @classmethod
    def parse(
        cls, raw_output: str, previous_scene_state: str
    ) -> tuple[str, bool, dict[str, Any] | None, list[str]]:
        clean_resp = raw_output.strip()

        if cls.is_no_change(clean_resp) or (
            "frame_index" in clean_resp and "image_width" in clean_resp
        ):
            return previous_scene_state or "No change.", True, None, []

        json_str = cls.extract_json_block(clean_resp)
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return cls.parse_json_event_payload(
                    parsed, previous_scene_state, clean_resp
                )
        except (json.JSONDecodeError, ValueError):
            pass

        extracted = cls.regex_extract_update(clean_resp)
        if extracted:
            if cls.is_no_change(extracted):
                return previous_scene_state or "No change.", True, None, []
            return extracted, False, None, [extracted]

        if clean_resp.startswith("{"):
            return previous_scene_state or "No change.", True, None, []

        cleaned = cls.clean_event_str(clean_resp)
        return cleaned, False, None, [cleaned]


class CaptionRenderer:
    """Handles text fitting, word wrapping, and Qt rendering for captions."""

    @staticmethod
    def fit_text(
        text: str, fm: QFontMetrics, band: QRect, flags: int
    ) -> tuple[str, QRect]:
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

        candidate = words[0] if words else ""
        rect = fm.boundingRect(band, flags, candidate)
        return candidate, rect.intersected(band)

    @staticmethod
    def render_text(
        painter: QPainter,
        bounding_rect: QRect,
        flags: int,
        text: str,
        matching_keywords: list[str],
        text_color: QColor,
        highlight_color: QColor,
    ) -> None:
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
            f'<div style="text-align: center; color: {text_hex};">{escaped}</div>'
        )

        painter.save()
        painter.translate(QPointF(bounding_rect.topLeft()))
        doc.drawContents(painter)
        painter.restore()


def _clear_mlx_cache() -> None:
    if "mlx" in sys.modules or "mlx.core" in sys.modules:
        try:
            import mlx.core as mx  # type: ignore[import-not-found]  # ty: ignore[import-not-found]

            mx.clear_cache()
        except Exception:
            logger.debug("Could not clear MLX cache", exc_info=True)


class _Captioner:
    """Loads a VLM once and captions many images, then releases it."""

    def __init__(self, model: CaptionModel | str, max_tokens: int) -> None:
        if isinstance(model, str):
            try:
                model = CaptionModel(model)
            except ValueError:
                model = CaptionModel[model]
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
            self._model = None
            self._processor = None
            raise

    def _load_mlx(self) -> None:
        try:
            import mlx.core as mx  # type: ignore[import-not-found]  # ty: ignore[import-not-found]

            if hasattr(mx, "set_cache_limit"):
                mx.set_cache_limit(128 * 1024 * 1024)
            elif hasattr(mx, "metal") and hasattr(mx.metal, "set_cache_limit"):
                mx.metal.set_cache_limit(128 * 1024 * 1024)
        except Exception:
            logger.debug("Could not set MLX cache limit", exc_info=True)

        from mlx_vlm import load  # type: ignore[import-not-found]  # ty: ignore[import-not-found]
        from mlx_vlm.utils import load_config  # type: ignore[import-not-found]  # ty: ignore[import-not-found]

        model, processor = load(self.repo)
        self._config = load_config(self.repo)
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="npp_visual_context_")
        self._tmp_image = Path(self._tmp_dir.name) / "frame.jpg"
        self._model, self._processor = model, processor

    def _load_transformers(self) -> None:
        import torch  # type: ignore[import-not-found]  # ty: ignore[import-not-found]
        from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore[import-not-found]  # ty: ignore[import-not-found]

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
        from mlx_vlm import generate  # type: ignore[import-not-found]  # ty: ignore[import-not-found]
        from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore[import-not-found]  # ty: ignore[import-not-found]

        assert self._tmp_image is not None
        image.save(self._tmp_image, format="JPEG", quality=92)

        payload: Any = (
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            if system_prompt
            else prompt
        )
        formatted = apply_chat_template(
            self._processor, self._config, payload, num_images=1
        )
        result = generate(
            self._model,
            self._processor,
            formatted,
            [str(self._tmp_image)],
            max_tokens=self._max_tokens,
            temperature=0.0,
            repetition_penalty=1.15,
            verbose=False,
        )
        _clear_mlx_cache()

        raw = getattr(result, "text", result)
        if system_prompt:
            return str(raw).strip() if raw is not None else ""
        return _clean_caption(raw)

    def _caption_transformers(
        self, image: Image.Image, prompt: str, system_prompt: str | None = None
    ) -> str:
        torch = self._torch
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        })
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self._max_tokens,
                do_sample=False,
                repetition_penalty=1.15,
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
            import torch  # type: ignore[import-not-found]  # ty: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        _clear_mlx_cache()


class VlmCaptioningPipeline:
    """Coordinates frame sampling, gaze mapping, and VLM batching."""

    @staticmethod
    def sample_indices(times_ns: np.ndarray, interval_s: float) -> list[int]:
        n = len(times_ns)
        if n == 0:
            return []

        order: np.ndarray | None = None
        if np.any(np.diff(times_ns) < 0):
            logger.warning(
                "Scene timestamps are not monotonic; sorting before sampling"
            )
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

    @staticmethod
    def assign_end_times(
        captions: list[dict[str, Any]], times_ns: np.ndarray, rec_start_ns: int
    ) -> None:
        if not captions or len(times_ns) == 0:
            return

        frame_duration_ns = (
            int(np.median(np.diff(times_ns))) if len(times_ns) > 1 else 0
        )
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

    @staticmethod
    def frame_to_pil(bgr: np.ndarray) -> Image.Image:
        if bgr.ndim == 2:
            rgb = np.stack([bgr] * 3, axis=-1)
        elif bgr.ndim == 3 and bgr.shape[2] >= 3:
            rgb = bgr[:, :, 2::-1]
        else:
            raise ValueError(f"Unexpected scene frame shape: {bgr.shape}")

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

    @staticmethod
    def thumbnail(image: Image.Image) -> np.ndarray:
        small = image.convert("L").resize(
            (THUMB_EDGE, THUMB_EDGE), Image.Resampling.BILINEAR
        )
        return np.asarray(small, dtype=np.int16)

    @staticmethod
    def thumbnail_difference(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    @staticmethod
    def extract_gaze_timeseries(
        recording: Any,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
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

    @staticmethod
    def lookup_gaze_point(
        g_times: np.ndarray | None,
        g_points_x: np.ndarray | None,
        g_points_y: np.ndarray | None,
        timestamp_ns: int,
        scene_w: int,
        scene_h: int,
        target_w: int,
        target_h: int,
    ) -> tuple[int | None, int | None, bool]:
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
                np.clip(np.searchsorted(g_times, timestamp_ns), 0, len(g_times) - 1)
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

    @classmethod
    def check_similarity_match(
        cls,
        image: Image.Image,
        position: int,
        reference_thumb: np.ndarray | None,
        reference_text: str | None,
        is_event_gaze_aware: bool,
        previous_scene_state: str,
        skip_similar: bool,
        similarity_threshold: float,
    ) -> tuple[np.ndarray | None, str | None, bool, str]:
        if not skip_similar or position == 0:
            return None, None, False, ""
        thumb = cls.thumbnail(image)
        if (
            reference_thumb is not None
            and cls.thumbnail_difference(thumb, reference_thumb) <= similarity_threshold
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

    @classmethod
    def caption_single_frame(
        cls,
        captioner: _Captioner,
        image: Image.Image,
        frame_index: int,
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
        if is_event_gaze_aware:
            gaze_str = (
                f"({vlm_gaze_x}, {vlm_gaze_y})"
                if gaze_valid and vlm_gaze_x is not None and vlm_gaze_y is not None
                else "unknown"
            )
            bounded_prev = (
                TemporalResponseParser.bound_scene_state(previous_scene_state)
                or "Initial frame."
            )

            prompt_lines = [
                f"Image dimensions: {image.width}x{image.height}.",
                f"Wearer gaze position: {gaze_str}.",
                f"Previous scene state: {bounded_prev}",
            ]
            if recent_events:
                prompt_lines.append(f"Recent events: {', '.join(recent_events[-3:])}.")
            if custom_instructions:
                prompt_lines.append(f"User instructions: {custom_instructions}")
            prompt_lines.append("Describe the current frame in JSON format.")
            user_prompt = "\\n".join(prompt_lines)

            try:
                raw_output = captioner.caption(
                    image,
                    user_prompt,
                    system_prompt=EVENT_GAZE_SYSTEM_PROMPT,
                )
            except Exception:
                logger.exception("Captioning failed for scene frame %s", frame_index)
                return "", "", False, None, []

            text, is_no_change, event_data, new_events = TemporalResponseParser.parse(
                raw_output, previous_scene_state
            )
            return text, raw_output, is_no_change, event_data, new_events

        gaze_str_x = str(vlm_gaze_x) if vlm_gaze_x is not None else "unknown"
        gaze_str_y = str(vlm_gaze_y) if vlm_gaze_y is not None else "unknown"
        current_prompt = (
            legacy_prompt
            .replace("{gaze_x}", gaze_str_x)
            .replace("{gaze_y}", gaze_str_y)
            .replace("{width}", str(image.width))
            .replace("{height}", str(image.height))
            .replace("{prev_desc}", reference_text or "None")
        )

        try:
            raw_output = captioner.caption(image, current_prompt)
        except Exception:
            logger.exception("Captioning failed for scene frame %s", frame_index)
            return "", "", False, None, []

        return raw_output, raw_output, False, None, []

    @classmethod
    def validate_scene(
        cls, recording: Any, interval_s: float
    ) -> tuple[Any, np.ndarray, list[int]]:
        scene = getattr(recording, "scene", None)
        if scene is None or len(scene) == 0:
            raise RuntimeError("This recording has no scene video to caption")

        times_ns = np.asarray(scene.time)
        indices = cls.sample_indices(times_ns, interval_s)
        if not indices:
            raise RuntimeError("Could not select any scene frames to caption")
        return scene, times_ns, indices

    @classmethod
    def run(
        cls,
        recording: Any,
        model: CaptionModel | str,
        prompt: str,
        prompt_template: str | PromptTemplate,
        interval_s: float,
        skip_similar: bool,
        similarity_threshold: float,
        max_tokens: int,
        save_cache_fn: Any,
        record_failure_fn: Any,
    ) -> Generator[ProgressUpdate, None, None]:
        yield ProgressUpdate(0.0)

        try:
            scene, times_ns, indices = cls.validate_scene(recording, interval_s)
        except RuntimeError as exc:
            record_failure_fn(str(exc))
            raise

        is_event_gaze_aware = (
            prompt_template == PromptTemplate.Event_Gaze_Aware
            or "temporal memory" in prompt.lower()
            or "event-oriented" in prompt.lower()
            or "EVENT_GAZE_SYSTEM_PROMPT" in prompt
        )
        effective_max_tokens = (
            max(max_tokens, 350) if is_event_gaze_aware else max_tokens
        )

        captioner = _Captioner(model, effective_max_tokens)
        total = len(indices)

        try:
            t_start = time.time()
            yield ProgressUpdate(0.02)
            try:
                captioner.load()
            except Exception as e:
                record_failure_fn(
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
            g_times, g_points_x, g_points_y = cls.extract_gaze_timeseries(recording)

            custom_instructions = ""
            if (
                is_event_gaze_aware
                and prompt.strip()
                != PROMPT_TEMPLATES["Event & Gaze Aware (Temporal)"].strip()
            ):
                custom_instructions = prompt.strip()

            for position, frame_index in enumerate(indices):
                image = cls.frame_to_pil(scene[frame_index].bgr)
                start_ns = int(times_ns[frame_index])

                vlm_gaze_x, vlm_gaze_y, gaze_valid = cls.lookup_gaze_point(
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
                ) = cls.check_similarity_match(
                    image,
                    position,
                    reference_thumb,
                    reference_text,
                    is_event_gaze_aware,
                    previous_scene_state,
                    skip_similar,
                    similarity_threshold,
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
                    ) = cls.caption_single_frame(
                        captioner,
                        image,
                        frame_index,
                        vlm_gaze_x,
                        vlm_gaze_y,
                        gaze_valid,
                        is_event_gaze_aware,
                        previous_scene_state,
                        recent_events,
                        custom_instructions,
                        prompt,
                        reference_text,
                    )
                    if not is_no_change and text:
                        previous_scene_state = TemporalResponseParser.bound_scene_state(
                            text
                        )
                    if new_events:
                        recent_events.extend(new_events)
                    if text and not is_no_change:
                        reference_text = text
                        reference_thumb = (
                            thumb if thumb is not None else cls.thumbnail(image)
                        )

                processing_time_s = time.time() - t0

                captions.append({
                    "frame_index": int(frame_index),
                    "start_ns": start_ns,
                    "start_s": (start_ns - recording.start_time) / 1e9,
                    "text": text or "",
                    "raw_output": raw_output,
                    "is_no_change": is_no_change,
                    "event_data": event_data,
                    "processing_time_s": processing_time_s,
                })

                yield ProgressUpdate(0.02 + 0.93 * (position + 1) / total)

            yield ProgressUpdate(0.97)
            save_cache_fn(captions, times_ns, total, t_start, reused, captioner)
            yield ProgressUpdate(1.0)

        finally:
            captioner.close()


def _proxy_property(target_attr: str, name: str) -> property:
    """Create a hidden backward-compatibility property forwarding to a subgroup."""

    @property_params(widget=None, dont_encode=True)
    def _get(self: Any) -> object:
        return getattr(getattr(self, target_attr), name)

    def _set(self: Any, value: Any) -> None:
        setattr(getattr(self, target_attr), name, value)

    return property(_get, _set)


class _SettingsGroup(PersistentPropertiesMixin, QObject):
    """Base class for persistent property subgroups."""

    changed = Signal()

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> Any:
        inst = cls()
        inst.__setstate__(state)
        return inst


class CaptionColors(_SettingsGroup):
    """Pair of text and highlight colors."""

    def __init__(
        self,
        text_color: QColor | str | dict[str, Any] = "#FFFFFF",
        highlight_color: QColor | str = "#FFD600",
    ) -> None:
        super().__init__()
        if isinstance(text_color, dict):
            data = text_color
            text_color = data.get("text_color", "#FFFFFF")
            highlight_color = data.get("highlight_color", "#FFD600")
        self._text_color = (
            QColor(*text_color)
            if isinstance(text_color, (list, tuple))
            else QColor(text_color)
        )
        self._highlight_color = (
            QColor(*highlight_color)
            if isinstance(highlight_color, (list, tuple))
            else QColor(highlight_color)
        )

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
        self._value = CaptionColors(col, self._value.highlight_color)
        self.value_changed.emit(self._value)

    def _on_hl_changed(self, col: QColor) -> None:
        if self._updating:
            return
        self._value = CaptionColors(self._value.text_color, col)
        self.value_changed.emit(self._value)

    @property
    def value(self) -> CaptionColors:
        return self._value

    @value.setter
    def value(self, val: CaptionColors | dict) -> None:
        self._updating = True
        try:
            if isinstance(val, dict):
                self._value = CaptionColors(val)
            elif isinstance(val, CaptionColors):
                self._value = CaptionColors(val.text_color, val.highlight_color)
            self.widget_text.value = self._value.text_color
            self.widget_hl.value = self._value.highlight_color
        finally:
            self._updating = False


class CaptionStyle(_SettingsGroup):
    """Caption appearance and style settings."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._font_size = 26
        self._colors = CaptionColors()
        self._colors.changed.connect(self.changed.emit)
        if isinstance(data, dict):
            self.__setstate__(data)

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
    @property_params(label="Color", widget=cast(Any, CaptionColorsWidget))
    def colors(self) -> CaptionColors:
        return self._colors

    @colors.setter
    def colors(self, v: CaptionColors | dict) -> None:
        if isinstance(v, dict):
            self._colors.__setstate__(v)
        elif isinstance(v, CaptionColors) and v is not self._colors:
            self._colors = v
            self._colors.changed.connect(self.changed.emit)
        self.changed.emit()

    text_color = _proxy_property("_colors", "text_color")
    highlight_color = _proxy_property("_colors", "highlight_color")


class CollapsibleGroupWidget(PropertyWidget):
    """Renders a custom type inside a collapsible Expander section."""

    value_changed = Signal(object)

    @staticmethod
    def from_property_impl(prop: property) -> CollapsibleGroupWidget:
        fget = getattr(prop, "fget", None)
        target_type: Any = None
        if fget is not None:
            try:
                import typing as T

                target_type = T.get_type_hints(fget).get("return")
            except (TypeError, NameError, AttributeError):
                target_type = None
            if not isinstance(target_type, type):
                ret = getattr(fget, "__annotations__", {}).get("return")
                if isinstance(ret, type):
                    target_type = ret
                elif isinstance(ret, str):
                    target_type = getattr(fget, "__globals__", {}).get(ret, object)
        if not isinstance(target_type, type):
            target_type = object

        params = getattr(fget, "parameters", {}) if fget is not None else {}
        title = params.get("label") or (
            fget.__name__.replace("_", " ").title() if fget is not None else ""
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
        if (exp_layout := self.expander.layout()) is not None:
            exp_layout.setContentsMargins(0, 0, 0, 0)
        self.setObjectName("CollapsibleGroupWidget")
        self.setStyleSheet("""
            CollapsibleGroupWidget Expander {
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
        if (
            enable_prop
            and hasattr(instance, enable_prop)
            and getattr(self, "bool_widget", None) is None
        ):
            self.bool_widget = BoolWidget()
            self.bool_widget.spacer.setVisible(False)
            self.bool_widget.value = bool(getattr(instance, enable_prop))
            self.bool_widget.value_changed.connect(
                lambda val: setattr(instance, enable_prop, val)
            )
            if hasattr(instance, "changed") and hasattr(instance.changed, "connect"):
                instance.changed.connect(
                    lambda: (
                        setattr(
                            self.bool_widget,
                            "value",
                            bool(getattr(instance, enable_prop)),
                        )
                        if self.bool_widget.value
                        != bool(getattr(instance, enable_prop))
                        else None
                    )
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
        if hasattr(getattr(self, "_target_instance", None), "open_prompt_editor"):
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


class CaptioningSettings(_SettingsGroup):
    """VLM captioning generation and prompt settings."""

    def __init__(
        self,
        prompt: str | dict[str, Any] = DETAILED_SCENE_PROMPT,
        max_tokens: int = 64,
        interval_s: float = 2.0,
        skip_similar: bool = True,
        similarity_threshold: float = 6.0,
    ) -> None:
        super().__init__()
        self._prompt = prompt if isinstance(prompt, str) else DETAILED_SCENE_PROMPT
        self._max_tokens = max_tokens
        self._interval_s = interval_s
        self._skip_similar = skip_similar
        self._similarity_threshold = similarity_threshold
        if isinstance(prompt, dict):
            self.__setstate__(prompt)

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
            try:
                v = PromptTemplate(v)
            except ValueError:
                v = PromptTemplate[v]
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
        self._prompt = text or DETAILED_SCENE_PROMPT
        self.changed.emit()

    @property
    @property_params(
        label="Edit Prompt", widget=cast(Any, EditPromptWidget), dont_encode=True
    )
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
        combo.addItems([*PROMPT_TEMPLATES.keys(), "Custom"])
        combo.setCurrentText(_match_prompt_template(self._prompt))
        tmpl_layout.addWidget(combo)
        layout.addLayout(tmpl_layout)

        layout.addWidget(QLabel("Prompt sent to the VLM for every frame:"))

        text_edit = QTextEdit()
        text_edit.setPlainText(self._prompt)
        layout.addWidget(text_edit)

        combo.currentTextChanged.connect(
            lambda text: (
                text_edit.setPlainText(PROMPT_TEMPLATES[text])
                if text in PROMPT_TEMPLATES
                else None
            )
        )
        text_edit.textChanged.connect(
            lambda: combo.setCurrentText(
                _match_prompt_template(text_edit.toPlainText())
            )
        )

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


class MCPInfoWidget(PropertyWidget):
    """Displays MCP endpoint URL and a button to copy configuration."""

    value_changed = Signal(object)

    @staticmethod
    def from_property_impl(prop: property) -> MCPInfoWidget:
        return MCPInfoWidget()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)

        self.endpoint_label = QLabel("http://127.0.0.1:8765/mcp")
        self.endpoint_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self.endpoint_label)

        self.btn_copy = QPushButton("Copy Config")
        self.btn_copy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_copy.clicked.connect(self._copy_config)
        layout.addWidget(self.btn_copy)

        self.grid_layout.addLayout(layout, 0, 0)
        self._target_instance: Any = None

    def on_bind(self, instance: Any) -> None:
        super().on_bind(instance)
        self._target_instance = instance
        self._update_url()
        if hasattr(instance, "changed") and hasattr(instance.changed, "connect"):
            instance.changed.connect(self._update_url)

    def _get_port(self) -> int:
        return int(getattr(self._target_instance, "port", 8765))

    def _update_url(self) -> None:
        self.endpoint_label.setText(f"http://127.0.0.1:{self._get_port()}/mcp")

    def _copy_config(self) -> None:
        port = self._get_port()
        config_text = json.dumps(
            {
                "mcpServers": {
                    "neon-visual-context": {"url": f"http://127.0.0.1:{port}/mcp"}
                }
            },
            indent=2,
        )
        QGuiApplication.clipboard().setText(config_text)
        self.btn_copy.setText("Copied!")
        QTimer.singleShot(1500, lambda: self.btn_copy.setText("Copy Config"))

    @property
    def value(self) -> Any:
        return None

    @value.setter
    def value(self, v: Any) -> None:
        pass


class MCPSettings(_SettingsGroup):
    """MCP server configuration and endpoint."""

    def __init__(self, port: int | dict[str, Any] = 8765) -> None:
        super().__init__()
        self._port = 8765 if isinstance(port, dict) else int(port)
        if isinstance(port, dict):
            self.__setstate__(port)

    @property
    @property_params(label="Server Port", min=1024, max=65535)
    def port(self) -> int:
        return self._port

    @port.setter
    def port(self, v: int) -> None:
        self._port = min(max(int(v), 1024), 65535)
        self.changed.emit()

    @property
    @property_params(
        label="Endpoint", widget=cast(Any, MCPInfoWidget), dont_encode=True
    )
    def endpoint(self) -> None:
        return None

    @endpoint.setter
    def endpoint(self, v: Any) -> None:
        pass


class _MacSayTTS:
    """Fallback TTS using macOS `/usr/bin/say` when `QtTextToSpeech` is absent."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None
        atexit.register(self.stop)

    def say(self, text: str) -> None:
        self.stop()
        if not text:
            return
        try:
            self._proc = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
                ["/usr/bin/say", "--", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            logger.debug("Failed to spawn /usr/bin/say", exc_info=True)

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.kill()

    def __del__(self) -> None:
        self.stop()


class TTSController:
    """Controls text-to-speech narration with deduplication and backend fallback."""

    def __init__(self) -> None:
        self._backend: Any = self.init_backend()
        self._last_narrated_ns: int = -1
        atexit.register(self.stop)

    @staticmethod
    def init_backend() -> Any:
        try:
            from PySide6.QtTextToSpeech import QTextToSpeech  # type: ignore[import-not-found]

            return QTextToSpeech()
        except Exception:
            if sys.platform == "darwin" and Path("/usr/bin/say").exists():
                return _MacSayTTS()
            logger.debug("QTextToSpeech not available in this build", exc_info=True)
            return None

    def narrate(self, text: str, start_ns: int, is_no_change: bool = False) -> None:
        if self._backend is None or start_ns == self._last_narrated_ns:
            return
        self._last_narrated_ns = start_ns
        if not is_no_change and text.strip():
            self.stop()
            self._backend.say(text.strip())

    def stop(self) -> None:
        if self._backend is not None and hasattr(self._backend, "stop"):
            with contextlib.suppress(Exception):
                self._backend.stop()

    def reset(self) -> None:
        self._last_narrated_ns = -1


class VisualContextMCPBridge:
    """Manages the FastMCP/Uvicorn server lifecycle, threading, and tool bindings."""

    def __init__(self, plugin: VisualContextPlugin) -> None:
        self.plugin = plugin
        self._uvicorn_server: Any = None
        self._thread: threading.Thread | None = None
        self._running_port: int | None = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def running_port(self) -> int | None:
        return self._running_port

    def start(self, port: int) -> None:
        if self.is_running():
            return
        server = self.create_server(self.plugin)
        if server is None:
            return

        host = "127.0.0.1"
        try:
            uvicorn_server = self.build_uvicorn_server(server, host, port)
        except Exception as exc:  # ruff: ignore[blind-except]
            logger.warning("Failed to initialize MCP uvicorn server: %s", exc)
            return
        self._uvicorn_server = uvicicorn = uvicorn_server
        self._running_port = port

        def _serve() -> None:
            try:
                uvicicorn.run()
            except Exception:
                logger.exception("MCP server stopped with error")

        self._thread = threading.Thread(
            target=_serve, daemon=True, name="VisualContextMCP"
        )
        self._thread.start()
        logger.info(
            "VisualContextPlugin MCP server started at http://%s:%s/mcp", host, port
        )

    def stop(self) -> None:
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
            self._uvicorn_server = None
        thread = self._thread
        self._thread = None
        self._running_port = None
        if (
            thread is not None
            and thread.is_alive()
            and threading.current_thread() is not thread
        ):
            thread.join(timeout=2.0)
        logger.info("VisualContextPlugin MCP server stopped")

    @classmethod
    def create_server(cls, plugin: VisualContextPlugin) -> Any:
        try:
            import typing_extensions

            if not hasattr(typing_extensions, "sentinel") and hasattr(
                typing_extensions, "Sentinel"
            ):
                typing_extensions.sentinel = typing_extensions.Sentinel  # type: ignore[attr-defined]
        except ImportError:
            logger.debug("typing_extensions not available", exc_info=True)

        try:
            try:
                from mcp.server.mcpserver import MCPServer  # type: ignore[import-not-found]  # ty: ignore[import-not-found]
            except ImportError:
                from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[import-not-found]  # ty: ignore[import-not-found]
        except ImportError as exc:
            logger.warning("MCP server dependencies not available: %s", exc)
            cls.notify_app(
                plugin,
                "MCP Server",
                "Dependencies not installed. Launch Neon Player to install them.",
            )
            return None

        instructions = (
            "Visual Context MCP provides VLM scene captions and timeline event tools "
            "for the loaded Neon recording. Captions include nanosecond timestamps "
            "(`start_ns`, `end_ns`) and relative seconds (`start_s`, `end_s`). "
            "To analyze eye-tracking metrics (pupil diameter, blinks, fixations, gaze, "
            "eyelid, IMU) over caption intervals, call `get_plugin_status()` for "
            "`recording_path`, then run a Python script with `pupil_labs.neon_recording` "
            "(`uv run --with pupil-labs-neon-recording python ...`): "
            "`rec = nr.load(recording_path)` exposes `rec.pupil` (`.time`, "
            "`.diameter_left`, `.diameter_right`), `rec.blinks` (`.start_time`, "
            "`.stop_time`), `rec.fixations`, `rec.gaze`, `rec.eyeball`, `rec.eyelid`, "
            "and `rec.imu` (all timestamps in nanoseconds)."
        )
        try:
            server = MCPServer("neon-visual-context", instructions=instructions)
        except TypeError:
            server = MCPServer("neon-visual-context")
        cls.register_mcp_tools(server, plugin)
        return server

    @staticmethod
    def build_uvicorn_server(server: Any, host: str, port: int) -> Any:
        if "logging.config" not in sys.modules:
            import importlib

            try:
                importlib.import_module("logging.config")
            except ImportError:
                import types

                log_cfg = types.ModuleType("logging.config")
                log_cfg.__dict__["dictConfig"] = lambda *args, **kwargs: None
                log_cfg.__dict__["fileConfig"] = lambda *args, **kwargs: None
                sys.modules["logging.config"] = log_cfg
                logging.__dict__["config"] = log_cfg

        import uvicorn  # type: ignore[import-not-found]  # ty: ignore[import-not-found]

        if hasattr(server, "streamable_http_app"):
            try:
                app = server.streamable_http_app(streamable_http_path="/mcp")
            except TypeError:
                app = server.streamable_http_app()
        elif hasattr(server, "sse_app"):
            app = server.sse_app()
        else:
            raise RuntimeError("Unsupported MCP server implementation")

        config = uvicorn.Config(
            app, host=host, port=port, log_level="warning", log_config=None
        )
        return uvicorn.Server(config)

    @classmethod
    def register_mcp_tools(cls, server: Any, plugin: VisualContextPlugin) -> None:
        cls.register_captions(server, plugin)
        cls.register_events(server, plugin)
        cls.register_lifecycle(server, plugin)

    @staticmethod
    def register_captions(server: Any, plugin: VisualContextPlugin) -> None:
        @server.tool()
        def get_captions(
            start_s: float | None = None,
            end_s: float | None = None,
            search: str | None = None,
        ) -> list[dict[str, Any]]:
            """Get scene captions with timestamps and gaze info."""

            def _action() -> list[dict[str, Any]]:
                results = []
                for c in plugin._captions:
                    if start_s is not None and c["start_s"] < start_s:
                        continue
                    if end_s is not None and c["end_s"] > end_s:
                        continue
                    if search is not None and search.lower() not in c["text"].lower():
                        continue
                    results.append({
                        "start_ns": c["start_ns"],
                        "end_ns": c["end_ns"],
                        "start_s": round(float(c["start_s"]), 3),
                        "end_s": round(float(c["end_s"]), 3),
                        "text": c["text"],
                        "frame_index": c.get("frame_index"),
                        "event_data": c.get("event_data", {}),
                    })
                return results

            return plugin._run_on_main(_action)

        @server.tool()
        def export_captions_csv(path: str) -> str:
            """Export all loaded captions to a CSV file at the specified path."""

            def _action() -> str:
                if not plugin._captions:
                    return "No captions to export."
                target = Path(path).expanduser()
                plugin._write_captions_csv(target)
                return f"Exported {len(plugin._captions)} captions to {target}"

            return plugin._run_on_main(_action)

    @classmethod
    def register_events(cls, server: Any, plugin: VisualContextPlugin) -> None:
        @server.tool()
        def add_events(events: dict[str, list[int | float | str]]) -> str:
            """Add custom timeline events {event_name: [timestamp_ns_or_s, ...]}."""

            def _action() -> str:
                events_plugin = cls.require_events_plugin()
                cleaned = {
                    str(name): [
                        cls.resolve_event_timestamp_ns(ts, plugin) for ts in timestamps
                    ]
                    for name, timestamps in events.items()
                }
                events_plugin.add_events(cleaned)
                cls.refresh_events_plugin_ui(events_plugin)
                summary = {k: len(v) for k, v in cleaned.items()}
                total = sum(summary.values())
                names = ", ".join(f"'{k}' ({v})" for k, v in summary.items())
                cls.notify_app(
                    plugin,
                    "Events Created",
                    f"Added {total} events: {names}.",
                )
                return f"Successfully added events: {summary}"

            return plugin._run_on_main(_action)

        @server.tool()
        def create_events_from_keyword(keyword: str) -> str:
            """Search captions for keyword and add 'Saw: <keyword>' events."""
            return plugin._run_on_main(lambda: plugin._add_keyword_events(keyword))

    @staticmethod
    def register_lifecycle(server: Any, plugin: VisualContextPlugin) -> None:
        @server.tool()
        def generate_captions(
            prompt: str | None = None, interval_s: float | None = None
        ) -> str:
            """Trigger caption generation in Neon Player for the current recording."""

            def _action() -> str:
                if prompt is not None:
                    plugin.prompt = prompt
                if interval_s is not None:
                    plugin.interval_s = interval_s
                plugin.generate_captions()  # ty: ignore[call-non-callable]
                return "Started caption generation background job."

            return plugin._run_on_main(_action)

        @server.tool()
        def get_plugin_status() -> dict[str, Any]:
            """Get status of VisualContextPlugin, loaded recording path/start time, and settings."""

            def _action() -> dict[str, Any]:
                rec = plugin._safe_recording()
                rec_dir = getattr(rec, "_rec_dir", None)
                rec_name = getattr(rec, "name", None) or (
                    rec_dir.name if rec_dir is not None else None
                )
                rec_path = str(rec_dir) if rec_dir is not None else None
                rec_start_time = getattr(rec, "start_time", None)
                rec_start_ns = (
                    int(rec_start_time) if rec_start_time is not None else None
                )
                job = getattr(plugin, "job", None) or getattr(plugin, "_job", None)
                is_running = (
                    job is not None and getattr(job, "is_running", lambda: False)()
                )
                port = plugin._mcp_settings.port
                return {
                    "recording_loaded": rec is not None,
                    "recording_name": rec_name,
                    "recording_path": rec_path,
                    "recording_start_time_ns": rec_start_ns,
                    "caption_count": len(plugin._captions),
                    "is_captioning_running": is_running,
                    "model": (
                        plugin.model.value
                        if hasattr(plugin.model, "value")
                        else str(plugin.model)
                    ),
                    "prompt": plugin.prompt,
                    "interval_s": plugin.interval_s,
                    "mcp_url": f"http://127.0.0.1:{port}/mcp",
                }

            return plugin._run_on_main(_action)

    @staticmethod
    def refresh_events_plugin_ui(events_plugin: Any) -> None:
        try:
            prop = getattr(type(events_plugin), "event_types", None)
            if hasattr(prop, "binds"):
                for widget in prop.binds.get(events_plugin, []):
                    widget.value = events_plugin.event_types
        except Exception:
            logger.debug("Could not refresh EventsPlugin widget", exc_info=True)

    @staticmethod
    def resolve_event_timestamp_ns(
        raw_ts: float | str, plugin: VisualContextPlugin
    ) -> int:
        rec = plugin._safe_recording()
        rec_start = int(getattr(rec, "start_time", 0) or 0)
        if isinstance(raw_ts, int) and not isinstance(raw_ts, bool):
            ts_ns = (
                rec_start + raw_ts * 1_000_000_000
                if raw_ts < 1_000_000_000_000
                else raw_ts
            )
        else:
            raw_str = str(raw_ts).strip()
            if raw_str.isdigit() and int(raw_str) >= 1_000_000_000_000:
                ts_ns = int(raw_str)
            else:
                val = float(raw_str)
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite event timestamp: {raw_ts!r}")
                ts_ns = (
                    rec_start + round(val * 1e9)
                    if val < 1_000_000_000_000
                    else round(val)
                )

        if plugin._starts_ns:
            idx = bisect.bisect_left(plugin._starts_ns, ts_ns)
            for candidate_idx in (idx - 1, idx):
                if 0 <= candidate_idx < len(plugin._starts_ns):
                    cand = plugin._starts_ns[candidate_idx]
                    if abs(cand - ts_ns) <= 10_000:
                        return int(cand)
        return ts_ns

    @staticmethod
    def get_events_plugin() -> EventsPlugin | None:
        inst = Plugin.get_instance_by_name("EventsPlugin") or EventsPlugin.instance()
        return inst if isinstance(inst, EventsPlugin) else None

    @classmethod
    def require_events_plugin(cls) -> EventsPlugin:
        events_plugin = cls.get_events_plugin()
        if events_plugin is None:
            raise RuntimeError("EventsPlugin is not loaded in Neon Player.")
        return events_plugin

    @staticmethod
    def notify_app(plugin: VisualContextPlugin, title: str, message: str) -> None:
        if getattr(plugin, "app", None) is not None:
            plugin.app.show_notification(title, message)


class VisualContextPlugin(Plugin):
    label = "Visual Context (Scene Captioning)"
    _mcp_dispatch = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.render_layer = 100
        self._captions: list[dict[str, Any]] = []
        self._starts_ns: list[int] = []
        self._show_captions = True
        self._model = CaptionModel.Balanced_4B
        self._settings = CaptioningSettings()
        self._settings.changed.connect(self._on_settings_or_model_changed)

        self._caption_style = CaptionStyle()
        self._caption_style.changed.connect(self.changed.emit)
        self._bg_color = QColor(0, 0, 0, 160)

        self._mcp_settings = MCPSettings()
        self._mcp_settings.changed.connect(self._on_mcp_settings_changed)
        self._mcp_enabled = True
        self._mcp_bridge = VisualContextMCPBridge(self)
        self._mcp_dispatch.connect(
            self._on_mcp_dispatch, Qt.ConnectionType.QueuedConnection
        )

        self._narrate_captions = False
        self._tts = TTSController()

    @Slot(object)
    def _on_mcp_dispatch(self, fn: Any) -> None:
        fn()

    def _safe_recording(self) -> Any:
        try:
            return self.recording
        except RuntimeError:
            return None

    def _safe_headless(self) -> bool:
        try:
            return self.headless
        except RuntimeError:
            return True

    def __setstate__(self, state: dict[str, Any]) -> None:
        ordered = dict(state)
        for junk in (
            "_mcp_uvicorn",
            "_mcp_server_thread",
            "_mcp_running_port",
            "_last_narrated_ns",
        ):
            ordered.pop(junk, None)
        for group_key in ("settings", "caption_style", "mcp_settings"):
            if group_key in ordered:
                ordered[group_key] = ordered.pop(group_key)
        super().__setstate__(ordered)

    def _active_model_repo(self) -> str:
        """Return the repository ID for the currently selected model preset."""
        if isinstance(self._model, CaptionModel):
            return self._model.mlx_repo if _is_apple_silicon() else self._model.hf_repo
        return str(self._model)

    def _current_cache_filename(self) -> str:
        """Return the settings-hashed cache filename for the active configuration."""
        return CaptionStorage.cache_filename_for(
            self._active_model_repo(),
            self.prompt,
            self.interval_s,
            self.skip_similar,
            self.similarity_threshold,
        )

    def _read_cache(self) -> dict[str, Any] | None:
        return CaptionStorage.read_cache(
            load_fn=self.load_cached_json,
            model=self._active_model_repo(),
            prompt=self.prompt,
            interval_s=self.interval_s,
            skip_similar=self.skip_similar,
            similarity_threshold=self.similarity_threshold,
        )

    def _on_settings_or_model_changed(self) -> None:
        rec = self._safe_recording()
        if rec is not None:
            self.on_recording_loaded(rec)
        self.changed.emit()

    @staticmethod
    def _valid_captions(data: dict[str, Any]) -> list[dict[str, Any]] | None:
        return CaptionStorage.validate_captions(data)

    def on_recording_loaded(self, recording: Any) -> None:
        self._captions = []
        self._starts_ns = []

        if self._mcp_enabled and not self._safe_headless():
            self._start_mcp_server()

        data = self._read_cache()
        if data is None:
            return

        if data.get("version") != CACHE_VERSION:
            logger.info("Discarding visual context cache from an older version")
            return

        if data.get("recording_start_time") != getattr(recording, "start_time", None):
            logger.warning("Visual context cache belongs to a different recording")
            return

        if "error" in data:
            return

        captions = self._valid_captions(data)
        if captions is None:
            logger.warning("Visual context cache is malformed; ignoring it")
            return

        self._captions = captions
        self._starts_ns = [c["start_ns"] for c in captions]

    @property
    @collapsible_property_params(
        primary=True,
        widget=cast(Any, CollapsibleGroupWidget),
        label="Display Captions",
        enable_prop="show_captions",
    )
    def caption_style(self) -> CaptionStyle:
        return self._caption_style

    @caption_style.setter
    def caption_style(self, value: CaptionStyle | dict) -> None:
        if isinstance(value, dict):
            self._caption_style.__setstate__(value)
        elif isinstance(value, CaptionStyle) and value is not self._caption_style:
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
        if not self._narrate_captions:
            self._tts.stop()
        self.changed.emit()

    @property
    @property_params(label="VLM Model")
    def model(self) -> CaptionModel:
        return self._model

    @model.setter
    def model(self, v: CaptionModel | str) -> None:
        if isinstance(v, str):
            try:
                v = CaptionModel(v)
            except ValueError:
                v = CaptionModel[v]
        if self._model == v:
            return
        self._model = v
        self._on_settings_or_model_changed()

    @property
    @property_params(
        primary=True,
        widget=cast(Any, CollapsibleGroupWidget),
        label="Settings",
    )
    def settings(self) -> CaptioningSettings:
        return self._settings

    @settings.setter
    def settings(self, value: CaptioningSettings | dict) -> None:
        if isinstance(value, dict):
            self._settings.__setstate__(value)
        elif isinstance(value, CaptioningSettings) and value is not self._settings:
            self._settings = value
            self._settings.changed.connect(self._on_settings_or_model_changed)
        self._on_settings_or_model_changed()

    prompt_template = _proxy_property("_settings", "prompt_template")
    prompt = _proxy_property("_settings", "prompt")
    max_tokens = _proxy_property("_settings", "max_tokens")
    interval_s = _proxy_property("_settings", "interval_s")
    skip_similar = _proxy_property("_settings", "skip_similar")
    similarity_threshold = _proxy_property("_settings", "similarity_threshold")
    font_size = _proxy_property("_caption_style", "font_size")
    text_color = _proxy_property("_caption_style", "text_color")
    highlight_color = _proxy_property("_caption_style", "highlight_color")

    def _get_saw_keywords(self) -> set[str]:
        keywords: set[str] = set()
        events_plugin = _get_events_plugin()
        if events_plugin is not None:
            for et in getattr(events_plugin, "event_types", ()):
                name = getattr(et, "name", "")
                if name.startswith("Saw: ") and name[5:].strip():
                    keywords.add(name[5:].strip())
            for name in getattr(events_plugin, "_events", ()):
                if name.startswith("Saw: ") and name[5:].strip():
                    keywords.add(name[5:].strip())
        return keywords

    @property
    @collapsible_property_params(
        primary=True,
        widget=cast(Any, CollapsibleGroupWidget),
        label="Model Context Protocol (MCP)",
        enable_prop="mcp_enabled",
    )
    def mcp_settings(self) -> MCPSettings:
        return self._mcp_settings

    @mcp_settings.setter
    def mcp_settings(self, value: MCPSettings | dict) -> None:
        if isinstance(value, dict):
            self._mcp_settings.__setstate__(value)
        elif isinstance(value, MCPSettings) and value is not self._mcp_settings:
            self._mcp_settings = value
            self._mcp_settings.changed.connect(self._on_mcp_settings_changed)
        self._on_mcp_settings_changed()

    def _on_mcp_settings_changed(self) -> None:
        if (
            self._mcp_enabled
            and not self._safe_headless()
            and self._mcp_bridge.is_running()
            and self._mcp_bridge.running_port != self._mcp_settings.port
        ):
            self._stop_mcp_server()
            self._start_mcp_server()
        self.changed.emit()

    @property
    @property_params(widget=None)
    def mcp_enabled(self) -> bool:
        return self._mcp_enabled

    @mcp_enabled.setter
    def mcp_enabled(self, v: bool) -> None:
        enabled = bool(v)
        if self._mcp_enabled == enabled:
            return
        self._mcp_enabled = enabled
        if self._mcp_enabled and not self._safe_headless():
            self._start_mcp_server()
        else:
            self._stop_mcp_server()
        self.changed.emit()

    mcp_port = _proxy_property("_mcp_settings", "port")

    def on_disabled(self) -> None:
        self._stop_mcp_server()
        self._tts.stop()

    def edit_prompt(self) -> None:
        """Edit Captioning Prompt (opens dialog)."""
        self._settings.open_prompt_editor()

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("media-playback-start"))
    def generate_captions(self) -> None:
        """Generate Captions (local VLM)"""
        if self._safe_headless():
            if hasattr(self.job_manager, "work_job"):
                self.job_manager.work_job(self.bg_generate_captions())
            else:
                self.job_manager.run_in_foreground(self.bg_generate_captions())
            self._on_captions_ready()
            return

        self.job = self.job_manager.run_background_action(
            "Captioning Scene Video",
            "VisualContextPlugin.bg_generate_captions",
        )
        if self.job is None:
            return
        self.job.finished.connect(self._on_captions_ready)

    def _on_captions_ready(self) -> None:
        cache = self._read_cache() or {}
        error = cache.get("error")
        total_time_s = cache.get("total_time_s")

        rec = self._safe_recording()
        if rec is not None:
            self.on_recording_loaded(rec)
        self.changed.emit()

        if error:
            _notify_app(self, "Captioning Failed", str(error))
        elif self._captions:
            msg = f"Generated {len(self._captions)} captions."
            if total_time_s:
                msg += f" (Took {total_time_s:.1f}s)"
            _notify_app(self, "Visual Context", msg)

    def _add_keyword_events(self, keyword: str) -> str:
        events_plugin = _require_events_plugin()
        kw = keyword.strip()
        timestamps = _keyword_event_timestamps(self._captions, kw)
        if not timestamps:
            _notify_app(self, "No Matches", f"No caption contains '{kw}'.")
            return f"No captions contain '{kw}'."
        event_name = f"Saw: {kw}"
        events_plugin.add_events({event_name: timestamps})
        _refresh_events_plugin_ui(events_plugin)
        _notify_app(
            self,
            "Events Created",
            f"Added {len(timestamps)} events for '{kw}'.",
        )
        return f"Added {len(timestamps)} events for '{kw}' ({event_name})."

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("edit-find"))
    def create_events_from_keyword(self) -> None:
        """Create Events from Keyword"""
        if not self._captions:
            _notify_app(self, "Notice", "Please generate captions first.")
            return

        keyword, ok = QInputDialog.getText(
            None, "Keyword Search", "Enter keyword to search for in the captions:"
        )
        if not ok or not keyword.strip():
            return

        try:
            self._add_keyword_events(keyword)
        except RuntimeError:
            _notify_app(self, "Error", "EventsPlugin not loaded.")

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("preferences-desktop-font"))
    def view_captions(self) -> None:
        """View All Captions"""
        if not self._captions:
            _notify_app(self, "Notice", "Please generate captions first.")
            return

        dialog = QDialog()
        dialog.setWindowTitle("Visual Context")
        dialog.resize(760, 520)

        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFontFamily("Courier New")
        text_edit.setPlainText(
            "\\n".join(
                f"[{CaptionStorage.format_time(c['start_s'])}] {c['text']}"
                for c in self._captions
            )
        )
        layout.addWidget(text_edit)

        btn = QPushButton("Close")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)

        dialog.setLayout(layout)
        dialog.exec()

    def _write_captions_csv(self, target: Path) -> None:
        CaptionStorage.write_csv(self._captions, target)

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("document-export"))
    def export_csv(self) -> None:
        """Export Captions to CSV"""
        if not self._captions:
            _notify_app(self, "Notice", "Please generate captions first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            None, "Export Captions", "visual_context.csv", "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            self._write_captions_csv(Path(path).expanduser())
        except OSError:
            logger.exception("Failed to export captions")
            _notify_app(self, "Error", "Could not write the CSV file.")
            return

        _notify_app(self, "Export Complete", f"Wrote {len(self._captions)} captions.")

    def _run_on_main(self, fn: Any, timeout: float = 15.0) -> Any:
        if QThread.currentThread() == self.thread():
            return fn()
        fut: Future[Any] = Future()

        def _work() -> None:
            try:
                fut.set_result(fn())
            except Exception as exc:  # ruff: ignore[blind-except]
                fut.set_exception(exc)

        self._mcp_dispatch.emit(_work)
        return fut.result(timeout=timeout)

    def _start_mcp_server(self) -> None:
        self._mcp_bridge.start(self._mcp_settings.port)

    def _stop_mcp_server(self) -> None:
        self._mcp_bridge.stop()

    def _check_similarity_match(
        self,
        image: Image.Image,
        position: int,
        reference_thumb: np.ndarray | None,
        reference_text: str | None,
        is_event_gaze_aware: bool,
        previous_scene_state: str,
    ) -> tuple[np.ndarray | None, str | None, bool, str]:
        return VlmCaptioningPipeline.check_similarity_match(
            image,
            position,
            reference_thumb,
            reference_text,
            is_event_gaze_aware,
            previous_scene_state,
            self.skip_similar,
            self.similarity_threshold,
        )

    def _validate_scene_frames(self) -> tuple[Any, np.ndarray, list[int]]:
        try:
            return VlmCaptioningPipeline.validate_scene(self.recording, self.interval_s)
        except RuntimeError as e:
            self._record_failure(str(e))
            raise

    def _record_failure(self, message: str) -> None:
        CaptionStorage.record_failure(
            load_fn=self.load_cached_json,
            save_fn=self.save_cached_json,
            start_time=getattr(self.recording, "start_time", None),
            model=self._active_model_repo(),
            prompt=self.prompt,
            interval_s=self.interval_s,
            skip_similar=self.skip_similar,
            similarity_threshold=self.similarity_threshold,
            message=message,
        )

    def _finalize_captions(
        self,
        captions: list[dict[str, Any]],
        times_ns: np.ndarray,
        total: int,
        t_start: float,
        reused: int,
        captioner: _Captioner,
    ) -> None:
        if not any(c["text"] for c in captions):
            message = (
                f"The model produced no usable caption for any of the {total} "
                "sampled frames. See the log for details."
            )
            self._record_failure(message)
            raise RuntimeError(message)

        rec = self._safe_recording()
        start_time = getattr(rec, "start_time", 0) if rec is not None else 0
        VlmCaptioningPipeline.assign_end_times(captions, times_ns, start_time)
        total_time_s = time.time() - t_start
        cache_filename = CaptionStorage.save_cache(
            save_fn=self.save_cached_json,
            start_time=start_time,
            model=captioner.repo,
            backend=captioner.backend,
            prompt=self.prompt,
            interval_s=self.interval_s,
            skip_similar=self.skip_similar,
            similarity_threshold=self.similarity_threshold,
            captions=captions,
            total_time_s=total_time_s,
        )
        logger.info(
            "Captioned %d scene frames (%d reused from near-identical frames) -> %s",
            len(captions),
            reused,
            cache_filename,
        )

    def _apply_job_overrides(
        self, model: str, prompt_template: str, interval_s: float
    ) -> None:
        if model:
            self.model = model
        if prompt_template:
            self.prompt_template = prompt_template
        if interval_s > 0.0:
            self.interval_s = interval_s

    def _is_event_gaze_aware_prompt(self) -> bool:
        return (
            self.prompt_template == PromptTemplate.Event_Gaze_Aware
            or "temporal memory" in self.prompt.lower()
            or "event-oriented" in self.prompt.lower()
            or "EVENT_GAZE_SYSTEM_PROMPT" in self.prompt
        )

    def bg_generate_captions(
        self,
        model: str = "",
        prompt_template: str = "",
        interval_s: float = 0.0,
    ) -> Generator[ProgressUpdate, None, None]:
        self._apply_job_overrides(model, prompt_template, interval_s)
        yield from VlmCaptioningPipeline.run(
            recording=self.recording,
            model=self._model,
            prompt=self.prompt,
            prompt_template=self.prompt_template,
            interval_s=self.interval_s,
            skip_similar=self.skip_similar,
            similarity_threshold=self.similarity_threshold,
            max_tokens=self.max_tokens,
            save_cache_fn=self._finalize_captions,
            record_failure_fn=self._record_failure,
        )

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
        rec = self._safe_recording()
        if not self._show_captions or not self._captions or not rec:
            return

        caption = self._caption_at(time_in_recording)
        if caption is None:
            self._tts.reset()
            return

        text = caption["text"].strip()
        if not text:
            return

        if self._narrate_captions:
            self._tts.narrate(
                text, caption["start_ns"], caption.get("is_no_change", False)
            )

        scene_width = rec.scene.width
        scene_height = rec.scene.height
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

            text, bounding_rect = CaptionRenderer.fit_text(text, fm, band, flags)
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

            matching_keywords = [
                kw
                for kw in self._get_saw_keywords()
                if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE)
            ]
            CaptionRenderer.render_text(
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


def _keyword_event_timestamps(
    captions: list[dict[str, Any]], keyword: str
) -> list[int]:
    """Start timestamps of each contiguous run of captions matching `keyword`."""
    needle = keyword.strip()
    if not needle:
        return []
    pattern = re.compile(rf"\b{re.escape(needle)}\b", re.IGNORECASE)

    timestamps: list[int] = []
    previous_matched = False
    for caption in captions:
        text = caption.get("text") or ""
        if not text:
            continue
        matched = bool(pattern.search(text))
        if matched and not previous_matched:
            timestamps.append(int(caption["start_ns"]))
        previous_matched = matched
    return timestamps


# Module-level backward-compatibility aliases
_settings_cache_key = CaptionStorage.settings_cache_key
_cache_filename_for = CaptionStorage.cache_filename_for
_format_time = CaptionStorage.format_time
_csv_safe = CaptionStorage.csv_safe
_clean_caption = TemporalResponseParser.clean_caption
_clean_event_str = TemporalResponseParser.clean_event_str
_is_no_change = TemporalResponseParser.is_no_change
_bound_scene_state = TemporalResponseParser.bound_scene_state
_extract_json_block = TemporalResponseParser.extract_json_block
_parse_json_event_payload = TemporalResponseParser.parse_json_event_payload
_regex_extract_update = TemporalResponseParser.regex_extract_update
_parse_temporal_response = TemporalResponseParser.parse
_sample_indices = VlmCaptioningPipeline.sample_indices
_assign_end_times = VlmCaptioningPipeline.assign_end_times
_frame_to_pil = VlmCaptioningPipeline.frame_to_pil
_thumbnail = VlmCaptioningPipeline.thumbnail
_thumbnail_difference = VlmCaptioningPipeline.thumbnail_difference
_extract_gaze_timeseries = VlmCaptioningPipeline.extract_gaze_timeseries
_lookup_gaze_point = VlmCaptioningPipeline.lookup_gaze_point
_caption_single_frame = VlmCaptioningPipeline.caption_single_frame
_fit_text = CaptionRenderer.fit_text
_render_caption_text = CaptionRenderer.render_text
_init_tts = TTSController.init_backend
_create_mcp_server = VisualContextMCPBridge.create_server
_build_uvicorn_server = VisualContextMCPBridge.build_uvicorn_server
_register_mcp_tools = VisualContextMCPBridge.register_mcp_tools
_reg_captions = VisualContextMCPBridge.register_captions
_reg_events = VisualContextMCPBridge.register_events
_reg_lifecycle = VisualContextMCPBridge.register_lifecycle
_refresh_events_plugin_ui = VisualContextMCPBridge.refresh_events_plugin_ui
_resolve_event_timestamp_ns = VisualContextMCPBridge.resolve_event_timestamp_ns
_get_events_plugin = VisualContextMCPBridge.get_events_plugin
_require_events_plugin = VisualContextMCPBridge.require_events_plugin
_notify_app = VisualContextMCPBridge.notify_app
