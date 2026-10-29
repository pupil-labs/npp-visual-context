# Visual Context Plugin

Automatic scene-camera captioning for Neon Player, running a state-of-the-art
vision-language model **entirely on your own machine**. No cloud, no API key,
no upload.

The plugin samples frames from the scene video, projects gaze coordinates onto
each frame, asks a local VLM what is visible and being done, and stores the
answers as timestamped captions. Those captions are drawn over the video during
playback, can be narrated via text-to-speech, exported to CSV, queried via an
embedded MCP server, and — like the
[Audio Transcription plugin](https://github.com/pupil-labs/npp-transcribe)
does for spoken words — can be turned into Neon Player **events** whenever a
keyword appears.

## Requirements

| | |
|---|---|
| **Apple Silicon (M1 and newer)** | Runs on the GPU through [MLX](https://github.com/ml-explore/mlx) via [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm), using 4-bit quantized weights. This is the recommended setup. |
| **Everything else** | Falls back to HuggingFace `transformers` with unquantized weights; needs a CUDA GPU with enough VRAM (see the table below) to be usable at a reasonable speed. |

Dependencies are declared inline (PEP 723) and installed by Neon Player when
the plugin is added. Model weights are downloaded from HuggingFace on first
use and cached by `huggingface_hub`.

On a MacBook with 16 GB unified memory, **Liquid_2B**, **Fast_2B**, and
**Balanced_4B** are the practical choices. Expect sub-second to a few seconds
of inference per captioned frame — *Skip Near-Identical Frames* (on by default)
usually cuts total processing time substantially on real recordings where the
scene holds still.

## Models

Four presets are available, supporting both [LiquidAI LFM3-VL](https://huggingface.co/LiquidAI/LFM3-VL-2B)
and [Qwen3-VL Instruct](https://huggingface.co/Qwen) checkpoints. Note the
difference in footprint between the two backends — the Apple Silicon path uses
4-bit quantized MLX weights, whereas the transformers path uses unquantized
weights:

| Preset | Apple Silicon (MLX, 4-bit) | Elsewhere (transformers, bf16) |
|---|---|---|
| `Liquid_2B` | `LiquidAI/LFM3-VL-2B-MLX-4bit` — ~1.5 GB | `LiquidAI/LFM3-VL-2B` — ~4.0 GB |
| `Fast_2B` | `mlx-community/Qwen3-VL-2B-Instruct-4bit` — ~1.8 GB | `Qwen/Qwen3-VL-2B-Instruct` — ~4.3 GB |
| `Balanced_4B` *(default)* | `mlx-community/Qwen3-VL-4B-Instruct-4bit` — ~3.0 GB | `Qwen/Qwen3-VL-4B-Instruct` — ~8.0 GB |
| `Quality_8B` | `mlx-community/Qwen3-VL-8B-Instruct-4bit` — ~5.8 GB | `Qwen/Qwen3-VL-8B-Instruct` — ~17.5 GB |

Budget a similar amount of free RAM (or VRAM) as disk. **On the transformers
path, `Quality_8B` needs roughly 17.5 GB of VRAM** — pick a smaller preset
unless you have it.

## Prompt Templates & Telegraphic Captioning

Captions are optimized for **telegraphic, action-first descriptions** (starting
directly with an active verb or object, such as *"Holding phone with map next
to parked bicycle"* or *"Pushing bicycle up brick stairs"*). The pipeline:

- **Projects gaze coordinates** from the recording's gaze stream into the
  downscaled VLM image frame (`(gaze_x, gaze_y)` in `width×height` pixels).
- **Strips third-person filler prefixes** (e.g., *"The wearer is..."*, *"In this
  image..."*, *"A person is seen..."*) and capitalizes the leading verb/noun so
  overlays and event logs stay scannable.
- **Applies repetition penalties (`1.15`) and bounded temporal context** to
  prevent repetitive loops on small 2B/4B quantized models and caps MLX Metal
  cache memory during long runs.

Available **Prompt Templates**:

| Template | Description |
|---|---|
| `Detailed Scene` *(default)* | Comprehensive description of visible objects, text, and environmental details. |
| `Default` | Telegraphic, action-first phrase (<15 words) describing what is visible and being done in the scene. |
| `Gaze & Action` | Telegraphic, action-first phrase (<15 words) incorporating the exact pixel gaze coordinates `({gaze_x}, {gaze_y})`. |
| `Event & Gaze Aware (Temporal)` | Structured JSON scene understanding with temporal memory (injects gaze coordinates, bounded prior scene state, and recent events; emits `NO_CHANGE` when the scene and action are unchanged). |
| `Gazed Object (One Word)` | Names the main object at the gaze point `({gaze_x}, {gaze_y})` in a single word. |
| `Delta Changes` | Compares against `{prev_desc}` and describes only what changed in the new frame. |
| `Custom` | User-defined prompt supporting `{gaze_x}`, `{gaze_y}`, `{width}`, `{height}`, and `{prev_desc}` placeholders. |

## Installation

For instructions on how to install and manage Neon Player plugins, please refer
to the [Neon Player Plugin Documentation](https://docs.pupil-labs.com/neon/neon-player/plugin-api/#adding-a-plugin).

## Usage

1. **Generate Captions** — captions the scene video. The model is downloaded on
   first run, so the first run takes noticeably longer. Progress is shown as a
   percentage in the status bar. Results are cached per recording, so you only
   pay this cost once. If the run fails, a notification says why and any
   previously generated captions are left untouched.
2. **Playback** — captions are drawn over the scene video automatically. Toggle
   them with *Show Captions*, enable *Narrate Captions (TTS)* for spoken
   readout, and adjust the font size and colours in the properties panel.
3. **Create Events From Keyword** — enter a word or phrase (for example
   `laptop`) and the plugin adds a `Saw: laptop` event at the start of every
   run of captions mentioning it. A subject looking at a laptop for a minute
   produces one event, not thirty. These behave like any other Neon Player
   event and are included in exports.
4. **View Captions** — opens the full timestamped caption list with search and
   double-click seeking.
5. **Export Csv** — writes start/end timestamps (both absolute nanoseconds and
   seconds relative to recording start), the scene frame index, and the caption
   text.

## Embedded MCP Server

The plugin includes a built-in [Model Context Protocol (MCP)](https://modelcontextprotocol.io)
server that lets external AI assistants (Claude Code, Cursor, etc.) query
captions, inspect plugin state, and create timeline events directly inside a
live Neon Player session.

In the **MCP Settings** collapsible section of the plugin sidebar:
- Toggle the header checkbox (**MCP Settings** / `mcp_enabled`) to start or stop
  the background HTTP server.
- Default endpoint: `http://127.0.0.1:8765/mcp` (port configurable via **Port**).
- Click **Copy Config** to copy a ready-to-use `.mcp.json` snippet to the
  clipboard.

### Exposed MCP Tools

| Tool | Description |
|---|---|
| `get_captions(start_s, end_s, search)` | Query loaded scene captions filtered by time range (`start_s`, `end_s`) and/or case-insensitive `search` substring, including frame index and structured `event_data`. |
| `add_events(events)` | Add custom timeline events (`{event_name: [timestamp_ns, ...]}`) directly to Neon Player's `EventsPlugin`. |
| `create_events_from_keyword(keyword)` | Search captions for `keyword` and create `Saw: <keyword>` events at the start of each matching run. |
| `generate_captions(prompt, interval_s)` | Trigger a background caption generation job in Neon Player with optional `prompt` and `interval_s` overrides. |
| `export_captions_csv(path)` | Export all loaded captions to a CSV file at `path`. |
| `get_plugin_status()` | Return current plugin status, loaded recording name, caption count, active VLM model, prompt, sampling interval, and MCP endpoint URL. |

## Headless CLI Generation

Neon Player supports running plugin jobs headlessly from the command line via `--job`, with a live `tqdm` progress bar and no GUI window:

```bash
# Run using the recording's saved VisualContextPlugin settings
neon-player "/path/to/recording" --job VisualContextPlugin.bg_generate_captions

# Or pass optional positional overrides: [model] [prompt_template] [interval_s]
neon-player "/path/to/recording" \
  --job VisualContextPlugin.bg_generate_captions Liquid_2B Event_Gaze_Aware 0.33

# Or supply a custom recording settings JSON file
neon-player "/path/to/recording" \
  --recording-settings "/path/to/settings.json" \
  --job VisualContextPlugin.bg_generate_captions
```

### Headless Job Arguments (`VisualContextPlugin.bg_generate_captions`)

1. `model` *(optional)* — Model preset name (`Liquid_2B`, `Fast_2B`, `Balanced_4B`, `Quality_8B`) or repository ID.
2. `prompt_template` *(optional)* — Template enum name or label (`Default`, `Gaze_Action`, `Event_Gaze_Aware`, `Gazed_Object`, `Detailed_Scene`, `Delta_Changes`, `Custom`).
3. `interval_s` *(optional)* — Seconds between sampled frames (e.g. `0.33` for ~3 Hz dense captioning, `2.0` default).

## Settings

| Setting | Default | Notes |
|---|---|---|
| Show Captions | on | Overlay captions during playback. |
| Narrate Captions (TTS) | off | Speak new captions aloud during playback. |
| VLM Model | Balanced (4B) | `Liquid_2B`, `Fast_2B`, `Balanced_4B`, or `Quality_8B`. |
| Prompt Template | Detailed Scene | Preset prompt strategy or `Custom`. |
| Seconds Between Captions | 2.0 | Lower is more detailed and slower (down to `0.1`s). |
| Max Tokens per Caption | 64 | Upper bound on caption length (automatically raised to 350 for structured JSON temporal mode). |
| Skip Near-Identical Frames | on | Reuses the previous caption when the scene has barely changed, instead of re-running the model. |
| Frame Difference Threshold | 6.0 | Mean absolute difference (0–255 grey levels) above which a frame is re-captioned. |
| Font Size | 26 | Overlay caption font size. |
| Text / Highlight / Background Color | white / yellow / dark grey | Appearance of the playback overlay. |
| MCP Settings | on (port 8765) | Embedded HTTP MCP server toggle, port, and config copy action. |

## How it works

- **Frame sampling** — frames are chosen by binary-searching the scene video's
  own timestamp array rather than assuming a constant frame rate, so dropped
  frames don't cause drift. Only the sampled frames are decoded.
- **Gaze coordinate projection** — the nearest gaze sample in `recording.gaze`
  is matched by timestamp and scaled to the resized VLM image dimensions.
- **Redundancy skipping** — each sampled frame is reduced to a 64×64 greyscale
  thumbnail and compared against the last frame that was actually captioned
  (not merely the last one sampled), so slow scene drift still eventually
  triggers a fresh caption.
- **Timestamps** — captions carry absolute nanosecond timestamps taken straight
  from the scene video's time array, on the same clock as `recording.start_time`,
  so events land exactly on the frame they describe.
- **Multi-configuration caching** — results are stored per recording under
  `.neon_player/cache/VisualContextPlugin/visual_context_<model_tag>_<hash>.json`,
  where `<hash>` is a 10-character SHA-256 digest of the model, prompt, sampling
  interval, and similarity-skip settings (alongside `visual_context.json` for
  backwards compatibility). Switching the **VLM Model** or **Settings** in the UI
  automatically loads any previously generated captions for that configuration
  without overwriting earlier runs.

## Caveats

Captions are generated by a small quantized language model and **will
occasionally be wrong or hallucinated**. Treat them as a search and
navigation aid over a recording, not as ground-truth annotation. Anything
used in analysis should be spot-checked against the video.
