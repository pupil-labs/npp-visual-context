# Visual Context Plugin

Automatic scene-camera captioning for Neon Player, running a state-of-the-art
vision-language model **entirely on your own machine**. No cloud, no API key,
no upload.

The plugin samples frames from the scene video, asks a local VLM what is
visible in each one, and stores the answers as timestamped captions. Those
captions are drawn over the video during playback, can be exported to CSV, and
— like the [Audio Transcription plugin](https://github.com/pupil-labs/npp-transcribe)
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

On a MacBook M1 with 16 GB, **Fast (2B)** and **Balanced (4B)** are the
practical choices. Expect on the order of one to a few seconds of inference
per captioned frame — a 10-minute recording at the default 2-second interval
is ~300 frames, so plan for a coffee break. *Skip Near-Identical Frames*
(on by default) usually cuts this substantially on real recordings, where the
scene often holds still.

## Models

Three presets, all [Qwen3-VL Instruct](https://huggingface.co/Qwen) checkpoints.
Note the large difference in footprint between the two backends — the Apple
Silicon path is 4-bit quantized, the transformers path is not:

| Preset | Apple Silicon (MLX, 4-bit) | Elsewhere (transformers, bf16) |
|---|---|---|
| Fast (2B) | `mlx-community/Qwen3-VL-2B-Instruct-4bit` — ~1.8 GB | `Qwen/Qwen3-VL-2B-Instruct` — ~4.3 GB |
| Balanced (4B) — default | `mlx-community/Qwen3-VL-4B-Instruct-4bit` — ~3 GB | `Qwen/Qwen3-VL-4B-Instruct` — ~8 GB |
| Quality (8B) | `mlx-community/Qwen3-VL-8B-Instruct-4bit` — ~5.8 GB | `Qwen/Qwen3-VL-8B-Instruct` — ~17.5 GB |

Budget a similar amount of free RAM (or VRAM) as disk. **On the transformers
path, Quality (8B) needs roughly 17.5 GB of VRAM** — pick a smaller preset
unless you have it.

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
   them with *Show Captions* and adjust the font size and text colour in the
   properties panel.
3. **Create Events From Keyword** — enter a word or phrase (for example
   `laptop`) and the plugin adds a `Saw: laptop` event at the start of every
   run of captions mentioning it. A subject looking at a laptop for a minute
   produces one event, not thirty. These behave like any other Neon Player
   event and are included in exports. Matching is plain case-insensitive
   substring matching, so `car` also matches "cardboard" — use a longer phrase
   if that matters.
4. **View Captions** — opens the full timestamped caption list.
5. **Export Csv** — writes start/end timestamps (both absolute nanoseconds and
   seconds relative to recording start), the scene frame index, and the caption
   text.

## Settings

| Setting | Default | Notes |
|---|---|---|
| Show Captions | on | Overlay captions during playback. |
| VLM Model | Balanced (4B) | See the table above. |
| Seconds Between Captions | 2.0 | Lower is more detailed and much slower. |
| Max Tokens per Caption | 64 | Upper bound on caption length. |
| Captioning Prompt | *(see below)* | The instruction sent to the model for every frame. |
| Skip Near-Identical Frames | on | Reuses the previous caption when the scene has barely changed, instead of re-running the model. |
| Frame Difference Threshold | 6.0 | Mean absolute difference (0–255 grey levels) above which a frame is re-captioned. Raise it to skip more aggressively; set it to 0 to caption every sampled frame. |
| Caption Font Size | 26 | |
| Caption Text Color | white | |

The prompt can be edited inline or, for longer text, through the **Edit Prompt**
action which opens a multi-line dialog. It is saved with the rest of the
plugin's settings. Narrowing it is often worthwhile — for example
*"Name only the object the wearer is holding."*

## How it works

- **Frame sampling** — frames are chosen by binary-searching the scene video's
  own timestamp array rather than assuming a constant frame rate, so dropped
  frames don't cause drift. Only the sampled frames are decoded.
- **Redundancy skipping** — each sampled frame is reduced to a 64×64 greyscale
  thumbnail and compared against the last frame that was actually captioned
  (not merely the last one sampled), so slow scene drift still eventually
  triggers a fresh caption.
- **Timestamps** — captions carry absolute nanosecond timestamps taken straight
  from the scene video's time array, on the same clock as `recording.start_time`,
  so events land exactly on the frame they describe.
- **Caching** — results are stored per recording under
  `.neon_player/cache/VisualContextPlugin/`. The cache records the recording's
  start time and a format version, and is ignored if either doesn't match or if
  the contents don't validate.

## Caveats

Captions are generated by a small quantized language model and **will
occasionally be wrong or hallucinated**. Treat them as a search and
navigation aid over a recording, not as ground-truth annotation. Anything
used in analysis should be spot-checked against the video.
