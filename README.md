
<h1 align="center">Frugan</h1>

<p align="center">
  Turns video/audio into transcripts, separates speakers by diarization and uses local LLM to summarize the transcript. Also has option prompting.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#use-cases">Use Cases</a> •
  <a href="#security--privacy">Security & Privacy</a> •
  <a href="#outputs">Outputs</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#troubleshooting">Troubleshooting</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img alt="Platform" src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-brightgreen" />
  <img alt="CLI" src="https://img.shields.io/badge/CLI-Yes-black" />
  <img alt="FFmpeg" src="https://img.shields.io/badge/FFmpeg-Required-orange" />
  <img alt="LM Studio" src="https://img.shields.io/badge/LM%20Studio-Optional-purple" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## What is Frugan?

Frugan is a production-oriented CLI that:
- Ingests local media (can also download directly from YT)
- extracts audio with FFmpeg
- transcribes with WhisperX (word-level timestamps)
- optionally diarizes speakers 
- exports standard artifacts teams can actually use: **SRT + TXT**, plus optional **summary.txt** and optional **lm_answer.txt** via LM Studio

It’s built for workflows where transcripts become data: compliance reviews, customer calls, incident retros, clinical ops interviews, usability sessions, research recordings, internal meetings, and more.

---

## Features

- **YouTube → transcript in one command** (via `yt-dlp`)
- **Local file support** (`.mp4`, `.mkv`, `.mp3`, `.wav`, etc.)
- **Timestamped SRT + readable TXT**
- **Optional speaker diarization** (HF token required)
- **Optional local LLM outputs** (LM Studio): `summary.txt` and/or `lm_answer.txt`
- **Clean output folder per input** (safe to archive, share internally, or attach to tickets)
- **Quiet mode** for noisy environments and CI logs

---

## Outputs

Frugan creates a folder per input under `outputs/`:

    outputs/<title_or_filename>/
    ├── transcript.srt
    ├── transcript.txt
    ├── transcript.speakers.srt        (if diarization enabled)
    ├── summary.txt                    (if LM Studio + --summary)
    ├── lm_answer.txt                  (if LM Studio + --ask)
    └── lm_prompt.txt                  (optional, if --save-prompt)

====================

## Quick Start

### 1) Prerequisites

Required:
- **Python 3.10+**
- **FFmpeg** (must be accessible)

FFmpeg options:
- Option A (recommended): place FFmpeg here:
  - `tools/ffmpeg.exe` (Windows)
  - `tools/ffmpeg` (macOS/Linux)
- Option B: install system-wide and ensure `ffmpeg` is on PATH

Optional:
- **CUDA + NVIDIA GPU** for faster transcription (auto-falls back to CPU)
- **Hugging Face token** for diarization (pyannote pipeline)
- **LM Studio** for local LLM summary/Q&A (OpenAI-compatible server)

---

### 2) Install (recommended: venv)

    git clone https://github.com/<YOUR_USERNAME>/frugan.git
    cd frugan

    python -m venv .venv

    # Windows:
    .venv\Scripts\activate

    # macOS/Linux:
    source .venv/bin/activate

    pip install -U pip
    pip install -r requirements.txt

---

### 3) Run

Local file:

    python frugan.py -i "path/to/video.mp4"

YouTube:

    python frugan.py -i "https://www.youtube.com/watch?v=VIDEO_ID"

---

## Use Cases

### Compliance / Risk / QA 
- Transcribe customer support calls or internal reviews
- Search transcripts for required disclosures
- Produce SRT for audit-friendly playback with timestamps

### HealthOps / Clinical Operations
- Transcribe interviews, site visits, investigator meetings
- Summarize long recordings into structured reports
- Keep processing local (no cloud upload required)

### Product & Research
- Turn usability sessions into timestamped notes
- Generate subtitles for internal demos
- Create quick summaries for stakeholders

---

## LM Studio (optional)

Frugan can generate:
- `summary.txt` — structured report (hook, evidence, outcomes, open questions)
- `lm_answer.txt` — answer a custom prompt against the transcript

Start LM Studio server:
- Run LM Studio
- Enable the OpenAI-compatible API server
- Default URL: `http://localhost:1234`

Use LM Studio with Frugan:

    python frugan.py -i "video.mp4" --lmstudio --summary

Custom prompt:

    python frugan.py -i "video.mp4" --lmstudio --ask --prompt "Who is speaking most and what are they claiming?"

Save the prompt too:

    python frugan.py -i "video.mp4" --lmstudio --ask --prompt "List the key allegations with timestamps." --save-prompt

====================

## Speaker diarization (optional)

Diarization requires a Hugging Face token.

1) Create a `.env` file:

    HF_TOKEN=your_token_here

2) Run with diarization:

    python frugan.py -i "video.mp4" --diarize

Optional tuning:

    python frugan.py -i "video.mp4" --diarize --min-speakers 2 --max-speakers 6

---

## Configuration

Common flags:

    python frugan.py -i INPUT ^
      --lang mix ^
      --vad silero ^
      --device cuda ^
      --model large-v2 ^
      --merge-gap 0.6 ^
      --quiet

LM Studio:

    python frugan.py -i INPUT --lmstudio --summary --lm-max-tokens 15000 --lm-timeout 900

---

## Security & Privacy

Frugan is designed to run locally:
- Media stays on your machine
- Transcripts stay on your machine
- Optional LM features can stay local via LM Studio

For regulated environments:
- Prefer local inputs (avoid downloading sensitive links)
- Store outputs on encrypted drives
- Keep `.env` out of version control (see `.gitignore`)

---

## Troubleshooting

ffmpeg not found:
- Put FFmpeg at `./tools/ffmpeg.exe` (Windows) or `./tools/ffmpeg` (macOS/Linux), or install FFmpeg and add it to PATH.

HF token missing:
- Add `.env` with `HF_TOKEN=...` and ensure `.env` is gitignored.

LM Studio context errors:
- Your model’s context window may be smaller than the transcript being sent.
- Use a larger-context model in LM Studio, or summarize shorter clips.

---

## Roadmap / Ideas

- N/A
---

## License

MIT — see `LICENSE`.

<p align="center">
  <sub>Built for real transcripts, real workflows, and real teams.</sub>
</p>
