#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()


# -----------------------------
# Quiet mode (reduce clutter)
# -----------------------------
def setup_quiet_mode(quiet: bool) -> None:
    if not quiet:
        return

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*torchcodec.*")
    warnings.filterwarnings("ignore", message=".*symlinks.*huggingface_hub.*")
    warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
    warnings.filterwarnings("ignore", message=".*triton not found.*")

    logging.getLogger().setLevel(logging.ERROR)

    for name in [
        "whisperx",
        "pyannote",
        "pytorch_lightning",
        "lightning",
        "torch",
        "huggingface_hub",
        "transformers",
        "faster_whisper",
        "yt_dlp",
        "urllib3",
        "requests",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")


@contextlib.contextmanager
def suppress_stderr(enabled: bool):
    """Hard-silence noisy libs that dump to stderr (torchcodec/pyannote)."""
    if not enabled:
        yield
        return
    devnull = open(os.devnull, "w")
    old_stderr = sys.stderr
    try:
        sys.stderr = devnull
        yield
    finally:
        sys.stderr = old_stderr
        devnull.close()


# -----------------------------
# Env / tokens (GitHub-safe)
# -----------------------------
def load_env() -> None:
    load_dotenv(dotenv_path=Path(".env"), override=False)


def get_hf_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


# -----------------------------
# ffmpeg
# -----------------------------
def is_windows() -> bool:
    return os.name == "nt"


def find_ffmpeg(user_ffmpeg: Optional[str]) -> str:
    if user_ffmpeg:
        return user_ffmpeg
    local = (
        Path(__file__).parent / "tools" / ("ffmpeg.exe" if is_windows() else "ffmpeg")
    )
    if local.exists():
        return str(local)
    on_path = shutil.which("ffmpeg")
    return on_path or "ffmpeg"


def check_ffmpeg(ffmpeg_bin: str) -> None:
    try:
        subprocess.run(
            [ffmpeg_bin, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        raise RuntimeError(
            "❌ ffmpeg not found.\n"
            "Fix one of these:\n"
            "  1) Put ffmpeg at ./tools/ffmpeg.exe (recommended)\n"
            "  2) Install ffmpeg system-wide (PATH)\n"
            "  3) Pass --ffmpeg C:\\path\\to\\ffmpeg.exe\n"
        )


def run_cmd(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{p.stderr}")


def extract_audio_to_wav(
    ffmpeg_bin: str, input_path: Path, wav_path: Path, sr: int = 16000
) -> None:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    run_cmd(cmd)


# -----------------------------
# YouTube download (yt-dlp) with Rich progress
# -----------------------------
def looks_like_url(s: str) -> bool:
    s = s.strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def download_youtube_with_progress(
    url: str, out_dir: Path, ffmpeg_bin: str, progress: Progress
) -> Path:
    try:
        import yt_dlp
    except Exception:
        raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "%(title).200s [%(id)s].%(ext)s")
    task_id = progress.add_task("⬇️ Downloading (yt-dlp)", total=100)

    def hook(d: Dict[str, Any]) -> None:
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes")
            if total and downloaded is not None and total > 0:
                pct = (downloaded / total) * 100.0
                progress.update(task_id, completed=max(0.0, min(100.0, pct)))
            spd = d.get("speed")
            if spd:
                progress.update(
                    task_id,
                    description=f"⬇️ Downloading (yt-dlp) • {spd / 1e6:.1f} MB/s",
                )
        elif status == "finished":
            progress.update(
                task_id,
                completed=100,
                description="✅ Downloaded (yt-dlp) • merging/converting",
            )

    ydl_opts = {
        "outtmpl": template,
        "format": "bv*+ba/best",
        "merge_output_format": "mp4",
        "ffmpeg_location": ffmpeg_bin,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if isinstance(info, dict):
            rd = info.get("requested_downloads")
            if rd and isinstance(rd, list) and rd[0].get("filepath"):
                return Path(rd[0]["filepath"]).resolve()
            fn = info.get("_filename")
            if fn:
                return Path(fn).resolve()

    raise RuntimeError(
        "yt-dlp download completed but output path could not be determined."
    )


# -----------------------------
# CLI prompts
# -----------------------------
def ask(prompt: str, default: Optional[str] = None) -> str:
    prompt = f"{prompt} [{default}]: " if default is not None else f"{prompt}: "
    s = input(prompt).strip().strip('"').strip("'")
    return s if s else (default or "")


def ask_yes_no(prompt: str, default_yes: bool) -> bool:
    default = "y" if default_yes else "n"
    while True:
        s = ask(f"{prompt} (y/n)", default=default).lower()
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False


def ask_lang_mode(existing: Optional[str]) -> str:
    if existing in ("auto", "tr", "en", "mix"):
        return existing
    console.print("🌍 Language mode?")
    console.print("  tr   = Turkish only")
    console.print("  en   = English only")
    console.print("  mix  = mixed TR/EN (auto-detect)")
    console.print("  auto = auto-detect")
    while True:
        s = ask("Choose", default="mix").lower()
        if s in ("auto", "tr", "en", "mix"):
            return s


def resolve_language(lang_mode: str) -> Optional[str]:
    if lang_mode == "tr":
        return "tr"
    if lang_mode == "en":
        return "en"
    return None


# -----------------------------
# Segment merging
# -----------------------------
def merge_consecutive_segments(
    segments: List[Dict[str, Any]], max_gap_s: float = 0.6
) -> List[Dict[str, Any]]:
    if not segments:
        return segments
    merged: List[Dict[str, Any]] = []
    cur = dict(segments[0])

    def norm_spk(s: Any) -> str:
        return str(s) if s is not None else ""

    for seg in segments[1:]:
        spk_cur = norm_spk(cur.get("speaker"))
        spk_next = norm_spk(seg.get("speaker"))
        gap = float(seg.get("start", 0.0)) - float(cur.get("end", 0.0))

        if spk_cur == spk_next and gap <= max_gap_s:
            cur["end"] = seg.get("end", cur.get("end"))
            t1 = (cur.get("text") or "").rstrip()
            t2 = (seg.get("text") or "").lstrip()
            if t1 and not t1.endswith((" ", "\n")):
                t1 += " "
            cur["text"] = (t1 + t2).strip()
        else:
            merged.append(cur)
            cur = dict(seg)

    merged.append(cur)
    return merged


# -----------------------------
# SRT + TXT
# -----------------------------
def srt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000.0))
    ms = ms_total % 1000
    s_total = ms_total // 1000
    s = s_total % 60
    m_total = s_total // 60
    m = m_total % 60
    h = m_total // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(
    segments: List[Dict[str, Any]], out_path: Path, speaker_prefix: bool
) -> None:
    lines: List[str] = []
    idx = 1
    speaker_map: Dict[str, str] = {}
    speaker_counter = 1

    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 0.2))
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        spk = seg.get("speaker")
        if speaker_prefix and spk:
            if spk not in speaker_map:
                speaker_map[spk] = f"Speaker {speaker_counter}"
                speaker_counter += 1
            text_out = f"{speaker_map[spk]}: {text}"
        else:
            text_out = text

        lines.append(str(idx))
        lines.append(f"{srt_ts(start)} --> {srt_ts(end)}")
        lines.append(text_out)
        lines.append("")
        idx += 1

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_transcript_txt(
    segments: List[Dict[str, Any]],
    out_path: Path,
    include_timestamps: bool = True,
    include_speakers: bool = True,
) -> None:
    speaker_map: Dict[str, str] = {}
    speaker_counter = 1
    out_lines: List[str] = []

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 0.2))
        ts = f"[{srt_ts(start)}–{srt_ts(end)}] " if include_timestamps else ""

        prefix = ""
        if include_speakers and seg.get("speaker"):
            spk = str(seg["speaker"])
            if spk not in speaker_map:
                speaker_map[spk] = f"Speaker {speaker_counter}"
                speaker_counter += 1
            prefix = f"{speaker_map[spk]}: "

        out_lines.append(f"{ts}{prefix}{text}")

    out_path.write_text("\n".join(out_lines), encoding="utf-8")


# -----------------------------
# Output folder naming
# -----------------------------
def sanitize_name(name: str, max_len: int = 120) -> str:
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name or "output"


def output_dir_for_input(input_path: Path) -> Path:
    folder = sanitize_name(input_path.stem)
    return Path("outputs") / folder


# -----------------------------
# LM Studio (JSON final only)
# -----------------------------
def lmstudio_is_up(base_url: str, timeout_s: float = 1.5) -> bool:
    try:
        r = requests.get(base_url.rstrip("/") + "/v1/models", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def lmstudio_pick_model(base_url: str, preferred: str) -> str:
    try:
        r = requests.get(base_url.rstrip("/") + "/v1/models", timeout=3.0)
        r.raise_for_status()
        data = r.json()
        ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
        if not ids:
            return preferred
        return preferred if preferred in ids else ids[0]
    except Exception:
        return preferred


def lmstudio_chat_raw(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 15000,
    timeout_s: float = 900.0,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def build_summary_prompt_json(transcript_text: str) -> Tuple[str, str]:
    system = (
        "Return ONLY valid JSON. No markdown, no code fences.\n"
        'JSON keys: "final".\n'
        '"final" must be the final report exactly in the requested format.\n'
    )
    user = f"""
Write a COMPLETE structured report for the transcript below.

Return ONLY JSON.

The final report MUST include ALL 6 sections with these exact headings:
1) Goal / Thesis / Hook
2) Main Arguments
3) Evidence / Examples
4) Conclusion / Result
5) Open Questions / Plot Twists
6) Practical Lessons

Rules for the FINAL report:
- Each section must have at least 2 bullet points.
- If something is missing, write: "Not stated in transcript."
- If transcript contains multiple distinct stories, separate them inside the bullets (Case A/B/C).
- Be faithful to the transcript. Do not invent facts.

Transcript:
{transcript_text}
""".strip()
    return system, user


def build_custom_prompt_json(
    transcript_text: str, custom_prompt: str
) -> Tuple[str, str]:
    system = (
        "Return ONLY valid JSON. No markdown, no code fences.\n"
        'JSON keys: "final".\n'
        '"final" must be the final answer only.\n'
        "Do NOT include chain-of-thought, steps, or analysis.\n"
        "If relevant, cite timestamps like [00:12:34].\n"
    )
    user = f"""
Custom prompt:
{custom_prompt}

Transcript:
{transcript_text}
""".strip()
    return system, user


def build_transcript_for_ai(
    segments: List[Dict[str, Any]], max_chars: int = 180_000
) -> str:
    lines: List[str] = []
    total = 0
    for s in segments:
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        start = srt_ts(float(s.get("start", 0.0)))
        spk = s.get("speaker")
        prefix = f"{spk}: " if spk else ""
        line = f"[{start}] {prefix}{txt}"
        if total + len(line) + 1 > max_chars:
            lines.append("\n[... transcript truncated for length ...]")
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="✨ Frugan: YouTube/local → outputs/<title>/ with SRT + diarization + optional LM Studio summary + optional custom prompt"
    )
    parser.add_argument("-i", "--input", help="Input file path OR YouTube URL")
    parser.add_argument(
        "--ffmpeg", help="Path to ffmpeg (default: ./tools/ffmpeg.exe or PATH)"
    )

    parser.add_argument(
        "--model", default=None, help="Whisper model (tiny/base/small/medium/large-v2)"
    )
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--batch-size", type=int, default=None)

    parser.add_argument(
        "--vad",
        default="silero",
        choices=["silero", "pyannote"],
        help="Default: silero",
    )
    parser.add_argument("--lang", default=None, help="auto|tr|en|mix")

    parser.add_argument("--diarize", action="store_true", help="Enable diarization")
    parser.add_argument(
        "--hf-token", default=None, help="HF token (prefer .env or env var)"
    )
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)

    parser.add_argument(
        "--merge-gap", type=float, default=0.6, help="Merge gap seconds (default 0.6)"
    )
    parser.add_argument("--quiet", action="store_true", help="Hide noisy warnings/logs")
    parser.add_argument(
        "--non-interactive", action="store_true", help="No prompts, use defaults/flags"
    )

    # LM Studio
    parser.add_argument(
        "--lmstudio", action="store_true", help="Enable LM Studio outputs"
    )
    parser.add_argument(
        "--lmstudio-url", default="http://localhost:1234", help="LM Studio base URL"
    )
    parser.add_argument(
        "--lmstudio-model",
        default="local-model",
        help="LM Studio model name (auto-picks if invalid)",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Generate summary.txt via LM Studio"
    )

    # Optional custom prompt -> lm_answer.txt (+ optional lm_prompt.txt)
    parser.add_argument(
        "--ask", action="store_true", help="Run a custom prompt and save lm_answer.txt"
    )
    parser.add_argument(
        "--prompt", default=None, help="Custom prompt text (used if --ask)."
    )
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="Also save lm_prompt.txt (the prompt you asked).",
    )

    parser.add_argument(
        "--lm-max-tokens", type=int, default=15000, help="LM Studio max_tokens."
    )
    parser.add_argument(
        "--lm-timeout", type=float, default=900.0, help="LM Studio timeout seconds."
    )

    args = parser.parse_args()

    setup_quiet_mode(args.quiet)
    load_env()

    ffmpeg_bin = find_ffmpeg(args.ffmpeg)
    try:
        check_ffmpeg(ffmpeg_bin)
    except Exception as ex:
        console.print(str(ex))
        return 2

    if not args.input and not args.non_interactive:
        console.print("📥 Input source?")
        console.print("  1) Local file path")
        console.print("  2) YouTube URL")
        choice = ask("Choose 1 or 2", default="1").strip()
        args.input = (
            ask("🔗 Paste YouTube URL")
            if choice == "2"
            else ask("🎬 Enter video/audio file path")
        )

    if not args.input:
        console.print("❌ No input provided.")
        return 2

    lang_mode = args.lang
    if not args.non_interactive:
        lang_mode = ask_lang_mode(lang_mode)
        if not args.diarize:
            args.diarize = ask_yes_no("🧑‍🤝‍🧑 Do you want speaker diarization?", False)

        if not args.lmstudio:
            args.lmstudio = ask_yes_no(
                "🧠 Use LM Studio outputs (summary/custom prompt)?", False
            )

        if args.lmstudio:
            if not args.summary:
                args.summary = ask_yes_no("📝 Generate summary.txt?", True)
            if not args.ask:
                args.ask = ask_yes_no(
                    "❓ Run a custom prompt and save lm_answer.txt?", False
                )
            if args.ask and not args.prompt:
                args.prompt = ask("✍️ Enter custom prompt")
            if args.ask and not args.save_prompt:
                args.save_prompt = ask_yes_no("💾 Also save lm_prompt.txt?", False)
    else:
        lang_mode = (lang_mode or "mix").lower()

    device = args.device or "cuda"
    compute_type = args.compute_type or ("float16" if device == "cuda" else "int8")
    batch_size = args.batch_size or (16 if device == "cuda" else 8)
    model_name = args.model or "large-v2"
    language = resolve_language(lang_mode)

    try:
        import torch
        import whisperx
    except Exception as ex:
        console.print(
            "❌ Missing deps. Install:\n  pip install whisperx rich python-dotenv yt-dlp requests"
        )
        console.print(str(ex))
        return 2

    if device == "cuda" and not torch.cuda.is_available():
        console.print("⚠️ CUDA requested but unavailable. Falling back to CPU.")
        device = "cpu"
        if compute_type.lower() in ("float16", "fp16"):
            compute_type = "int8"
        batch_size = min(batch_size, 8)

    hf_token = None
    if args.diarize:
        hf_token = get_hf_token(args.hf_token)
        if not hf_token:
            console.print(
                "❌ Diarization requires HF token. Put HF_TOKEN=... in .env (gitignored)."
            )
            return 2

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    console.print(
        Panel.fit(
            "🚀 Frugan",
            subtitle="YouTube/local → outputs/<title>/",
            style="bold cyan",
        )
    )

    input_is_url = looks_like_url(args.input)
    downloaded_path: Optional[Path] = None

    with progress:
        # Resolve input path
        if input_is_url:
            downloaded_path = download_youtube_with_progress(
                args.input,
                out_dir=Path("outputs") / "_downloads_tmp",
                ffmpeg_bin=ffmpeg_bin,
                progress=progress,
            )
            in_path = downloaded_path
            console.print(f"✅ Downloaded: [bold]{in_path}[/bold]")
        else:
            in_path = Path(args.input).expanduser().resolve()
            if not in_path.exists():
                console.print(f"❌ File not found: {in_path}")
                return 2

        # Output folder
        out_dir = output_dir_for_input(in_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Move downloaded video into output folder
        if downloaded_path:
            target_video = out_dir / in_path.name
            if not target_video.exists():
                t = progress.add_task("📦 Organizing output folder", total=100)
                shutil.move(str(in_path), str(target_video))
                progress.update(t, completed=100)
            video_path = target_video
        else:
            video_path = in_path

        # Output files
        srt_path = out_dir / "transcript.srt"
        srt_speakers_path = out_dir / "transcript.speakers.srt"
        txt_path = out_dir / "transcript.txt"
        summary_path = out_dir / "summary.txt"
        answer_path = out_dir / "lm_answer.txt"
        prompt_path = out_dir / "lm_prompt.txt"

        # Temp audio
        tmp_dir = Path(tempfile.mkdtemp(prefix="frugan_"))
        wav_path = tmp_dir / "audio.wav"

        detected = None
        base_segments: List[Dict[str, Any]] = []
        spk_segments: Optional[List[Dict[str, Any]]] = None

        try:
            t = progress.add_task("🎧 Extracting audio (ffmpeg)", total=100)
            extract_audio_to_wav(ffmpeg_bin, video_path, wav_path)
            progress.update(t, completed=100)

            t = progress.add_task("🧠 Loading ASR model", total=100)
            with suppress_stderr(args.quiet):
                model = whisperx.load_model(
                    model_name,
                    device,
                    compute_type=compute_type,
                    vad_method=args.vad,
                )
            progress.update(t, completed=100)

            t = progress.add_task("✍️ Transcribing", total=100)
            audio = whisperx.load_audio(str(wav_path))
            result = model.transcribe(audio, batch_size=batch_size, language=language)
            progress.update(t, completed=100)

            detected = result.get("language")
            console.print(f"🗣️ Language detected/used: [bold]{detected}[/bold]")

            t = progress.add_task("📍 Aligning timestamps", total=100)
            with suppress_stderr(args.quiet):
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected, device=device
                )
                aligned = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False,
                )
            progress.update(t, completed=100)

            base_segments = merge_consecutive_segments(
                aligned.get("segments", []), max_gap_s=args.merge_gap
            )

            t = progress.add_task("📄 Writing transcript outputs", total=100)
            write_srt(base_segments, srt_path, speaker_prefix=False)
            write_transcript_txt(
                base_segments, txt_path, include_timestamps=True, include_speakers=True
            )
            progress.update(t, completed=100)

            if args.diarize:
                t = progress.add_task("🧑‍🤝‍🧑 Diarizing speakers", total=100)
                from whisperx.diarize import DiarizationPipeline

                with suppress_stderr(args.quiet):
                    diar = DiarizationPipeline(token=hf_token, device=device)

                diar_kwargs: Dict[str, Any] = {}
                if args.min_speakers is not None:
                    diar_kwargs["min_speakers"] = args.min_speakers
                if args.max_speakers is not None:
                    diar_kwargs["max_speakers"] = args.max_speakers

                with suppress_stderr(args.quiet):
                    diar_segs = (
                        diar(audio, **diar_kwargs) if diar_kwargs else diar(audio)
                    )
                    with_spk = whisperx.assign_word_speakers(diar_segs, aligned)

                progress.update(t, completed=100)

                spk_segments = merge_consecutive_segments(
                    with_spk.get("segments", []), max_gap_s=args.merge_gap
                )

                t = progress.add_task("🏷️ Writing speaker SRT", total=100)
                write_srt(spk_segments, srt_speakers_path, speaker_prefix=True)
                progress.update(t, completed=100)

            # LM Studio outputs
            if args.lmstudio and (args.summary or args.ask):
                if not lmstudio_is_up(args.lmstudio_url):
                    console.print(
                        "⚠️ LM Studio server not reachable. Skipping LM outputs."
                    )
                else:
                    args.lmstudio_model = lmstudio_pick_model(
                        args.lmstudio_url, args.lmstudio_model
                    )
                    console.print(
                        f"🧠 LM Studio model: [bold]{args.lmstudio_model}[/bold]"
                    )

                    use_segments = spk_segments if spk_segments else base_segments
                    transcript_for_ai = build_transcript_for_ai(use_segments)

                    if args.summary:
                        t = progress.add_task(
                            "📝 LM Studio: generating summary.txt", total=100
                        )
                        raw = ""
                        try:
                            sys_p, usr_p = build_summary_prompt_json(transcript_for_ai)
                            raw = lmstudio_chat_raw(
                                base_url=args.lmstudio_url,
                                model=args.lmstudio_model,
                                system_prompt=sys_p,
                                user_prompt=usr_p,
                                temperature=0.2,
                                max_tokens=args.lm_max_tokens,
                                timeout_s=args.lm_timeout,
                            )
                            obj = try_parse_json(raw)
                            if obj and isinstance(obj, dict):
                                final = str(obj.get("final") or "").strip()
                                summary_path.write_text(
                                    final if final else raw.strip(), encoding="utf-8"
                                )
                            else:
                                summary_path.write_text(raw.strip(), encoding="utf-8")
                        except Exception as ex:
                            summary_path.write_text(
                                f"LM Studio summary failed:\n{type(ex).__name__}: {ex}\n\nRaw:\n{raw}",
                                encoding="utf-8",
                            )
                        progress.update(t, completed=100)

                    if args.ask and (args.prompt or "").strip():
                        if args.save_prompt:
                            prompt_path.write_text(
                                args.prompt.strip(), encoding="utf-8"
                            )

                        t = progress.add_task(
                            "❓ LM Studio: running custom prompt", total=100
                        )
                        raw = ""
                        try:
                            sys_p, usr_p = build_custom_prompt_json(
                                transcript_for_ai, args.prompt.strip()
                            )
                            raw = lmstudio_chat_raw(
                                base_url=args.lmstudio_url,
                                model=args.lmstudio_model,
                                system_prompt=sys_p,
                                user_prompt=usr_p,
                                temperature=0.2,
                                max_tokens=min(args.lm_max_tokens, 15000),
                                timeout_s=max(120.0, min(args.lm_timeout, 900.0)),
                            )
                            obj = try_parse_json(raw)
                            if obj and isinstance(obj, dict):
                                final = str(obj.get("final") or "").strip()
                                answer_path.write_text(
                                    final if final else raw.strip(), encoding="utf-8"
                                )
                            else:
                                answer_path.write_text(raw.strip(), encoding="utf-8")
                        except Exception as ex:
                            answer_path.write_text(
                                f"LM Studio custom prompt failed:\n{type(ex).__name__}: {ex}\n\nRaw:\n{raw}",
                                encoding="utf-8",
                            )
                        progress.update(t, completed=100)

        finally:
            try:
                for p in tmp_dir.glob("*"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
                tmp_dir.rmdir()
            except Exception:
                pass

        # Clean temp download folder if empty
        tmp_dl = Path("outputs") / "_downloads_tmp"
        try:
            if tmp_dl.exists() and tmp_dl.is_dir() and not any(tmp_dl.iterdir()):
                tmp_dl.rmdir()
        except Exception:
            pass

    console.print(f"✅ Output folder: [bold]{out_dir}[/bold]")
    console.print(f"✅ transcript.srt: {srt_path}")
    if args.diarize:
        console.print(f"✅ transcript.speakers.srt: {srt_speakers_path}")
    console.print(f"✅ transcript.txt: {txt_path}")
    if args.lmstudio and args.summary:
        console.print(f"✅ summary.txt: {summary_path}")
    if args.lmstudio and args.ask and (args.prompt or "").strip():
        console.print(f"✅ lm_answer.txt: {answer_path}")
        if args.save_prompt:
            console.print(f"✅ lm_prompt.txt: {prompt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
