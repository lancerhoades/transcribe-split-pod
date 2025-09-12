import shutil
import os
import json
import subprocess
import requests
import orjson
from typing import Dict, Any, List, Optional

import runpod
from rapidfuzz import fuzz

# --- GPU diag ---
def _log_gpu_info():
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True)
        print("[GPU] nvidia-smi available.\n" + out)
    except Exception as e:
        print(f"[GPU] nvidia-smi not available: {e}")



# ----------- Config / Defaults -----------
DEFAULT_LANGUAGE = os.getenv("TRANSCRIBE_LANG", "en")
DEFAULT_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")  # base|small|medium|large-v3, etc.
DEFAULT_SPLIT_PHRASE = os.getenv("SPLIT_PHRASE_DEFAULT", "sermon")
DEFAULT_ANNOUNCEMENTS_PHRASE = os.getenv("ANNOUNCEMENTS_PHRASE_DEFAULT", "")

STORAGE_ROOT = "/runpod-volume"  # RunPod NAS mount

# ----------- Utilities -----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def storage_path(job_id: str, *parts: str) -> str:
    return os.path.join(STORAGE_ROOT, job_id, *parts)

def ffprobe_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", path
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    return float(data["format"]["duration"])

def ffmpeg_cut(input_path: str, out_path: str, ss: float, to: Optional[float] = None) -> None:
    # try fast stream copy first
    args = ["ffmpeg", "-y", "-ss", f"{ss}"]
    if to is not None:
        duration = max(0.0, to - ss)
        args += ["-t", f"{duration}"]
    args += ["-i", input_path, "-c", "copy", out_path]
    try:
        subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return
    except subprocess.CalledProcessError:
        pass
    # fallback to re-encode for precision
    args = ["ffmpeg", "-y", "-ss", f"{ss}"]
    if to is not None:
        duration = max(0.0, to - ss)
        args += ["-t", f"{duration}"]
    args += ["-i", input_path, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-c:a", "aac", "-b:a", "192k", out_path]
    subprocess.run(args, check=True)

def ts_to_hhmmss(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = t - (h*3600 + m*60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def download_to(path: str, url: str):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        ensure_dir(os.path.dirname(path))
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)

# ----------- Transcription -----------
def do_transcribe(input_path: str, language: str) -> Dict[str, Any]:
    # import here so the module loads only when needed
    from faster_whisper import WhisperModel

    # pick device automatically
    device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"  # good defaults
    model_size = DEFAULT_MODEL_SIZE
    print(f"[WHISPER] device={device} compute_type={compute_type} CUDA_VISIBLE_DEVICES={os.getenv("CUDA_VISIBLE_DEVICES")}", flush=True)

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments_it, _info = model.transcribe(
        input_path,
        language=language or DEFAULT_LANGUAGE,
        beam_size=5,
        vad_filter=True
    )

    segments: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []
    for seg in segments_it:
        item = {
            "id": getattr(seg, "id", len(segments)),
            "start": float(seg.start) if seg.start is not None else None,
            "end": float(seg.end) if seg.end is not None else None,
            "text": (seg.text or "").strip()
        }
        segments.append(item)
        if item["text"]:
            full_text_parts.append(item["text"])

    return {"segments": segments, "text": " ".join(full_text_parts).strip()}

def write_transcripts(job_id: str, data: Dict[str, Any]) -> Dict[str, str]:
    out_dir = storage_path(job_id, "transcripts")
    ensure_dir(out_dir)

    txt_path = os.path.join(out_dir, "transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(data["text"] + "\n")

    json_path = os.path.join(out_dir, "transcript.json")
    with open(json_path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    ts_path = os.path.join(out_dir, "timestamped.txt")
    with open(ts_path, "w", encoding="utf-8") as f:
        for seg in data["segments"]:
            s = seg.get("start"); e = seg.get("end"); t = seg.get("text", "")
            if s is None or e is None:
                continue
            f.write(f"{ts_to_hhmmss(s)} --> {ts_to_hhmmss(e)} | {t}\n")

    return {"text": txt_path, "json": json_path, "timestamped": ts_path}

# ----------- Phrase Location -----------
def normalize(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()

def find_phrase_time(segments: List[Dict[str, Any]], phrase: str, min_ratio: float = 85.0) -> Optional[float]:
    if not phrase:
        return None
    target = normalize(phrase)
    if not target:
        return None
    for seg in segments:
        text_norm = normalize(seg.get("text", ""))
        if not text_norm:
            continue
        ratio = fuzz.token_set_ratio(text_norm, target)
        if ratio >= min_ratio:
            st = seg.get("start")
            if st is not None:
                return float(st)
    return None

# ----------- Main Handler -----------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    _log_gpu_info()
    """
    Input:
      {
        "job_id": "20250911-120000-abc123-video42",
        "local_video_path": "/runpod-volume/{job_id}/raw/full.mp4",  # optional
        "input_url": "https://....mp4",                        # optional
        "language": "en",                                      # optional
        "split_phrase": "today we begin our sermon",           # optional
        "announcements_phrase": "announcements"                # optional
      }
    """
    inp = (event or {}).get("input") or {}

    job_id = inp.get("job_id")
    if not job_id:
        return {"ok": False, "error": "job_id is required"}

    local_video_path = inp.get("local_video_path") or storage_path(job_id, "raw", "full.mp4")
    input_url = inp.get("input_url")
    language = inp.get("language") or DEFAULT_LANGUAGE
    split_phrase = inp.get("split_phrase") or DEFAULT_SPLIT_PHRASE
    announcements_phrase = inp.get("announcements_phrase") or DEFAULT_ANNOUNCEMENTS_PHRASE

    # Ensure dirs
    ensure_dir(storage_path(job_id, "raw"))
    ensure_dir(storage_path(job_id, "transcripts"))
    ensure_dir(storage_path(job_id, "splits"))

    # Download if input_url provided
    if input_url:
        download_to(local_video_path, input_url)

    if not os.path.exists(local_video_path):
        return {"ok": False, "error": f"Input video not found: {local_video_path}"}

    # 1) Transcribe
    tr_data = do_transcribe(local_video_path, language=language)
    tr_paths = write_transcripts(job_id, tr_data)

    # 2) Split points
    segments = tr_data["segments"]
    sermon_start = find_phrase_time(segments, split_phrase) or None
    ann_start = find_phrase_time(segments, announcements_phrase) if announcements_phrase else None
    duration = ffprobe_duration_seconds(local_video_path)

    # 3) Split video files
    splits_dir = storage_path(job_id, "splits")
    created = []
    warnings = []

    worship_out = os.path.join(splits_dir, "worship.mp4")
    sermon_out = os.path.join(splits_dir, "sermon.mp4")
    announcements_out = os.path.join(splits_dir, "announcements.mp4")

    try:
        if sermon_start is not None and sermon_start > 1.0:
            # worship: [0, sermon_start]
            ffmpeg_cut(local_video_path, worship_out, ss=0.0, to=sermon_start)
            created.append("worship.mp4")

            if ann_start is not None and (ann_start > sermon_start + 1.0) and (ann_start < duration - 1.0):
                # sermon: [sermon_start, ann_start]
                ffmpeg_cut(local_video_path, sermon_out, ss=sermon_start, to=ann_start)
                created.append("sermon.mp4")
                # announcements: [ann_start, end]
                ffmpeg_cut(local_video_path, announcements_out, ss=ann_start, to=None)
                created.append("announcements.mp4")
            else:
                # sermon: [sermon_start, end]
                ffmpeg_cut(local_video_path, sermon_out, ss=sermon_start, to=None)
                created.append("sermon.mp4")
        else:
            warnings.append("split_phrase not found with sufficient confidence; exporting full as 'sermon.mp4'.")
            ffmpeg_cut(local_video_path, sermon_out, ss=0.0, to=None)
            created.append("sermon.mp4")
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": f"ffmpeg split error: {e}"}

    return {
        "ok": True,
        "paths": {
            "input": local_video_path,
            "transcripts": tr_paths,
            "splits_dir": splits_dir
        },
        "times": {
            "sermon_start": sermon_start,
            "announcements_start": ann_start,
            "duration": duration
        },
        "created": created,
        "warnings": warnings
    }

# RunPod glue
runpod.serverless.start({"handler": handler})

# Log GPU info at import
_log_gpu_info()

