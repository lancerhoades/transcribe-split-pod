import os
import json
import subprocess
import requests
import orjson
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import runpod
from rapidfuzz import fuzz
from openai import OpenAI

# ---------- Config / Defaults ----------
DEFAULT_LANGUAGE = os.getenv("TRANSCRIBE_LANG", "en")
DEFAULT_SPLIT_PHRASE = os.getenv("SPLIT_PHRASE_DEFAULT", "sermon")
DEFAULT_ANNOUNCEMENTS_PHRASE = os.getenv("ANNOUNCEMENTS_PHRASE_DEFAULT", "")
SPLIT_PHRASES = [p.strip() for p in os.getenv("SPLIT_PHRASES", "").split("|") if p.strip()]

# OpenAI
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "whisper-1")
OPENAI_CHUNK_SEC = int(os.getenv("OPENAI_CHUNK_SEC", "600"))       # 10 min chunks
OPENAI_MAX_PARALLEL = int(os.getenv("OPENAI_MAX_PARALLEL", "1"))   # 1 = sequential (safe)
AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "64k")                  # keeps uploads small

# Storage (RunPod Serverless network volume lives here)
STORAGE_ROOT = "/runpod-volume"

# ---------- Utilities ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def storage_path(job_id: str, *parts: str) -> str:
    return os.path.join(STORAGE_ROOT, job_id, *parts)

def ffprobe_duration_seconds(path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    return float(data["format"]["duration"])

def ffmpeg_cut(input_path: str, out_path: str, ss: float, to: Optional[float] = None) -> None:
    # try fast stream-copy
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
    # fallback re-encode for precision
    args = ["ffmpeg", "-y", "-ss", f"{ss}"]
    if to is not None:
        duration = max(0.0, to - ss)
        args += ["-t", f"{duration}"]
    args += [
        "-i", input_path,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", out_path
    ]
    subprocess.run(args, check=True)

def ts_to_hhmmss(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - (h * 3600 + m * 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def download_to(path: str, url: str):
    with requests.get(url, stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        ensure_dir(os.path.dirname(path))
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)

# ---------- OpenAI transcription helpers ----------
client = OpenAI()

def extract_audio_m4a(input_mp4: str, out_m4a: str):
    ensure_dir(os.path.dirname(out_m4a))
    cmd = [
        "ffmpeg", "-y", "-i", input_mp4,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE, out_m4a
    ]
    subprocess.run(cmd, check=True)

def _ffprobe_duration(path: str) -> float:
    try:
        return ffprobe_duration_seconds(path)
    except Exception:
        return 0.0

def split_audio_into_chunks(in_m4a: str, work_dir: str, sec: int) -> List[Dict[str, Any]]:
    ensure_dir(work_dir)
    pattern = os.path.join(work_dir, "chunk_%03d.m4a")
    cmd = [
        "ffmpeg", "-y", "-i", in_m4a,
        "-f", "segment", "-segment_time", str(sec),
        "-reset_timestamps", "1",
        "-c", "copy", pattern
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # collect chunk files and assign offsets by cumulative duration
    files = sorted([os.path.join(work_dir, f) for f in os.listdir(work_dir) if f.startswith("chunk_") and f.endswith(".m4a")])
    chunks = []
    offset = 0.0
    for fp in files:
        dur = _ffprobe_duration(fp)
        chunks.append({"path": fp, "offset": offset, "duration": dur})
        offset += dur
    return chunks

def transcribe_chunk_openai(file_path: str, offset: float, language: str) -> Dict[str, Any]:
    with open(file_path, "rb") as f:
        tr = client.audio.transcriptions.create(
            model=OPENAI_MODEL,
            file=f,
            response_format="verbose_json",
            language=language,
            timestamp_granularities=["segment", "word"]
        )
    text = (getattr(tr, "text", None) or tr.get("text") or "").strip()
    raw_segments = getattr(tr, "segments", None) or tr.get("segments") or []
    segs = []
    if raw_segments:
        for s in raw_segments:
            s_start = float(s.get("start", 0.0)) + offset
            s_end = float(s.get("end", 0.0)) + offset
            s_text = (s.get("text") or "").strip()
            segs.append({"id": 0, "start": s_start, "end": s_end, "text": s_text})
    else:
        segs.append({"id": 0, "start": offset, "end": offset, "text": text})
    return {"offset": offset, "text": text, "segments": segs}

def do_transcribe_openai(input_path: str, language: str) -> Dict[str, Any]:
    # 1) extract to m4a (small)
    work = os.path.join(os.path.dirname(input_path), "_openai_audio")
    m4a = os.path.join(work, "audio.m4a")
    extract_audio_m4a(input_path, m4a)

    # 2) chunk
    chunks = split_audio_into_chunks(m4a, os.path.join(work, "chunks"), OPENAI_CHUNK_SEC)
    if not chunks:
        # fall back to transcribing the whole thing
        chunks = [{"path": m4a, "offset": 0.0, "duration": _ffprobe_duration(m4a)}]

    # 3) transcribe (optional parallel)
    results = []
    if OPENAI_MAX_PARALLEL > 1 and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=OPENAI_MAX_PARALLEL) as pool:
            futs = [pool.submit(transcribe_chunk_openai, c["path"], c["offset"], language) for c in chunks]
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for c in chunks:
            results.append(transcribe_chunk_openai(c["path"], c["offset"], language))

    # 4) stitch
    results.sort(key=lambda r: r["offset"])
    segments: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []
    for r in results:
        if r["text"]:
            full_text_parts.append(r["text"])
        segments.extend(r["segments"])

    return {"segments": segments, "text": " ".join(full_text_parts).strip()}

# ---------- Phrase Location ----------
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

def find_first_of_phrases(segments: List[Dict[str, Any]], phrases: List[str], min_ratio: float = 85.0) -> Optional[float]:
    best = None
    for p in phrases:
        t = find_phrase_time(segments, p, min_ratio=min_ratio)
        if t is not None and (best is None or t < best):
            best = t
    return best

# ---------- Transcript writers ----------
def write_transcripts(job_id: str, data: Dict[str, Any]) -> Dict[str, str]:
    out_dir = storage_path(job_id, "transcripts")
    ensure_dir(out_dir)

    txt_path = os.path.join(out_dir, "transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write((data.get("text") or "") + "\n")

    json_path = os.path.join(out_dir, "transcript.json")
    with open(json_path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    ts_path = os.path.join(out_dir, "timestamped.txt")
    with open(ts_path, "w", encoding="utf-8") as f:
        for seg in data.get("segments") or []:
            s = seg.get("start"); e = seg.get("end"); t = seg.get("text", "")
            if s is None or e is None:
                continue
            f.write(f"{ts_to_hhmmss(s)} --> {ts_to_hhmmss(e)} | {t}\n")

    return {"text": txt_path, "json": json_path, "timestamped": ts_path}

# ---------- Main Handler ----------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
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
    ensure_dir(storage_path(job_id, "clips"))

    # Download if input_url provided
    if input_url:
        download_to(local_video_path, input_url)

    if not os.path.exists(local_video_path):
        return {"ok": False, "error": f"Input video not found: {local_video_path}"}

    # 1) Transcribe via OpenAI
    tr_data = do_transcribe_openai(local_video_path, language=language)
    tr_paths = write_transcripts(job_id, tr_data)

    # 2) Split points
    segments = tr_data["segments"]
    if SPLIT_PHRASES:
        sermon_start = find_first_of_phrases(segments, SPLIT_PHRASES) or None
    else:
        sermon_start = find_phrase_time(segments, split_phrase) or None
    ann_start = find_phrase_time(segments, announcements_phrase) if announcements_phrase else None
    duration = ffprobe_duration_seconds(local_video_path)

    # 2b) sections.json for compatibility
    clips_dir = storage_path(job_id, "clips")
    sections = {
        "worship": {
            "start": 0.0,
            "end": float(sermon_start) if sermon_start is not None else float(duration),
            "file": "worship.mp4"
        },
        "sermon": {
            "start": float(sermon_start) if sermon_start is not None else 0.0,
            "end": float(ann_start) if ann_start is not None else float(duration),
            "file": "sermon.mp4"
        }
    }
    if ann_start is not None:
        sections["announcements"] = {"start": float(ann_start), "end": float(duration), "file": "announcements.mp4"}

    with open(os.path.join(clips_dir, "sections.json"), "w", encoding="utf-8") as jf:
        json.dump(sections, jf, indent=2)

    # 3) Split video files
    splits_dir = storage_path(job_id, "splits")
    created = []
    warnings = []

    worship_out = os.path.join(splits_dir, "worship.mp4")
    sermon_out = os.path.join(splits_dir, "sermon.mp4")
    announcements_out = os.path.join(splits_dir, "announcements.mp4")

    try:
        if sermon_start is not None and sermon_start > 1.0:
            ffmpeg_cut(local_video_path, worship_out, ss=0.0, to=sermon_start)
            created.append("worship.mp4")

            if ann_start is not None and (ann_start > sermon_start + 1.0) and (ann_start < duration - 1.0):
                ffmpeg_cut(local_video_path, sermon_out, ss=sermon_start, to=ann_start)
                created.append("sermon.mp4")
                ffmpeg_cut(local_video_path, announcements_out, ss=ann_start, to=None)
                created.append("announcements.mp4")
            else:
                ffmpeg_cut(local_video_path, sermon_out, ss=sermon_start, to=None)
                created.append("sermon.mp4")
        else:
            warnings.append("split phrase(s) not found confidently; exporting full as 'sermon.mp4'.")
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
