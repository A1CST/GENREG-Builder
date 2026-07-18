"""Video editor backend — library + ffmpeg job runner for the /video page.

A plain utility (nothing evolved): manages a library of video files under
runs/video/library, probes them with ffprobe, renders cached thumbnails, and
runs cut / stitch / convert / export operations as background ffmpeg jobs with
live progress (parsed from `-progress pipe:1`).

ffmpeg/ffprobe resolution order:
  1. tools/ffmpeg-*/bin in this repo (full gyan.dev build, has ffprobe)
  2. anything on PATH
  3. imageio-ffmpeg's bundled binary (ffmpeg only, no ffprobe)
"""

import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid

BASE = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(BASE, "runs", "video", "library")
THUMB_DIR = os.path.join(BASE, "runs", "video", "thumbs")
os.makedirs(LIB_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".wmv", ".flv",
              ".mpg", ".mpeg", ".ts", ".gif"}
# voiceover / sfx tracks for the animation studio live in the same library
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
# formats the browser's <video> tag can actually play back
BROWSER_PLAYABLE = {".mp4", ".webm", ".m4v", ".mp3", ".wav", ".m4a", ".ogg"}


def _find_binaries():
    """Locate ffmpeg + ffprobe (repo tools dir > PATH > imageio bundle)."""
    hits = sorted(glob.glob(os.path.join(BASE, "tools", "ffmpeg-*", "bin", "ffmpeg.exe")),
                  reverse=True)
    if hits:
        ffmpeg = hits[0]
        ffprobe = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe")
        return ffmpeg, (ffprobe if os.path.isfile(ffprobe) else None)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg, shutil.which("ffprobe")
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe(), None
    except Exception:
        return None, None


FFMPEG, FFPROBE = _find_binaries()


def available():
    return FFMPEG is not None


def _run(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          creationflags=CREATE_NO_WINDOW, encoding="utf-8",
                          errors="replace")


# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------
_SAFE = re.compile(r"[^A-Za-z0-9._ -]+")


def safe_name(filename):
    """Basename only, safe charset, must keep a video extension."""
    name = os.path.basename(filename or "").strip()
    name = _SAFE.sub("_", name)
    root, ext = os.path.splitext(name)
    root = root.strip(" .") or "clip"
    return root + ext.lower()


def unique_path(name):
    """Path in the library, suffixed -2, -3, ... if the name is taken."""
    root, ext = os.path.splitext(name)
    cand, n = name, 1
    while os.path.exists(os.path.join(LIB_DIR, cand)):
        n += 1
        cand = f"{root}-{n}{ext}"
    return os.path.join(LIB_DIR, cand)


def probe(path):
    """Duration / streams / codec metadata for one file (ffprobe json)."""
    if FFPROBE:
        r = _run([FFPROBE, "-v", "quiet", "-print_format", "json",
                  "-show_format", "-show_streams", path])
        try:
            info = json.loads(r.stdout or "{}")
        except ValueError:
            info = {}
        fmt = info.get("format", {})
        out = {"duration": float(fmt.get("duration", 0) or 0),
               "size": int(fmt.get("size", 0) or 0),
               "container": fmt.get("format_name", ""),
               "video": None, "audio": None}
        for s in info.get("streams", []):
            if s.get("codec_type") == "video" and out["video"] is None:
                fr = s.get("avg_frame_rate", "0/1")
                try:
                    num, den = fr.split("/")
                    fps = float(num) / float(den) if float(den) else 0.0
                except (ValueError, ZeroDivisionError):
                    fps = 0.0
                out["video"] = {"codec": s.get("codec_name", "?"),
                                "width": s.get("width", 0),
                                "height": s.get("height", 0),
                                "fps": round(fps, 3)}
            elif s.get("codec_type") == "audio" and out["audio"] is None:
                out["audio"] = {"codec": s.get("codec_name", "?"),
                                "channels": s.get("channels", 0),
                                "sample_rate": s.get("sample_rate", "")}
        return out
    # fallback: parse `ffmpeg -i` banner (no ffprobe available)
    r = _run([FFMPEG, "-hide_banner", "-i", path])
    text = r.stderr or ""
    out = {"duration": 0.0, "size": os.path.getsize(path) if os.path.exists(path) else 0,
           "container": os.path.splitext(path)[1].lstrip("."), "video": None, "audio": None}
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", text)
    if m:
        out["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"Video: (\w+).*?(\d{2,5})x(\d{2,5})", text)
    if m:
        out["video"] = {"codec": m.group(1), "width": int(m.group(2)),
                        "height": int(m.group(3)), "fps": 0.0}
    m = re.search(r"Audio: (\w+)", text)
    if m:
        out["audio"] = {"codec": m.group(1), "channels": 0, "sample_rate": ""}
    return out


_meta_cache = {}          # name -> (mtime, meta)
_meta_lock = threading.Lock()


def _meta(name, path):
    st = os.stat(path)
    with _meta_lock:
        hit = _meta_cache.get(name)
        if hit and hit[0] == st.st_mtime:
            return hit[1]
    meta = probe(path)
    with _meta_lock:
        _meta_cache[name] = (st.st_mtime, meta)
    return meta


def list_library():
    out = []
    for fn in sorted(os.listdir(LIB_DIR)):
        full = os.path.join(LIB_DIR, fn)
        ext = os.path.splitext(fn)[1].lower()
        if not os.path.isfile(full) or ext not in VIDEO_EXTS | AUDIO_EXTS:
            continue
        try:
            st = os.stat(full)
            meta = _meta(fn, full)
        except OSError:
            continue
        out.append({"name": fn, "size": st.st_size, "mtime": st.st_mtime,
                    "is_audio": ext in AUDIO_EXTS,
                    "playable": ext in BROWSER_PLAYABLE, **meta})
    out.sort(key=lambda f: -f["mtime"])
    return out


def delete(name):
    name = safe_name(name)
    path = os.path.join(LIB_DIR, name)
    if not os.path.isfile(path):
        return False
    os.unlink(path)
    thumb = os.path.join(THUMB_DIR, name + ".jpg")
    if os.path.isfile(thumb):
        try:
            os.unlink(thumb)
        except OSError:
            pass
    with _meta_lock:
        _meta_cache.pop(name, None)
    return True


def thumbnail(name):
    """Cached jpg poster frame (grabbed ~10% into the clip)."""
    name = safe_name(name)
    src = os.path.join(LIB_DIR, name)
    if not os.path.isfile(src):
        return None
    dst = os.path.join(THUMB_DIR, name + ".jpg")
    if os.path.isfile(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    dur = _meta(name, src).get("duration", 0) or 0
    at = max(0.0, min(dur * 0.1, dur - 0.1)) if dur > 1 else 0.0
    r = _run([FFMPEG, "-y", "-ss", f"{at:.2f}", "-i", src, "-frames:v", "1",
              "-vf", "scale=320:-2", "-q:v", "5", dst], timeout=120)
    return dst if r.returncode == 0 and os.path.isfile(dst) else None


# --------------------------------------------------------------------------
# Jobs — background ffmpeg with live progress
# --------------------------------------------------------------------------
JOBS = {}                 # id -> job dict
_jobs_lock = threading.Lock()

# encoder settings per target container
_CODECS = {
    ".mp4":  ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"],
    ".m4v":  ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"],
    ".mov":  ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"],
    ".mkv":  ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"],
    ".webm": ["-c:v", "libvpx-vp9", "-row-mt", "1", "-c:a", "libopus"],
    ".avi":  ["-c:v", "mpeg4", "-q:v", "5", "-c:a", "mp3"],
    ".gif":  [],          # handled specially (palette filter, no audio)
}
FORMATS = sorted(e.lstrip(".") for e in _CODECS)


def _encode_args(ext, crf, scale_h, fps):
    """Video/audio encoder args + optional scale/fps filters for one output."""
    args = list(_CODECS[ext])
    if ext in (".mp4", ".m4v", ".mov", ".mkv"):
        args += ["-crf", str(crf), "-preset", "medium", "-b:a", "192k"]
    elif ext == ".webm":
        args += ["-crf", str(crf), "-b:v", "0", "-b:a", "160k"]
    vf = []
    if scale_h:
        vf.append(f"scale=-2:{int(scale_h)}")
    if fps:
        vf.append(f"fps={fps}")
    if ext == ".gif":
        vf.append("split[a][b];[a]palettegen[p];[b][p]paletteuse")
        args = ["-an"]
    if vf:
        args += ["-vf", ",".join(vf)] if ext != ".gif" else ["-filter_complex", ",".join(vf)]
    return args


def _job_thread(job, cmd, total_duration):
    job["status"] = "running"
    job["started"] = time.time()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace",
                                creationflags=CREATE_NO_WINDOW)
    except OSError as exc:
        job.update(status="error", message=str(exc))
        return
    job["proc"] = proc
    tail = []

    def read_stderr():
        for line in proc.stderr:
            tail.append(line.rstrip())
            del tail[:-30]
    threading.Thread(target=read_stderr, daemon=True).start()

    for line in proc.stdout:              # -progress pipe:1 key=value stream
        line = line.strip()
        if line.startswith("out_time_us=") and total_duration > 0:
            try:
                done = int(line.split("=", 1)[1]) / 1e6
                job["progress"] = max(0.0, min(1.0, done / total_duration))
            except ValueError:
                pass
        elif line == "progress=end":
            job["progress"] = 1.0
    proc.wait()
    job["proc"] = None
    if job.get("cancelled"):
        job.update(status="cancelled", message="cancelled")
        try:
            os.unlink(job["output_path"])
        except OSError:
            pass
    elif proc.returncode == 0:
        job.update(status="done", progress=1.0, message="")
        with _meta_lock:                  # output landed in the library
            _meta_cache.pop(job["output"], None)
    else:
        job.update(status="error",
                   message="\n".join(tail[-8:]) or f"ffmpeg exit {proc.returncode}")
        try:
            os.unlink(job["output_path"])
        except OSError:
            pass
    job["finished"] = time.time()


def _start_job(op, label, cmd, total_duration, output_path):
    job = {"id": uuid.uuid4().hex[:12], "op": op, "label": label,
           "status": "queued", "progress": 0.0, "message": "",
           "output": os.path.basename(output_path), "output_path": output_path,
           "created": time.time(), "proc": None, "cancelled": False}
    with _jobs_lock:
        JOBS[job["id"]] = job
    threading.Thread(target=_job_thread, args=(job, cmd, total_duration),
                     daemon=True, name=f"video-{op}").start()
    return job


def invalidate_meta(name):
    """Drop the cached probe for a file that was just (re)written."""
    with _meta_lock:
        _meta_cache.pop(name, None)


def custom_job(op, label, output_path):
    """Register an externally-driven job (e.g. the animation renderer). The
    caller owns the thread and updates status/progress/proc itself; cancel
    and the /api/video/jobs list work the same as ffmpeg-command jobs."""
    job = {"id": uuid.uuid4().hex[:12], "op": op, "label": label,
           "status": "running", "progress": 0.0, "message": "",
           "output": os.path.basename(output_path), "output_path": output_path,
           "created": time.time(), "started": time.time(),
           "proc": None, "cancelled": False}
    with _jobs_lock:
        JOBS[job["id"]] = job
    return job


def job_view(job):
    return {k: job[k] for k in ("id", "op", "label", "status", "progress",
                                "message", "output", "created")}


def list_jobs():
    with _jobs_lock:
        jobs = sorted(JOBS.values(), key=lambda j: -j["created"])
    return [job_view(j) for j in jobs[:50]]


def cancel(job_id):
    with _jobs_lock:
        job = JOBS.get(job_id)
    if not job or job["status"] not in ("queued", "running"):
        return False
    job["cancelled"] = True
    proc = job.get("proc")
    if proc:
        try:
            proc.kill()
        except OSError:
            pass
    return True


def _src(name):
    path = os.path.join(LIB_DIR, safe_name(name))
    if not os.path.isfile(path):
        raise ValueError(f"no such library file: {name}")
    return path


def _out_path(base_root, ext):
    return unique_path(safe_name(base_root + ext))


# ── operations ─────────────────────────────────────────────────────────────
def convert(name, fmt, crf=23, scale_h=None, fps=None):
    """Re-encode/convert one file to another container (mp4 -> mkv etc.)."""
    ext = "." + str(fmt).lower().lstrip(".")
    if ext not in _CODECS:
        raise ValueError(f"unsupported format: {fmt}")
    src = _src(name)
    dur = _meta(os.path.basename(src), src).get("duration", 0)
    out = _out_path(os.path.splitext(os.path.basename(src))[0], ext)
    cmd = [FFMPEG, "-y", "-i", src, *_encode_args(ext, crf, scale_h, fps),
           "-progress", "pipe:1", "-nostats", out]
    return _start_job("convert", f"{os.path.basename(src)} -> {ext.lstrip('.')}",
                      cmd, dur, out)


def trim(name, start, end, precise=True, crf=23):
    """Cut [start, end] seconds out of a clip into a new library file.

    precise=True re-encodes (frame-exact); False stream-copies (instant, but
    cuts snap to the nearest keyframe).
    """
    src = _src(name)
    meta = _meta(os.path.basename(src), src)
    dur = meta.get("duration", 0) or 0
    start = max(0.0, float(start))
    end = min(float(end), dur) if dur else float(end)
    if end - start <= 0.01:
        raise ValueError("end must be after start")
    root, ext = os.path.splitext(os.path.basename(src))
    if ext not in _CODECS or ext == ".gif":
        ext = ".mp4"
    out = _out_path(f"{root}_cut", ext)
    if precise:
        enc = _encode_args(ext, crf, None, None)
    else:
        enc = ["-c", "copy"]
    cmd = [FFMPEG, "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src,
           *enc, "-progress", "pipe:1", "-nostats", out]
    return _start_job("cut", f"{root}{ext} [{start:.1f}s - {end:.1f}s]",
                      cmd, end - start, out)


def stitch(clips, fmt="mp4", crf=23, scale_h=None, fps=None, out_name=""):
    """Concatenate timeline clips [{name, start, end}] into one file.

    Everything is re-encoded through a concat filter so mixed codecs,
    resolutions and framerates all work. Video is normalised to the first
    clip's height (or scale_h); inputs missing audio get silence so the
    audio track stays in sync.
    """
    ext = "." + str(fmt).lower().lstrip(".")
    if ext not in _CODECS or ext == ".gif":
        raise ValueError(f"unsupported stitch format: {fmt}")
    if not clips:
        raise ValueError("no clips given")

    parsed, total = [], 0.0
    for c in clips:
        src = _src(c.get("name", ""))
        meta = _meta(os.path.basename(src), src)
        dur = meta.get("duration", 0) or 0
        start = max(0.0, float(c.get("start", 0)))
        end = float(c.get("end", 0)) or dur
        end = min(end, dur) if dur else end
        if end - start <= 0.01:
            raise ValueError(f"clip {os.path.basename(src)}: end must be after start")
        parsed.append({"src": src, "start": start, "end": end,
                       "has_audio": meta.get("audio") is not None,
                       "height": (meta.get("video") or {}).get("height", 0)})
        total += end - start

    # concat requires every input at the SAME WxH: scale to fit, pad to centre
    height = int(scale_h or next((c["height"] for c in parsed if c["height"]), 720))
    height -= height % 2
    first = next((c for c in parsed if c["height"]), None)
    first_meta = _meta(os.path.basename(first["src"]), first["src"]) if first else {}
    fw = (first_meta.get("video") or {}).get("width", 0)
    fh = (first_meta.get("video") or {}).get("height", 0)
    width = int(round(fw * height / fh)) if fw and fh else int(round(height * 16 / 9))
    width -= width % 2
    rate = fps or 30

    cmd = [FFMPEG, "-y"]
    for c in parsed:
        cmd += ["-i", c["src"]]
    fc, pairs = [], []
    for i, c in enumerate(parsed):
        fc.append(f"[{i}:v]trim={c['start']:.3f}:{c['end']:.3f},setpts=PTS-STARTPTS,"
                  f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                  f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={rate}[v{i}]")
        if c["has_audio"]:
            fc.append(f"[{i}:a]atrim={c['start']:.3f}:{c['end']:.3f},asetpts=PTS-STARTPTS,"
                      f"aresample=44100,aformat=channel_layouts=stereo[a{i}]")
        else:
            fc.append(f"aevalsrc=0:c=stereo:s=44100:d={c['end'] - c['start']:.3f}[a{i}]")
        pairs.append(f"[v{i}][a{i}]")
    fc.append(f"{''.join(pairs)}concat=n={len(parsed)}:v=1:a=1[vout][aout]")

    root = safe_name(out_name or "timeline").rsplit(".", 1)[0] or "timeline"
    out = _out_path(root, ext)
    enc = _encode_args(ext, crf, None, None)   # scale/fps already in the graph
    cmd += ["-filter_complex", ";".join(fc), "-map", "[vout]", "-map", "[aout]",
            *enc, "-progress", "pipe:1", "-nostats", out]
    label = f"{len(parsed)} clip(s) -> {os.path.basename(out)}"
    return _start_job("stitch", label, cmd, total, out)
