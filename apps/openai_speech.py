"""
VieNeu-TTS — OpenAI-compatible speech API (streaming), CPU or GPU.
==================================================================
Drop-in for ``POST /v1/audio/speech``: any OpenAI SDK / Pipecat / LiveKit /
Vercel AI client works by changing ``base_url``. Audio streams as it is
generated — the first bytes leave ~115 ms after the request on an RTX 3060,
~600 ms on a 6-core CPU (see docs/streaming.vi.md for every number).

    uv run python -m apps.openai_speech                 # http://127.0.0.1:8000
    VIENEU_BACKEND=onnx uv run python -m apps.openai_speech   # force CPU

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x")
    with client.audio.speech.with_streaming_response.create(
        model="vieneu-v3-nano", voice="Mai Anh", input="Xin chào! Đây là chế độ streaming của VieNeu, phát tới đâu nghe tới đó.",
        response_format="wav",              # 24 kHz s16le mono (Nano native)
    ) as r:
        for chunk in r.iter_bytes(4096):
            play(chunk)

Request body (OpenAI fields + a few extras, all optional but ``input``):
    model            any string; reported back as-is
    input            text, chunked internally at ``max_chars`` (256)
    voice            preset name (GET /v1/voices) or one added with POST /v1/voices
    response_format  "pcm" (s16le, no header) | "wav" (header with unknown length,
                     plays as it streams). mp3/opus/aac/flac → 400.
    stream_format    "audio" (default: raw chunked body) | "sse" (Server-Sent
                     Events: {"type":"speech.audio.delta","audio":"<base64>"} ...
                     {"type":"speech.audio.done","usage":{...}})
    speed            accepted for compatibility, ignored (header X-VieNeu-Ignored)
    sample_rate      24000 (Nano native default) | 16000 | 8000 —
                     resampled on the fly (soxr) when below native rate
    temperature, top_k, top_p, repetition_penalty, max_chars   sampling extras,
                     range-checked (out of range → 400): 0-2 / 1-1024 / (0, 1] / 1-2 / 64-512

POST /v1/voices refuses (409) the name of a built-in voice or alias, so a client
cannot swap out a voice other clients rely on (e.g. the default one).

Concurrency: GPU serves ``VIENEU_MAX_STREAMS`` streams at once (continuous
batching, default 16); CPU serves one (fp32) or two (int8) — the ONNX engine
interleaves them frame by frame but they share the cores.
Extra requests wait up to ``VIENEU_QUEUE_TIMEOUT`` s in a queue of at most
``VIENEU_QUEUE`` entries, then get 429 with Retry-After.

Environment:
    VIENEU_MODE=v3nano|v3turbo         (this host defaults to v3nano / 24 kHz)
    VIENEU_BACKEND=auto|onnx|pytorch   VIENEU_DEVICE=auto|cuda|cpu
    VIENEU_PRECISION=fp32|int8 (CPU/Turbo) VIENEU_ONNX_DIR=... (local ONNX export)
    VIENEU_MAX_STREAMS=16 (GPU)        VIENEU_QUEUE=16   VIENEU_QUEUE_TIMEOUT=10
    VIENEU_API_KEY=...  (Bearer auth; unset = open — logged as a warning when
                        HOST is not a loopback address)
    VIENEU_WATERMARK=1                 HOST=127.0.0.1  PORT=8000
                                       (HOST=0.0.0.0 to serve other machines; the
                                       Docker profiles set it)
    VieNeuTTS_WEB_ENABLED=true|false   public browser UI at / (default false).
                                       When false, / shows "Webapp đang đóng".
                                       /v1/* always needs Bearer when VIENEU_API_KEY is set.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import hmac
import io
import json
import logging
import os
import re
import struct
import threading
import time
import uuid
import wave
import weakref
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from vieneu import Vieneu

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vieneu.api")

# CPU Instabox default: VieNeu-TTS v3 Nano @ 24 kHz.
MODE = os.environ.get("VIENEU_MODE", "v3nano").strip().lower() or "v3nano"
SAMPLE_RATE = 24_000 if MODE == "v3nano" else 48_000
MODEL_ID = "vieneu-v3-nano" if MODE == "v3nano" else "vieneu-v3-turbo"
RATES = (24_000, 16_000, 8_000) if MODE == "v3nano" else (48_000, 24_000, 16_000, 8_000)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}




def web_history_dir() -> Path:
    root = Path(os.environ.get("VIENEU_WEB_HISTORY_DIR", "/data/workspace/vieneu-web-history"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_voice_slug(voice: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", (voice or "default").strip(), flags=re.UNICODE)
    return (slug[:40] or "default")
def web_enabled() -> bool:
    # Prefer the exact user-facing name; accept a few aliases.
    for name in ("VieNeuTTS_WEB_ENABLED", "VIENEUTTS_WEB_ENABLED", "VIENEU_WEB_ENABLED"):
        if name in os.environ:
            return _env_bool(name, False)
    return False


# ── engine + admission ───────────────────────────────────────────────────────

class Engine:
    """The model plus the admission gate (how many streams run at once)."""

    def __init__(self):
        mode = MODE
        backend = os.environ.get("VIENEU_BACKEND", "auto")
        device = os.environ.get("VIENEU_DEVICE", "auto")
        precision = os.environ.get("VIENEU_PRECISION", "fp32")
        onnx_dir = os.environ.get("VIENEU_ONNX_DIR") or None
        t = time.perf_counter()
        if mode == "v3nano":
            # Nano is ONNX/CPU-only; ignore Turbo GPU kwargs.
            log.info("⏳ loading VieNeu-TTS v3 Nano (ONNX/CPU, 24 kHz)")
            self.tts = Vieneu(mode="v3nano", onnx_dir=onnx_dir)
            self.backend = getattr(self.tts, "backend", "onnx")
            aliases = getattr(self.tts, "_voice_aliases", {}) or {}
            self.builtin_voices = frozenset(self.tts._preset_voices) | frozenset(aliases)
            self.sched = None
            self.max_streams = _env_int("VIENEU_MAX_STREAMS", 1) if "VIENEU_MAX_STREAMS" in os.environ else 1
        else:
            kw: dict = dict(backend=backend, device=device,
                            precision=precision,
                            onnx_dir=onnx_dir,
                            max_streams=_env_int("VIENEU_MAX_STREAMS", 16))
            log.info("⏳ loading VieNeu-TTS v3 Turbo (backend=%s device=%s)", backend, device)
            self.tts = Vieneu(mode="v3turbo", **kw)
            self.backend = self.tts.backend
            self.builtin_voices = frozenset(self.tts._preset_voices) | frozenset(self.tts._voice_aliases)
            self.sched = self.tts._get_stream_scheduler() if self.backend == "pytorch" else None
            if self.sched is not None:
                self.max_streams = self.sched.B
            else:
                cpu_default = 2 if precision == "int8" else 1
                self.max_streams = _env_int("VIENEU_MAX_STREAMS", cpu_default) if "VIENEU_MAX_STREAMS" in os.environ else cpu_default
        self.mode = mode
        self.sample_rate = int(getattr(self.tts, "sample_rate", SAMPLE_RATE))
        self.watermark = os.environ.get("VIENEU_WATERMARK", "1") != "0"
        self.max_queue = _env_int("VIENEU_QUEUE", self.max_streams)
        self.queue_timeout = float(os.environ.get("VIENEU_QUEUE_TIMEOUT", "10"))
        self._gate = threading.BoundedSemaphore(self.max_streams)
        self._lock = threading.Lock()
        self.active = 0
        self.waiting = 0
        # Warm-up: capture the CUDA graphs / load the ONNX graphs before the
        # first client pays for it.
        for _ in self.tts.infer_stream("Xin chào.", apply_watermark=False):
            pass
        log.info("✅ ready in %.1fs: backend=%s max_streams=%d queue=%d", time.perf_counter() - t,
                 self.backend, self.max_streams, self.max_queue)

    def acquire(self) -> None:
        with self._lock:
            if self.waiting >= self.max_queue:
                raise HTTPException(429, "server busy: queue full", headers={"Retry-After": "1"})
            self.waiting += 1
        try:
            if not self._gate.acquire(timeout=self.queue_timeout):
                raise HTTPException(429, "server busy: timed out waiting for a stream slot",
                                    headers={"Retry-After": "2"})
        finally:
            with self._lock:
                self.waiting -= 1
        with self._lock:
            self.active += 1

    def release(self) -> None:
        with self._lock:
            self.active -= 1
        self._gate.release()

    def stream_error(self) -> Optional[BaseException]:
        """Why the GPU stream scheduler died, or None. A dead scheduler refuses
        every request until the process restarts."""
        return getattr(self.sched, "error", None) if self.sched is not None else None

    def voices(self) -> list:
        out = []
        aliases = getattr(self.tts, "_voice_aliases", {}) or {}
        for name, v in self.tts._preset_voices.items():
            out.append({"id": name, "name": name, "description": v.get("description", ""),
                        "gender": v.get("gender", ""), "featured": v.get("featured"),
                        "aliases": [a for a, t in aliases.items() if t == name] or list(v.get("aliases") or [])})
        return out

    def resolve_voice(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        resolver = getattr(self.tts, "resolve_voice_name", None)
        if callable(resolver):
            return resolver(name)
        if name in self.tts._preset_voices:
            return name
        aliases = getattr(self.tts, "_voice_aliases", {}) or {}
        return aliases.get(name)


class _Slot:
    """One ``Engine`` stream slot, released exactly once: by the stream's own
    ``finally``, or — when the body is never iterated because the client left
    while queued — when the dropped body is garbage-collected."""

    def __init__(self, eng: Engine):
        self._eng = eng
        self._lock = threading.Lock()
        self._held = True

    def release(self) -> None:
        with self._lock:
            held, self._held = self._held, False
        if held:
            self._eng.release()


ENGINE: Optional[Engine] = None


def engine() -> Engine:
    global ENGINE
    if ENGINE is None:
        ENGINE = Engine()
    return ENGINE


# ── auth / errors in OpenAI's shape ───────────────────────────────────────────

def _auth(authorization: Optional[str] = Header(default=None)) -> None:
    key = os.environ.get("VIENEU_API_KEY")
    if key and not hmac.compare_digest((authorization or "").encode(), f"Bearer {key}".encode()):
        raise HTTPException(401, "invalid api key")


def _error(status: int, message: str, code: Optional[str] = None, headers=None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error",
                                   "code": code or status}}, status_code=status, headers=headers)


app = FastAPI(title="VieNeu-TTS speech API", version="1.0")

_WEB_CLOSED_HTML = """<!doctype html>
<html lang="vi"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>VieNeu TTS</title>
<style>
body{margin:0;font-family:system-ui,sans-serif;background:#0b1220;color:#e8eefc;display:grid;place-items:center;min-height:100vh}
.card{max-width:34rem;padding:2rem;border:1px solid #24314d;border-radius:16px;background:#121a2b;box-shadow:0 10px 40px #0008}
h1{margin:0 0 .5rem;font-size:1.4rem}p{margin:0;color:#9db0d0;line-height:1.5}
code{color:#9fd0ff}
</style></head>
<body><div class="card">
<h1>Webapp đang đóng</h1>
<p>Giao diện VieNeu TTS hiện không mở cho truy cập công khai.<br/>
API vẫn dùng được với <code>Authorization: Bearer &lt;VIENEU_API_KEY&gt;</code>.</p>
</div></body></html>
"""

_WEB_APP_HTML = """<!doctype html>
<html lang="vi"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>VieNeu TTS</title>
<style>
:root{
  color-scheme:dark;
  --bg:#140d08;
  --bg2:#1c130c;
  --card:#24170f;
  --line:#4a2f1d;
  --text:#fff4ea;
  --muted:#d2b39a;
  --orange:#ff7a18;
  --orange2:#ff9a4d;
  --orange-deep:#d45600;
}
*{box-sizing:border-box}
body{
  margin:0;min-height:100vh;color:var(--text);
  font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  background:
    radial-gradient(1100px 520px at 12% -10%, rgba(255,122,24,.28), transparent 55%),
    radial-gradient(900px 480px at 95% 0%, rgba(255,154,77,.16), transparent 50%),
    linear-gradient(180deg, var(--bg), #0d0907 70%);
}
.wrap{max-width:1100px;margin:0 auto;padding:2rem 1.25rem 3rem}
.hero{display:flex;justify-content:space-between;gap:1rem;align-items:flex-end;margin-bottom:1.5rem;flex-wrap:wrap}
.brand h1{margin:0;font-size:clamp(1.8rem,3vw,2.35rem);letter-spacing:-.02em}
.tag{margin:.45rem 0 0;color:var(--muted);font-size:1rem}
.tag a,.footer a,.hist a{color:var(--orange2);text-decoration:none}
.tag a:hover,.footer a:hover,.hist a:hover{text-decoration:underline}
.pill{display:inline-flex;align-items:center;gap:.4rem;padding:.45rem .8rem;border:1px solid var(--line);border-radius:999px;background:rgba(255,122,24,.08);color:var(--orange2);font-size:.9rem;white-space:nowrap}
.grid{display:grid;grid-template-columns:1.4fr .95fr;gap:1.1rem}
@media (max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:linear-gradient(180deg,rgba(36,23,15,.96),rgba(28,19,12,.94));border:1px solid var(--line);border-radius:22px;padding:1.2rem 1.2rem 1.3rem;box-shadow:0 18px 50px rgba(0,0,0,.28)}
.card h2{margin:0 0 1rem;font-size:1.05rem;color:var(--orange2);font-weight:700}
label{display:block;margin:0 0 .4rem;color:var(--muted);font-size:.92rem}
select,textarea,button{width:100%;border-radius:14px;border:1px solid var(--line);background:#1a110c;color:var(--text);padding:.85rem 1rem;font:inherit}
select:focus,textarea:focus{outline:2px solid rgba(255,122,24,.35);border-color:var(--orange)}
textarea{min-height:12rem;resize:vertical;line-height:1.5}
.field{margin-bottom:1rem}
.actions{display:flex;gap:.75rem;flex-wrap:wrap;margin-top:.2rem}
button{cursor:pointer;background:linear-gradient(180deg,var(--orange),var(--orange-deep));border-color:transparent;font-weight:700;color:#1a0d05}
button.secondary{background:transparent;border:1px solid var(--line);color:var(--text);font-weight:600}
button:disabled{opacity:.55;cursor:wait}
.status{min-height:1.3rem;margin-top:.9rem;color:var(--muted);font-size:.92rem}
.err{color:#ff9b8f}
audio{width:100%;margin-top:1rem;border-radius:12px}
.hist{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:.7rem;max-height:34rem;overflow:auto}
.hist li{display:grid;grid-template-columns:1fr auto;gap:.7rem;align-items:center;padding:.85rem .9rem;border:1px solid var(--line);border-radius:16px;background:rgba(255,122,24,.05)}
.hist .meta{min-width:0}
.hist .title{font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hist .sub{color:var(--muted);font-size:.84rem;margin-top:.2rem}
.hist .dl{display:inline-flex;align-items:center;justify-content:center;padding:.55rem .8rem;border-radius:999px;border:1px solid rgba(255,122,24,.35);color:var(--orange2);background:rgba(255,122,24,.08);font-size:.86rem;font-weight:650}
.empty{color:var(--muted);padding:.4rem 0}
.footer{margin-top:1.6rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--muted);font-size:.9rem}
.heart{color:#ff5d5d}
</style></head>
<body>
<div class="wrap">
  <div class="hero">
    <div class="brand">
      <h1>VieNeu TTS</h1>
      <p class="tag">chạy trên <a href="https://orangepi.vn" target="_blank" rel="noopener noreferrer">Orange Pi</a></p>
    </div>
    <div class="pill">24 kHz · 11 preset voices</div>
  </div>
  <div class="grid">
    <section class="card">
      <h2>Tạo giọng nói</h2>
      <div class="field">
        <label for="voice">Voice</label>
        <select id="voice"></select>
      </div>
      <div class="field">
        <label for="text">Nội dung</label>
        <textarea id="text">Xin chào, đây là VieNeu TTS trên Orange Pi.</textarea>
      </div>
      <div class="actions">
        <button id="speak" type="button">Tạo giọng nói</button>
        <button id="refresh" class="secondary" type="button">Làm mới danh sách</button>
      </div>
      <div class="status" id="status"></div>
      <audio id="player" controls></audio>
    </section>
    <section class="card">
      <h2>File WAV đã tạo</h2>
      <ul class="hist" id="history"></ul>
      <div class="empty" id="empty">Chưa có file nào.</div>
    </section>
  </div>
  <footer class="footer">Copyright 2026 © · Được tạo ra cùng với <span class="heart">♥</span> bởi <a href="https://tonytran.id.vn" target="_blank" rel="noopener noreferrer">Tony Trần</a></footer>
</div>
<script>
function fmtBytes(n){
  if(n<1024) return n+' B';
  if(n<1048576) return (n/1024).toFixed(1)+' KB';
  return (n/1048576).toFixed(2)+' MB';
}
function esc(s){
  return String(s||'').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
async function loadVoices(){
  const st=document.getElementById('status');
  st.textContent='Đang tải danh sách voice…';
  const r=await fetch('/web/voices');
  if(!r.ok){ st.innerHTML='<span class="err">Không tải được voices ('+r.status+')</span>'; return; }
  const data=await r.json();
  const sel=document.getElementById('voice');
  sel.innerHTML='';
  for(const v of (data.data||[])){
    const opt=document.createElement('option');
    opt.value=v.id; opt.textContent=v.id+(v.description?(' — '+v.description):'');
    if(v.id===data.default_voice) opt.selected=true;
    sel.appendChild(opt);
  }
  st.textContent= (data.count||0)+' preset voices · '+(data.sample_rate||24000)+' Hz';
}
async function loadHistory(){
  const list=document.getElementById('history');
  const empty=document.getElementById('empty');
  const r=await fetch('/web/history');
  if(!r.ok){ list.innerHTML=''; empty.style.display='block'; empty.textContent='Không tải được lịch sử ('+r.status+')'; return; }
  const data=await r.json();
  const items=data.data||[];
  list.innerHTML='';
  if(!items.length){ empty.style.display='block'; empty.textContent='Chưa có file nào.'; return; }
  empty.style.display='none';
  for(const it of items){
    const li=document.createElement('li');
    li.innerHTML = '<div class="meta"><div class="title">'+esc(it.voice)+' · '+esc(it.preview||it.filename)+'</div>'+
      '<div class="sub">'+esc(it.created_at||'')+' · '+fmtBytes(it.bytes||0)+' · '+(it.sample_rate||24000)+' Hz</div></div>'+
      '<a class="dl" href="/web/history/'+encodeURIComponent(it.id)+'" download="'+esc(it.filename)+'">Tải về</a>';
    list.appendChild(li);
  }
}
async function speak(){
  const st=document.getElementById('status');
  const btn=document.getElementById('speak');
  const text=document.getElementById('text').value.trim();
  const voice=document.getElementById('voice').value;
  if(!text){ st.innerHTML='<span class="err">Nhập nội dung trước</span>'; return; }
  btn.disabled=true; st.textContent='Đang tổng hợp…';
  try{
    const r=await fetch('/web/speech',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({input:text,voice,response_format:'wav',sample_rate:24000})});
    if(!r.ok){ const t=await r.text(); throw new Error(t||('HTTP '+r.status)); }
    const blob=await r.blob();
    const url=URL.createObjectURL(blob);
    const player=document.getElementById('player');
    player.src=url; await player.play().catch(()=>{});
    st.textContent='Xong · WAV 24 kHz';
    await loadHistory();
  }catch(e){ st.innerHTML='<span class="err">'+(e.message||e)+'</span>'; }
  finally{ btn.disabled=false; }
}
document.getElementById('speak').onclick=speak;
document.getElementById('refresh').onclick=loadHistory;
loadVoices();
loadHistory();
</script>
</body></html>
"""



@app.exception_handler(HTTPException)
async def _http_exc(_req: Request, exc: HTTPException):
    return _error(exc.status_code, str(exc.detail), headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def _invalid_request(_req: Request, exc: RequestValidationError):
    # Field and reason only: FastAPI's default echoes the rejected value, and a
    # NaN (which json.loads accepts) cannot be encoded back — the 422 became a 500.
    errs = exc.errors()
    first = errs[0] if errs else {}
    field = ".".join(str(p) for p in first.get("loc", ()) if p not in ("body", "query", "header"))
    msg = first.get("msg", "invalid request")
    return _error(400, f"{field}: {msg}" if field else msg)


@app.on_event("startup")
def _startup() -> None:
    engine()


# ── audio helpers ────────────────────────────────────────────────────────────

def _pcm16(x: np.ndarray) -> bytes:
    return (np.asarray(x, dtype=np.float32) * 32767).clip(-32768, 32767).astype("<i2").tobytes()


def _wav_header(rate: int) -> bytes:
    """A WAV header claiming an unknown (huge) length, so players start at once."""
    h = io.BytesIO()
    with wave.open(h, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.setnframes(0)
    b = bytearray(h.getvalue())
    struct.pack_into("<I", b, 4, 0xFFFFFFFF)      # RIFF size
    struct.pack_into("<I", b, len(b) - 4, 0xFFFFFFFF)   # data size
    return bytes(b)


class _Resampler:
    """Native engine rate → ``rate`` chunk by chunk with carried state (soxr)."""

    def __init__(self, rate: int, native_rate: int = SAMPLE_RATE):
        self.rate = rate
        self.native_rate = native_rate
        self._rs = None
        if rate != native_rate:
            import soxr
            self._rs = soxr.ResampleStream(native_rate, rate, 1, dtype="float32", quality="HQ")

    def __call__(self, x: np.ndarray, last: bool = False) -> np.ndarray:
        if self._rs is None:
            return x
        return self._rs.resample_chunk(np.asarray(x, dtype=np.float32), last=last)


# ── /v1/audio/speech ─────────────────────────────────────────────────────────

class SpeechRequest(BaseModel):
    model: str = MODEL_ID
    input: str = Field(min_length=1, max_length=20_000)
    voice: Optional[str] = None
    response_format: str = "wav"
    stream_format: str = "audio"
    speed: Optional[float] = None
    instructions: Optional[str] = None
    sample_rate: int = SAMPLE_RATE
    # Turbo sampling knobs (ignored by Nano).
    temperature: float = Field(0.8, ge=0.0, le=2.0, allow_inf_nan=False)
    top_k: int = Field(25, ge=1, le=1024)   # 1024 = the codec's vocabulary
    top_p: float = Field(0.95, gt=0.0, le=1.0, allow_inf_nan=False)
    repetition_penalty: float = Field(1.2, ge=1.0, le=2.0, allow_inf_nan=False)
    # Nano was trained on short clips; default 140. Turbo keeps up to 512.
    max_chars: int = Field(140 if MODE == "v3nano" else 256, ge=64, le=512)
    # Nano quality/speed knobs
    steps: Optional[int] = Field(None, ge=1, le=64)
    cfg: Optional[float] = Field(None, ge=0.0, le=10.0, allow_inf_nan=False)


def _speech_chunks(eng: Engine, req: SpeechRequest, rid: str, slot: _Slot) -> Iterator[np.ndarray]:
    """float32 chunks at ``req.sample_rate``; logs TTFA / RTF; holds one stream slot."""
    t0 = time.perf_counter()
    first = None
    emitted = 0
    native = eng.sample_rate
    rs = _Resampler(req.sample_rate, native_rate=native)
    try:
        kwargs = dict(voice=req.voice or None, apply_watermark=eng.watermark, max_chars=req.max_chars)
        if eng.mode == "v3nano":
            if req.steps is not None:
                kwargs["steps"] = req.steps
            if req.cfg is not None:
                kwargs["cfg"] = req.cfg
        else:
            kwargs.update(temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
                          repetition_penalty=req.repetition_penalty)
        for chunk in eng.tts.infer_stream(req.input, **kwargs):
            if chunk is None or len(chunk) == 0:
                continue
            if first is None:
                first = time.perf_counter() - t0
            emitted += len(chunk)
            out = rs(chunk)
            if len(out):
                yield out
        tail = rs(np.zeros(0, np.float32), last=True)
        if len(tail):
            yield tail
    finally:
        slot.release()
        total = time.perf_counter() - t0
        audio_s = emitted / native if native else 0
        log.info("%s done: ttfa=%s total=%.2fs audio=%.2fs rtf=%s active=%d", rid,
                 f"{first*1000:.0f}ms" if first is not None else "-", total, audio_s,
                 f"{total/audio_s:.2f}" if audio_s else "-", eng.active)


def _audio_body(chunks: Iterator[np.ndarray], fmt: str, rate: int) -> Iterator[bytes]:
    if fmt == "wav":
        yield _wav_header(rate)
    for c in chunks:
        yield _pcm16(c)


def _sse_body(chunks: Iterator[np.ndarray], fmt: str, rate: int) -> Iterator[bytes]:
    def ev(obj: dict) -> bytes:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")
    n = 0
    if fmt == "wav":
        yield ev({"type": "speech.audio.delta", "audio": base64.b64encode(_wav_header(rate)).decode()})
    for c in chunks:
        n += len(c)
        yield ev({"type": "speech.audio.delta", "audio": base64.b64encode(_pcm16(c)).decode()})
    yield ev({"type": "speech.audio.done",
              "usage": {"output_samples": n, "sample_rate": rate, "seconds": round(n / rate, 3)}})


@app.post("/v1/audio/speech", dependencies=[Depends(_auth)])
def speech(req: SpeechRequest):
    eng = engine()
    fmt = req.response_format.lower()
    if fmt not in ("pcm", "wav"):
        raise HTTPException(400, f"response_format '{fmt}' not supported; use 'pcm' or 'wav'")
    if req.stream_format not in ("audio", "sse"):
        raise HTTPException(400, "stream_format must be 'audio' or 'sse'")
    if req.sample_rate not in RATES:
        raise HTTPException(400, f"sample_rate must be one of {RATES}")
    if req.voice and eng.resolve_voice(req.voice) is None:
        raise HTTPException(400, f"unknown voice '{req.voice}'; see GET /v1/voices/presets")
    if eng.stream_error() is not None:
        # Say so now: once the 200 headers are out, a failure only truncates the body.
        raise HTTPException(503, "speech engine is down; restart the server (see /health)")
    rid = f"spk-{uuid.uuid4().hex[:8]}"
    eng.acquire()   # 429 if the server is full; released when the stream ends
    slot = _Slot(eng)
    chunks = _speech_chunks(eng, req, rid, slot)
    # A generator that never starts never runs its finally: if the client is gone
    # before the body is read, the slot is freed when the body is dropped.
    weakref.finalize(chunks, slot.release)
    headers = {"X-Request-Id": rid, "X-Sample-Rate": str(req.sample_rate), "Cache-Control": "no-store"}
    ignored = [k for k in ("speed", "instructions") if getattr(req, k) is not None]
    if ignored:
        headers["X-VieNeu-Ignored"] = ",".join(ignored)
    if req.stream_format == "sse":
        return StreamingResponse(_sse_body(chunks, fmt, req.sample_rate),
                                 media_type="text/event-stream", headers=headers)
    media = "audio/wav" if fmt == "wav" else "audio/pcm"
    return StreamingResponse(_audio_body(chunks, fmt, req.sample_rate), media_type=media, headers=headers)




# ── public web UI (no Bearer). API under /v1/* still requires VIENEU_API_KEY. ─

@app.get("/", response_class=HTMLResponse)
def web_home():
    if not web_enabled():
        return HTMLResponse(_WEB_CLOSED_HTML, status_code=403)
    return HTMLResponse(_WEB_APP_HTML)


@app.get("/web/voices")
def web_voices():
    if not web_enabled():
        raise HTTPException(403, "Webapp đang đóng")
    eng = engine()
    data = eng.voices()
    return {
        "object": "list",
        "model": MODEL_ID,
        "mode": eng.mode,
        "sample_rate": eng.sample_rate,
        "count": len(data),
        "default_voice": getattr(eng.tts, "_default_voice", None),
        "data": data,
    }


@app.get("/web/history")
def web_history():
    if not web_enabled():
        raise HTTPException(403, "Webapp đang đóng")
    root = web_history_dir()
    items = []
    for meta_path in sorted(root.glob("*.json"), reverse=True):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        wav = root / f"{meta_path.stem}.wav"
        if not wav.exists():
            continue
        items.append({
            "id": meta_path.stem,
            "filename": meta.get("filename") or wav.name,
            "voice": meta.get("voice") or "",
            "preview": meta.get("preview") or "",
            "created_at": meta.get("created_at") or "",
            "bytes": wav.stat().st_size,
            "sample_rate": meta.get("sample_rate") or 24000,
        })
        if len(items) >= 50:
            break
    return {"object": "list", "count": len(items), "data": items}


@app.get("/web/history/{item_id}")
def web_history_download(item_id: str):
    if not web_enabled():
        raise HTTPException(403, "Webapp đang đóng")
    if not re.fullmatch(r"[A-Za-z0-9_\-]{8,80}", item_id or ""):
        raise HTTPException(400, "invalid id")
    wav = web_history_dir() / f"{item_id}.wav"
    meta_path = web_history_dir() / f"{item_id}.json"
    if not wav.exists():
        raise HTTPException(404, "file not found")
    filename = f"{item_id}.wav"
    if meta_path.exists():
        try:
            filename = json.loads(meta_path.read_text(encoding="utf-8")).get("filename") or filename
        except Exception:
            pass
    return FileResponse(path=str(wav), media_type="audio/wav", filename=filename)


@app.post("/web/speech")
def web_speech(req: SpeechRequest):
    if not web_enabled():
        raise HTTPException(403, "Webapp đang đóng")
    # Force downloadable WAV history for the browser UI.
    req.response_format = "wav"
    if req.sample_rate not in RATES:
        req.sample_rate = SAMPLE_RATE

    eng = engine()
    if req.voice and eng.resolve_voice(req.voice) is None:
        raise HTTPException(400, f"unknown voice '{req.voice}'; see GET /web/voices")
    if eng.stream_error() is not None:
        raise HTTPException(503, "speech engine is down; restart the server (see /health)")

    rid = f"web-{uuid.uuid4().hex[:10]}"
    eng.acquire()
    slot = _Slot(eng)
    chunks = _speech_chunks(eng, req, rid, slot)
    weakref.finalize(chunks, slot.release)

    pcm = bytearray()
    for c in chunks:
        pcm.extend(_pcm16(c))
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(req.sample_rate)
        w.writeframes(bytes(pcm))
    data = wav_buf.getvalue()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    voice = eng.resolve_voice(req.voice) or (req.voice or "default")
    item_id = f"{stamp}-{_safe_voice_slug(voice)}-{uuid.uuid4().hex[:6]}"
    filename = f"{item_id}.wav"
    preview = " ".join((req.input or "").split())
    if len(preview) > 72:
        preview = preview[:69] + "..."
    root = web_history_dir()
    (root / filename).write_bytes(data)
    (root / f"{item_id}.json").write_text(json.dumps({
        "id": item_id,
        "filename": filename,
        "voice": voice,
        "preview": preview,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_rate": req.sample_rate,
        "bytes": len(data),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    headers = {
        "X-Request-Id": rid,
        "X-Sample-Rate": str(req.sample_rate),
        "X-History-Id": item_id,
        "Cache-Control": "no-store",
        "Content-Disposition": f'attachment; filename="{filename}"',
    }
    return StreamingResponse(iter([data]), media_type="audio/wav", headers=headers)


# ── discovery ────────────────────────────────────────────────────────────────

@app.get("/v1/models", dependencies=[Depends(_auth)])
def models():
    eng = engine()
    return {"object": "list", "data": [{
        "id": MODEL_ID, "object": "model", "created": 0, "owned_by": "vieneu",
        "mode": eng.mode, "sample_rate": eng.sample_rate, "backend": eng.backend,
        "max_streams": eng.max_streams, "response_formats": ["pcm", "wav"],
        "stream_formats": ["audio", "sse"], "sample_rates": list(RATES),
        "preset_count": len(eng.tts._preset_voices),
    }]}


@app.get("/v1/voices", dependencies=[Depends(_auth)])
def voices():
    return {"object": "list", "data": engine().voices()}


@app.get("/v1/voices/presets", dependencies=[Depends(_auth)])
def voice_presets():
    """List the built-in preset voices for the loaded engine (11 for v3 Nano)."""
    eng = engine()
    data = eng.voices()
    return {
        "object": "list",
        "model": MODEL_ID,
        "mode": eng.mode,
        "sample_rate": eng.sample_rate,
        "count": len(data),
        "default_voice": getattr(eng.tts, "_default_voice", None),
        "data": data,
    }


# A voice name is a label: letters/digits (any script), spaces, dots, dashes,
# underscores; 1-64 chars, no control characters. It ends up in logs and in the
# voices file, so anything else is rejected at the boundary.
_VOICE_NAME = re.compile(r"[^\W_][\w .\-]{0,63}")
_CLIP_EXTS = {".wav": ".wav", ".mp3": ".mp3", ".flac": ".flac", ".ogg": ".ogg", ".m4a": ".m4a"}
_MAX_CLIP_BYTES = 20 * 1024 * 1024


@app.post("/v1/voices", dependencies=[Depends(_auth)])
def add_voice(name: str = Form(...), file: UploadFile = File(...), denoise: bool = Form(True),
              description: str = Form("")):
    """Enroll a 3-8 s reference clip as ``voice=name`` (in memory, for this process).

    Plain ``def``: FastAPI runs it in a thread, so the blocking upload read,
    temp file and enrolment do not stall the event loop.
    """
    import tempfile
    eng = engine()
    if not _VOICE_NAME.fullmatch(name):
        raise HTTPException(400, "voice name: 1-64 letters, digits, spaces, '.', '-' or '_'")
    if not _VOICE_NAME.fullmatch(description or "x"):
        raise HTTPException(400, "description: 1-64 letters, digits, spaces, '.', '-' or '_'")
    if name in eng.builtin_voices:
        # add_voice would replace it for every client, including the default voice.
        raise HTTPException(409, f"'{name}' is a built-in voice; choose another name")
    # The suffix only picks the decoder; it is mapped to a constant, never taken from the client.
    ext = _CLIP_EXTS.get(os.path.splitext(file.filename or "")[1].lower(), ".wav")
    data = file.file.read(_MAX_CLIP_BYTES + 1)
    if len(data) > _MAX_CLIP_BYTES:
        raise HTTPException(413, "reference clip larger than 20 MB")
    fd, path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        eng.tts.add_voice(name, path, denoise=denoise, description=description)
    except HTTPException:
        raise
    except Exception as e:   # noqa: BLE001
        raise HTTPException(400, f"could not enroll voice: {e}")
    finally:
        os.unlink(path)
    return {"id": name, "name": name, "description": description}


@app.get("/health")
def health():
    eng = engine()
    body = {"status": "ok", "mode": eng.mode, "model": MODEL_ID, "backend": eng.backend,
            "max_streams": eng.max_streams, "active": eng.active, "waiting": eng.waiting,
            "sample_rate": eng.sample_rate, "preset_count": len(eng.tts._preset_voices),
            "web_enabled": web_enabled()}
    err = eng.stream_error()
    if err is not None:
        # 503 so a container healthcheck restarts us: the scheduler does not recover.
        return JSONResponse({**body, "status": "error", "error": f"stream scheduler is down: {err!r}"},
                            status_code=503)
    return body


def main() -> None:
    # Loopback unless told otherwise: the API is open when VIENEU_API_KEY is unset.
    host = os.environ.get("HOST", "127.0.0.1")
    port = _env_int("PORT", 8000)
    if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("VIENEU_API_KEY"):
        log.warning("⚠️ listening on %s:%d without VIENEU_API_KEY: anyone who can reach this port "
                    "can use the API and enroll voices", host, port)
    # One worker: the model (and on GPU the scheduler) is process-local.
    uvicorn.run(app, host=host, port=port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
