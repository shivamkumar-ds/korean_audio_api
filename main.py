"""
Korean Audio Dataset API (Q6)

POST /answer-audio   (and POST /  as a fallback, in case the grader hits the base URL)
  Accepts EITHER:
    - JSON body: {"audio_id": "...", "audio_base64": "..."}
    - multipart/form-data with a file field
    - raw binary body (Content-Type: audio/wav, audio/mpeg, application/octet-stream, ...)

  Pipeline:
    1. Decode audio, detect WAV/MP3 from magic bytes.
    2. Send the audio directly (as base64 "input_audio" content) to an audio-capable
       chat model via aipipe.org's /chat/completions endpoint, and ask it in one shot
       to listen + extract a structured table ({"columns": [...], "rows": [{...}]}).
       (aipipe's proxy requires a plain JSON body with a "model" field for cost
       tracking, so we do NOT use the multipart /audio/transcriptions endpoint.)
    3. Compute every statistic ourselves in Python (never trust the LLM's arithmetic).
    4. Return the full required JSON shape, every key always present.

Env vars:
  AIPIPE_TOKEN   - your aipipe.org bearer token (same one used for Q2)
  EXTRACT_MODEL  - default "gpt-4o-audio-preview" (must support audio input)
"""

import base64
import json
import os
import statistics
from itertools import combinations
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
EXTRACT_MODEL = os.environ.get("EXTRACT_MODEL", "gpt-audio")

AIPIPE_CHAT_URL = "https://aipipe.org/openai/v1/chat/completions"

app = FastAPI(title="Korean Audio Dataset API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUIRED_KEYS = [
    "rows",
    "columns",
    "mean",
    "std",
    "variance",
    "min",
    "max",
    "median",
    "mode",
    "range",
    "allowed_values",
    "value_range",
    "correlation",
]


def empty_result() -> dict:
    return {
        "rows": 0,
        "columns": [],
        "mean": {},
        "std": {},
        "variance": {},
        "min": {},
        "max": {},
        "median": {},
        "mode": {},
        "range": {},
        "allowed_values": {},
        "value_range": {},
        "correlation": [],
    }


def detect_audio_format(raw: bytes) -> str:
    """Return 'wav' or 'mp3' based on magic bytes (the two formats input_audio supports)."""
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "wav"
    if raw[:3] == b"ID3" or raw[:2] == b"\xff\xfb" or raw[:2] == b"\xff\xf3":
        return "mp3"
    # Fallback - the task spec only promises WAV (RIFF) or MP3 (ID3) inputs
    return "wav"


async def get_audio_bytes(request: Request) -> bytes:
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        b64 = body.get("audio_base64") or body.get("audio") or ""
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        b64 = "".join(b64.split())  # strip whitespace/newlines
        try:
            return base64.b64decode(b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid audio_base64")

    if "multipart/form-data" in content_type:
        form = await request.form()
        for value in form.values():
            if hasattr(value, "read"):
                return await value.read()
        raise HTTPException(status_code=400, detail="No file found in multipart form")

    # Raw binary body (audio/wav, audio/mpeg, application/octet-stream, ...)
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty request body")
    return raw


EXTRACT_SYSTEM_PROMPT = """You listen to an audio clip (often Korean speech) that describes a \
small tabular dataset - rows and columns of data, spoken aloud (numbers and/or category labels). \
Transcribe it mentally and convert any Korean numerals/words for numbers into actual numbers.

Return ONLY a JSON object of this exact shape, nothing else, no markdown fences:
{"columns": ["col1", "col2", ...], "rows": [{"col1": value, "col2": value, ...}, ...]}

Rules:
- Every row must have a value for every column (use null if genuinely not stated).
- Numeric values must be JSON numbers (int or float), not strings.
- Category/label values must be JSON strings.
- Do not compute or include any statistics yourself - only the raw extracted rows."""


async def extract_table_from_audio(raw: bytes) -> dict:
    if not AIPIPE_TOKEN:
        raise HTTPException(status_code=500, detail="AIPIPE_TOKEN is not configured on the server")

    audio_format = detect_audio_format(raw)
    audio_b64 = base64.b64encode(raw).decode()

    payload = {
        "model": EXTRACT_MODEL,
        "modalities": ["text"],
        "temperature": 0,
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": audio_format}},
                    {"type": "text", "text": "Extract the table as instructed."},
                ],
            },
        ],
    }
    headers = {"Authorization": f"Bearer {AIPIPE_TOKEN}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(AIPIPE_CHAT_URL, json=payload, headers=headers)

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"extraction error: {resp.status_code} {resp.text[:300]}")

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(status_code=502, detail=f"Unexpected extraction response: {data}")

    # Strip markdown code fences if the model added them despite instructions.
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"Extractor did not return valid JSON: {content[:300]}")

    if "columns" not in parsed or "rows" not in parsed:
        raise HTTPException(status_code=502, detail=f"Extractor JSON missing columns/rows: {parsed}")

    return parsed


def is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compute_stats(columns: list[str], rows: list[dict]) -> dict:
    result = empty_result()
    result["rows"] = len(rows)
    result["columns"] = list(columns)

    numeric_cols: dict[str, list[float]] = {}
    categorical_cols: dict[str, list[Any]] = {}

    for col in columns:
        values = [r.get(col) for r in rows if r.get(col) is not None]
        if values and all(is_number(v) for v in values):
            numeric_cols[col] = [float(v) for v in values]
        else:
            categorical_cols[col] = values

    for col, vals in numeric_cols.items():
        if not vals:
            continue
        result["mean"][col] = statistics.mean(vals)
        result["std"][col] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        result["variance"][col] = statistics.pvariance(vals) if len(vals) > 1 else 0.0
        result["min"][col] = min(vals)
        result["max"][col] = max(vals)
        result["median"][col] = statistics.median(vals)
        try:
            result["mode"][col] = statistics.mode(vals)
        except statistics.StatisticsError:
            result["mode"][col] = statistics.multimode(vals)[0]
        result["range"][col] = max(vals) - min(vals)
        result["value_range"][col] = [min(vals), max(vals)]

    for col, vals in categorical_cols.items():
        uniq = sorted(set(str(v) for v in vals))
        result["allowed_values"][col] = uniq
        result["value_range"][col] = uniq

    correlations = []
    numeric_names = list(numeric_cols.keys())
    for a, b in combinations(numeric_names, 2):
        # Only rows where both columns have numeric values, pair them up positionally.
        pairs = [
            (r.get(a), r.get(b))
            for r in rows
            if is_number(r.get(a)) and is_number(r.get(b))
        ]
        if len(pairs) < 2:
            continue
        xs = [float(p[0]) for p in pairs]
        ys = [float(p[1]) for p in pairs]
        try:
            r_value = statistics.correlation(xs, ys)
        except (statistics.StatisticsError, ZeroDivisionError):
            continue
        if r_value > 0.3:
            corr_type = "positive"
        elif r_value < -0.3:
            corr_type = "negative"
        else:
            corr_type = "none"
        correlations.append({"x": a, "y": b, "type": corr_type})

    result["correlation"] = correlations
    return result


async def handle_audio_request(request: Request) -> dict:
    raw = await get_audio_bytes(request)
    table = await extract_table_from_audio(raw)
    return compute_stats(table["columns"], table["rows"])


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/answer-audio")
async def answer_audio(request: Request):
    return await handle_audio_request(request)


@app.post("/")
async def answer_audio_root(request: Request):
    return await handle_audio_request(request)
