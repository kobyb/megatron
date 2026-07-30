"""Pipeline orchestration + FastAPI SSE dashboard."""
import asyncio
import itertools
import json
import queue
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from . import audio, control, stt
from .claims import ClaimExtractor
from .config import Config
from .verify import verify

STATIC = Path(__file__).parent / "static"


class Bus:
    def __init__(self):
        self.subs: set[asyncio.Queue] = set()
        self.history: list[dict] = []

    def publish(self, event: dict):
        event["ts"] = round(time.time(), 2)
        self.history = (self.history + [event])[-300:]
        for q in self.subs:
            q.put_nowait(event)


def run(cfg: Config):
    app = FastAPI()
    bus = Bus()
    transcripts: asyncio.Queue = asyncio.Queue()
    claim_q: asyncio.Queue = asyncio.Queue()
    stop_flag = threading.Event()
    claim_ids = itertools.count(1)

    @app.get("/")
    def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/events")
    async def events():
        async def gen():
            q: asyncio.Queue = asyncio.Queue()
            for e in bus.history:
                q.put_nowait(e)
            bus.subs.add(q)
            try:
                while True:
                    yield f"data: {json.dumps(await q.get())}\n\n"
            finally:
                bus.subs.discard(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def extract_loop():
        extractor = ClaimExtractor(cfg)
        while True:
            try:
                seg = await transcripts.get()
                bus.publish(seg)
                # wake word — mic only, so audio playing "control ..." can't run commands
                instruction = control.match(seg["text"]) if seg["source"] == "mic" else None
                if instruction:
                    bus.publish(await control.handle(cfg, instruction))
                    continue
                for c in await extractor.extract(seg["text"]):
                    c["id"] = next(claim_ids)
                    bus.publish({"type": "claim", **c})
                    claim_q.put_nowait(c)
            except Exception as e:
                print(f"[extract] error: {e}", flush=True)

    async def verify_loop():
        while True:
            c = await claim_q.get()
            try:
                v = await verify(cfg, c)
            except Exception as e:
                print(f"[verify] failed: {e}", flush=True)
                v = {"verdict": "unverifiable", "confidence": 0,
                     "explanation": f"verifier error: {e}", "suggested_response": "", "evidence": []}
            bus.publish({"type": "verdict", "claim_id": c["id"], **v})

    @app.on_event("startup")
    async def startup():
        loop = asyncio.get_running_loop()
        emit = lambda seg: loop.call_soon_threadsafe(transcripts.put_nowait, seg)
        for device in cfg.audio_devices:
            frames_q: queue.Queue = queue.Queue()
            threading.Thread(target=audio.capture, args=(device, frames_q, stop_flag),
                             daemon=True).start()
            threading.Thread(target=stt.transcribe_stream,
                             args=(frames_q, emit, cfg.stt_model, device, cfg.stt_hint),
                             daemon=True).start()
        asyncio.create_task(extract_loop())
        for _ in range(cfg.verify_workers):
            asyncio.create_task(verify_loop())
        print(f"Listening on: {', '.join(cfg.audio_devices)}  "
              f"→ http://localhost:{cfg.port}", flush=True)

    @app.on_event("shutdown")
    def shutdown():
        stop_flag.set()

    uvicorn.run(app, host="127.0.0.1", port=cfg.port, log_level="warning")
