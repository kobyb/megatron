"""VAD-segmented speech-to-text: silero-vad cuts utterances, mlx-whisper transcribes."""
import queue
import re
import threading
import traceback

import numpy as np

from .config import SAMPLE_RATE

_whisper_lock = threading.Lock()  # MLX graph eval is not thread-safe across sources

WIN = 512  # silero window @ 16kHz
MAX_UTTERANCE_S = 15
PAD_PRE, PAD_POST = 1600, 800  # 100ms / 50ms
# whisper hallucinates these on near-silence
_JUNK = ("subtitles by", "subs by", "thank you for watching", "thanks for watching")


def transcribe_stream(frames_q: queue.Queue, emit, model_repo: str, source: str = "system",
                      hint: str = "") -> None:
    """Blocking worker: read 16kHz frames, emit(text, t0, t1) per utterance."""
    try:
        _transcribe_stream(frames_q, emit, model_repo, source, hint)
    except Exception:
        print(f"[stt] {source} worker CRASHED:\n{traceback.format_exc()}", flush=True)


def _transcribe_stream(frames_q: queue.Queue, emit, model_repo: str, source: str,
                       hint: str = "") -> None:
    import mlx_whisper
    import torch
    from silero_vad import VADIterator, load_silero_vad

    with _whisper_lock:  # serialize model downloads/loads across source threads
        vad = VADIterator(load_silero_vad(), sampling_rate=SAMPLE_RATE,
                          min_silence_duration_ms=500)
    print(f"[stt] {source}: ready", flush=True)

    def norm(s):
        return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())

    hint_tokens = set(norm(hint).split())

    def transcribe(seg: np.ndarray, t0: float, t1: float):
        with _whisper_lock:
            result = mlx_whisper.transcribe(seg, path_or_hf_repo=model_repo,
                                            initial_prompt=hint or None)
        text = result["text"].strip()
        print(f"[stt] {source}: {t1 - t0:.1f}s utterance → {text[:80]!r}", flush=True)
        if len(text) < 4 or any(j in text.lower() for j in _JUNK):
            return
        # whisper echoes hint fragments on noise: drop utterances made (almost)
        # entirely of hint vocabulary — real speech has words beyond the glossary
        toks = norm(text).split()
        if hint_tokens and toks and sum(t in hint_tokens for t in toks) / len(toks) >= 0.8:
            print(f"[stt] {source}: dropped hint hallucination {text[:60]!r}", flush=True)
            return
        emit({"type": "transcript", "text": text, "lang": result.get("language", ""),
              "source": source, "t0": round(t0, 1), "t1": round(t1, 1)})

    audio = np.empty(0, np.float32)
    base = 0      # absolute sample index of audio[0]
    fed = 0       # absolute samples fed to VAD
    start = None  # absolute sample where current speech began
    peak, last_meter = 0.0, 0

    while True:
        chunk = frames_q.get()
        if chunk is None:
            return
        audio = np.concatenate([audio, chunk])
        peak = max(peak, float(np.abs(chunk).max()) if len(chunk) else 0.0)
        if fed - last_meter > 10 * SAMPLE_RATE:  # level meter every ~10s
            print(f"[stt] {source}: level peak={peak:.3f} "
                  f"{'(silence — check permissions/routing)' if peak < 0.01 else ''}", flush=True)
            peak, last_meter = 0.0, fed

        while base + len(audio) - fed >= WIN:
            window = audio[fed - base: fed - base + WIN]
            r = vad(torch.from_numpy(window))
            fed += WIN
            if r and "start" in r:
                start = r["start"]
            ended = r is not None and "end" in r
            if start is not None and (ended or fed - start > MAX_UTTERANCE_S * SAMPLE_RATE):
                end = r["end"] if ended else fed
                seg = audio[max(0, start - base - PAD_PRE): end - base + PAD_POST]
                transcribe(seg, start / SAMPLE_RATE, end / SAMPLE_RATE)
                start = None if ended else fed  # force-cut: speech continues

        if start is None and len(audio) > 2 * SAMPLE_RATE:  # trim silence tail, keep 1s
            cut = len(audio) - SAMPLE_RATE
            audio = audio[cut:]
            base += cut


if __name__ == "__main__":  # self-check: python -m megatron.stt <file.wav>
    import sys

    import mlx_whisper

    print(mlx_whisper.transcribe(
        sys.argv[1], path_or_hf_repo="mlx-community/whisper-large-v3-turbo")["text"])
