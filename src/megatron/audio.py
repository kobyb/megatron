"""Capture audio: system output via a Core Audio process tap (default), or any
input device (mic, BlackHole) via sounddevice."""
import queue
import subprocess
from pathlib import Path

import numpy as np
import sounddevice as sd

from .config import SAMPLE_RATE


def list_devices() -> str:
    return '  "system" — system output via Core Audio tap (default)\n' + str(sd.query_devices())


def _resample(mono: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE:
        return mono
    n = int(len(mono) * SAMPLE_RATE / rate)
    return np.interp(np.linspace(0, len(mono) - 1, n),
                     np.arange(len(mono)), mono).astype(np.float32)


def _systap_binary() -> Path:
    src = Path(__file__).parent / "systap.swift"
    binary = Path.home() / ".megatron" / "systap"
    if not binary.exists() or binary.stat().st_mtime < src.stat().st_mtime:
        binary.parent.mkdir(parents=True, exist_ok=True)
        print("[audio] compiling systap helper (one-time)...", flush=True)
        subprocess.run(
            ["swiftc", "-O", str(src), "-o", str(binary),
             "-framework", "CoreAudio", "-framework", "AudioToolbox",
             "-Xlinker", "-sectcreate", "-Xlinker", "__TEXT", "-Xlinker", "__info_plist",
             "-Xlinker", str(src.with_suffix(".plist"))],
            check=True)
    return binary


def _capture_system(frames_q: queue.Queue, stop_flag) -> None:
    proc = subprocess.Popen([str(_systap_binary())], stdout=subprocess.PIPE)
    try:
        header = proc.stdout.readline().split()
        rate, ch = int(header[0]), int(header[1])
        block = int(rate * 0.032) * ch * 4
        blocks = peak = 0.0
        while not stop_flag.is_set():
            if blocks == 300:  # ~10s: all-zero output = TCC permission denied
                blocks = -1
                if peak == 0.0:
                    print("[audio] tap delivers pure silence — grant your terminal "
                          "'System Audio Recording' under System Settings > Privacy & "
                          "Security > Screen & System Audio Recording, then restart",
                          flush=True)
            data = proc.stdout.read(block)
            if not data:
                err = proc.wait()
                print(f"[audio] systap exited ({err}) — grant 'System Audio Recording' "
                      f"permission in System Settings > Privacy & Security if prompted/denied",
                      flush=True)
                break
            mono = np.frombuffer(data, np.float32).reshape(-1, ch).mean(axis=1)
            if blocks >= 0:
                blocks += 1
                peak = max(peak, float(np.abs(mono).max()))
            frames_q.put(_resample(mono, rate))
    except (IndexError, ValueError):
        print("[audio] systap produced no header — is swiftc/macOS 14.4+ available?", flush=True)
    finally:
        proc.kill()
        frames_q.put(None)


def capture(device: str | None, frames_q: queue.Queue, stop_flag) -> None:
    """Blocking capture loop; pushes float32 mono chunks at 16kHz into frames_q."""
    if device in (None, "system"):
        return _capture_system(frames_q, stop_flag)
    if device == "mic":
        device = None  # sounddevice default input

    def make_stream(samplerate):
        channels = min(2, sd.query_devices(device, "input")["max_input_channels"] or 1)
        def callback(indata, _frames, _time, status):
            if status:
                print(f"[audio] {status}", flush=True)
            frames_q.put(_resample(indata.mean(axis=1).astype(np.float32), samplerate))

        return sd.InputStream(device=device, channels=channels, samplerate=samplerate,
                              blocksize=int(samplerate * 0.032), callback=callback)

    try:
        try:
            stream = make_stream(SAMPLE_RATE)
            stream.start()
        except sd.PortAudioError:
            # ponytail: device rejected 16kHz — use its default rate and linear-resample
            rate = int(sd.query_devices(device, "input")["default_samplerate"])
            stream = make_stream(rate)
            stream.start()
    except Exception as e:
        print(f"[audio] cannot open device {device!r}: {e}\n"
              f"[audio] run `megatron devices` to list inputs (or omit --device for system audio)",
              flush=True)
        frames_q.put(None)
        return

    with stream:
        while not stop_flag.is_set():
            sd.sleep(200)
    frames_q.put(None)
