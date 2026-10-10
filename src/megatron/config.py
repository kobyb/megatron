import os
from dataclasses import dataclass, field
from pathlib import Path


# ponytail: 6-line .env loader (cwd only) instead of a python-dotenv dep
if Path(".env").exists():
    for _line in Path(".env").read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))


def _env(key: str, default: str) -> str:
    return os.environ.get(f"MEGATRON_{key}", default)


@dataclass
class Config:
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    stt_model: str = field(default_factory=lambda: _env("STT_MODEL", "mlx-community/whisper-large-v3-turbo"))
    # vocabulary bias for whisper — bare glossary only, NEVER command-like phrases
    # (whisper can hallucinate the hint verbatim as a transcript on noise segments).
    # Add your project/product names via MEGATRON_STT_HINT.
    stt_hint: str = field(default_factory=lambda: _env(
        "STT_HINT", "Glossary: control, megatron, iTerm."))
    # comma-separated project names under ~/dev/, used by voice commands
    projects: str = field(default_factory=lambda: _env("PROJECTS", "megatron"))
    extract_model: str = field(default_factory=lambda: _env("EXTRACT_MODEL", "qwen3.6:latest"))
    verify_model: str = field(default_factory=lambda: _env("VERIFY_MODEL", "gpt-oss:120b"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "bge-m3"))
    embed_dim: int = 1024
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(Path.home() / ".megatron"))))
    # audio sources captured in parallel: "system" (Core Audio tap), "mic"
    # (default input), or any input device name
    audio_devices: list = field(default_factory=lambda: _env("DEVICES", "system,mic").split(","))
    port: int = field(default_factory=lambda: int(_env("PORT", "8765")))
    web_search: bool = field(default_factory=lambda: _env("WEB_SEARCH", "1") == "1")
    retrieve_k: int = 6
    verify_workers: int = 2

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)


SAMPLE_RATE = 16000
