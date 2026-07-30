"""Voice commands: 'control, <instruction>' (mic only) → shell command → run it."""
import asyncio
import os
import re
import subprocess

from .config import Config
from .llm import chat

WAKE = re.compile(r"^\W*(?:control|ctrl|kontrol)\b\W*(.+)", re.IGNORECASE | re.DOTALL)

SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string"}, "description": {"type": "string"}},
    "required": ["command", "description"],
}

def _prompt(cfg: Config) -> str:
    return f"""Convert a spoken macOS instruction (may be Hebrew or English, transcribed \
speech — tolerate transcription errors) into ONE zsh command. Home dir: {os.path.expanduser('~')}. \
Dev projects live in ~/dev/ ({cfg.projects}).
- URLs/files/folders: just `open <target>` (opens in default browser/app).
- "terminal" always means iTerm. To open a terminal and run things in it: osascript \
-e 'tell application "iTerm" to activate' -e 'tell application "iTerm" to create window \
with default profile' -e 'tell application "iTerm" to tell current session of current \
window to write text "cd ~/dev/megatron && ls"'. Never use Terminal.app.
- Specific browser: open -a "Google Chrome" <url>; apps: open -a "App Name"
- The megatron dashboard is http://localhost:8765
- Chain multiple spoken steps into one command.
description: <=8 words, what it does."""


def match(text: str) -> str | None:
    m = WAKE.match(text)
    return m.group(1).strip() if m else None


async def handle(cfg: Config, instruction: str) -> dict:
    try:
        result = await chat(cfg, cfg.extract_model,
                            [{"role": "system", "content": _prompt(cfg)},
                             {"role": "user", "content": instruction}],
                            schema=SCHEMA, think=False)
    except Exception as e:
        return {"type": "command", "instruction": instruction, "command": "",
                "description": "", "output": f"command generation failed: {e}", "ok": False}

    def run():
        return subprocess.run(["zsh", "-lc", result["command"]],
                              capture_output=True, text=True, timeout=60)
    try:
        proc = await asyncio.get_running_loop().run_in_executor(None, run)
        output = (proc.stdout + proc.stderr).strip()[-2000:]
        ok = proc.returncode == 0
    except Exception as e:
        output, ok = str(e), False
    return {"type": "command", "instruction": instruction, "command": result["command"],
            "description": result["description"], "output": output, "ok": ok}
