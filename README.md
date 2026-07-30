# Megatron

Real-time, fully local fact-checker for whatever plays through your Mac's speakers (or mic).
Transcribes (Hebrew + English), extracts check-worthy claims, verifies them against your
indexed repos/docs + local model knowledge (+ optional web snippets for fresh news), and
suggests what you could say back. Live dashboard at http://localhost:8765.

Models (all via Ollama + MLX, nothing leaves the machine except claim text on web search):
whisper-large-v3-turbo (STT) · qwen3.6 (claim extraction) · gpt-oss:120b (verification) · bge-m3 (embeddings).

## One-time setup

1. **Deps:** `uv sync` and `ollama pull bge-m3` (needs Xcode CLT for `swiftc` — the
   system-audio helper compiles itself on first run)
2. **Permission:** on first `megatron run`, approve the **System Audio Recording**
   prompt (or add your terminal under System Settings → Privacy & Security →
   Screen & System Audio Recording). Denied permission = silent capture; megatron
   warns after 10s of pure silence.
3. **Keep models hot** (optional but recommended): quit the Ollama menu-bar app, then
   ```
   OLLAMA_MAX_LOADED_MODELS=3 OLLAMA_KEEP_ALIVE=-1 ollama serve
   ```

## Usage

```bash
uv run megatron index ~/dev/my-repo ~/Documents/notes   # index your evidence (repos, docs) — rerun after big changes
uv run megatron run                                # listen to system audio (Core Audio tap) + mic in parallel
uv run megatron run --no-mic                       # system audio only
uv run megatron run --device mic                   # mic only (any input device name works too)
uv run megatron devices                            # list audio sources
uv run megatron run --no-web                       # strictly offline verification
```

**Voice commands:** say "control, ..." into the mic and the rest is executed as a shell
command (e.g. "control, open terminal, go to megatron folder, run ls"). Mic-source only —
audio playing the word "control" can never trigger it. Ran commands + output appear as
purple cards in the dashboard.

Transcript lines are tagged 🔊 (system) / 🎤 (mic). On open speakers the mic re-hears
system audio — duplicate transcripts are expected; duplicate claims are deduped.
Open http://localhost:8765 — left pane live transcript, right pane claim cards
(green supported / red refuted / amber unverifiable / blue needs-context), each with
evidence citations and a suggested response.

Config via env: `MEGATRON_DEVICES` (comma-separated sources), `MEGATRON_VERIFY_MODEL`,
`MEGATRON_EXTRACT_MODEL`, `MEGATRON_STT_MODEL`, `MEGATRON_DATA_DIR`, `MEGATRON_PORT`,
`MEGATRON_WEB_SEARCH=0`, `MEGATRON_STT_HINT` (vocabulary whisper should recognize —
your project/product names), `MEGATRON_PROJECTS` (comma-separated ~/dev/ project names
for voice commands).

Hebrew accuracy low? Swap STT: `MEGATRON_STT_MODEL=ivrit-ai/whisper-large-v3-turbo-ct2` →
use an MLX-converted ivrit.ai model. RAM tight? `MEGATRON_EXTRACT_MODEL=gemma4:latest`.
