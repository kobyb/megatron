"""Thin Ollama client: chat (optionally schema-constrained) and embeddings."""
import json

import httpx

from .config import Config

_client = httpx.AsyncClient(timeout=180)


async def chat(cfg: Config, model: str, messages: list[dict], schema: dict | None = None,
               think: bool | str | None = None, num_ctx: int = 8192):
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": num_ctx},
    }
    if schema:
        payload["format"] = schema
    if think is not None:
        payload["think"] = think
    r = await _client.post(f"{cfg.ollama_url}/api/chat", json=payload)
    if r.status_code == 400 and think is not None:
        # ponytail: model may not support the `think` param — retry without it
        payload.pop("think")
        r = await _client.post(f"{cfg.ollama_url}/api/chat", json=payload)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    return json.loads(content) if schema else content


def embed_sync(cfg: Config, texts: list[str]) -> list[list[float]]:
    r = httpx.post(f"{cfg.ollama_url}/api/embed",
                   json={"model": cfg.embed_model, "input": texts}, timeout=120)
    r.raise_for_status()
    return r.json()["embeddings"]
