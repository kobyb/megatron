"""Verify a claim against local index + optional web snippets using the big model."""
import asyncio
import time

from .config import Config
from .llm import chat
from .retrieve import search
from .websearch import search_web

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string",
                    "enum": ["supported", "refuted", "unverifiable", "needs-context"]},
        "confidence": {"type": "number"},
        "explanation": {"type": "string"},
        "suggested_response": {"type": "string"},
    },
    "required": ["verdict", "confidence", "explanation", "suggested_response"],
}

PROMPT = """You are a rigorous real-time fact-checker. Assess the claim using the provided \
evidence and your own knowledge. Evidence from LOCAL INDEX is the user's private \
docs/code — treat it as authoritative for company/code claims. WEB snippets may be \
noisy. If evidence conflicts with your prior knowledge, prefer the evidence.
Verdicts: supported / refuted / unverifiable (no way to check) / needs-context (depends \
on missing specifics). confidence: 0-1. explanation: <=2 sentences, cite which evidence.
suggested_response: 1-3 sentences the listener could actually say out loud in reply — \
correcting, confirming, or asking the right clarifying question. Write \
suggested_response in the same language as the SOURCE UTTERANCE."""


async def verify(cfg: Config, claim: dict) -> dict:
    loop = asyncio.get_running_loop()
    local = await loop.run_in_executor(None, search, cfg, claim["claim"])
    web = []
    if cfg.web_search and claim.get("time_sensitive"):
        web = await loop.run_in_executor(
            None, lambda: search_web(claim["claim"], news=True))

    parts = [f"CURRENT DATE/TIME: {time.strftime('%A, %Y-%m-%d %H:%M')}",
             f"CLAIM: {claim['claim']}", f"CATEGORY: {claim['category']}",
             f"SOURCE UTTERANCE: {claim.get('source_text', '')}"]
    if local:
        parts.append("LOCAL INDEX EVIDENCE:")
        parts += [f"[L{i}] ({h['source']}) {h['text'][:600]}" for i, h in enumerate(local, 1)]
    if web:
        parts.append("WEB EVIDENCE:")
        parts += [f"[W{i}] ({h['url']}) {h.get('date', '')} {h['title']}: {h['snippet']}"
                  for i, h in enumerate(web, 1)]
    if not local and not web:
        parts.append("NO EXTERNAL EVIDENCE — use your own knowledge; be honest about limits.")

    result = await chat(cfg, cfg.verify_model,
                        [{"role": "system", "content": PROMPT},
                         {"role": "user", "content": "\n".join(parts)}],
                        schema=SCHEMA, think="low", num_ctx=16384)
    result["evidence"] = ([{"kind": "local", "ref": h["source"], "text": h["text"][:300]} for h in local]
                          + [{"kind": "web", "ref": h["url"], "text": h["snippet"]} for h in web])
    return result
