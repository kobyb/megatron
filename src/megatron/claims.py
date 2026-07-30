"""Extract check-worthy factual claims from the rolling transcript (fast model)."""
import time

from .config import Config
from .llm import chat

SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "category": {"type": "string", "enum": ["generic", "company", "code"]},
                    "time_sensitive": {"type": "boolean"},
                },
                "required": ["claim", "category", "time_sensitive"],
            },
        }
    },
    "required": ["claims"],
}

PROMPT = """You monitor a live audio transcript (may be Hebrew or English) and extract \
check-worthy factual claims: statements that are verifiably true or false and worth checking.
Skip opinions, questions, small talk, hedged speculation, and anything already extracted.
Rewrite each claim as a self-contained English sentence (resolve pronouns from context, \
and convert relative time like "last night"/"yesterday" into absolute dates using the \
current date given below).
Categories: "company" = about the speaker's company/product/team, "code" = about a codebase \
or technical implementation, "generic" = everything else (news, science, history, stats).
time_sensitive = true if verifying it needs current information (news, prices, recent events).
Return {"claims": []} if there is nothing check-worthy — most utterances have nothing."""


class ClaimExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.window: list[tuple[float, str]] = []  # (wall_time, text)
        self.seen: list[str] = []

    async def extract(self, utterance: str) -> list[dict]:
        now = time.time()
        self.window = [(t, x) for t, x in self.window if now - t < 45]
        context = " ".join(x for _, x in self.window)
        self.window.append((now, utterance))

        user = (f"Current date/time: {time.strftime('%A, %Y-%m-%d %H:%M')}\n"
                f"Context (already processed): {context or '(none)'}\n"
                f"Already extracted claims: {self.seen[-15:] or '(none)'}\n"
                f"New utterance: {utterance}")
        try:
            result = await chat(self.cfg, self.cfg.extract_model,
                                [{"role": "system", "content": PROMPT},
                                 {"role": "user", "content": user}],
                                schema=SCHEMA, think=False)
        except Exception as e:
            print(f"[claims] extraction failed: {e}", flush=True)
            return []

        fresh = []
        for c in result.get("claims", []):
            key = c["claim"].strip().lower()
            if key and key not in (s.lower() for s in self.seen):
                self.seen.append(c["claim"].strip())
                c["source_text"] = utterance
                fresh.append(c)
        return fresh
