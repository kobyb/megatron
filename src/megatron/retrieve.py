"""Hybrid (vector + full-text) search over the indexed docs in LanceDB."""
import lancedb

from .config import Config
from .llm import embed_sync


def get_table(cfg: Config):
    db = lancedb.connect(cfg.data_dir / "lance")
    return db.open_table("chunks") if "chunks" in db.table_names() else None


def search(cfg: Config, query: str, k: int | None = None) -> list[dict]:
    """Sync (call via executor from async code). Returns [{text, source}]."""
    tbl = get_table(cfg)
    if tbl is None:
        return []
    vec = embed_sync(cfg, [query])[0]
    hits = (tbl.search(query_type="hybrid")
            .vector(vec).text(query)
            .limit(k or cfg.retrieve_k).to_list())
    return [{"text": h["text"], "source": h["source"]} for h in hits]
