"""Index repos/docs into LanceDB: walk, chunk, embed (Ollama), upsert by file hash."""
import hashlib
import json
from pathlib import Path

import lancedb
import pyarrow as pa

from .config import Config
from .llm import embed_sync

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
             ".next", ".nx", "coverage", ".cache", "target"}
SKIP_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "uv.lock", "poetry.lock"}
EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".mdx", ".txt", ".json", ".yaml",
        ".yml", ".toml", ".sql", ".graphql", ".sh", ".go", ".rs", ".java", ".rb",
        ".html", ".css", ".scss", ".prisma"}
CHUNK, OVERLAP = 1500, 200


def _chunks(text: str):
    step = CHUNK - OVERLAP
    for i in range(0, max(len(text) - OVERLAP, 1), step):
        yield text[i:i + CHUNK]


def _files(paths: list[Path]):
    for root in paths:
        if root.is_file():
            yield root
            continue
        for p in sorted(root.rglob("*")):
            if (p.is_file() and p.suffix.lower() in EXTS and p.name not in SKIP_FILES
                    and not p.name.startswith(".")
                    and not any(d in SKIP_DIRS or d.startswith(".") for d in p.parts[:-1])
                    and p.stat().st_size < 2_000_000):
                yield p


def index(cfg: Config, paths: list[str]) -> None:
    db = lancedb.connect(cfg.data_dir / "lance")
    schema = pa.schema([("text", pa.string()), ("source", pa.string()),
                        ("vector", pa.list_(pa.float32(), cfg.embed_dim))])
    tbl = (db.open_table("chunks") if "chunks" in db.table_names()
           else db.create_table("chunks", schema=schema))

    manifest_path = cfg.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    n_files = n_chunks = 0
    for f in _files([Path(p).expanduser().resolve() for p in paths]):
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue
        digest = hashlib.sha1(text.encode()).hexdigest()
        key = str(f)
        if manifest.get(key) == digest:
            continue
        tbl.delete(f"source = '{key}'")
        chunks = list(_chunks(text))
        vecs = []
        for i in range(0, len(chunks), 32):  # batch embeds
            vecs += embed_sync(cfg, chunks[i:i + 32])
        tbl.add([{"text": c, "source": key, "vector": v} for c, v in zip(chunks, vecs)])
        manifest[key] = digest
        n_files += 1
        n_chunks += len(chunks)
        print(f"  indexed {key} ({len(chunks)} chunks)", flush=True)

    manifest_path.write_text(json.dumps(manifest, indent=0))
    if n_files:
        tbl.create_fts_index("text", replace=True, use_tantivy=False)
    print(f"Done: {n_files} files, {n_chunks} chunks. Total rows: {tbl.count_rows()}")


if __name__ == "__main__":  # self-check: index a planted fact, retrieve it
    import tempfile

    from .retrieve import search

    with tempfile.TemporaryDirectory() as td:
        doc = Path(td) / "facts.md"
        doc.write_text("The megatron pipeline uses LanceDB hybrid search for evidence.")
        cfg = Config(data_dir=Path(td) / "data")
        index(cfg, [td])
        hits = search(cfg, "what does megatron use for evidence search?")
        assert hits and "LanceDB" in hits[0]["text"], hits
        print("self-check OK:", hits[0]["text"][:60])
