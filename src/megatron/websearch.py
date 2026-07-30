"""DuckDuckGo snippet search (sync — call via executor). No key, snippets only."""


def search_web(query: str, k: int = 6, news: bool = False) -> list[dict]:
    from ddgs import DDGS

    try:
        ddgs = DDGS()
        results = []
        if news:  # fresh, dated results for time-sensitive claims (last week)
            results = [{"title": r["title"], "url": r["url"],
                        "snippet": r["body"], "date": r.get("date", "")}
                       for r in ddgs.news(query, timelimit="w", max_results=k)]
        if not results:
            results = [{"title": r["title"], "url": r["href"],
                        "snippet": r["body"], "date": ""}
                       for r in ddgs.text(query, max_results=k)]
        return results
    except Exception as e:
        print(f"[websearch] failed: {e}", flush=True)
        return []
