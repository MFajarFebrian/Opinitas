"""Google Play Store review scraper for the Opinitas dashboard."""

from google_play_scraper import reviews, Sort, search
import pandas as pd
from datetime import datetime
from difflib import SequenceMatcher


def _query_variants(query):
    """Generate fallback query variants for better search hits.

    Google Play's search endpoint is notoriously poor at exact keyword
    matching (e.g. 'grok' returns generic AI apps). We try a few variants
    to surface the intended app.
    """
    q = query.strip().lower()
    variants = [q]
    if "grok" in q:
        variants += ["xai", "x ai", "grok x"]
    elif "gemini" in q:
        variants += ["google gemini", "gemini google"]
    elif "chatgpt" in q or "chat gpt" in q:
        variants += ["chatgpt openai", "openai chatgpt"]
    elif "deepseek" in q:
        variants += ["deep seek", "deepseek ai"]
    elif "copilot" in q:
        variants += ["microsoft copilot", "copilot microsoft"]
    elif "perplexity" in q:
        variants += ["perplexity ai"]
    elif "claude" in q:
        variants += ["anthropic claude", "claude anthropic"]
    return variants


# Known apps that Google Play search often misses but users expect.
# Injected when the query loosely matches.
_KNOWN_AI_APPS = [
    {"title": "Google Gemini", "appId": "com.google.android.apps.bard",
     "score": 4.3, "installs": "1,000,000,000+", "developer": "Google LLC"},
    {"title": "ChatGPT", "appId": "com.openai.chatgpt",
     "score": 4.6, "installs": "500,000,000+", "developer": "OpenAI"},
    {"title": "Grok", "appId": "ai.x.grok",
     "score": 4.4, "installs": "100,000,000+", "developer": "xAI"},
    {"title": "DeepSeek - Asisten AI", "appId": "com.deepseek.chat",
     "score": 4.7, "installs": "100,000,000+", "developer": "DeepSeek"},
    {"title": "Claude by Anthropic", "appId": "com.anthropic.claude",
     "score": 4.6, "installs": "50,000,000+", "developer": "Anthropic"},
    {"title": "Microsoft Copilot", "appId": "com.microsoft.copilot",
     "score": 4.3, "installs": "50,000,000+", "developer": "Microsoft Corporation"},
    {"title": "Perplexity - Ask Anything", "appId": "ai.perplexity.app.android",
     "score": 4.5, "installs": "50,000,000+", "developer": "Perplexity AI"},
]


def _relevance_score(r, query):
    """Score how relevant an app result is to the query.

    Exact substring match in title/appId/developer scores highest,
    followed by fuzzy similarity of title to query.
    """
    q = query.strip().lower()
    title = (r.get("title") or "").lower()
    appid = (r.get("appId") or "").lower()
    dev = (r.get("developer") or "").lower()
    score = 0.0
    for field in (title, appid, dev):
        if q and q in field:
            score += 2.0
        if q and q.split()[0] in field:
            score += 0.5
    score += SequenceMatcher(None, q, title).ratio() * 0.5
    return score


def search_apps(query, n=10, lang="id", country="id"):
    """Search Google Play for apps matching the query (Indonesian store).

    Strategy:
    1. Inject known AI apps that Google Play search often misses.
    2. Query the store with the original query + fallback variants.
    3. Rerank results by relevance to the query (substring + fuzzy).
    4. Return the top ``n`` most relevant apps.

    Returns a list of dicts: [{title, appId, score, installs, developer}, ...]
    """
    q = query.strip().lower()
    seen = {}

    # Inject known AI apps first (matched by substring in title/appId/developer)
    for known in _KNOWN_AI_APPS:
        ktext = (known["title"] + known["appId"] + known["developer"]).lower()
        if q in ktext or any(w in ktext for w in q.split() if len(w) >= 3):
            seen[known["appId"]] = known

    # Query Google Play with variants
    for v in _query_variants(q):
        try:
            results = search(v, n_hits=n, lang=lang, country=country)
            for r in results:
                appid = r.get("appId") or ""
                if not appid or appid in seen:
                    continue
                seen[appid] = {
                    "title": r.get("title", ""),
                    "appId": appid,
                    "score": r.get("score", 0),
                    "installs": r.get("installs", ""),
                    "developer": r.get("developer", ""),
                }
        except Exception:
            continue

    ranked = sorted(seen.values(),
                    key=lambda r: _relevance_score(r, query),
                    reverse=True)
    return ranked[:n]


def scrape_reviews(app_id, count=500, lang="id", country="id"):
    """Scrape ``count`` newest reviews for ``app_id`` from Google Play.

    Returns (DataFrame, error_message). On success error_message is None.
    """
    try:
        result, _ = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=count,
        )
        if not result:
            return pd.DataFrame(), "Tidak ada ulasan ditemukan."

        df = pd.DataFrame(result)
        # Keep only columns we need
        keep = ["reviewId", "userName", "content", "score", "at",
                "thumbsUpCount", "appVersion"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df["at"] = pd.to_datetime(df["at"], errors="coerce")
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Scraping gagal: {e}"
