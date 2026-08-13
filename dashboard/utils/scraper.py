"""Google Play Store review scraper for the Opinitas dashboard."""

from google_play_scraper import reviews, Sort, search
import pandas as pd
from datetime import datetime


def search_apps(query, n=10):
    """Search Google Play for apps matching the query.

    Returns a list of dicts: [{title, appId, score, installs, developer}, ...]
    """
    try:
        results = search(query, n_hits=n)
        out = []
        for r in results:
            out.append({
                "title": r.get("title", ""),
                "appId": r.get("appId", ""),
                "score": r.get("score", 0),
                "installs": r.get("installs", ""),
                "developer": r.get("developer", ""),
            })
        return out
    except Exception as e:
        print(f"search_apps error: {e}")
        return []


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
