"""
AI Daily Digest - Data Collectors
Collects AI-related content from Hacker News, Reddit, and RSS feeds.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List
from urllib.parse import quote

import feedparser
import requests
from bs4 import BeautifulSoup

from config import (
    BJT,
    HN_API_BASE,
    HN_MIN_POINTS,
    HN_RESULTS_PER_QUERY,
    HN_SEARCH_QUERIES,
    REDDIT_LIMIT,
    REDDIT_MIN_SCORE,
    REDDIT_SKIP_DOMAINS,
    REDDIT_SORT,
    REDDIT_SUBREDDITS,
    REDDIT_TIME_FILTER,
    RSS_FEEDS,
    YESTERDAY_END,
    YESTERDAY_START,
)

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "ai-daily-digest/1.0 (github.com/your-repo)"}


@dataclass
class RawItem:
    title: str
    url: str
    source: str  # "HN", "Reddit", "RSS"
    score: int = 0
    description: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(BJT))
    subreddit: str = ""

    def __hash__(self):
        return hash((self.title, self.url))


def fetch_with_retry(url: str, max_retries: int = 3, backoff: float = 2.0,
                     timeout: int = 15, **kwargs) -> requests.Response:
    """Fetch URL with exponential backoff on failure."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout, headers=HEADERS, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt == max_retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            logger.warning(f"Retry {attempt+1}/{max_retries} for {url} in {wait}s: {e}")
            time.sleep(wait)
    raise last_exc  # type: ignore


def _strip_html(text: str) -> str:
    """Strip HTML tags from text."""
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)


# ============================================================
# Hacker News Collector (Algolia Search API)
# ============================================================

def collect_hackernews() -> List[RawItem]:
    """Collect AI-related stories from Hacker News via Algolia API."""
    start_unix = int(YESTERDAY_START.timestamp())
    end_unix = int(YESTERDAY_END.timestamp())
    logger.info(f"HN: searching for stories from {YESTERDAY_START.date()} "
                f"(unix {start_unix} - {end_unix})")

    seen_ids = set()
    items: List[RawItem] = []

    for query in HN_SEARCH_QUERIES:
        try:
            encoded_q = quote(query)
            url = (
                f"{HN_API_BASE}/search_by_date"
                f"?query={encoded_q}"
                f"&tags=story"
                f"&numericFilters=created_at_i>{start_unix},"
                f"created_at_i<{end_unix},"
                f"points>={HN_MIN_POINTS}"
                f"&hitsPerPage={HN_RESULTS_PER_QUERY}"
            )
            resp = fetch_with_retry(url, max_retries=2)
            data = resp.json()

            for hit in data.get("hits", []):
                story_id = str(hit.get("objectID", ""))
                if story_id in seen_ids:
                    continue
                seen_ids.add(story_id)

                title = hit.get("title") or ""
                if not title:
                    continue

                story_url = hit.get("url") or f"https://news.ycombinator.com/item?id={story_id}"
                points = hit.get("points") or 0

                items.append(RawItem(
                    title=title,
                    url=story_url,
                    source="HN",
                    score=int(points),
                    description="",
                    timestamp=datetime.fromtimestamp(
                        int(hit.get("created_at_i", 0)), tz=timezone.utc
                    ).astimezone(BJT),
                ))

        except Exception as e:
            logger.warning(f"HN query '{query}' failed: {e}")
            continue

    logger.info(f"HN: collected {len(items)} unique stories")
    return items


# ============================================================
# Reddit Collector (Public JSON API)
# ============================================================

def collect_reddit() -> List[RawItem]:
    """Collect top AI-related posts from Reddit subreddits."""
    items: List[RawItem] = []
    seen_urls = set()

    for sub in REDDIT_SUBREDDITS:
        try:
            url = f"https://www.reddit.com/r/{sub}/{REDDIT_SORT}/.json?t={REDDIT_TIME_FILTER}&limit={REDDIT_LIMIT}"
            resp = fetch_with_retry(url, max_retries=2)
            data = resp.json()

            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                score = post.get("score", 0)
                if score < REDDIT_MIN_SCORE:
                    continue

                domain = post.get("domain", "")
                if domain in REDDIT_SKIP_DOMAINS:
                    continue

                # Skip image/video posts and memes
                flair = (post.get("link_flair_text") or "").lower()
                if any(skip in flair for skip in ["meme", "shitpost", "off-topic"]):
                    continue

                title = post.get("title", "")
                post_url = post.get("url", "")

                # For self posts, prefer the Reddit permalink
                if post.get("is_self") and post.get("selftext"):
                    post_url = post.get("permalink", "")
                    if post_url.startswith("/"):
                        post_url = f"https://www.reddit.com{post_url}"
                    desc = _strip_html(post.get("selftext", ""))[:300]
                else:
                    desc = _strip_html(post.get("selftext", ""))[:300] if post.get("selftext") else ""

                if post_url in seen_urls:
                    continue
                seen_urls.add(post_url)

                items.append(RawItem(
                    title=title,
                    url=post_url,
                    source="Reddit",
                    score=int(score),
                    description=desc,
                    subreddit=sub,
                    timestamp=datetime.fromtimestamp(
                        int(post.get("created_utc", 0)), tz=timezone.utc
                    ).astimezone(BJT),
                ))

            # Be polite to Reddit's API
            time.sleep(2)

        except Exception as e:
            logger.warning(f"Reddit r/{sub} failed: {e}")
            continue

    logger.info(f"Reddit: collected {len(items)} posts from {len(REDDIT_SUBREDDITS)} subreddits")
    return items


# ============================================================
# RSS Collector
# ============================================================

def collect_rss() -> List[RawItem]:
    """Collect AI news from RSS feeds, filtering to yesterday's entries."""
    items: List[RawItem] = []

    for feed_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url, request_headers=HEADERS)

            for entry in feed.entries:
                # Parse publish time
                pub_time = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        pub_time = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).astimezone(BJT)
                    except Exception:
                        pass
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    try:
                        pub_time = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc).astimezone(BJT)
                    except Exception:
                        pass

                # Filter to yesterday (if we have a date)
                if pub_time:
                    if not (YESTERDAY_START <= pub_time <= YESTERDAY_END):
                        continue

                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                if not title or not link:
                    continue

                # Extract description
                summary = ""
                if hasattr(entry, "content") and entry.content:
                    summary = _strip_html(entry.content[0].get("value", ""))[:300]
                elif hasattr(entry, "summary"):
                    summary = _strip_html(entry.summary)[:300]

                items.append(RawItem(
                    title=title,
                    url=link,
                    source="RSS",
                    score=0,
                    description=summary,
                    timestamp=pub_time or YESTERDAY_START,
                ))

        except Exception as e:
            logger.warning(f"RSS feed '{feed_name}' failed: {e}")
            continue

    logger.info(f"RSS: collected {len(items)} entries from {len(RSS_FEEDS)} feeds")
    return items


def collect_all() -> List[RawItem]:
    """Run all collectors with graceful degradation. Returns deduplicated items."""
    all_items: List[RawItem] = []
    sources = [
        ("HN", collect_hackernews),
        ("Reddit", collect_reddit),
        ("RSS", collect_rss),
    ]

    for name, collector in sources:
        try:
            items = collector()
            all_items.extend(items)
        except Exception as e:
            logger.error(f"{name} collector FAILED: {e}", exc_info=True)

    # Global dedup by URL
    seen_urls = set()
    deduped: List[RawItem] = []
    for item in all_items:
        if item.url not in seen_urls:
            seen_urls.add(item.url)
            deduped.append(item)

    logger.info(f"Total: {len(deduped)} unique items (from {len(all_items)} raw)")
    return deduped
