"""
AI Daily Digest - Configuration
All configurable values in one place.
"""

import os
from datetime import datetime, timedelta, timezone

# ============================================================
# Time window — yesterday in Beijing time
# ============================================================
BJT = timezone(timedelta(hours=8))
NOW_BJT = datetime.now(BJT)
YESTERDAY_START = (NOW_BJT - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
YESTERDAY_END = YESTERDAY_START.replace(hour=23, minute=59, second=59)

# ============================================================
# LLM API (OpenAI 兼容格式)
# ============================================================
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://eoeo.xyz/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 16000
CHUNK_SIZE = 20           # items per LLM call (avoid output truncation)

# ============================================================
# QQ Mail SMTP
# ============================================================
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.qq.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))  # SSL
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")  # QQ 邮箱授权码, 非 QQ 密码
EMAIL_TO = os.environ.get("EMAIL_TO", "")
EMAIL_SUBJECT_PREFIX = "🤖 AI 日报 — "

# ============================================================
# Hacker News (Algolia Search API)
# ============================================================
HN_API_BASE = "https://hn.algolia.com/api/v1"
HN_SEARCH_QUERIES = [
    "AI", "LLM", "GPT", "Claude", "Gemini",
    "large language model", "machine learning",
    "OpenAI", "Anthropic",
    "deep learning", "neural network", "AGI",
    "fine-tuning", "RAG", "embedding", "RLHF",
]
HN_MIN_POINTS = 15
HN_RESULTS_PER_QUERY = 20

# ============================================================
# Reddit (public JSON API)
# ============================================================
REDDIT_SUBREDDITS = [
    "MachineLearning",
    "artificial",
    "LocalLLaMA",
]
REDDIT_SORT = "top"
REDDIT_TIME_FILTER = "day"
REDDIT_LIMIT = 25
REDDIT_MIN_SCORE = 20
REDDIT_SKIP_DOMAINS = {"i.redd.it", "v.redd.it", "youtube.com", "youtu.be"}

# ============================================================
# RSS Feeds
# ============================================================
RSS_FEEDS = {
    "OpenAI Blog": "https://openai.com/blog/rss.xml",
    "Anthropic Blog": "https://www.anthropic.com/rss.xml",
    "Google AI Blog": "https://blog.google/technology/ai/rss/",
    "DeepMind Blog": "https://deepmind.google/blog/rss.xml",
    "Hugging Face Blog": "https://huggingface.co/blog/feed.xml",
    "MIT Tech Review AI": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
    "The Verge AI": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "VentureBeat AI": "https://venturebeat.com/category/ai/feed/",
    "Ars Technica AI": "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "TLDR AI": "https://tldr.tech/ai/rss",
    "MarkTechBlog": "https://www.marktechpost.com/feed/",
    "Synced Review": "https://syncedreview.com/feed/",
}

# ============================================================
# Content categories (used in Claude prompt + email template)
# ============================================================
CATEGORIES = [
    "📄 研究论文 (Research Papers)",
    "🚀 产品发布 (Product Launches)",
    "🏢 行业动态 (Industry News)",
    "💻 开源项目 (Open Source)",
    "🛠️ 工具与框架 (Tools & Frameworks)",
    "📊 数据集与基准 (Datasets & Benchmarks)",
    "🧠 观点与讨论 (Opinions & Discussions)",
]
