"""
AI Daily Digest - Content Curator
Uses LLM API (OpenAI-compatible) to filter, categorize, and summarize AI news.
"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List

from openai import OpenAI

from collectors import RawItem
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, CATEGORIES, MAX_TOKENS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert AI news curator. Your task is to process a list of AI-related news items collected from Hacker News, Reddit, and RSS feeds.

## Your Responsibilities

1. **Filter**: Remove irrelevant, low-quality, or duplicate content
2. **Categorize**: Assign each item to exactly one category
3. **Summarize**: Write a concise Chinese summary (2-3 sentences) for each item
4. **Sort**: Order items within each category by importance (highest score first)

## Output Format

Respond with valid JSON in this exact structure:

```json
{
  "categories": {
    "📄 研究论文 (Research Papers)": [
      {
        "title": "Original English Title",
        "url": "https://...",
        "summary_zh": "中文摘要，2-3句话",
        "source": "HN",
        "score": 42
      }
    ],
    "🚀 产品发布 (Product Launches)": [],
    ...
  },
  "stats": {
    "total_processed": 80,
    "total_kept": 45,
    "by_source": {"HN": 20, "Reddit": 15, "RSS": 10}
  }
}
```

## Categories

- 📄 研究论文 (Research Papers): Academic papers, technical reports, preprints (arXiv, papers with code)
- 🚀 产品发布 (Product Launches): New model releases, product updates, API launches, demos
- 🏢 行业动态 (Industry News): Funding rounds, acquisitions, partnerships, policy changes
- 💻 开源项目 (Open Source): GitHub repos, open-source tools, model weights, frameworks
- 🛠️ 工具与框架 (Tools & Frameworks): Developer tools, libraries, platforms, infrastructure
- 📊 数据集与基准 (Datasets & Benchmarks): New datasets, benchmark results, leaderboards, evaluations
- 🧠 观点与讨论 (Opinions & Discussions): Opinion pieces, interviews, debates, trend analysis

## Filtering Rules

**Keep** items that are:
- Directly related to AI/ML research, products, or industry
- High-quality content with substantive information
- From credible sources (academic, official blogs, reputable news)

**Remove** items that are:
- General tech news that only incidentally mentions AI
- Job postings, hiring threads, career advice
- Pure memes, jokes, or low-effort posts
- Advertisements, promotions, or sponsored content
- Duplicates (same event covered multiple times — keep the most authoritative source)
- Off-topic discussions (e.g., programming in general, non-AI tech)

## Summary Style

Write summaries that are:
- **Concise**: 2-3 sentences, under 100 Chinese characters
- **Information-dense**: Include key facts (who, what, why it matters)
- **Objective**: Avoid subjective judgments or hype
- **Contextual**: Explain significance for someone tracking AI developments

Examples:
✅ "OpenAI 发布 GPT-4o，支持文本、语音、视觉多模态输入输出。新模型推理速度提升 2 倍，API 价格降低 50%，标志着多模态模型进入实用化阶段。"
✅ "Meta 开源 Llama 3.1 405B 模型，参数量超越 GPT-4。在多项基准测试中表现接近闭源模型，为开源社区提供最强基座模型。"
❌ "OpenAI 发布了新模型，看起来很厉害。" (too vague, lacks specifics)
❌ "这篇论文提出了一种新的 Transformer 变体，改进了注意力机制。" (too generic, no specifics)

## Important Notes

- Empty categories are fine — don't force items into wrong categories
- If an item could fit multiple categories, pick the most specific one
- Preserve the original English title exactly as provided
- For items with high scores (>100), they are likely significant — give them more detailed summaries
- If you're unsure about relevance, err on the side of exclusion
- IMPORTANT: Respond ONLY with the JSON object, no other text before or after"""


def format_items_for_prompt(items: List[RawItem]) -> str:
    """Format RawItem list into a compact text representation."""
    lines = []
    for i, item in enumerate(items, 1):
        source_info = item.source
        if item.subreddit:
            source_info = f"{item.source}/r/{item.subreddit}"

        line = f"{i}. [{source_info}|score:{item.score}] "
        line += f"Title: {item.title} | "
        line += f"URL: {item.url}"

        if item.description:
            desc = item.description[:200].replace("\n", " ")
            line += f" | Desc: {desc}"

        lines.append(line)

    return "\n".join(lines)


def curate(items: List[RawItem]) -> Dict[str, Any]:
    """
    Use LLM to filter, categorize, and summarize items.

    Returns: {
        "categories": {"📄 研究论文 (Research Papers)": [...], ...},
        "stats": {"total_processed": N, "total_kept": M, ...}
    }
    """
    if not items:
        logger.warning("No items to curate")
        return {"categories": {cat: [] for cat in CATEGORIES}, "stats": {}}

    if not LLM_API_KEY:
        logger.error("LLM_API_KEY not set, falling back to basic grouping")
        return fallback_curation(items)

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    items_text = format_items_for_prompt(items)
    user_prompt = f"""Please curate the following {len(items)} AI news items collected from yesterday.

Apply the filtering rules carefully — remove irrelevant content and duplicates.
Categorize each kept item and write a Chinese summary.

Items:
{items_text}"""

    logger.info(f"Calling LLM API with {len(items)} items (~{len(items_text)} chars)")
    logger.info(f"Using model: {LLM_MODEL}, base_url: {LLM_BASE_URL}")

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        response_text = response.choices[0].message.content.strip()
        logger.info(f"LLM API response: {len(response_text)} chars")

        # Extract JSON from response (handle markdown code fences)
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.strip().startswith("```json") or line.strip().startswith("```"):
                    if not in_json:
                        in_json = True
                        continue
                    else:
                        break
                if in_json:
                    json_lines.append(line)
            response_text = "\n".join(json_lines)

        result = json.loads(response_text)

        if "categories" not in result:
            raise ValueError("Response missing 'categories' key")

        for cat in CATEGORIES:
            if cat not in result["categories"]:
                result["categories"][cat] = []

        logger.info(
            f"Curation complete: {result.get('stats', {}).get('total_kept', 'N/A')} items kept "
            f"from {result.get('stats', {}).get('total_processed', len(items))} processed"
        )

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.debug(f"Raw response: {response_text[:500]}")
        return fallback_curation(items)
    except Exception as e:
        logger.error(f"LLM API call failed: {e}", exc_info=True)
        return fallback_curation(items)


def fallback_curation(items: List[RawItem]) -> Dict[str, Any]:
    """
    Basic curation without LLM — just group by source.
    Used when LLM API is unavailable.
    """
    logger.info("Using fallback curation (basic grouping by source)")

    categories = {cat: [] for cat in CATEGORIES}

    for item in items:
        title_lower = item.title.lower()

        if any(kw in title_lower for kw in ["paper", "arxiv", "study", "research"]):
            cat = "📄 研究论文 (Research Papers)"
        elif any(kw in title_lower for kw in ["launch", "release", "announce", "introduce", "new"]):
            cat = "🚀 产品发布 (Product Launches)"
        elif any(kw in title_lower for kw in ["funding", "acquire", "invest", "partner"]):
            cat = "🏢 行业动态 (Industry News)"
        elif any(kw in title_lower for kw in ["github", "open source", "open-source", "repo"]):
            cat = "💻 开源项目 (Open Source)"
        elif any(kw in title_lower for kw in ["tool", "framework", "library", "api", "platform"]):
            cat = "🛠️ 工具与框架 (Tools & Frameworks)"
        elif any(kw in title_lower for kw in ["dataset", "benchmark", "leaderboard", "evaluation"]):
            cat = "📊 数据集与基准 (Datasets & Benchmarks)"
        else:
            cat = "🧠 观点与讨论 (Opinions & Discussions)"

        categories[cat].append({
            "title": item.title,
            "url": item.url,
            "summary_zh": item.description[:150] if item.description else "[AI 整理服务暂时不可用]",
            "source": item.source,
            "score": item.score,
        })

    for cat in categories:
        categories[cat].sort(key=lambda x: x["score"], reverse=True)

    stats = {
        "total_processed": len(items),
        "total_kept": len(items),
        "by_source": {"HN": 0, "Reddit": 0, "RSS": 0},
    }
    for item in items:
        if item.source in stats["by_source"]:
            stats["by_source"][item.source] += 1

    return {"categories": categories, "stats": stats, "fallback": True}
