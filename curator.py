"""
AI Daily Digest - Content Curator
Uses LLM API (OpenAI-compatible) to filter, categorize, and summarize AI news.
"""

import json
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List

import httpx
from openai import OpenAI

from collectors import RawItem
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, CATEGORIES, MAX_TOKENS

logger = logging.getLogger(__name__)

# ============================================================
# Retry / timeout configuration
# ============================================================
LLM_TIMEOUT = 120        # seconds per request
LLM_MAX_RETRIES = 3      # number of retries on transient failures
LLM_RETRY_BACKOFF = 5.0   # seconds base backoff between retries

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


def _test_api_connectivity(base_url: str, api_key: str, model: str) -> bool:
    """Quick connectivity test before the real call. Logs detailed diagnostics."""
    logger.info(f"[diag] Testing API connectivity: {base_url}")
    try:
        resp = httpx.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        logger.info(f"[diag] GET /models -> HTTP {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("id", "") for m in data.get("data", [])]
            if model in models:
                logger.info(f"[diag] Model '{model}' is available")
            else:
                logger.warning(f"[diag] Model '{model}' NOT found in available models")
                # Find similar models
                similar = [m for m in models if any(
                    kw in m.lower() for kw in model.lower().split("/")
                )]
                if similar:
                    logger.info(f"[diag] Similar models: {similar[:10]}")
            return True
        elif resp.status_code == 401:
            logger.warning("[diag] API key may be invalid (401 Unauthorized)")
            return False
        else:
            logger.warning(f"[diag] Unexpected status: {resp.status_code} - {resp.text[:200]}")
            return resp.status_code < 500
    except httpx.ConnectError as e:
        logger.error(f"[diag] Connection failed: {e}")
        return False
    except Exception as e:
        logger.error(f"[diag] Connectivity test error: {e}")
        return False


def _call_llm_with_retry(client: OpenAI, model: str, messages: list,
                          max_tokens: int) -> str:
    """Call LLM API with retry logic for transient failures."""
    last_error = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            logger.info(f"LLM API call attempt {attempt}/{LLM_MAX_RETRIES}...")
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty response")
            response_text = content.strip()
            logger.info(f"LLM API success: {len(response_text)} chars")

            # Log usage info if available
            if response.usage:
                logger.info(
                    f"Token usage: prompt={response.usage.prompt_tokens}, "
                    f"completion={response.usage.completion_tokens}, "
                    f"total={response.usage.total_tokens}"
                )
            return response_text

        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            last_error = e
            logger.warning(
                f"LLM API attempt {attempt} network error: {type(e).__name__}: {e}"
            )
        except Exception as e:
            last_error = e
            err_str = str(e)
            # Check for specific error types
            if "401" in err_str or "Unauthorized" in err_str or "INVALID_API_KEY" in err_str:
                logger.error(f"LLM API authentication failed: {e}")
                raise  # No point retrying auth errors
            elif "404" in err_str or "not found" in err_str.lower():
                logger.error(f"LLM API model not found: {e}")
                raise  # No point retrying model errors
            elif "429" in err_str or "rate" in err_str.lower():
                logger.warning(f"LLM API rate limited, attempt {attempt}: {e}")
            else:
                logger.warning(f"LLM API attempt {attempt} error: {type(e).__name__}: {e}")

        if attempt < LLM_MAX_RETRIES:
            wait = LLM_RETRY_BACKOFF * (2 ** (attempt - 1))
            logger.info(f"Waiting {wait}s before retry...")
            time.sleep(wait)

    raise last_error  # type: ignore


def _extract_json_from_response(response_text: str) -> str:
    """Extract JSON from LLM response, handling markdown code fences."""
    if not response_text.startswith("```"):
        return response_text

    lines = response_text.split("\n")
    json_lines = []
    in_json = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```json") or (stripped.startswith("```") and not in_json):
            in_json = True
            continue
        if stripped.startswith("```") and in_json:
            break
        if in_json:
            json_lines.append(line)
    return "\n".join(json_lines)


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

    logger.info(f"LLM config: model={LLM_MODEL}, base_url={LLM_BASE_URL}, "
                f"api_key_len={len(LLM_API_KEY)}, max_tokens={MAX_TOKENS}")

    # Step 1: Quick connectivity check
    api_reachable = _test_api_connectivity(LLM_BASE_URL, LLM_API_KEY, LLM_MODEL)
    if not api_reachable:
        logger.warning("API connectivity check failed, will still attempt LLM call...")

    # Step 2: Build the OpenAI client
    # The OpenAI SDK sets User-Agent to "OpenAI/Python x.x" which triggers
    # Cloudflare WAF 403 on some API proxies (e.g. eoeo.xyz). We use a request
    # event hook to force a generic User-Agent after the SDK sets its own.
    def _override_ua(request: httpx.Request) -> None:
        request.headers["User-Agent"] = "Mozilla/5.0 (compatible; ai-daily-digest/1.0)"

    http_client = httpx.Client(
        timeout=LLM_TIMEOUT,
        event_hooks={"request": [_override_ua]},
    )
    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        http_client=http_client,
    )

    items_text = format_items_for_prompt(items)
    user_prompt = f"""Please curate the following {len(items)} AI news items collected from yesterday.

Apply the filtering rules carefully — remove irrelevant content and duplicates.
Categorize each kept item and write a Chinese summary.

Items:
{items_text}"""

    logger.info(f"Prompt size: {len(items)} items, ~{len(items_text)} chars, "
                f"~{len(SYSTEM_PROMPT) + len(user_prompt)} total prompt chars")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response_text = None
    try:
        response_text = _call_llm_with_retry(client, LLM_MODEL, messages, MAX_TOKENS)
    except Exception as e:
        logger.error(f"LLM API call failed after {LLM_MAX_RETRIES} attempts: {e}", exc_info=True)
        return fallback_curation(items)
    finally:
        http_client.close()

    # Step 3: Parse JSON response with robust error handling
    cleaned = _extract_json_from_response(response_text)
    result = None

    # Try parsing as-is first
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}, attempting repair...")
        logger.warning(f"Error at line {e.lineno}, col {e.colno}, pos {e.pos}")

        # Try common JSON repairs
        # 1. Remove trailing commas
        repaired = cleaned.replace(",\n  ]", "\n  ]").replace(",\n}", "\n}")
        repaired = repaired.replace(",]", "]").replace(",}", "}")
        try:
            result = json.loads(repaired)
            logger.info("JSON repaired successfully (removed trailing commas)")
        except json.JSONDecodeError:
            pass

        # 2. Try finding JSON object boundaries
        if result is None:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    result = json.loads(cleaned[start:end+1])
                    logger.info("JSON repaired successfully (extracted object)")
                except json.JSONDecodeError:
                    pass

        # 3. Try to fix truncated JSON by completing open structures
        if result is None:
            try:
                # Find the last complete item in each category array
                fixed = cleaned
                # Count open brackets/braces
                open_braces = fixed.count("{") - fixed.count("}")
                open_brackets = fixed.count("[") - fixed.count("]")

                # Try to close at the last complete JSON object
                # Find last "}" that ends an item (has "score": N before it)
                import re
                # Find all complete items ending with score
                last_complete = fixed.rfind('"score"')
                if last_complete > 0:
                    # Find the closing brace of this item
                    item_end = fixed.find("}", last_complete)
                    if item_end > 0:
                        # Truncate after this item
                        truncated = fixed[:item_end+1]
                        # Close any open arrays/objects
                        truncated += "]" * max(0, truncated.count("[") - truncated.count("]"))
                        truncated += "}" * max(0, truncated.count("{") - truncated.count("}"))
                        result = json.loads(truncated)
                        logger.info(f"JSON repaired (truncated at last complete item)")
            except Exception as repair_err:
                logger.warning(f"Truncation repair failed: {repair_err}")

        # 4. If all repairs fail, log and use fallback
        if result is None:
            logger.error(f"All JSON repair attempts failed")
            logger.error(f"Raw response (first 2000 chars):\n{response_text[:2000]}")
            return fallback_curation(items)

    if "categories" not in result:
        logger.error("LLM response missing 'categories' key, using fallback")
        logger.error(f"Response keys: {list(result.keys())}")
        return fallback_curation(items)

    # Ensure all expected categories exist
    for cat in CATEGORIES:
        if cat not in result["categories"]:
            result["categories"][cat] = []

    logger.info(
        f"Curation complete: {result.get('stats', {}).get('total_kept', 'N/A')} items kept "
        f"from {result.get('stats', {}).get('total_processed', len(items))} processed"
    )

    return result


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
