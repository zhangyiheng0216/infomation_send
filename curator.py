"""
AI Daily Digest - Content Curator
Uses LLM API (OpenAI-compatible) to filter, categorize, and summarize AI news.
Splits large item sets into chunks to avoid output truncation.
"""

import json
import logging
import re
import time
from collections import defaultdict
from typing import Any, Dict, List

import httpx
from openai import OpenAI

from collectors import RawItem
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, CATEGORIES, MAX_TOKENS, CHUNK_SIZE

logger = logging.getLogger(__name__)

# ============================================================
# Retry / timeout configuration
# ============================================================
LLM_TIMEOUT = 180        # seconds per request
LLM_MAX_RETRIES = 3      # number of retries on transient failures
LLM_RETRY_BACKOFF = 5.0   # seconds base backoff between retries

SYSTEM_PROMPT = """You are an expert AI news curator. Your task is to process a batch of AI-related news items.

## Your Responsibilities

1. **Filter**: Remove irrelevant, low-quality, or duplicate content
2. **Categorize**: Assign each kept item to exactly one category from the list below
3. **Summarize**: Write a concise Chinese summary (2-3 sentences) for each kept item

## Output Format

Respond with ONLY a valid JSON object. No markdown code fences, no explanation text.

{
  "items": [
    {
      "title": "Original English Title",
      "url": "https://...",
      "summary_zh": "中文摘要，2-3句话",
      "source": "HN",
      "score": 42,
      "category": "🚀 产品发布 (Product Launches)"
    }
  ],
  "stats": {
    "processed": 20,
    "kept": 15,
    "filtered": 5
  }
}

## Categories (pick exactly one per item)

- 📄 研究论文 (Research Papers): Academic papers, technical reports, preprints (arXiv, papers with code)
- 🚀 产品发布 (Product Launches): New model releases, product updates, API launches, demos
- 🏢 行业动态 (Industry News): Funding rounds, acquisitions, partnerships, policy changes
- 💻 开源项目 (Open Source): GitHub repos, open-source tools, model weights, frameworks
- 🛠️ 工具与框架 (Tools & Frameworks): Developer tools, libraries, platforms, infrastructure
- 📊 数据集与基准 (Datasets & Benchmarks): New datasets, benchmark results, leaderboards, evaluations
- 🧠 观点与讨论 (Opinions & Discussions): Opinion pieces, interviews, debates, trend analysis

## Filtering Rules

**Keep**: AI/ML research, products, industry news from credible sources.
**Remove**: General tech news, job postings, memes, ads, off-topic content, duplicates.

## Summary Style

Concise (2-3 sentences, under 100 Chinese characters), information-dense, objective.

Examples:
✅ "OpenAI 发布 GPT-4o，支持文本、语音、视觉多模态输入输出。新模型推理速度提升 2 倍，API 价格降低 50%。"
✅ "Meta 开源 Llama 3.1 405B 模型，参数量超越 GPT-4。在多项基准测试中表现接近闭源模型。"
❌ "OpenAI 发布了新模型，看起来很厉害。" (too vague)

## Rules

- Empty items array if nothing qualifies
- Preserve original English title exactly
- High-score items (>100) get more detailed summaries
- Err on exclusion when unsure
- IMPORTANT: Output ONLY the JSON object, no code fences or other text"""


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
            try:
                data = resp.json()
                models = [m.get("id", "") for m in data.get("data", [])]
                if model in models:
                    logger.info(f"[diag] Model '{model}' is available")
                else:
                    logger.warning(f"[diag] Model '{model}' NOT found in available models")
                    similar = [m for m in models if any(
                        kw in m.lower() for kw in model.lower().split("/")
                    )]
                    if similar:
                        logger.info(f"[diag] Similar models: {similar[:10]}")
            except Exception as json_err:
                logger.warning(f"[diag] Failed to parse /models response: {json_err}")
                logger.debug(f"[diag] Response text: {resp.text[:200]}")
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

            # Handle non-standard responses (some proxies return raw strings)
            if isinstance(response, str):
                logger.warning("API returned string instead of response object, treating as raw content")
                response_text = response.strip()
            else:
                if not hasattr(response, 'choices') or not response.choices:
                    raise ValueError(f"Invalid response format: {type(response)}")

                # Check finish_reason for truncation
                finish_reason = response.choices[0].finish_reason
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("LLM returned empty response")
                response_text = content.strip()

                if finish_reason == "length":
                    logger.warning(f"LLM output was truncated (finish_reason=length). "
                                   f"Got {len(response_text)} chars")

                if hasattr(response, 'usage') and response.usage:
                    logger.info(
                        f"Token usage: prompt={response.usage.prompt_tokens}, "
                        f"completion={response.usage.completion_tokens}, "
                        f"total={response.usage.total_tokens}"
                    )

            logger.info(f"LLM API success: {len(response_text)} chars")
            return response_text

        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            last_error = e
            logger.warning(
                f"LLM API attempt {attempt} network error: {type(e).__name__}: {e}"
            )
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "401" in err_str or "Unauthorized" in err_str or "INVALID_API_KEY" in err_str:
                logger.error(f"LLM API authentication failed: {e}")
                raise
            elif "404" in err_str or "not found" in err_str.lower():
                logger.error(f"LLM API model not found: {e}")
                raise
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
    # Strip code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
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
        text = "\n".join(json_lines)

    # If no code fences, try to find JSON object boundaries
    if not text.startswith("{"):
        start = text.find("{")
        if start != -1:
            text = text[start:]

    return text


def _fix_unescaped_quotes(text: str) -> str:
    """Fix unescaped double quotes inside JSON string values.

    The LLM sometimes outputs Chinese quotes like: 这种"看护机器人"工作
    where the inner " are content quotes, not JSON delimiters.
    This function escapes them: 这种\\"看护机器人\\"工作
    """
    # Fix quote preceded by CJK char or CJK punctuation (opening content quote)
    # e.g. 这种"看护 → 这种\"看护
    text = re.sub(r'(?<=[一-鿿㐀-䶿　-〿＀-￯])"', r'\\"', text)
    # Fix quote followed by CJK char or CJK punctuation (closing content quote)
    # e.g. 机器人"工作 → 机器人\"工作
    text = re.sub(r'"(?=[一-鿿㐀-䶿　-〿＀-￯])', r'\\"', text)
    # Handle double-escaped (idempotent): \\" -> \"
    text = text.replace('\\\\"', '\\"')
    return text


def _repair_json(text: str) -> Dict:
    """Attempt to parse JSON with multiple repair strategies."""
    # Strategy 1: Try as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Remove trailing commas
    repaired = re.sub(r',(\s*[}\]])', r'\1', text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Fix unescaped quotes inside string values
    quoted = _fix_unescaped_quotes(text)
    try:
        return json.loads(quoted)
    except json.JSONDecodeError:
        pass

    # Strategy 3b: Fix unescaped quotes + remove trailing commas
    quoted_clean = re.sub(r',(\s*[}\]])', r'\1', quoted)
    try:
        return json.loads(quoted_clean)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        bounded = text[start:end+1]
        bounded_fixed = _fix_unescaped_quotes(bounded)
        try:
            return json.loads(bounded_fixed)
        except json.JSONDecodeError:
            pass

    # Strategy 5: Truncate at last complete item and close brackets
    last_score = text.rfind('"score"')
    if last_score > 0:
        item_end = text.find("}", last_score)
        if item_end > 0:
            truncated = text[:item_end+1]
            open_brackets = truncated.count("[") - truncated.count("]")
            open_braces = truncated.count("{") - truncated.count("}")
            truncated += "]" * max(0, open_brackets)
            truncated += "}" * max(0, open_braces)
            try:
                result = json.loads(truncated)
                logger.info("JSON repaired via truncation at last complete item")
                return result
            except json.JSONDecodeError:
                pass

    return None


def _curate_chunk(client: OpenAI, chunk: List[RawItem],
                  chunk_idx: int, total_chunks: int) -> Dict[str, Any]:
    """Curate a single chunk of items via LLM."""
    items_text = format_items_for_prompt(chunk)
    user_prompt = f"""Curate the following {len(chunk)} AI news items (batch {chunk_idx}/{total_chunks}).

Filter out irrelevant content and duplicates. Categorize each kept item and write a Chinese summary.

Items:
{items_text}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response_text = _call_llm_with_retry(client, LLM_MODEL, messages, MAX_TOKENS)
    except Exception as e:
        logger.error(f"Chunk {chunk_idx}/{total_chunks} LLM call failed: {e}")
        return {"items": [], "stats": {"processed": len(chunk), "kept": 0, "filtered": len(chunk)}}

    cleaned = _extract_json_from_response(response_text)
    result = _repair_json(cleaned)

    if result is None:
        logger.error(f"Chunk {chunk_idx}/{total_chunks}: all JSON repair attempts failed")
        logger.error(f"Raw response (first 1000 chars):\n{response_text[:1000]}")
        return {"items": [], "stats": {"processed": len(chunk), "kept": 0, "filtered": len(chunk)}}

    # Validate structure
    if "items" not in result:
        logger.warning(f"Chunk {chunk_idx}/{total_chunks}: response missing 'items', "
                       f"keys={list(result.keys())}")
        return {"items": [], "stats": {"processed": len(chunk), "kept": 0, "filtered": len(chunk)}}

    kept = len(result.get("items", []))
    logger.info(f"Chunk {chunk_idx}/{total_chunks}: processed {len(chunk)}, kept {kept}")
    return result


def curate(items: List[RawItem]) -> Dict[str, Any]:
    """
    Use LLM to filter, categorize, and summarize items.
    Splits into chunks of CHUNK_SIZE to avoid output truncation.

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

    # Step 3: Split items into chunks
    chunks = [items[i:i+CHUNK_SIZE] for i in range(0, len(items), CHUNK_SIZE)]
    total_chunks = len(chunks)
    logger.info(f"Splitting {len(items)} items into {total_chunks} chunks "
                f"(chunk_size={CHUNK_SIZE})")

    # Step 4: Process each chunk
    all_kept_items = []
    total_processed = 0
    total_filtered = 0
    failed_chunks = 0

    for idx, chunk in enumerate(chunks, 1):
        result = _curate_chunk(client, chunk, idx, total_chunks)

        chunk_items = result.get("items", [])
        chunk_stats = result.get("stats", {})
        total_processed += chunk_stats.get("processed", len(chunk))
        total_filtered += chunk_stats.get("filtered", 0)

        if chunk_items:
            all_kept_items.extend(chunk_items)
        else:
            failed_chunks += 1

        # Small delay between chunks to be polite to the API
        if idx < total_chunks:
            time.sleep(1)

    http_client.close()

    logger.info(f"All chunks complete: {total_processed} processed, "
                f"{len(all_kept_items)} kept, {failed_chunks} failed chunks")

    # If too many chunks failed, fall back entirely
    if failed_chunks > total_chunks / 2:
        logger.error(f"More than half of chunks failed ({failed_chunks}/{total_chunks}), "
                     f"using fallback")
        return fallback_curation(items)

    # Step 5: Group items by category
    categories = {cat: [] for cat in CATEGORIES}

    for item in all_kept_items:
        cat = item.get("category", "")
        if cat not in categories:
            # Try fuzzy match
            matched = False
            for valid_cat in CATEGORIES:
                if cat in valid_cat or valid_cat in cat:
                    cat = valid_cat
                    matched = True
                    break
            if not matched:
                cat = "🧠 观点与讨论 (Opinions & Discussions)"

        categories[cat].append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "summary_zh": item.get("summary_zh", ""),
            "source": item.get("source", ""),
            "score": item.get("score", 0),
        })

    # Sort each category by score
    for cat in categories:
        categories[cat].sort(key=lambda x: x.get("score", 0), reverse=True)

    total_kept = sum(len(v) for v in categories.values())

    # Build stats
    by_source = {"HN": 0, "Reddit": 0, "RSS": 0}
    for item in all_kept_items:
        src = item.get("source", "")
        if src in by_source:
            by_source[src] += 1

    result = {
        "categories": categories,
        "stats": {
            "total_processed": total_processed,
            "total_kept": total_kept,
            "by_source": by_source,
        }
    }

    logger.info(
        f"Curation complete: {total_kept} items kept from {total_processed} processed"
    )

    return result


def fallback_curation(items: List[RawItem]) -> Dict[str, Any]:
    """
    Basic curation without LLM — keyword-based grouping.
    Used when LLM API is unavailable.
    """
    logger.info("Using fallback curation (keyword-based grouping)")

    categories = {cat: [] for cat in CATEGORIES}

    # Use word-boundary matching to avoid false positives
    patterns = {
        "📄 研究论文 (Research Papers)": re.compile(r'\b(paper|arxiv|study|research|paper)\b', re.I),
        "🚀 产品发布 (Product Launches)": re.compile(r'\b(launch|release|announce|introduce)\b', re.I),
        "🏢 行业动态 (Industry News)": re.compile(r'\b(funding|acquire|invest|partner|acquisition)\b', re.I),
        "💻 开源项目 (Open Source)": re.compile(r'\b(github|open.?source|repo)\b', re.I),
        "🛠️ 工具与框架 (Tools & Frameworks)": re.compile(r'\b(tool|framework|library|platform)\b', re.I),
        "📊 数据集与基准 (Datasets & Benchmarks)": re.compile(r'\b(dataset|benchmark|leaderboard|evaluation)\b', re.I),
    }

    for item in items:
        cat = "🧠 观点与讨论 (Opinions & Discussions)"  # default
        for cat_name, pattern in patterns.items():
            if pattern.search(item.title):
                cat = cat_name
                break

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
