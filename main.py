"""
AI Daily Digest - Main Entry Point
Orchestrates: collect → curate → email
"""

import logging
import sys
from datetime import datetime

from collectors import collect_all
from config import EMAIL_SUBJECT_PREFIX, YESTERDAY_START
from curator import curate
from emailer import build_email, send_email

# ============================================================
# Logging setup
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ai-digest")


def main():
    """Main workflow: collect → curate → send."""
    date_str = YESTERDAY_START.strftime("%Y-%m-%d")
    logger.info(f"=" * 60)
    logger.info(f"AI Daily Digest — Processing news for {date_str}")
    logger.info(f"=" * 60)

    # Step 1: Collect raw items from all sources
    logger.info("Step 1: Collecting news from HN, Reddit, RSS...")
    items = collect_all()

    if not items:
        logger.warning("No items collected from any source. Skipping email.")
        return

    logger.info(f"Collected {len(items)} unique items")

    # Step 2: Curate with Claude (filter + categorize + summarize)
    logger.info("Step 2: Curating with Claude API...")
    curated = curate(items)

    categories = curated.get("categories", {})
    total_kept = sum(len(v) for v in categories.values())
    logger.info(f"Curated into {total_kept} items across {len(categories)} categories")

    # Log category breakdown
    for cat, cat_items in categories.items():
        if cat_items:
            logger.info(f"  {cat}: {len(cat_items)} items")

    # Step 3: Build and send email
    logger.info("Step 3: Building and sending email...")
    html = build_email(curated, date_str)
    subject = f"{EMAIL_SUBJECT_PREFIX}{date_str}"

    try:
        send_email(html, subject)
        logger.info("✅ Email sent successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to send email: {e}", exc_info=True)
        # Save HTML for manual inspection
        output_file = f"digest-{date_str}.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Saved HTML to {output_file} for manual inspection")
        raise

    logger.info(f"=" * 60)
    logger.info("Done!")
    logger.info(f"=" * 60)


if __name__ == "__main__":
    main()
