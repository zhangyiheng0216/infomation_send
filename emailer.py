"""
AI Daily Digest - Email Builder & Sender
Renders HTML email with Jinja2 and sends via QQ SMTP.
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader

from config import EMAIL_SUBJECT_PREFIX, EMAIL_TO, SMTP_HOST, SMTP_PASSWORD, SMTP_PORT, SMTP_USER, YESTERDAY_START

logger = logging.getLogger(__name__)


def build_email(curated: Dict[str, Any], date_str: str) -> str:
    """Render the Jinja2 template with curated data."""
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("template.html")

    # Extract stats
    stats = curated.get("stats", {"total_kept": 0, "by_source": {"HN": 0, "Reddit": 0, "RSS": 0}})
    categories = curated.get("categories", {})
    fallback = curated.get("fallback", False)

    # Ensure stats has all keys
    by_source = stats.get("by_source", {})
    by_source.setdefault("HN", 0)
    by_source.setdefault("Reddit", 0)
    by_source.setdefault("RSS", 0)
    stats["by_source"] = by_source
    stats.setdefault("total_kept", sum(len(v) for v in categories.values()))

    html = template.render(
        date=date_str,
        subject=f"{EMAIL_SUBJECT_PREFIX}{date_str}",
        categories=categories,
        stats=stats,
        fallback=fallback,
    )

    logger.info(f"Email HTML rendered: {len(html)} chars")
    return html


def send_email(html: str, subject: str) -> None:
    """Send HTML email via QQ SMTP with SSL."""
    if not SMTP_USER or not SMTP_PASSWORD:
        raise ValueError("SMTP_USER and SMTP_PASSWORD must be set")
    if not EMAIL_TO:
        raise ValueError("EMAIL_TO must be set")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO

    # Plain text fallback
    msg.attach(MIMEText("Your email client doesn't support HTML.", "plain"))
    # HTML body
    msg.attach(MIMEText(html, "html", "utf-8"))

    logger.info(f"Connecting to {SMTP_HOST}:{SMTP_PORT} (SSL)...")

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, [EMAIL_TO], msg.as_string())

    logger.info(f"Email sent to {EMAIL_TO}")
