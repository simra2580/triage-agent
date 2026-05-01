"""
src/corpus_loader.py  —  Corpus Management

Two modes:
  1. File mode  → reads text/markdown files from data/corpus/
  2. Scrape mode → fetches live support pages and saves them

Each chunk: { "text": str, "source": str, "section": str }
"""

from __future__ import annotations

import os
import re
import time
import json
import textwrap
from pathlib import Path

# ── Support URL registry ───────────────────────────────────────────────────────
SUPPORT_SOURCES = {
    "hackerrank": [
        "https://support.hackerrank.com/hc/en-us",
        "https://support.hackerrank.com/hc/en-us/categories/115001811208-Developers",
        "https://support.hackerrank.com/hc/en-us/categories/115001810968-Recruiters",
    ],
    "claude": [
        "https://support.claude.ai/hc/en-us",
        "https://support.claude.ai/hc/en-us/categories/14111985867540-Claude-ai",
    ],
    "visa": [
        "https://www.visa.co.in/support.html",
        "https://usa.visa.com/support/consumer/visa-cards.html",
    ],
}

CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 80    # word overlap between chunks


# ── Public API ─────────────────────────────────────────────────────────────────

class CorpusLoader:
    def __init__(self, corpus_dir: str):
        self.corpus_dir = Path(corpus_dir)

    def load(self) -> list[dict]:
        """
        Load corpus from local files.
        Falls back to a minimal built-in corpus if no files found.
        """
        chunks = []

        if self.corpus_dir.exists():
            for fpath in sorted(self.corpus_dir.rglob("*.txt")):
                source = fpath.stem.split("_")[0].lower()
                text   = fpath.read_text(encoding="utf-8", errors="ignore")
                chunks.extend(self._chunk_text(text, source=source, filename=fpath.name))

            for fpath in sorted(self.corpus_dir.rglob("*.md")):
                source = fpath.stem.split("_")[0].lower()
                text   = fpath.read_text(encoding="utf-8", errors="ignore")
                # strip markdown syntax
                text = re.sub(r"#+ ", "", text)
                text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
                chunks.extend(self._chunk_text(text, source=source, filename=fpath.name))

        if not chunks:
            # Fallback: use the built-in minimal corpus
            print("[WARN] No corpus files found — using built-in minimal corpus.")
            print("       Run with --scrape-corpus to download live support docs.")
            chunks = _BUILTIN_CORPUS

        return chunks

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _chunk_text(text: str, source: str, filename: str = "") -> list[dict]:
        """Split text into overlapping word-based chunks."""
        words  = text.split()
        chunks = []
        start  = 0
        while start < len(words):
            end   = min(start + CHUNK_SIZE, len(words))
            chunk = " ".join(words[start:end])
            if len(chunk.strip()) > 40:   # ignore tiny shards
                chunks.append({
                    "text":    chunk,
                    "source":  source,
                    "section": filename,
                    "score":   0.0,
                })
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks


def scrape_all_corpus(output_dir: str):
    """
    Scrape live support pages and save as .txt files.
    Requires: requests, beautifulsoup4
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("Install scraping deps: pip install requests beautifulsoup4")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (compatible; TriageAgent/1.0)"}

    for domain, urls in SUPPORT_SOURCES.items():
        all_text = []
        for url in urls:
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                # Remove nav/footer/script noise
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                text = soup.get_text(separator=" ", strip=True)
                text = re.sub(r"\s{2,}", " ", text)
                all_text.append(f"# Source: {url}\n\n{text}\n\n")
                print(f"  ✓ Scraped: {url}")
                time.sleep(1)
            except Exception as e:
                print(f"  ✗ Failed {url}: {e}")

        if all_text:
            outfile = out / f"{domain}_support.txt"
            outfile.write_text("\n".join(all_text), encoding="utf-8")
            print(f"  → Saved: {outfile}")


# ── Built-in minimal corpus ────────────────────────────────────────────────────
# Used when no local corpus files are present.
# Covers the most common support topics across all 3 domains.

_BUILTIN_CORPUS: list[dict] = [
    # ── HackerRank ─────────────────────────────────────────────────────────────
    {
        "source": "hackerrank",
        "section": "assessments",
        "text": (
            "HackerRank Assessments: Candidates invited to a coding assessment receive an email invitation. "
            "The test link is valid for a limited time window set by the recruiter. "
            "If your test timer is still running you can resume from the same browser. "
            "Proctoring features may record webcam, screen, and mouse movement. "
            "Switching tabs or minimizing the window may flag your test. "
            "If you face technical issues during the test, contact the company recruiter directly or use the "
            "Help button inside the test interface. HackerRank support cannot extend or reset test timers "
            "without recruiter authorization."
        ),
    },
    {
        "source": "hackerrank",
        "section": "account",
        "text": (
            "HackerRank Account: To reset your password, go to hackerrank.com and click Forgot Password. "
            "An email with a reset link will be sent within a few minutes. "
            "If you do not receive the email, check your spam folder. "
            "If your account is locked due to multiple failed attempts, wait 30 minutes or contact support. "
            "Social login (Google/LinkedIn/GitHub) accounts do not have a separate HackerRank password."
        ),
    },
    {
        "source": "hackerrank",
        "section": "plagiarism",
        "text": (
            "Plagiarism and Code Similarity: HackerRank uses automated tools to detect code similarity. "
            "A plagiarism flag is raised when code closely matches other submissions. "
            "The final hiring decision rests with the recruiter, not HackerRank. "
            "If you believe the flag is incorrect, contact the recruiter who sent you the test."
        ),
    },
    {
        "source": "hackerrank",
        "section": "billing",
        "text": (
            "HackerRank for Work Billing: Recruiters and companies are billed for HackerRank plans. "
            "Billing issues should be raised through the company admin account. "
            "Candidates do not pay HackerRank for taking assessments. "
            "For invoicing or subscription changes, contact HackerRank sales or billing support."
        ),
    },
    # ── Claude ─────────────────────────────────────────────────────────────────
    {
        "source": "claude",
        "section": "account",
        "text": (
            "Claude.ai Account & Login: To reset your Claude.ai password, visit claude.ai and click "
            "Forgot password on the login page. An email will be sent to your registered address. "
            "If you signed up with Google or Apple, you cannot set a separate Claude password; "
            "log in using those providers instead. "
            "Account deletion can be requested from Settings > Privacy."
        ),
    },
    {
        "source": "claude",
        "section": "subscription",
        "text": (
            "Claude.ai Subscription & Billing: Claude offers a free tier and a Claude Pro subscription. "
            "Pro subscribers get higher message limits and priority access. "
            "To cancel your subscription, go to Settings > Billing > Cancel Plan. "
            "Refunds are handled on a case-by-case basis; contact Anthropic support with your order details. "
            "Billing is managed through Stripe."
        ),
    },
    {
        "source": "claude",
        "section": "usage_limits",
        "text": (
            "Claude Usage Limits: Free users have a limited number of messages per day. "
            "Pro users get significantly higher limits. "
            "When you hit your limit, Claude will notify you and indicate when it resets. "
            "Usage resets every 24 hours. Heavy usage during peak times may result in temporary slowdowns."
        ),
    },
    {
        "source": "claude",
        "section": "safety",
        "text": (
            "Claude Safety & Content Policy: Claude is designed to be safe and helpful. "
            "It declines requests that could cause harm, involve illegal activity, or violate Anthropic policy. "
            "Claude does not store conversation history between sessions unless memory features are enabled. "
            "To report a safety concern or policy violation, use the thumbs-down feedback or contact Anthropic."
        ),
    },
    {
        "source": "claude",
        "section": "bugs",
        "text": (
            "Claude Bugs and Technical Issues: If Claude stops responding mid-conversation, try refreshing. "
            "Persistent issues can be reported through the feedback button in the interface. "
            "Known issues are tracked on the Anthropic status page. "
            "API users experiencing errors should check the Anthropic status page and their API key validity."
        ),
    },
    # ── Visa ───────────────────────────────────────────────────────────────────
    {
        "source": "visa",
        "section": "fraud",
        "text": (
            "Visa Fraud & Unauthorized Transactions: If you see charges on your Visa card that you did not make, "
            "contact your card-issuing bank immediately to dispute the transaction. "
            "Your bank will investigate and may issue a provisional credit while the dispute is open. "
            "Visa's Zero Liability Policy protects cardholders from unauthorized transactions "
            "when reported promptly. Do not share your card number, CVV, or PIN with anyone."
        ),
    },
    {
        "source": "visa",
        "section": "lost_stolen",
        "text": (
            "Lost or Stolen Visa Card: If your Visa card is lost or stolen, contact your issuing bank immediately. "
            "Most banks have a 24/7 emergency line. You can request a replacement card. "
            "Freeze your card instantly through your bank's mobile app if available. "
            "Visa Global Customer Assistance Services: +1-800-847-2911 (USA), or the local Visa helpline."
        ),
    },
    {
        "source": "visa",
        "section": "disputes",
        "text": (
            "Visa Dispute Resolution: To dispute a charge, contact your card-issuing bank, not Visa directly. "
            "Your bank handles the chargeback process. Provide the transaction date, merchant name, and amount. "
            "Disputes must typically be filed within 60-120 days of the transaction. "
            "The merchant has the right to respond. Resolution may take up to 90 days."
        ),
    },
    {
        "source": "visa",
        "section": "payment_failure",
        "text": (
            "Visa Payment Failures: A declined Visa transaction can occur due to: insufficient funds, "
            "exceeded credit limit, expired card, incorrect CVV or billing address, "
            "or the bank flagging an unusual purchase pattern. "
            "Contact your bank to understand the specific reason. "
            "International transactions may require notifying your bank in advance."
        ),
    },
    {
        "source": "visa",
        "section": "general",
        "text": (
            "Visa Support: Visa does not issue cards directly to consumers. "
            "All card-specific issues — billing, limits, rewards, account access — "
            "must be resolved through your card-issuing bank. "
            "Visa provides the payment network; your bank manages your account."
        ),
    },
]