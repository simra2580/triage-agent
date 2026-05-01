"""
src/classifier.py  —  Rule-Based Risk & Safety Pre-Classifier

Runs BEFORE the LLM call. Fast keyword matching to catch:
  - Fraud / unauthorized transaction signals        → HIGH
  - Account compromise signals                      → HIGH
  - Sensitive financial matters                     → MEDIUM (handle carefully, NOT auto-escalate)
  - Prompt injection / jailbreak attempts           → CRITICAL (always escalate)
  - Clearly irrelevant / empty inputs               → flag only

IMPORTANT ESCALATION LOGIC:
  CRITICAL → always escalate (prompt injection)
  HIGH     → always escalate (confirmed fraud / account compromise)
  MEDIUM   → pass to LLM with a "handle carefully" flag — do NOT auto-escalate
  LOW      → normal flow
"""

from __future__ import annotations
import re


FRAUD_KEYWORDS = [
    "fraud", "unauthorized", "unauthorised", "not me", "didn't make",
    "didn't authorize", "stolen card", "card stolen", "hacked my card",
    "scam", "phishing", "identity theft", "account takeover",
    "suspicious charge", "suspicious transaction", "i didn't buy",
    "not my transaction", "someone else used my card", "unauthorized charge",
]

ACCOUNT_COMPROMISE_KEYWORDS = [
    "account hacked", "someone else logged in", "unauthorized login",
    "unknown device signed in", "account taken over", "login from unknown",
    "someone accessed my account", "my account was breached",
]

# NOTE: These are MEDIUM — careful handling, but NOT automatic escalation
SENSITIVE_FINANCIAL_KEYWORDS = [
    "billing error", "double charged", "charged twice", "overcharged",
    "refund", "money back", "bank dispute", "payment reversed",
    "wrong amount charged", "fee dispute", "chargeback",
]

# Account access issues — MEDIUM, let LLM decide (password reset is answerable)
ACCOUNT_ACCESS_KEYWORDS = [
    "can't login", "cannot login", "locked out", "account suspended",
    "account disabled", "lost access", "forgot password", "reset password",
    "2fa issue", "two factor", "phone lost",
]

PROMPT_INJECTION_PATTERNS = [
    r"ignore (previous|all|above) instruction",
    r"you are now",
    r"act as (a|an) (different|new|other|unrestricted)",
    r"disregard (your|all) (rules|instructions|guidelines)",
    r"jailbreak",
    r"pretend (you are|to be)",
    r"do anything now",
    r"dan mode",
    r"override (safety|guidelines)",
    r"new prompt:",
    r"system prompt:",
]

IRRELEVANT_PATTERNS = [
    r"^\s*$",
    r"^(test|hello|hi|hey|ok|lol|asdf|foo|bar)\s*$",
    r"(.)\1{15,}",
]

URGENCY_KEYWORDS = [
    "urgent", "immediately", "asap", "emergency", "critical",
    "right now", "losing money", "lost money",
]


class RiskClassifier:
    def check(self, text: str) -> dict:
        t = text.lower()

        is_fraud        = self._match_any(t, FRAUD_KEYWORDS)
        is_compromised  = self._match_any(t, ACCOUNT_COMPROMISE_KEYWORDS)
        is_sensitive_fin = self._match_any(t, SENSITIVE_FINANCIAL_KEYWORDS)
        is_acct_access  = self._match_any(t, ACCOUNT_ACCESS_KEYWORDS)
        is_injected     = self._match_patterns(t, PROMPT_INJECTION_PATTERNS)
        is_irrelevant   = self._match_patterns(t, IRRELEVANT_PATTERNS)
        is_urgent       = self._match_any(t, URGENCY_KEYWORDS)

        # ── Risk level ────────────────────────────────────────────────────────
        if is_injected:
            level = "CRITICAL"
        elif is_fraud or is_compromised:
            level = "HIGH"
        elif is_sensitive_fin or is_acct_access or is_urgent:
            level = "MEDIUM"
        else:
            level = "LOW"

        # ── Escalation logic (FIXED) ──────────────────────────────────────────
        # ONLY CRITICAL and HIGH force escalation.
        # MEDIUM means "handle carefully" — the LLM decides.
        # This fixes the bug where all tickets were being escalated.
        must_escalate = level in ("CRITICAL", "HIGH")

        flags = []
        if is_fraud:          flags.append("potential_fraud")
        if is_compromised:    flags.append("account_compromise")
        if is_sensitive_fin:  flags.append("sensitive_financial")
        if is_acct_access:    flags.append("account_access")
        if is_injected:       flags.append("prompt_injection_attempt")
        if is_irrelevant:     flags.append("potentially_irrelevant")
        if is_urgent:         flags.append("urgent")

        return {
            "level":         level,
            "must_escalate": must_escalate,
            "flags":         flags,
            "is_injected":   is_injected,
            "is_irrelevant": is_irrelevant,
        }

    @staticmethod
    def _match_any(text: str, keywords: list) -> bool:
        return any(kw in text for kw in keywords)

    @staticmethod
    def _match_patterns(text: str, patterns: list) -> bool:
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)