"""
src/classifier.py  —  Rule-Based Risk & Safety Pre-Classifier

Runs BEFORE the LLM call.  Fast keyword matching to catch:
  - Fraud / unauthorized transaction signals
  - Account compromise signals
  - PII exposure risk
  - Malicious / prompt injection attempts
  - Out-of-scope / nonsense inputs

Returns a risk dict that the LLM prompt uses to guide its decision.
"""

from __future__ import annotations
import re


# ── Risk signal categories ─────────────────────────────────────────────────────

FRAUD_KEYWORDS = [
    "fraud", "unauthorized", "unauthorised", "not me", "didn't make",
    "didn't authorize", "stolen", "hacked", "compromised", "scam",
    "phishing", "identity theft", "account takeover", "suspicious charge",
    "suspicious transaction", "i didn't buy", "not my transaction",
    "dispute", "chargeback", "someone else used my card",
]

ACCOUNT_COMPROMISE_KEYWORDS = [
    "can't login", "cannot login", "locked out", "account suspended",
    "account disabled", "lost access", "forgot password", "reset my account",
    "2fa issue", "two factor", "phone lost", "account hacked",
    "someone else logged in", "unauthorized login", "unknown device",
]

SENSITIVE_FINANCIAL_KEYWORDS = [
    "billing error", "double charged", "charged twice", "overcharged",
    "refund", "money back", "bank dispute", "payment reversed",
    "wrong amount", "fee dispute",
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
]

IRRELEVANT_PATTERNS = [
    r"^\s*$",                           # blank
    r"^(test|hello|hi|hey|ok|lol)\s*$", # trivial
    r"(.)\1{10,}",                      # repeated chars (e.g. "aaaaaaaaaa")
]

URGENCY_KEYWORDS = [
    "urgent", "immediately", "asap", "emergency", "critical", "right now",
    "losing money", "lost money", "account at risk",
]


class RiskClassifier:
    """
    Lightweight, zero-LLM risk assessment.
    Returns a structured dict consumed by the triage prompt.
    """

    def check(self, text: str) -> dict:
        text_lower = text.lower()

        is_fraud         = self._match_any(text_lower, FRAUD_KEYWORDS)
        is_compromised   = self._match_any(text_lower, ACCOUNT_COMPROMISE_KEYWORDS)
        is_sensitive_fin = self._match_any(text_lower, SENSITIVE_FINANCIAL_KEYWORDS)
        is_injected      = self._match_patterns(text_lower, PROMPT_INJECTION_PATTERNS)
        is_irrelevant    = self._match_patterns(text_lower, IRRELEVANT_PATTERNS)
        is_urgent        = self._match_any(text_lower, URGENCY_KEYWORDS)

        # Determine overall risk level
        if is_injected:
            level = "CRITICAL"
        elif is_fraud or is_compromised:
            level = "HIGH"
        elif is_sensitive_fin or is_urgent:
            level = "MEDIUM"
        else:
            level = "LOW"

        # Mandatory escalation if HIGH or CRITICAL
        must_escalate = level in ("HIGH", "CRITICAL") or is_injected

        flags = []
        if is_fraud:         flags.append("potential_fraud")
        if is_compromised:   flags.append("account_compromise")
        if is_sensitive_fin: flags.append("sensitive_financial")
        if is_injected:      flags.append("prompt_injection_attempt")
        if is_irrelevant:    flags.append("potentially_irrelevant")
        if is_urgent:        flags.append("urgent")

        return {
            "level":         level,
            "must_escalate": must_escalate,
            "flags":         flags,
            "is_injected":   is_injected,
            "is_irrelevant": is_irrelevant,
        }

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _match_any(text: str, keywords: list[str]) -> bool:
        return any(kw in text for kw in keywords)

    @staticmethod
    def _match_patterns(text: str, patterns: list[str]) -> bool:
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)
    