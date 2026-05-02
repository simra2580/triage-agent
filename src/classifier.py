import re

FRAUD_KEYWORDS = [
    "fraud", "unauthorized", "not me", "stolen",
    "scam", "phishing", "suspicious transaction",
    "someone used my card", "not my transaction"
]

ACCOUNT_ISSUES = [
    "cannot login", "can't login", "forgot password",
    "account locked", "reset password"
]

PAYMENT_ISSUES = [
    "payment failed", "transaction failed", "charged",
    "refund", "money deducted"
]

URGENCY = [
    "urgent", "immediately", "asap", "right now"
]


class RiskClassifier:
    def check(self, text: str):
        t = text.lower()

        is_fraud = any(k in t for k in FRAUD_KEYWORDS)
        is_account = any(k in t for k in ACCOUNT_ISSUES)
        is_payment = any(k in t for k in PAYMENT_ISSUES)
        is_urgent = any(k in t for k in URGENCY)

        # Risk logic
        if is_fraud:
            level = "high"
            must_escalate = True

        elif is_account:
            level = "medium"
            must_escalate = False

        elif is_payment:
            level = "medium"
            must_escalate = False

        elif is_urgent:
            level = "medium"
            must_escalate = True

        else:
            level = "low"
            must_escalate = False

        flags = []
        if is_fraud:
            flags.append("fraud")
        if is_account:
            flags.append("account_issue")
        if is_payment:
            flags.append("payment_issue")
        if is_urgent:
            flags.append("urgent")

        return {
            "level": level,
            "must_escalate": must_escalate,
            "flags": flags
        }