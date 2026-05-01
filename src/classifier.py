def classify_request(issue):
    if "error" in issue or "not working" in issue:
        return "bug"
    elif "feature" in issue or "add" in issue:
        return "feature_request"
    elif "how" in issue or "help" in issue or "unable" in issue:
        return "product_issue"
    return "invalid"


def detect_product_area(issue, company):
    if company == "Visa":
        return "payments"
    elif company == "Claude":
        return "ai_usage"
    elif company == "HackerRank":
        return "coding_tests"

    if "card" in issue or "payment" in issue:
        return "payments"
    if "test" in issue or "code" in issue:
        return "coding_tests"
    if "ai" in issue:
        return "ai_usage"

    return "general"