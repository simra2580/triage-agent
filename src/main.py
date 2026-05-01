import pandas as pd

# Fake corpus (no CSV needed)
corpus = [
    "Reset your password using forgot password option",
    "Visa card unauthorized transaction contact support immediately",
    "Payment failed retry later",
    "HackerRank test not loading refresh page",
    "Claude usage depends on plan limits",
    "Login issues clear cookies or reset password"
]

# Load issues
issues_df = pd.read_csv("../data/support_issues.csv")

results = []

for _, row in issues_df.iterrows():
    issue = str(row["issue"]).lower()

    # simple classification
    if "error" in issue or "not working" in issue:
        request_type = "bug"
    elif "feature" in issue:
        request_type = "feature_request"
    else:
        request_type = "product_issue"

    # simple product area
    if "card" in issue or "payment" in issue:
        product_area = "payments"
    elif "test" in issue or "code" in issue:
        product_area = "coding_tests"
    else:
        product_area = "general"

    # simple retrieval
    response = None
    for doc in corpus:
        if any(word in doc.lower() for word in issue.split()):
            response = doc
            break

    # decision
    if response:
        status = "replied"
    else:
        status = "escalated"
        response = "Escalated to human support"

    results.append({
        "status": status,
        "product_area": product_area,
        "response": response,
        "justification": "basic logic applied",
        "request_type": request_type
    })

# Save output
pd.DataFrame(results).to_csv("../output/results.csv", index=False)

print("✅ WORKING! Check output/results.csv")