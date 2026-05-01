def assess_risk(issue):
    high_risk_keywords = ["fraud", "hacked", "unauthorized", "stolen"]
    return "high" if any(k in issue for k in high_risk_keywords) else "low"


def decide_action(risk, retrieved):
    if risk == "high":
        return "escalated"
    if not retrieved:
        return "escalated"
    return "replied"