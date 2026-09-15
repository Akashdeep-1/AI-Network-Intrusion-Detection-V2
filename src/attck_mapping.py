"""MITRE ATT&CK mapping — rule-based vs model-derived, separated."""
RULE_MAP = {
    "Benign": {"tactic":"—","technique":"—","id":"—","source":"Rule-Based"},
    "DDoS": {"tactic":"Impact","technique":"Network Denial of Service","id":"T1498","source":"Rule-Based"},
    "DoS": {"tactic":"Impact","technique":"Network Denial of Service","id":"T1498","source":"Rule-Based"},
    "Mirai": {"tactic":"Execution","technique":"Botnet","id":"T1584","source":"Rule-Based"},
    "Recon": {"tactic":"Reconnaissance","technique":"Active Scanning","id":"T1595","source":"Rule-Based"},
    "Spoofing": {"tactic":"Defense Evasion","technique":"Spoofing","id":"T1036","source":"Rule-Based"},
    "WebAttack": {"tactic":"Initial Access","technique":"Exploit Public-Facing Application","id":"T1190","source":"Rule-Based"},
    "BruteForce": {"tactic":"Credential Access","technique":"Brute Force","id":"T1110","source":"Rule-Based"},
    "OtherAttack": {"tactic":"Discovery","technique":"Vulnerability Scanning","id":"T1046","source":"Rule-Based"},
}
def map_attack(category: str, model_confidence: float | None = None) -> dict:
    base = RULE_MAP.get(category, {"tactic":"Unknown","technique":"Unknown","id":"—","source":"Rule-Based"})
    out = dict(base)
    out["category"]=category
    if model_confidence is not None:
        out["model_confidence"]=model_confidence
        out["attribution"]="Heuristic rule (static) — model provides Predicted_Category & Confidence, not ATT&CK"
    return out
