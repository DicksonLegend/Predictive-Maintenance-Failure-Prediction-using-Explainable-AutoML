"""Maintenance Recommendation Service.

Generates rule-based engineering actions based on top risk contributors
and machine risk level.
"""
from typing import List, Dict, Any, Union


def generate_maintenance_recommendations(
    top_contributors: List[Union[str, Dict[str, Any]]],
    risk_level: str = "Low",
    failure_probability: float = 0.0
) -> str:
    """Generate prescriptive maintenance recommendations based on top risk contributors.

    Parameters
    ----------
    top_contributors : list
        List of feature names or feature dicts (e.g. [{"feature": "torque", ...}]).
    risk_level : str
        "Low", "Medium", or "High".
    failure_probability : float
        Predicted failure probability between 0 and 1.

    Returns
    -------
    str : Bulleted or summarized maintenance action plan.
    """
    # Extract feature names
    features = []
    for item in top_contributors:
        if isinstance(item, dict) and "feature" in item:
            features.append(item["feature"].lower())
        elif isinstance(item, str):
            features.append(item.lower())

    recommendations = []

    # Check each known risk driver
    has_torque = any("torque" in f for f in features)
    has_tool_wear = any("tool_wear" in f or "wear" in f for f in features)
    has_temp = any("temp" in f for f in features)
    has_speed = any("speed" in f or "rotational" in f for f in features)

    if has_torque:
        recommendations.append("Reduce mechanical load; check torque limits; inspect drive system and gearbox.")
    
    if has_tool_wear:
        recommendations.append("Replace cutting tool immediately; schedule tool recalibration; inspect tool usage logs.")

    if has_temp:
        recommendations.append("Check heat dissipation and cooling system; inspect ventilation ducts; monitor thermal differential.")

    if has_speed:
        recommendations.append("Check motor spindle alignment; inspect bearing vibration and lubrication; verify dynamic load balance.")

    if not recommendations:
        if risk_level == "High":
            recommendations.append("Conduct comprehensive mechanical and electrical diagnostic overhaul; halt non-critical operation.")
        elif risk_level == "Medium":
            recommendations.append("Increase telemetry sampling frequency; inspect high-stress components during next shift change.")
        else:
            recommendations.append("All primary parameters within nominal operating bounds; continue automated telemetry monitoring.")

    # Prefix with urgency header
    urgency_tag = f"[{risk_level.upper()} RISK - P(Failure) = {failure_probability:.1%}]"
    action_text = " | ".join(recommendations)
    return f"{urgency_tag} Recommended Action: {action_text}"
