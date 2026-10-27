"""What-If Counterfactual Simulator for Industrial Telemetry.

Analyzes how hypothetical adjustments to machine operating parameters
(e.g., reducing torque, cooling operating temperatures, replacing tools)
shift failure probability and equipment risk profiles.
"""
from typing import Dict, Any, List
from app.api.config import get_risk_level
from app.api.schemas import SensorInput, WhatIfResponse
from app.api.model_service import model_service
from app.api.recommendation_service import generate_maintenance_recommendations


def analyze_what_if(original: SensorInput, modified: SensorInput) -> WhatIfResponse:
    """Computes before-and-after predictions and synthesizes natural language interpretation."""
    orig_prob = model_service.predict_probability(original)
    mod_prob = model_service.predict_probability(modified)
    prob_change = round(mod_prob - orig_prob, 4)

    orig_risk = get_risk_level(orig_prob)
    mod_risk = get_risk_level(mod_prob)

    # Detect altered features
    orig_dict = original.model_dump()
    mod_dict = modified.model_dump()

    changes: List[str] = []
    unit_map = {
        "air_temp": "K",
        "process_temp": "K",
        "rotational_speed": "rpm",
        "torque": "Nm",
        "tool_wear": "min",
        "type": "tier"
    }

    for k in ["type", "air_temp", "process_temp", "rotational_speed", "torque", "tool_wear"]:
        v_orig = orig_dict[k]
        v_mod = mod_dict[k]
        if v_orig != v_mod:
            u = unit_map.get(k, "")
            name = k.replace("_", " ")
            if isinstance(v_orig, float):
                changes.append(f"{name} from {v_orig:.1f}{u} to {v_mod:.1f}{u}")
            else:
                changes.append(f"{name} from '{v_orig}' to '{v_mod}'")

    if not changes:
        interp = (
            f"No parameter modifications detected. Failure probability remains constant at {orig_prob:.1%} "
            f"({orig_risk} risk)."
        )
    else:
        direction = "decreases" if prob_change < 0 else ("increases" if prob_change > 0 else "maintains")
        delta_str = f"{abs(prob_change):.1%}"
        change_summary = "; ".join(changes)
        interp = (
            f"Modifying {change_summary} {direction} machine failure probability from "
            f"{orig_prob:.1%} ({orig_risk} risk) to {mod_prob:.1%} ({mod_risk} risk), "
            f"a net change of {delta_str}."
        )

    # Generate recommendation based on modified state
    top_contributors = [k for k, v in mod_dict.items() if k in ["torque", "tool_wear", "process_temp", "rotational_speed"]]
    rec = generate_maintenance_recommendations(top_contributors, risk_level=mod_risk, failure_probability=mod_prob)

    return WhatIfResponse(
        original_probability=round(orig_prob, 4),
        modified_probability=round(mod_prob, 4),
        probability_change=prob_change,
        original_risk_level=orig_risk,
        modified_risk_level=mod_risk,
        interpretation=interp,
        recommendation=rec
    )
