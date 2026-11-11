"""Live rule chosen by the user on 2026-11-11 ("Verified + near-misses").

A model is live only if it is Verified, or it was tested on real outside data (another hospital, country or survey year) and reached
sensitivity and AUC >= 0.77 there. Every other model moves to "In testing": still listed with its real scores, never used for
patient results or offered by the AI. Run from backend/ after the add_* scripts (they call it themselves).
"""

import json
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REGISTRIES = [BACKEND / "registry" / "model_registry.json", BACKEND.parent / "docs" / "model_registry.json"]
# Unverified but tested on real outside data at >= 0.77 sensitivity and AUC (fixed on 2026-11-11)
NEAR_MISS = {
    "survey_skin_cancer": "BRFSS 2025: sensitivity 0.842, AUC 0.785",
    "survey_vision_loss": "BRFSS 2025: sensitivity 0.829, AUC 0.777",
    "survey_hearing_difficulty": "BRFSS 2025: sensitivity 0.835, AUC 0.793",
    "survey_high_blood_pressure": "BRFSS 2025: sensitivity 0.829, AUC 0.797",
    "nhanes_low_bone_density": "NHANES 2009-10: sensitivity 0.823, AUC 0.778",
}


def apply() -> tuple[int, int, int]:
    counts = (0, 0, 0)
    for reg in REGISTRIES:
        d = json.loads(reg.read_text(encoding="utf-8"))
        live = verified = testing = 0
        for m in d["models"]:
            if m.get("in_testing"):
                m["integrated"] = True  # re-evaluate below
            if not m.get("integrated"):
                continue
            if m.get("verified") or m["id"] in NEAR_MISS:
                m.pop("in_testing", None)
                live += 1
                verified += bool(m.get("verified"))
                if m["id"] in NEAR_MISS:
                    m["near_miss"] = NEAR_MISS[m["id"]]
            else:
                m["integrated"] = False
                m["in_testing"] = True
                testing += 1
        reg.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        counts = (live, verified, testing)
    return counts


if __name__ == "__main__":
    live, verified, testing = apply()
    print(f"live {live} (Verified {verified}, near-miss {live - verified}), in testing {testing}")
