"""Doctor module answers, built only from the Curanex library and checked statement by statement (api/knowledge/verify.py).

Workflow (owner's example: "fever for 3 days, a lot of cough, difficulty breathing"):
  1. suggest_tests(case)      symptoms -> library -> tests to order, each with a cited reason
  2. differential(case)       symptoms + report values -> AI proposes candidates -> library verifies each one
                              -> "supported" (with evidence + citations) and "also_consider" (names only, no treatment)
  3. treatment(case, disease) library treatment passages -> cited plan + medicines -> doses copied from official labels (drugs.py)
Before 2 and 3, missing patient details (age, sex, pregnancy, kidney/liver disease, allergies, current medicines) are asked for.
Red flags use the app's fixed rules (no AI). Nothing here is shown to patients.
"""

from api.knowledge.drugs import medicine_card
from api.knowledge.llm import ask_json
from api.knowledge.search import search
from api.knowledge.verify import _words, keep_supported
from api.safety.rules import red_flag

MAX_CANDIDATES = 8
PASSAGE_WORDS = 220
RULES = ("Use ONLY the numbered passages. Every item must cite the passage ids it comes from (\"cite\": [\"id\"]). "
         "Write each item's text close to the passage wording; do not add facts, numbers or doses that are not in the cited passage. "
         "If the passages do not support something, leave it out. Reply with JSON only.")


def case_text(case: dict) -> str:
    p = case.get("patient") or {}
    bits = [f"{p.get('age', '?')}-year-old {p.get('sex', 'person')}"]
    if p.get("pregnant"):
        bits.append("pregnant")
    bits.append(f"symptoms: {case.get('symptoms', '')}")
    if case.get("history"):
        bits.append(f"history: {case['history']}")
    if case.get("report"):
        bits.append("report: " + ", ".join(f"{k} {v}" for k, v in case["report"].items()))
    if case.get("model_results"):
        bits.append("Verified model results: " + "; ".join(case["model_results"]))
    return ". ".join(bits)


def missing_info(case: dict, stage: str) -> list[str]:
    p = case.get("patient") or {}
    need = []
    if not p.get("age"):
        need.append("age")
    if not p.get("sex"):
        need.append("sex")
    if stage == "treatment":
        if str(p.get("sex", "")).lower().startswith("f") and 12 <= (p.get("age") or 0) <= 50 and "pregnant" not in p:
            need.append("pregnant (yes/no)")
        for k in ("weight_kg", "kidney_disease", "liver_disease", "allergies", "current_medicines"):
            if k not in p:
                need.append(k)
    return need


def _passages(hits: list[dict]) -> dict[str, dict]:
    return {h["id"]: h for h in hits}


def _numbered(passages: dict[str, dict]) -> str:
    return "\n\n".join(f"[{pid}] {p['title']} - {p['section']}: {' '.join(p['text'].split()[:PASSAGE_WORDS])}" for pid, p in passages.items())


def _cite(passages: dict[str, dict], ids: list[str]) -> list[dict]:
    return [{"id": i, "source": passages[i]["source"], "title": passages[i]["title"], "section": passages[i]["section"],
             "url": passages[i]["url"], "licence": passages[i]["licence"]} for i in ids]


def _emergency(case: dict) -> str | None:
    return red_flag(f"{case.get('symptoms', '')} {case.get('history', '')}")


def suggest_tests(case: dict) -> dict:
    passages = _passages(search(f"{case.get('symptoms', '')} evaluation diagnosis tests", k=10, sources=("StatPearls",)))
    out = {"stage": "tests", "emergency": _emergency(case), "missing_info": missing_info(case, "tests"), "tests": [], "dropped": 0}
    if not passages:
        return out | {"note": "No matching passages in the library."}
    reply = ask_json(f"You help a doctor choose investigations. {RULES} JSON: "
                     '{"tests": [{"test": "name", "text": "why, from the passage", "cite": ["id"]}]}',
                     f"Case: {case_text(case)}\n\nPassages:\n{_numbered(passages)}") or {}
    kept, dropped = keep_supported(reply.get("tests") or [], passages)
    out["tests"] = [{"test": t.get("test", ""), "why": t["text"], "sources": _cite(passages, t["cite"])} for t in kept]
    out["dropped"] = dropped
    return out


def differential(case: dict) -> dict:
    out = {"stage": "differential", "emergency": _emergency(case), "missing_info": missing_info(case, "differential"),
           "supported": [], "also_consider": [], "dropped": 0}
    proposed = ask_json("You are helping a doctor. List the most likely diagnoses for this case, most likely first. "
                        f"At most {MAX_CANDIDATES}. JSON only: " '{"candidates": ["disease name"]}', f"Case: {case_text(case)}") or {}
    candidates = [str(c).strip() for c in (proposed.get("candidates") or []) if str(c).strip()][:MAX_CANDIDATES]
    findings = f"{case.get('symptoms', '')} " + " ".join(f"{k} {v}" for k, v in (case.get("report") or {}).items())
    for disease in candidates:
        passages = _passages(search(f"{disease} {findings}", k=4, sources=("StatPearls",)))
        reply = ask_json(f"Check whether the passages support this diagnosis for this case. {RULES} JSON: "
                         '{"evidence": [{"text": "finding from the case that the passage links to this disease", "cite": ["id"]}]}',
                         f"Diagnosis: {disease}\nCase: {case_text(case)}\n\nPassages:\n{_numbered(passages)}") if passages else None
        kept, dropped = keep_supported((reply or {}).get("evidence") or [], passages)
        # evidence must also be about THIS patient: it has to share a finding word with the case (symptoms or report)
        relevant = [e for e in kept if _words(e["text"]) & _words(findings)]
        dropped += len(kept) - len(relevant)
        kept = relevant
        out["dropped"] += dropped
        if kept:
            out["supported"].append({"disease": disease, "evidence": [{"text": e["text"], "sources": _cite(passages, e["cite"])} for e in kept],
                                     "support": len(kept)})
        else:
            out["also_consider"].append(disease)
    out["supported"].sort(key=lambda d: -d["support"])
    out["note"] = "'Also consider' names were suggested by the AI but not confirmed by the library; no treatment is given for them."
    return out


def treatment(case: dict, disease: str) -> dict:
    out = {"stage": "treatment", "disease": disease, "emergency": _emergency(case), "missing_info": missing_info(case, "treatment"),
           "plan": [], "medicines": [], "referral": [], "dropped": 0}
    if out["missing_info"]:
        return out | {"note": "Please give the missing patient details first; medicine advice depends on them."}
    passages = _passages(search(f"{disease} treatment management first-line therapy", k=8, sources=("StatPearls",)))
    if not passages:
        return out | {"note": "No treatment passages for this diagnosis in the library."}
    reply = ask_json(f"You help a doctor plan treatment for {disease}. {RULES} Do not write any dose. JSON: "
                     '{"plan": [{"text": "step", "cite": ["id"]}], "medicines": [{"name": "generic medicine name", "text": "when it is used, from the passage", "cite": ["id"]}], '
                     '"referral": [{"text": "when surgery or a specialist is needed, from the passage", "cite": ["id"]}]}',
                     f"Case: {case_text(case)}\n\nPassages:\n{_numbered(passages)}") or {}
    for key in ("plan", "referral"):
        kept, dropped = keep_supported(reply.get(key) or [], passages)
        out[key] = [{"text": k["text"], "sources": _cite(passages, k["cite"])} for k in kept]
        out["dropped"] += dropped
    meds, dropped = keep_supported(reply.get("medicines") or [], passages)
    out["dropped"] += dropped
    patient = case.get("patient") or {}
    for m in meds:
        name = str(m.get("name") or "").strip()
        if name and name.lower() in m["text"].lower():  # the medicine must be named in the supported statement itself
            out["medicines"].append({"why": m["text"], "sources": _cite(passages, m["cite"]), "label": medicine_card(name, patient)})
    return out
