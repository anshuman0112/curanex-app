"""Official drug labels (US FDA, via openFDA; public domain) - the ONLY source of doses in the doctor module.
From backend/: python -m knowledge.fetch_druglabels

For each medicine we keep one label whose active ingredient(s) match exactly (so "metformin" never returns a combination
product's dose), preferring prescription labels, newest first. Sections are copied word for word; nothing is generated.
"""

import json
import time
import urllib.parse
import urllib.request

from knowledge.common import RAW, chunk, write_chunks

LICENCE = "Public domain (US FDA label via openFDA / DailyMed)"
# Indian name -> US label name. Combinations are written "A AND B" as openFDA does.
ALIASES = {"paracetamol": "acetaminophen", "salbutamol": "albuterol", "rifampicin": "rifampin", "frusemide": "furosemide",
           "glibenclamide": "glyburide", "adrenaline": "epinephrine", "lignocaine": "lidocaine", "cholecalciferol": "cholecalciferol",
           "psyllium": "psyllium husk", "fluticasone": "fluticasone propionate"}
MEDICINES = [
    # OTC list used by the patient app
    "paracetamol", "ibuprofen", "cetirizine", "clotrimazole", "psyllium", "loratadine", "aluminum hydroxide and magnesium hydroxide",
    # diabetes, heart, BP, lipids, blood
    "metformin", "glipizide", "glimepiride", "glibenclamide", "sitagliptin", "empagliflozin", "dapagliflozin", "pioglitazone", "insulin glargine",
    "amlodipine", "telmisartan", "losartan", "enalapril", "ramipril", "atenolol", "metoprolol", "propranolol", "carvedilol", "hydrochlorothiazide",
    "chlorthalidone", "furosemide", "spironolactone", "torsemide", "nifedipine", "labetalol", "methyldopa", "atorvastatin", "rosuvastatin",
    "aspirin", "clopidogrel", "warfarin", "apixaban", "heparin", "enoxaparin", "digoxin", "isosorbide mononitrate", "nitroglycerin",
    # thyroid, steroids, bone, gout
    "levothyroxine", "methimazole", "propylthiouracil", "prednisolone", "prednisone", "dexamethasone", "hydrocortisone",
    "alendronate", "allopurinol", "febuxostat", "colchicine",
    # infections (incl. TB, malaria, HIV, hepatitis)
    "amoxicillin", "amoxicillin and clavulanate potassium", "azithromycin", "doxycycline", "ciprofloxacin", "levofloxacin", "cefixime",
    "ceftriaxone", "cephalexin", "metronidazole", "nitrofurantoin", "sulfamethoxazole and trimethoprim", "isoniazid", "rifampicin",
    "pyrazinamide", "ethambutol", "artemether and lumefantrine", "chloroquine", "primaquine", "albendazole", "ivermectin", "fluconazole",
    "acyclovir", "oseltamivir", "tenofovir disoproxil fumarate", "entecavir", "sofosbuvir and velpatasvir", "dolutegravir",
    # stomach, lungs, allergy
    "omeprazole", "pantoprazole", "famotidine", "ondansetron", "loperamide", "lactulose", "albuterol", "budesonide", "montelukast",
    "fluticasone", "tiotropium",
    # pain, nerves, mind
    "diclofenac", "naproxen", "tramadol", "gabapentin", "pregabalin", "amitriptyline", "sertraline", "fluoxetine", "escitalopram",
    "alprazolam", "diazepam", "phenytoin", "levetiracetam", "valproic acid", "carbamazepine", "haloperidol", "risperidone", "olanzapine",
    # kidney, urology, women's health, supplements
    "tamsulosin", "finasteride", "sildenafil", "misoprostol", "folic acid", "ferrous sulfate", "calcium carbonate", "potassium chloride",
]
KEEP = ["boxed_warning", "indications_and_usage", "dosage_and_administration", "dosage_forms_and_strengths", "contraindications",
        "warnings_and_cautions", "warnings", "drug_interactions", "use_in_specific_populations", "pregnancy", "pediatric_use",
        "geriatric_use", "overdosage", "purpose", "uses", "directions", "do_not_use", "ask_doctor", "stop_use"]
TITLES = {k: k.replace("_", " ").capitalize() for k in KEEP}


def query(name: str) -> list[dict]:
    q = urllib.parse.quote(f'openfda.generic_name.exact:"{name.upper()}"')
    url = f"https://api.fda.gov/drug/label.json?search={q}&limit=20&sort=effective_time:desc"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r).get("results", [])
    except Exception as e:  # 404 = no label
        print("no label:", name, type(e).__name__)
        return []


def query_salt(name: str) -> list[dict]:
    """Fallback for medicines listed under a salt name (e.g. ATORVASTATIN CALCIUM, LEVOTHYROXINE SODIUM)."""
    q = urllib.parse.quote(f'openfda.generic_name:"{name}"')
    url = f"https://api.fda.gov/drug/label.json?search={q}&limit=50&sort=effective_time:desc"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r).get("results", [])
    except Exception:
        return []


SALTS = ("SODIUM", "POTASSIUM", "CALCIUM", "HYDROCHLORIDE", "HCL", "MESYLATE", "TARTRATE", "SUCCINATE", "BESYLATE", "MALEATE",
         "SULFATE", "PHOSPHATE", "FUMARATE", "BROMIDE", "MAGNESIUM", "ACETATE", "DISODIUM", "MONOHYDRATE", "TRIHYDRATE", "CITRATE")


def same_medicine(generic: str, name: str) -> bool:
    """Exactly the asked medicine, allowing only a salt word after it (never a different or combined active ingredient)."""
    g, n = generic.upper().strip(), name.upper().strip()
    if g == n:
        return True
    if not g.startswith(n + " ") or " AND " in g or "," in g:
        return False
    return all(w in SALTS for w in g[len(n):].split())


def pick(name: str, results: list[dict]) -> dict | None:
    want = sorted(p.strip().upper() for p in name.upper().split(" AND "))
    good = []
    for r in results:
        o = r.get("openfda", {})
        gen = [g.upper() for g in o.get("generic_name", [])]
        subs = sorted(s.upper() for s in o.get("substance_name", []))
        if len(gen) != 1 or not same_medicine(gen[0], name) or (subs and len(subs) != len(want)):
            continue
        rx = "PRESCRIPTION" in " ".join(o.get("product_type", []))
        good.append((rx, r.get("effective_time", ""), r))
    good.sort(key=lambda g: (g[0], g[1]), reverse=True)
    return good[0][2] if good else None


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    chunks, found, missing = [], [], []
    for asked in MEDICINES:
        name = ALIASES.get(asked, asked)
        label = pick(name, query(name))
        time.sleep(1.6)  # stay under openFDA's anonymous rate limit
        if not label and " and " not in name:
            label = pick(name, query_salt(name))
            time.sleep(1.6)
        if not label:
            missing.append(asked)
            continue
        set_id = label.get("set_id", "")
        url = f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}"
        title = f"{asked.title()} - official drug label" + (f" (US name: {name})" if name != asked else "")
        for key in KEEP:
            text = " ".join(label.get(key) or [])
            if text.strip():
                chunks += chunk("Drug label (FDA, public domain)", title, TITLES[key], url, LICENCE, text, drug=asked, us_name=name,
                                field=key, effective=label.get("effective_time", ""), keep_short=True)
        found.append(asked)
    n = write_chunks("druglabels", chunks)
    print(f"labels found {len(found)}, missing {len(missing)}: {missing}; chunks {n}")


if __name__ == "__main__":
    main()
