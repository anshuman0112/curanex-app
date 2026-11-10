"""Registry entries for image models trained by the Curanex team on 2026-11-10 (macular oedema, dermoscopy, oesophagitis).
Writes backend/registry/model_registry.json and docs/model_registry.json (kept identical). Run from backend/."""

import json
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
MULTI = BACKEND / "workers" / "torch" / "models" / "multi"
REGISTRIES = [BACKEND / "registry" / "model_registry.json", BACKEND.parent / "docs" / "model_registry.json"]


def metrics(name: str, finding: str) -> dict:
    m = json.loads((MULTI / f"{name}.json").read_text())
    t = m["per_class"][finding] if "per_class" in m else m["test"]
    return {
        "accuracy": m.get("accuracy", t.get("accuracy")),
        "sensitivity": t["sensitivity"],
        "specificity": t["specificity"],
        "auc": t["auc"],
        "n": t["n"],
    }


def entry(**k) -> dict:
    met = k.pop("met")
    return {
        "repo": "Curanex team",
        "url": None,
        "modality": "medical_image_upload",
        "status": "W",
        "app_tier": "C",
        "tier": "C",
        "wave": "replacement",
        "worker": "torch",
        "input_schema": {"type": "object", "required": [], "properties": {}},
        "integrated": True,
        "verified": False,
        "license": "See dataset",
        "metrics": {
            "accuracy": met["accuracy"],
            "sensitivity": met["sensitivity"],
            "specificity": met["specificity"],
            "auc": met["auc"],
            "eval_set": k.pop("eval_set").format(n=met["n"]),
            "measured": "2026-11-10",
        },
        **k,
    }


def main() -> None:
    dme, derm, eso = metrics("idrid_dme", "Macular_edema"), metrics("dermoscopy", "suspicious"), metrics("esophagitis", "Esophagitis")
    entries = [
        entry(
            id="macular_edema_check",
            categories=["Eye"],
            tasks=["Diabetic macular oedema"],
            input="Retina photo",
            name="Macular oedema check (retina photo)",
            adapter="workers/torch/adapters/macular_edema_check.py",
            upload_kind="fundus",
            conditions=["Diabetic macular oedema"],
            output_labels=["Possible diabetic macular oedema", "No clear signs of macular oedema"],
            trained_on="375 retina photos from an eye clinic in Nanded, India (IDRiD), graded for macular oedema by experts",
            limitations="Not a diagnosis. Needs a retina photo from an eye clinic's fundus camera; phone photos of an eye won't work. "
            f"Tested only on IDRiD's own test images, where it flagged about {round(100 * (1 - dme['specificity']))}% of eyes without oedema. "
            "An eye doctor confirms macular oedema with an OCT scan. People with diabetes need a yearly retina check.",
            credit="Trained by the Curanex team on the Indian Diabetic Retinopathy Image Dataset (IDRiD, Porwal et al.), using ImageNet-pretrained EfficientNet-B0",
            eval_set="IDRiD official test images (n={n} with oedema), scored once",
            met=dme,
            status_note="Unverified: tested only on the same dataset's test images.",
            description="Diabetic macular oedema check from a retina photo. Swelling of the macula, blurred central vision with diabetes, macular edema.",
        ),
        entry(
            id="dermoscopy_check",
            categories=["Cancer"],
            tasks=["Suspicious skin lesion on dermoscopy"],
            input="Dermoscope image",
            name="Dermoscopy skin lesion check (clinic)",
            adapter="workers/torch/adapters/dermoscopy_check.py",
            upload_kind="dermoscopy",
            conditions=["Suspicious skin lesion on dermoscopy"],
            output_labels=[
                "Possible suspicious lesion (melanoma, basal cell carcinoma or actinic keratosis)",
                "Lesion doesn't look suspicious on this image",
            ],
            trained_on="7,002 dermoscope images from Vienna and Queensland (HAM10000), split by lesion, mostly biopsy-confirmed",
            limitations="Clinic-grade: needs a dermoscope (magnified skin) image, not an ordinary phone photo; for phone photos use the skin lesion check. "
            f"Not a diagnosis. Tested only on held-out HAM10000 lesions, where it flagged about {round(100 * (1 - derm['specificity']))}% of harmless lesions. "
            "Mostly fair-skinned patients. Any changing, bleeding or new mole needs a dermatologist.",
            credit="Trained by the Curanex team on HAM10000 (Tschandl, Rosendahl, Kittler), using ImageNet-pretrained EfficientNet-B0",
            eval_set="Held-out HAM10000 lesions, split by lesion (n={n}), scored once on 2026-11-06",
            met=derm,
            status_note="Unverified: tested only on held-out images from the same dataset.",
            description="For doctors: dermoscope (dermatoscopy) image of a mole or skin spot, is this lesion suspicious for melanoma.",
        ),
        entry(
            id="esophagitis_check",
            categories=["Gastroenterology"],
            tasks=["Oesophagitis on endoscopy"],
            input="Endoscopy image",
            name="Oesophagitis check (endoscopy)",
            adapter="workers/torch/adapters/esophagitis_check.py",
            upload_kind="endoscopy",
            conditions=["Oesophagitis"],
            output_labels=["Possible oesophagitis", "No clear signs of oesophagitis"],
            trained_on="700 endoscopy images of the oesophagus-stomach junction from Vestre Viken hospitals, Norway (Kvasir), labelled by endoscopists",
            limitations="Not a diagnosis. Needs an endoscopy image of the lower oesophagus. Tested only on held-out Kvasir images, "
            f"where it flagged about {round(100 * (1 - eso['specificity']))}% of normal junctions. The endoscopist's report decides.",
            credit="Trained by the Curanex team on the Kvasir dataset (Pogorelov et al.), using ImageNet-pretrained EfficientNet-B0",
            eval_set="Held-out Kvasir images (n={n} with oesophagitis), scored once",
            met=eso,
            status_note="Unverified: on an outside set (HyperKvasir, Kvasir copies removed) sensitivity 0.766, AUC 0.848, below the 0.80 Verified bar.",
            description="Endoscopy image of the food pipe: oesophagitis, reflux damage, acid reflux, GERD on endoscopy, Z-line.",
        ),
    ]
    entries = [e for e in entries if e["id"] != "dermoscopy_check"]  # failed the BCN20000 outside test (AUC 0.697): off
    ids = {e["id"] for e in entries} | {"dermoscopy_check"}
    for reg in REGISTRIES:
        data = json.loads(reg.read_text(encoding="utf-8"))
        data["models"] = [m for m in data["models"] if m["id"] not in ids] + entries
        reg.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(len(entries), "image entries written")


if __name__ == "__main__":
    main()
