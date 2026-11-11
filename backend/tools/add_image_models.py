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
    # outside tests run later (tools/evaluate_hyperkvasir.py; Messidor-2 check) stay on the card
    outside = {
        "esophagitis_check": (
            "data/eval/hyperkvasir/result_esophagitis.json",
            "esophagitis_check",
            "HyperKvasir (Baerum Hospital, Norway), Kvasir copies removed, scored once on 2026-11-10",
        ),
        "macular_edema_check": (
            "data/eval/messidor2/result_dme.json",
            None,
            "Messidor-2 (France), official adjudicated DME grades, 1,057 gradable photos, scored once on 2026-11-11",
        ),
    }
    for e in entries:
        if e["id"] in outside:
            path, key, label = outside[e["id"]]
            f = BACKEND / path
            if f.exists():
                r = json.loads(f.read_text())
                r = r[key] if key else r
                e["metrics"]["outside_test"] = {"set": label, **r}
                e["status_note"] = (
                    f"Unverified: on an outside set ({label.split(',')[0]}) sensitivity {r['sensitivity']}, "
                    f"specificity {r['specificity']}, AUC {r['auc']}; below the 0.80 Verified bar on sensitivity."
                )
    # 2026-11-11: macular oedema and oesophagitis were replaced by fully retrained models that passed outside tests
    swapped = {
        "macular_edema_check": (
            "dme_messidor",
            None,
            "Messidor-2 (France): 1,057 retina photos with adjudicated DME grades; whole network fine-tuned on a GPU",
            "all 455 graded IDRiD photos (India), scored once",
            "Trained by the Curanex team on Messidor-2 (Decenciere et al.; grades by Krause et al.), tested on IDRiD (Porwal et al.); ImageNet EfficientNet-B0 fully fine-tuned on a Kaggle GPU",
        ),
        "esophagitis_check": (
            "endoscopy_hk",
            "oesophagitis",
            "About 7,500 HyperKvasir endoscopy images from Baerum Hospital, Norway, labelled by endoscopists; whole network fine-tuned on a GPU",
            "Kvasir v2 (Vestre Viken, Norway), copies of training images removed: oesophagitis vs normal Z-line, scored once",
            "Trained by the Curanex team on HyperKvasir (Borgli et al. 2020), tested on Kvasir v2 (Pogorelov et al.); ImageNet EfficientNet-B0 fully fine-tuned on a Kaggle GPU",
        ),
    }
    for e in entries:
        if e["id"] in swapped:
            name, key, trained, eval_set, credit = swapped[e["id"]]
            m = json.loads((MULTI / f"{name}.json").read_text())
            t = m["outside_test"][key] if key else m["outside_test"]
            e.update(
                {
                    "trained_on": trained,
                    "credit": credit,
                    "verified": True,
                    "status_note": f"Verified on a separate dataset: sensitivity {t['sensitivity']}, AUC {t['auc']}.",
                }
            )
            e["metrics"] = {
                "accuracy": None,
                "sensitivity": t["sensitivity"],
                "specificity": t["specificity"],
                "auc": t["auc"],
                "eval_set": f"{eval_set} (n={t['n']}, {t['positives']} positive)",
                "measured": "2026-11-11",
            }
            e["limitations"] = e["limitations"].split(" Tested only")[0].split(" Tested on")[0] + (
                f" On a separate dataset it found {round(100 * t['sensitivity'])}% of cases and flagged {round(100 * (1 - t['specificity']))}% of normal images."
                " A specialist confirms it."
            )
    # 2026-11-11: AMD check from a retinal OCT scan, fine-tuned on Kermany 2018; Verified on OCTDL and confirmed on Duke
    oct_meta = json.loads((MULTI / "oct_ft.json").read_text())
    amd, duke = oct_meta["outside_test"]["AMD"], oct_meta["confirmation_test"]["amd"]["eye_level"]
    entries.append(
        {
            **entry(
                id="amd_oct_check",
                categories=["Eye"],
                tasks=["Age-related macular degeneration on OCT"],
                input="Retinal OCT scan",
                name="AMD check (retinal OCT scan)",
                adapter="workers/torch/adapters/amd_oct_check.py",
                upload_kind="fundus",
                conditions=["Age-related macular degeneration"],
                output_labels=["Possible age-related macular degeneration (AMD)", "No clear signs of AMD"],
                trained_on=f"{oct_meta['images']:,} retinal OCT scans (Kermany 2018: wet AMD, drusen, diabetic macular oedema, normal), "
                "graded by eye specialists; whole network fine-tuned on a Kaggle GPU",
                limitations="Not a diagnosis. Needs an OCT scan (the grey cross-section image from an eye clinic), not a retina photo. "
                f"On a separate clinic's scans it found {round(100 * amd['sensitivity'])}% of AMD scans and flagged "
                f"{round(100 * (1 - amd['specificity']))}% of normal ones. It does not check for diabetic macular oedema. "
                "Sudden distorted or lost central vision needs an eye doctor the same day.",
                credit="Trained by the Curanex team on Kermany et al. 2018 (OCT2017), tested on OCTDL (Kulyabin et al. 2024) and "
                "Duke / Srinivasan et al. 2014; ImageNet EfficientNet-B0 fully fine-tuned on a Kaggle GPU",
                eval_set="x",
                met={"accuracy": None, **amd, "n": amd["n"]},
                status_note=f"Verified on a separate dataset (OCTDL, Russia): sensitivity {amd['sensitivity']}, AUC {amd['auc']}; "
                f"confirmed on Duke (USA), eye level: sensitivity {duke['sensitivity']}, AUC {duke['auc']}.",
                description="AMD check from a retinal OCT scan. Age-related macular degeneration, drusen, wet AMD, choroidal "
                "neovascularisation, distorted central vision, older adults.",
            ),
            "verified": True,
        }
    )
    entries[-1]["metrics"].update(
        {
            "eval_set": f"OCTDL (Russia), AMD vs normal scans (n={amd['n']}, {amd['positives']} AMD), scored once; "
            f"confirmed once on Duke / Srinivasan 2014 (USA), {duke['n']} eyes: sensitivity {duke['sensitivity']}, AUC {duke['auc']}",
            "measured": "2026-11-11",
        }
    )
    # 2026-11-12: DME from an OCT scan (same model); threshold set on OCTDL, Verified on Duke
    dme = oct_meta["dme_test"]["duke"]
    de, db = dme["eye_level"], dme["bscan_level"]
    entries.append(
        {
            **entry(
                id="dme_oct_check",
                categories=["Eye"],
                tasks=["Diabetic macular oedema on OCT"],
                input="Retinal OCT scan",
                name="DME check (retinal OCT scan)",
                adapter="workers/torch/adapters/dme_oct_check.py",
                upload_kind="fundus",
                conditions=["Diabetic macular oedema"],
                output_labels=["Possible diabetic macular oedema (DME)", "No clear signs of DME"],
                trained_on=f"{oct_meta['images']:,} retinal OCT scans (Kermany 2018), graded by eye specialists; whole network fine-tuned on a Kaggle GPU; "
                "flagging threshold set once on OCTDL (Russia)",
                limitations="Not a diagnosis. Needs an OCT scan (the grey cross-section image), not a retina photo. "
                f"On a separate hospital's scans it found {round(100 * db['sensitivity'])}% of DME scans and flagged "
                f"{round(100 * (1 - db['specificity']))}% of normal scans. People with diabetes need a yearly eye check.",
                credit="Trained by the Curanex team on Kermany et al. 2018; threshold set on OCTDL (Kulyabin et al. 2024); tested on "
                "Duke / Srinivasan et al. 2014; ImageNet EfficientNet-B0 fully fine-tuned on a Kaggle GPU",
                eval_set="x",
                met={"accuracy": None, **db, "n": db["n"]},
                status_note=f"Verified on Duke (USA), never used before: eye level sensitivity {de['sensitivity']}, AUC {de['auc']}; "
                f"scan level sensitivity {db['sensitivity']}, AUC {db['auc']}.",
                description="Diabetic macular oedema (DME) check from a retinal OCT scan. Swelling of the macula, fluid in the retina, "
                "blurred central vision with diabetes.",
            ),
            "verified": True,
        }
    )
    entries[-1]["metrics"].update(
        {
            "eval_set": f"Duke / Srinivasan 2014 (USA), scored once: {db['n']:,} scans ({db['positives']:,} DME); eye level "
            f"{de['n']} eyes, sensitivity {de['sensitivity']}, AUC {de['auc']}. Threshold set beforehand on OCTDL.",
            "measured": "2026-11-12",
        }
    )
    ids |= {"amd_oct_check", "dme_oct_check"}
    for reg in REGISTRIES:
        data = json.loads(reg.read_text(encoding="utf-8"))
        data["models"] = [m for m in data["models"] if m["id"] not in ids] + entries
        reg.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(len(entries), "image entries written")


if __name__ == "__main__":
    main()
    from apply_live_rule import apply  # the user's live rule (Verified + near-misses)

    apply()
