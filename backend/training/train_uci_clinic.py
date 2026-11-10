"""Clinic-grade models from public UCI datasets: values a doctor or lab enters, not a patient. Trained by the Curanex team.
From backend/: python training/train_uci_clinic.py [name ...]

- fetal_ctg: fetal distress from cardiotocography (CTG) summary values. UCI Cardiotocography (Ayres de Campos et al.,
  2,126 CTGs, Portugal). Suspect or pathological (three obstetricians' consensus) against normal.
- thyroid_cancer_recurrence: recurrence of treated differentiated thyroid cancer. UCI 915 (383 patients, Iran).
- breast_fna: malignant breast lump from fine-needle aspirate cytology scores. UCI Breast Cancer Wisconsin (Original),
  699 samples, University of Wisconsin.

Rules fixed before testing: 60/20/20 stratified split; logistic regression on a small feature set chosen up front (the
form can ask five questions); threshold highest with validation sensitivity >= 0.85; test scored once. No outside set,
so a pass at the user's 0.75 bar makes a model live but Unverified.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
UCI = BACKEND / "data" / "eval" / "uci"
OUT = BACKEND / "workers" / "tabular" / "models" / "clinic"


def fetal_ctg():
    d = pd.read_excel(UCI / "193" / "CTG.xls", sheet_name="Raw Data").dropna(subset=["NSP"])
    num = {"ASTV": "astv", "ALTV": "altv", "AC": "accelerations", "DP": "prolonged_decelerations", "LB": "baseline_fhr",
           "MSTV": "mstv", "UC": "contractions", "DL": "light_decelerations"}
    x = pd.DataFrame({v: d[k].astype(float) for k, v in num.items()})
    return x, (d["NSP"] >= 2).astype(int).to_numpy(), {}


def thyroid_cancer_recurrence():
    d = pd.read_csv(UCI / "915" / "Thyroid_Diff.csv")
    cats = {"Risk": ["Low", "Intermediate", "High"], "N": ["N0", "N1a", "N1b"],
            "Response": ["Excellent", "Indeterminate", "Biochemical Incomplete", "Structural Incomplete"],
            "Focality": ["Uni-Focal", "Multi-Focal"], "T": ["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"]}
    x = pd.DataFrame({"age": d["Age"].astype(float), "male": (d["Gender"] == "M").astype(float)})
    for k, levels in cats.items():
        for lv in levels[1:]:
            x[f"{k}={lv}"] = (d[k] == lv).astype(float)
    return x, (d["Recurred"] == "Yes").astype(int).to_numpy(), cats


def breast_fna():
    cols = ["id", "clump_thickness", "cell_size_uniformity", "cell_shape_uniformity", "marginal_adhesion", "epithelial_cell_size",
            "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "class"]
    d = pd.read_csv(UCI / "15" / "breast-cancer-wisconsin.data", names=cols, na_values="?")
    d = d.drop_duplicates(subset=cols[1:])  # identical rows would sit on both sides of the split
    x = d[["clump_thickness", "cell_size_uniformity", "cell_shape_uniformity", "bare_nuclei", "bland_chromatin", "normal_nucleoli",
           "marginal_adhesion", "mitoses"]].astype(float)
    return x, (d["class"] == 4).astype(int).to_numpy(), {}


def appendicitis():
    """Regensburg Pediatric Appendicitis (Marcinkevičs et al. 2023, 782 children, Germany). Symptoms and blood tests only:
    ultrasound findings, Alvarado/PAS scores and the doctors' presumptive diagnosis are left out."""
    d = pd.read_excel(UCI / "938" / "app_data.xlsx").dropna(subset=["Diagnosis"])
    yn = lambda c: d[c].map({"yes": 1.0, "no": 0.0})  # noqa: E731
    x = pd.DataFrame({
        "lower_right_pain": yn("Lower_Right_Abd_Pain"), "migratory_pain": yn("Migratory_Pain"), "wbc": d["WBC_Count"].astype(float),
        "crp": d["CRP"].astype(float), "age": d["Age"].astype(float), "nausea": yn("Nausea"), "loss_of_appetite": yn("Loss_of_Appetite"),
        "temperature": d["Body_Temperature"].astype(float), "neutrophils": d["Neutrophil_Percentage"].astype(float),
        "pain_on_coughing": yn("Coughing_Pain"),
    })
    return x, (d["Diagnosis"] == "appendicitis").astype(int).to_numpy(), {}


def mammo_mass():
    """UCI Mammographic Mass (Elter et al. 2007, 961 masses, Erlangen): BI-RADS, age, shape, margin, density; biopsy truth."""
    cols = ["birads", "age", "shape", "margin", "density", "severity"]
    d = pd.read_csv(UCI / "161" / "mammographic_masses.data", names=cols, na_values="?")
    d = d[d["birads"].between(0, 6)]  # a few entries are typos (55)
    return d[["birads", "age", "shape", "margin", "density"]].astype(float), d["severity"].astype(int).to_numpy(), {}


def glioma_grade():
    """UCI Glioma Grading (TCGA LGG and GBM, 839 patients): glioblastoma (grade 4) against lower-grade glioma from age and
    the most informative gene tests. Race is left out."""
    d = pd.read_csv(UCI / "759" / "TCGA_InfoWithGrade.csv")
    x = d[["IDH1", "Age_at_diagnosis", "ATRX", "PTEN", "EGFR", "TP53", "CIC", "NF1", "Gender"]].astype(float)
    x.columns = ["idh1", "age", "atrx", "pten", "egfr", "tp53", "cic", "nf1", "male"]
    return x, d["Grade"].astype(int).to_numpy(), {}


def psoriasis():
    """UCI Dermatology (Güvenir et al., 366 patients, Turkey): psoriasis against five other scaly red skin diseases, from the
    dermatologist's clinical grades only (0-3); the biopsy (histopathology) features are left out."""
    cols = ["erythema", "scaling", "definite_borders", "itching", "koebner", "polygonal_papules", "follicular_papules", "oral_mucosa",
            "knee_elbow", "scalp", "family_history"]
    d = pd.read_csv(UCI / "33" / "dermatology.data", header=None, na_values="?")
    x = pd.DataFrame({c: d[i].astype(float) for i, c in enumerate(cols)})
    x["age"] = d[33].astype(float)
    order = ["scaling", "definite_borders", "knee_elbow", "scalp", "koebner", "erythema", "itching", "polygonal_papules", "follicular_papules",
             "oral_mucosa", "family_history", "age"]
    return x[order], (d[34] == 1).astype(int).to_numpy(), {}


def gallstones():
    """UCI Gallstone (Esen et al. 2024, 319 people, Turkey): gallstone disease from age, BMI and routine blood tests.
    Body-composition scanner values are left out (not routine); sex is left out because its coding is not documented."""
    d = pd.read_excel(UCI / "1150" / "dataset-uci.xlsx", engine="calamine")
    cols = {"C-Reactive Protein (CRP)": "crp", "Vitamin D": "vitamin_d", "Age": "age", "Body Mass Index (BMI)": "bmi",
            "High Density Lipoprotein (HDL)": "hdl", "Hemoglobin (HGB)": "hemoglobin", "Triglyceride": "triglycerides",
            "Total Cholesterol (TC)": "cholesterol", "Glucose": "glucose", "Alkaline Phosphatase (ALP)": "alp",
            "Aspartat Aminotransferaz (AST)": "ast", "Alanin Aminotransferaz (ALT)": "alt"}
    x = pd.DataFrame({v: d[k].astype(float) for k, v in cols.items()})
    return x, d["Gallstone Status"].astype(int).to_numpy(), {}


def cirrhosis():
    """UCI Cirrhosis Patient Survival (Mayo Clinic PBC trial, 418 patients with primary biliary cholangitis): biopsy stage 4
    (cirrhosis) against stages 1-3, from blood tests and examination findings. Treatment, survival and copper are left out."""
    d = pd.read_csv(UCI / "878" / "cirrhosis.csv").dropna(subset=["Stage"])
    yn = lambda c: d[c].map({"Y": 1.0, "N": 0.0})  # noqa: E731
    x = pd.DataFrame({
        "platelets": d["Platelets"].astype(float), "albumin": d["Albumin"].astype(float), "bilirubin": d["Bilirubin"].astype(float),
        "prothrombin": d["Prothrombin"].astype(float), "hepatomegaly": yn("Hepatomegaly"), "ascites": yn("Ascites"), "spiders": yn("Spiders"),
        "edema": d["Edema"].map({"N": 0.0, "S": 0.5, "Y": 1.0}), "age": d["Age"].astype(float) / 365.25, "ast": d["SGOT"].astype(float),
        "alp": d["Alk_Phos"].astype(float),
    })
    return x, (d["Stage"] == 4).astype(int).to_numpy(), {}


MODELS = {"fetal_ctg": fetal_ctg, "thyroid_cancer_recurrence": thyroid_cancer_recurrence, "breast_fna": breast_fna,
          "appendicitis": appendicitis, "mammo_mass": mammo_mass, "glioma_grade": glioma_grade, "psoriasis": psoriasis,
          "gallstones": gallstones, "cirrhosis": cirrhosis}


def report(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "positives": int(y.sum()), "accuracy": round(float((tp + tn) / len(y)), 3),
            "sensitivity": round(float(tp / max(tp + fn, 1)), 3), "specificity": round(float(tn / max(tn + fp, 1)), 3),
            "auc": round(float(roc_auc_score(y, p)), 3)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or list(MODELS):
        xdf, y, cats = MODELS[name]()
        x = xdf.to_numpy(float)
        idx = np.arange(len(y))
        tr, rest = train_test_split(idx, test_size=0.4, random_state=42, stratify=y)
        va, te = train_test_split(rest, test_size=0.5, random_state=42, stratify=y[rest])
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(x[tr], y[tr])
        pv = model.predict_proba(x[va])[:, 1]
        t = float(max(v for v in np.unique(pv) if (pv[y[va] == 1] >= v).mean() >= 0.85))
        test = report(y[te], model.predict_proba(x[te])[:, 1], t)
        live = test["sensitivity"] >= 0.75 and test["auc"] >= 0.75
        print(f"{name:26s} rows {len(y)} ({int(y.sum())}+) test once {test} {'LIVE' if live else 'off'}", flush=True)
        imp, sc, clf = (model.steps[i][1] for i in range(3))
        (OUT / f"{name}.json").write_text(json.dumps({
            "features": list(xdf.columns), "categories": cats, "impute": imp.statistics_.tolist(), "mean": sc.mean_.tolist(),
            "scale": sc.scale_.tolist(), "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]), "threshold": t,
            "metrics": {"test": test}, "live": live, "training_rows": int(len(tr)),
        }, indent=1))


if __name__ == "__main__":
    main()
