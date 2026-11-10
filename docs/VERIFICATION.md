# Curanex verification

Final run on 2026-11-10 against the exported web build, the API and the three Docker workers. Earlier runs are in git history.

```bash
python tools/verify_e2e.py safety          # emergency messages
python tools/verify_care.py out.json       # symptom engine cases
```

## Summary

| Check | Result |
|---|---|
| Live models | 50 (21 Verified, 29 Unverified) covering 84 conditions (updated 2026-11-10; live bar 0.75, Verified 0.80) |
| Experimental models | 8, kept out of patient results |
| Live end-to-end check, one call per model through the API | 21 of 22 calls gave the expected result (see below) |
| Emergency red-team set, English, Hinglish and Hindi script | 59 of 59 escalate without reaching the AI |
| Symptom engine, internal cases | 30 of 30 plans; causes 30, tests 30, urgency 29; no prescription drugs or doses |
| Backend tests and model search test | Pass (89 tests after Phase 8) |
| Type check and web build | Clean |

## Live models

Verified means the model reached sensitivity >= 0.80 and AUC >= 0.80 on data it never saw, with thresholds fixed beforehand.

| Model | Status | Tested on | Sensitivity | Specificity | AUC |
|---|---|---|---:|---:|---:|
| Chest X-ray (6 findings) | Verified | VinBigData, Vietnam | 0.908 | 0.904 | 0.968 |
| Type 2 diabetes | Verified | Held-out Pima | 0.944 | 0.560 | 0.807 |
| Heart disease | Verified | UCI, four hospitals | 0.855 | 0.661 | 0.835 |
| Underactive thyroid | Verified | UCI test file | 0.873 | 0.989 | 0.995 |
| Tooth decay | Verified | DentalAI | 0.894 | 0.798 | 0.929 |
| Knee arthritis | Verified | Repo test set | 0.809 | 0.963 | 0.965 |
| Cataract | Verified | Independent retina set | 0.893 | 0.829 | 0.939 |
| Glaucoma | Verified | Independent retina set | 0.907 | 0.752 | 0.869 |
| Arm and hand X-ray | Verified | MURA validation | 0.836 | 0.861 | 0.903 |
| Skin lesion | Verified | Smartphone photos, PAD-UFES-20 | 0.880 | 0.860 | 0.943 |
| Brain MRI | Verified | Separate MRI collection | 0.871 | 0.812 | 0.935 |
| Diabetic retinopathy | Verified | IDRiD, India | 0.966 | 0.473 | 0.943 |
| Liver | Unverified | India held-out | 0.934 | 0.224 | 0.770 |
| Gestational diabetes | Unverified | Few inputs | 0.805 | 0.495 | 0.752 |
| Malaria cell | Unverified | Failed another lab (AUC 0.661) | | | |
| Skin shortlist (19 groups) | Unverified | DermNet only; first choice right 40% | | | |
| Endoscopy | Unverified | Kvasir only | | | |
| Mouth and tooth | Unverified | One collection; possible source bias | | | |
| Lung CT | Unverified | Failed a separate collection | | | |
| Kidney CT | Unverified | No independent set found | | | |

## Final end-to-end check

Every live model was called once through the real upload or form path, and each result carried its training data, limitations and credit.

- 20 of 22 calls returned the expected result.
- Malaria missed one real parasite cell. That is a model miss, consistent with its Unverified status.
- Gestational diabetes refused a male test profile, which is correct; with a female profile it returned a result.

## Removed or never switched on

TB (failed across sources), the repo glaucoma model (AUC 0.47 to 0.57 on independent photos), maternal risk (labels built from inputs), chronic kidney disease (hand-picked healthy group), overactive thyroid (sensitivity 0.60), breast ultrasound (best AUC 0.788 on another hospital), oral cancer (106 images) and bone age (13.9 months against a 12-month bar). Reasons are in INTEGRATION_REPORT.md.

## Phase 8 additions (2026-11-10)

| Feature | What was checked | Result |
|---|---|---|
| Voice health check (Parkinson's voice screening) | Test once on 51 unseen people (UCI, Sakar 2018) | Sensitivity 0.886, specificity 0.562, AUC 0.764. **Experimental only**, below the bar and never tested on phone microphones |
| Voice check pipeline | Synthetic sustained vowels as webm/opus through the live API; junk audio, silence, missing sex | Measured and answered; bad input refused with a message; recording deleted |
| Health alert in your area | Dengue weeks in 27 Karnataka districts, 2022-2023, weather only | AUC 0.667. The card leads with measured weather and advice; the estimate is marked experimental |
| Area alert live | Open-Meteo for Mysuru, Pune, Bengaluru, Kolkata, Leh, Chennai; unknown name | All answered; Leh gets no estimate (outside training weather); unknown name gets a message |
| DICOM scan upload | Real kidney-CT and brain-MRI test images wrapped in DICOM with fake patient tags | Same labels and scores as the original images; names and IDs absent from the stored file; burned-in text refused |
| Hinglish "koi bukhar nahi" | Live chat, 30-year-old | "Home care is usually enough" (it used to raise urgency) |
| Emergency red-team set after these changes | 59 phrases | 59 of 59 escalate |

## Known limits

- **Not tested:** a real microphone, a phone in Expo Go, and choosing a file in the real file dialogs. The path behind each of them was tested through the API.
- **Slow first use:** only 5 image models stay loaded. A model loaded again after a while can take minutes on this laptop; the teeth check took 347 s once in the final run.
- **Worker image:** the image worker was saved from the running container, because rebuilding it needs about 1 GB of downloads that fail on this connection. A clean rebuild from the Dockerfile still needs a good connection.
- **Emergency rules are pattern lists:** they catch every tested wording, not every possible one, and they lean towards over-escalating.
- **Voice check and DICOM picker:** the real microphone and file dialogs are untested here; the API paths behind them were tested.
- **Area alerts:** based on weather, not reported cases, and the model was trained only on Karnataka.
