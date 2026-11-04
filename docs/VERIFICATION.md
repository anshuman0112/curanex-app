# Curanex verification (Phase 9)

Run on 2026-11-04 against the exported web build, the API on :8001 and the three Docker workers. Rerun with:

```bash
python tools/verify_e2e.py safety
python tools/verify_e2e.py flows out.json
```

## Summary

| Check | Result |
|---|---|
| Backend tests | 48 passed, lint clean |
| `tsc --noEmit` and `export:web` | Clean |
| Worker and API `/health` | All OK |
| Typed and voice conversations | 12 of 12 reached the right model and returned a full result card |
| General questions that must not route | 10 of 10 stayed general |
| Red-team safety set | 51 of 51 escalate without reaching the AI (after fixes) |
| Attach menu | All 6 live models open their form or upload card with no AI call |
| Report reading | PDF and photo read; values pre-fill the matching check |
| Layout at 390 px and 1366 px | No sideways scrolling on Home, Doctor, Voice and Models |

## Live models tested

Each ran one typed and one voice conversation. Voice clips came from Windows speech, so they test the pipeline, not accents.

| Model | Status | Typed | Voice | Result on the test case |
|---|---|---|---|---|
| Chest X-ray | Verified | Pass | Pass | Possible lung opacity signs |
| Type 2 diabetes | Verified | Pass | Pass | Higher risk |
| Teeth photo | Verified | Pass | Pass | Possible tooth decay spotted |
| Knee X-ray | Verified | Pass | Pass | No clear moderate arthritis signs |
| Liver | Unverified | Pass | Pass | Liver tests need a closer look |
| Gestational diabetes | Unverified | Pass | Pass | Higher risk |

No reply contained "you have" plus a condition, or a medicine dose. Every card showed what the model was trained on, its limitations and the credit.

## Speed

Seconds, measured through the API. Small samples, on a laptop, with the free LLM.

| Route | n | p50 | p90 | Max |
|---|---|---:|---:|---:|
| Red flag | 20 | 0.01 | 0.01 | 0.01 |
| Direct answer | 10 | 2.27 | 3.70 | 3.70 |
| Ask for inputs | 12 | 2.20 | 2.52 | 27.2 |
| Model run and explanation | 12 | 4.88 | 12.81 | 13.12 |
| Speech to text | 6 | 4.55 | 5.46 | 5.46 |

The single 27.2 s reading was not investigated. Other runs of that route stayed under 5 s.

## Worker memory

| Worker | Memory |
|---|---|
| Tabular | 50 MiB |
| TensorFlow | 122 MiB |
| Image | 318 MiB |

## Problems found and fixed

1. **22 of 51 emergency messages missed the red-flag rules**, including "I want to die", "severe chest pain" and "my baby is not breathing". Rules now cover paraphrases, misspellings and Hinglish on the app and the backend. A test keeps both lists identical.
2. **The app flagged the word "benefits" as a seizure** because its rule matched "fits". Fixed, with a test.
3. **No upload option in the chat.** Added a + menu with Medical report, every live image check and every quick check, built from the live model list.
4. **Rejected images looked like offline demo mode.** The card now shows the reason.
5. **Lab values were deleted as doses** ("142 mg/dL"). Lab units are now kept; "mg/kg" and plain doses are still removed.
6. **The diabetes card said "Higher risk" with 42%.** Scores are now scaled so 50% is the flagging point, as gestational diabetes already does. Registry limitations say so. This is a display change.
7. **Spoken replies ran to 5 sentences.** They now ask for 2 to 4. One of six still reached 5.
8. **Form introductions could be a sentence fragment.** Fragments now fall back to a full sentence.
9. **Chat gave up after 20 s** and looked offline. The limit is now 60 s.

## Known limits

- **Not tested:** a real microphone, a phone in Expo Go, Hindi speech, and choosing a real file in the upload and report dialogs. The steps behind each dialog were tested through the API.
- **Screenshots:** only the 390 px upload menu was captured, because the browser pane was hidden. Layout at both widths was measured instead.
- **Docker API image:** not rebuilt with the two new report-reading packages. The API was tested from the local environment. Rebuild it before running the whole stack in Docker.
- **Red-flag rules are pattern lists.** They catch the tested phrases, not every possible wording. Some messages over-escalate, such as "food poisoning" or a general question about stroke, which is the safer direction.
- **Report values can be misread.** They are shown in editable fields with a note to check them against the report.
- **Eye checks and skin are in the backlog**, and 120 of the 126 models are not live.
