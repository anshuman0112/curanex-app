"""The doctor module's evidence check: unsupported statements and wrong numbers are dropped."""

from api.knowledge.verify import keep_supported, numbers_supported, supported

PASSAGE = ("The starting dose of metformin is 500 mg orally once daily with the evening meal. Metformin is contraindicated in "
           "patients with severe renal impairment (eGFR below 30 mL/min/1.73 m2). Obtain an eGFR before starting.")


def test_supported_statement_passes():
    assert supported("Metformin starting dose is 500 mg once daily with the evening meal.", PASSAGE)


def test_wrong_number_fails():
    assert not numbers_supported("Starting dose is 1000 mg once daily.", PASSAGE)
    assert not supported("Metformin starting dose is 1000 mg once daily with the evening meal.", PASSAGE)


def test_unrelated_claim_fails():
    assert not supported("Metformin cures tuberculosis within two weeks.", PASSAGE)


def test_unicode_hyphens_and_greek_letters_match():
    book = "Empiric therapy combines a beta-lactam with a macrolide; use the CURB-65 score to assess severity."
    assert supported("Start a β‑lactam with a macrolide and assess severity with CURB‑65.", book)


def test_keep_supported_drops_uncited_and_unsupported():
    passages = {"p1": {"text": PASSAGE}}
    items = [{"text": "Check eGFR before starting metformin.", "cite": ["p1"]},
             {"text": "Metformin is safe in severe renal impairment.", "cite": []},
             {"text": "Give 2000 mg on day one.", "cite": ["p1"]},
             {"text": "Metformin is contraindicated in severe renal impairment.", "cite": ["p9"]}]
    kept, dropped = keep_supported(items, passages)
    assert [k["text"] for k in kept] == ["Check eGFR before starting metformin."]
    assert dropped == 3
