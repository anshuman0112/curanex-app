"""Report reading and the attach menu chain: file to text to explanation to a pre-filled check."""
import io
from unittest.mock import patch

from PIL import Image, ImageDraw

from api.reports import find_values

LAB = ("Fasting Blood Sugar 142 mg/dL 70-110\nSGPT (ALT) 88 U/L 7-56\nSGOT (AST) 76 U/L 8-48\nBilirubin Total 2.1 mg/dL 0.1-1.2\n"
       "Triglycerides 238 mg/dL <150\nHDL Cholesterol 34 mg/dL >40")
LIVER = "nhanes_fatty_liver"
DIABETES = "nhanes_diabetes_any_labs"


def pdf_with(text: str) -> bytes:
    """Smallest valid one-page PDF holding the text."""
    stream = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode()
    objs = [b"<</Type/Catalog/Pages 2 0 R>>", b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
            b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
            b"<</Length %d>>stream\n%s\nendstream" % (len(stream), stream), b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, obj)
    xref = len(out)
    out += b"xref\n0 6\n0000000000 65535 f \n" + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    return out + b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF" % xref


def test_find_values_reads_the_result_not_the_range():
    assert find_values(LAB) == {"glucose": 142, "alt": 88, "ast": 76, "bilirubin": 2.1, "triglycerides": 238, "hdl": 34}


def test_find_values_skips_other_units():
    assert "glucose" not in find_values("Fasting glucose 7.8 mmol/L")


def test_pdf_report_is_read(client):
    r = client.post("/report", files={"file": ("lab.pdf", pdf_with("Fasting Blood Sugar 142 mg/dL range 70-110"), "application/pdf")})
    assert r.status_code == 200
    assert r.json()["values"] == {"glucose": 142}


def test_photo_report_is_read(client):
    img = Image.new("RGB", (900, 160), "white")
    ImageDraw.Draw(img).text((20, 40), "Fasting Blood Sugar 142 mg/dL", fill="black", font_size=48)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    r = client.post("/report", files={"file": ("lab.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200, r.text
    assert r.json()["values"].get("glucose") == 142


def test_bad_files_are_refused_with_a_reason(client):
    assert client.post("/report", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 415
    blank = io.BytesIO()
    Image.new("RGB", (400, 400), "white").save(blank, "PNG")
    assert client.post("/report", files={"file": ("b.png", blank.getvalue(), "image/png")}).status_code == 415


def test_report_chat_explains_then_offers_checks_with_values(client):
    with patch("api.orchestrator.loop.call_llm", return_value="Your sugar is above the printed range.") as llm:
        r = client.post("/chat", json={"session_id": "s", "message": "Please explain my report.", "report_text": LAB})
    blocks = r.json()["blocks"]
    assert blocks[0]["type"] == "text"
    assert "<report>" in llm.call_args.args[1]
    forms = {b["model_id"]: b for b in blocks if b["type"] == "form_request"}
    assert forms[LIVER]["defaults"] == {"alt": 88, "ast": 76}
    assert forms[DIABETES]["defaults"] == {"triglycerides": 238, "hdl": 34}
    assert {"alt", "age"} <= {f["key"] for f in forms[LIVER]["fields"]}


def test_every_live_model_opens_from_the_attach_menu_without_ai(client):
    # own-screen models (the voice check) open their own screen in the app instead of the chat
    models = [m for m in client.get("/models").json()["models"] if m.get("integrated") and not m.get("own_screen")]
    assert len(models) >= 6
    with patch("api.orchestrator.loop.call_llm") as llm:
        for m in models:
            blocks = client.post("/chat", json={"session_id": "s", "message": "", "model_id": m["id"]}).json()["blocks"]
            kinds = [b for b in blocks if b["type"] in ("form_request", "upload_request")]
            assert kinds and kinds[0]["model_id"] == m["id"], m["id"]
    llm.assert_not_called()
