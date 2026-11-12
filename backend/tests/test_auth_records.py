"""Accounts, roles, manual doctor approval, consent sharing, encrypted records, audit."""

import os
import sqlite3

PW = "a-strong-pass-1"
CASE = {"title": "Fever and cough", "symptoms": "fever for 3 days, cough, difficulty breathing", "patient": {"age": 45, "sex": "male"}}
DOCTOR_ROUTES = [("post", "/doctor/tests", {"patient": {}, "symptoms": "cough"}), ("post", "/doctor/differential", {"patient": {}, "symptoms": "cough"}),
                 ("get", "/doctor/cases", None)]
ADMIN_ROUTES = [("get", "/admin/doctors", None), ("get", "/admin/audit", None)]


def signup(client, email, role, name="Test User"):
    r = client.post("/auth/signup", json={"email": email, "password": PW, "name": name, "role": role})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}, r.json()["user"]


def call(client, method, path, body, headers):
    return getattr(client, method)(path, json=body, headers=headers) if body is not None else getattr(client, method)(path, headers=headers)


def test_patient_and_pending_doctor_are_locked_out(client):
    patient, _ = signup(client, "p@x.test", "patient")
    pending, user = signup(client, "d@x.test", "doctor")
    assert user["status"] == "pending"
    for method, path, body in DOCTOR_ROUTES + ADMIN_ROUTES:
        assert call(client, method, path, body, patient).status_code == 403, path
        assert call(client, method, path, body, pending).status_code == 403, path
        assert call(client, method, path, body, {}).status_code in (401, 403), path


def test_admin_approves_doctor_then_doctor_gets_in(client):
    admin, a = signup(client, "admin@curanex.test", "patient", "Owner")
    assert a["role"] == "admin"
    doctor, d = signup(client, "dr@x.test", "doctor", "Dr Mehta")
    assert client.get("/doctor/cases", headers=doctor).status_code == 403
    assert client.post(f"/admin/doctors/{d['id']}/approve", headers=admin).json()["status"] == "approved"
    assert client.get("/doctor/cases", headers=doctor).status_code == 200
    client.post(f"/admin/doctors/{d['id']}/suspend", headers=admin)
    assert client.get("/doctor/cases", headers=doctor).status_code == 401  # suspended: sessions ended


def test_share_plan_and_revoke(client):
    admin, _ = signup(client, "admin@curanex.test", "patient")
    doctor, d = signup(client, "dr@x.test", "doctor", "Dr Mehta")
    other, o = signup(client, "other@x.test", "doctor", "Dr Other")
    client.post(f"/admin/doctors/{d['id']}/approve", headers=admin)
    client.post(f"/admin/doctors/{o['id']}/approve", headers=admin)
    patient, _ = signup(client, "p@x.test", "patient")
    case = client.post("/records/cases", json=CASE, headers=patient).json()
    assert client.get(f"/doctor/cases/{case['id']}", headers=doctor).status_code == 404  # not shared yet
    shared = client.post(f"/records/cases/{case['id']}/share", json={"doctor_email": "dr@x.test"}, headers=patient).json()
    assert [s["email"] for s in shared["shared_with"]] == ["dr@x.test"]
    listed = client.get("/records/doctors", headers=patient).json()
    assert listed and "email" not in listed[0]  # patients pick a doctor by name; doctor emails are not exposed
    assert client.post(f"/records/cases/{case['id']}/share", json={"doctor_id": listed[0]["id"]}, headers=patient).status_code == 200
    assert client.get(f"/doctor/cases/{case['id']}", headers=doctor).json()["symptoms"] == CASE["symptoms"]
    assert client.get(f"/doctor/cases/{case['id']}", headers=other).status_code == 404  # other doctor never sees it
    plan = {"diagnosis": "Community-acquired pneumonia", "tests": ["Chest X-ray"], "medicines": ["Amoxicillin"], "advice": "Rest, fluids, review in 48 hours."}
    assert client.post(f"/doctor/cases/{case['id']}/plan", json=plan, headers=doctor).status_code == 200
    mine = client.get("/records/cases", headers=patient).json()[0]
    assert mine["doctor_plans"][0]["diagnosis"] == "Community-acquired pneumonia" and mine["doctor_plans"][0]["doctor_name"] == "Dr Mehta"
    client.delete(f"/records/cases/{case['id']}/share/{d['id']}", headers=patient)
    assert client.get(f"/doctor/cases/{case['id']}", headers=doctor).status_code == 404  # consent withdrawn


def test_records_are_encrypted_and_audited(client):
    patient, _ = signup(client, "p@x.test", "patient")
    client.post("/records/cases", json=CASE, headers=patient)
    con = sqlite3.connect(os.environ["CURANEX_DB"])
    raw = con.execute("SELECT data FROM cases").fetchone()[0]
    actions = [r[0] for r in con.execute("SELECT action FROM audit")]
    con.close()
    assert "fever" not in raw and "difficulty" not in raw  # unreadable on disk
    assert "case:create" in actions and "login" in actions


def test_wrong_password_and_duplicate_email(client):
    signup(client, "p@x.test", "patient")
    assert client.post("/auth/login", json={"email": "p@x.test", "password": "wrong-password"}).status_code == 401
    assert client.post("/auth/signup", json={"email": "p@x.test", "password": PW, "name": "Again", "role": "patient"}).status_code == 409
