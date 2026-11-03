from fastapi.testclient import TestClient

from workers.common.server import Registry, create_app, discover


def test_worker_serves_dummy_echo():
    c = TestClient(create_app("tabular"))
    assert c.get("/health").json()["ok"] is True
    assert "dummy_echo" in c.get("/models").json()
    r = c.post("/predict/dummy_echo", json={"inputs": {"echo_score": 8, "age": 34}})
    assert r.status_code == 200
    assert r.json()["label"] == "Higher" and r.json()["probabilities"]["Higher"] == 0.8


def test_worker_validation_and_unknown_model():
    c = TestClient(create_app("tabular"))
    r = c.post("/predict/dummy_echo", json={"inputs": {"echo_score": 99}})
    assert r.status_code == 422
    assert set(r.json()["detail"]["errors"]) == {"age: missing", "echo_score: out of range 0–10"}
    assert c.post("/predict/nope", json={"inputs": {}}).status_code == 404


def test_registry_is_lazy_and_lru():
    adapters = discover("tabular")
    reg = Registry({f"m{i}": adapters["dummy_echo"] for i in range(3)}, max_loaded=2)
    assert not reg.loaded
    for i in range(3):
        reg.get(f"m{i}")
    assert list(reg.loaded) == ["m1", "m2"]
