from fastapi.testclient import TestClient

from evalforge import cli
from evalforge.review.app import app


def test_review_flow(tmp_path):
    client = TestClient(app)
    assert "No runs yet" in client.get("/").text
    cli.main(["run", "--prompt-version", "v2", "--no-mlflow", "--limit", "20"])
    q = client.get("/api/queue").json()
    assert q["run_id"] and len(q["items"]) == 20 and all(i["label"] is None for i in q["items"])
    first = q["items"][0]
    r = client.post("/api/labels", json={"run_id": q["run_id"], "case_id": first["case_id"], "label": True, "reviewer": "ana"})
    assert r.status_code == 200
    q2 = client.get("/api/queue").json()
    labelled = [i for i in q2["items"] if i["label"] is not None]
    assert [i["case_id"] for i in labelled] == [first["case_id"]] and labelled[0]["label"] is True
    assert q2["items"][-1]["case_id"] == first["case_id"]  # labelled items sink to the bottom
    page = client.get("/")
    assert page.status_code == 200 and first["case_id"] in page.text and "labelled PASS" in page.text
    assert client.post("/api/labels", json={"run_id": q["run_id"], "case_id": "nope", "label": True}).status_code == 404
    assert client.post("/api/labels", json={"run_id": "zzz", "case_id": "x", "label": True}).status_code == 404


def test_calibration_endpoint_uses_seed_and_ui_labels():
    client = TestClient(app)
    rep = client.get("/api/calibration").json()
    assert rep["n_labels"] >= 20 and -1 <= rep["kappa"] <= 1
