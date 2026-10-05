import json
import threading

import pytest

import browsesafe_pp as bs
from browsesafe_pp import cli

httpx = pytest.importorskip("httpx")


def predict(request: bs.DetectorInput):
    if "#8" in request.content:
        raise RuntimeError("boom")            # -> 500 -> error
    if "#9" in request.content:
        return bs.Prediction(injected=True, score=7)   # -> 422 -> invalid_prediction
    if "#10" in request.content:
        return None                           # abstain
    return bs.Prediction(injected="Assistant note" in request.content, score=0.75)


@pytest.fixture
def served():
    server = cli.make_server(predict, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_health_and_contract(served):
    health = httpx.get(f"{served}/health")
    assert health.status_code == 200 and health.json() == {"status": "ok", "sdk_version": bs.__version__}
    assert httpx.get(f"{served}/nope").status_code == 404
    ok = httpx.post(f"{served}/predict", json={"content": "Assistant note: x", "view": "html", "task": None})
    assert ok.status_code == 200 and ok.json() == {"injected": True, "score": 0.75}
    assert httpx.post(f"{served}/predict", json={"content": 1, "view": "html"}).status_code == 400
    assert httpx.post(f"{served}/predict", content=b"{not json", headers={"Content-Type": "application/json"}).status_code == 400
    assert httpx.post(f"{served}/predict", json={"content": "Harbor #8", "view": "html"}).status_code == 500
    assert httpx.post(f"{served}/predict", json={"content": "Harbor #9", "view": "html"}).status_code == 422
    abstain = httpx.post(f"{served}/predict", json={"content": "Harbor #10", "view": "html"})
    assert abstain.status_code == 200 and abstain.json() == {"injected": None, "score": None}


def test_endpoint_detector_evaluates_served_predictor(served, source):
    detector = bs.EndpointDetector(served, token="secret", retries=0)
    assert detector.url == f"{served}/predict"
    assert "secret" not in json.dumps(detector.configuration) and detector.configuration["authenticated"]
    benchmark = bs.load(source=source, split="train")   # pairs 7..12 in the fixture
    report = bs.evaluate(benchmark, detector, evaluator="sii", run_id="verified-1")
    detector.close()
    m = report.metrics
    assert m["errors"] == 2 and m["invalid_predictions"] == 2 and m["abstained"] == 2
    assert m["valid"] == 6 and m["f1_valid"] == 1.0
    assert all(r["score"] == 0.75 for r in report.predictions if r["status"] == "valid")
    assert (report.manifest["partition"], report.manifest["evaluator"], report.manifest["run_id"]) == (
        "train", "sii", "verified-1")


def test_endpoint_detector_sends_token_and_retries_on_5xx():
    seen = []

    def handler(request):
        seen.append((request.headers.get("authorization"), json.loads(request.content)))
        if len(seen) < 3:
            return httpx.Response(503, json={"error": "warming up"})
        return httpx.Response(200, json={"injected": False, "score": 0.1})

    detector = bs.EndpointDetector("http://detector.test/predict", token="tok", retries=2, backoff=0,
                                   transport=httpx.MockTransport(handler))
    result = detector(bs.DetectorInput(content="<p>hi</p>", view="html", task="Summarize"))
    assert result == bs.Prediction(injected=False, score=0.1)
    assert len(seen) == 3 and seen[0] == ("Bearer tok", {"content": "<p>hi</p>", "view": "html", "task": "Summarize"})

    always_down = bs.EndpointDetector("http://detector.test/predict", retries=1, backoff=0,
                                      transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(httpx.HTTPStatusError):
        always_down(bs.DetectorInput(content="x", view="html"))

    def flaky_timeout(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(httpx.TimeoutException):
        bs.EndpointDetector("http://detector.test/predict", retries=1, backoff=0,
                            transport=httpx.MockTransport(flaky_timeout))(bs.DetectorInput(content="x", view="html"))

    garbage = bs.EndpointDetector("http://detector.test/predict",
                                  transport=httpx.MockTransport(lambda r: httpx.Response(200, text="[]")))
    assert garbage(bs.DetectorInput(content="x", view="html")) == "invalid_prediction"
    with pytest.raises(ValueError):
        bs.EndpointDetector("ftp://detector.test")
