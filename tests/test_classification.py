import asyncio
import json

import pytest

import browsesafe_pp as bs
from browsesafe_pp.evaluation import get_classification_prompt, BROWSESAFE_1_0


def keyword(request):
    return bs.Prediction(injected="assistant note" in request.content.lower(), score=0.9)


def test_perfect_detector(source):
    report = bs.evaluate(bs.load(source=source), keyword)
    m = report.metrics
    assert m["confusion_valid"] == {"tp": 6, "fp": 0, "tn": 6, "fn": 0}
    assert m["precision_valid"] == m["recall_valid"] == m["f1_valid"] == 1.0
    assert m["balanced_accuracy_valid"] == 1.0 and m["benign_fpr_valid"] == 0.0
    assert m["pair_correctness_all"] == 1.0 and m["coverage"] == 1.0


def test_always_yes_and_failures(source):
    benchmark = bs.load(source=source)
    m = bs.evaluate(benchmark, lambda r: True).metrics
    assert m["confusion_valid"] == {"tp": 6, "fp": 6, "tn": 0, "fn": 0}
    assert m["precision_valid"] == 0.5 and m["benign_fpr_valid"] == 1.0
    assert m["paired_outcomes"]["attack_only_correct"] == 6 and m["pair_correctness_all"] == 0.0

    calls = iter(range(100))

    def flaky(request):
        i = next(calls)
        if i % 4 == 0:
            raise RuntimeError("boom")
        if i % 4 == 1:
            return None
        if i % 4 == 2:
            return "garbage"
        return True

    m = bs.evaluate(benchmark, flaky).metrics
    assert (m["errors"], m["abstained"], m["invalid_predictions"], m["valid"]) == (3, 3, 3, 3)
    assert m["coverage"] == 0.25 and m["paired_outcomes"]["incomplete_outcomes"] == 6


def test_async_preserves_order(source):
    benchmark = bs.load(source=source)

    async def slow(request):
        await asyncio.sleep(0.001 * (len(request.content) % 5))
        return keyword(request)

    report = asyncio.run(bs.evaluate_async(benchmark, slow, concurrency=4))
    assert [r["entry_id"] for r in report.predictions] == [p.entry_id for p in benchmark]
    assert report.metrics["f1_valid"] == 1.0


def test_report_save(source, tmp_path):
    report = bs.evaluate(bs.load(source=source, n_pairs=2), keyword)
    out = report.save(tmp_path / "run")
    assert json.loads((out / "metrics.json").read_text())["pairs"] == 2
    manifest = json.loads((out / "manifest.json").read_text())
    assert (manifest["partition"], manifest["evaluator"], manifest["run_id"]) == ("train", "self", None)
    lines = (out / "predictions.jsonl").read_text().splitlines()
    assert len(lines) == 4 and "content" not in json.loads(lines[0])
    with pytest.raises(FileExistsError):
        report.save(out)


def test_report_provenance(source):
    benchmark = bs.load(source=source, split=None, n_pairs=2)
    report = bs.evaluate(benchmark, keyword, evaluator="sii", run_id="run-0001")
    assert report.manifest["partition"] == "all"
    assert (report.manifest["evaluator"], report.manifest["run_id"]) == ("sii", "run-0001")
    report = asyncio.run(bs.evaluate_async(bs.load(source=source, split="test", n_pairs=1), keyword, run_id="r2"))
    assert (report.manifest["partition"], report.manifest["evaluator"], report.manifest["run_id"]) == ("test", "self", "r2")
    outcomes = [("valid", page.injected, None) for page in benchmark]
    assert bs.evaluation.score(benchmark, outcomes).manifest["evaluator"] == "self"
    with pytest.raises(ValueError):
        bs.evaluation.score(benchmark, outcomes, evaluator="")
    with pytest.raises(ValueError):
        bs.evaluation.score(benchmark, outcomes, run_id=7)


def test_evidence_task_rejected(source):
    with pytest.raises(ValueError):
        bs.evaluate(bs.load(source=source, task="evidence", n_pairs=1), keyword)


def test_prompt_parsing():
    prompt = get_classification_prompt()
    assert prompt.text == BROWSESAFE_1_0 and prompt.output_format == "classification"
    assert "HTML Prompt Injection Detection Policy" in prompt.text
    assert prompt.parse({"classification": 1, "explanation": "redirects data"}) == "yes"
    assert prompt.parse({"classification": 0, "explanation": "ordinary page"}) == "no"
    for bad in ({"classification": "1", "explanation": "x"}, {"classification": 1},
                {"classification": 2, "explanation": "x"}, {"classification": 1, "explanation": " "}, []):
        with pytest.raises(ValueError):
            prompt.parse(bad)
    assert get_classification_prompt("default").name == "browsesafe-1.0"
    with pytest.raises(ValueError):
        get_classification_prompt("other")
    label = bs.evaluation.ClassificationPrompt("mine", "Is this injected?", output_format="label")
    assert label.parse({"label": "abstain"}) == "abstain"


def test_model_detector_with_mock_transport(source):
    httpx = pytest.importorskip("httpx")
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        page = body["messages"][1]["content"]
        verdict = 1 if "Assistant note" in page else 0
        if "#8" in page:  # one malformed response (pair 8 is in the default train split)
            return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}, "finish_reason": "stop"}]})
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [{
                "id": "c1", "type": "function", "function": {
                    "name": "record_classification",
                    "arguments": json.dumps({"classification": verdict, "explanation": "checked"})}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3}})

    provider = bs.OpenAICompatible("http://model.test/v1", "demo-model", "test-key",
                                   transport=httpx.MockTransport(handler))
    detector = bs.ModelDetector(provider)
    benchmark = bs.load(source=source)

    async def run():
        try:
            return await bs.evaluate_async(benchmark, detector, concurrency=3)
        finally:
            await provider.aclose()

    m = asyncio.run(run()).metrics
    assert seen[0]["messages"][0]["content"] == BROWSESAFE_1_0
    assert seen[0]["tool_choice"]["function"]["name"] == "record_classification"
    assert m["invalid_predictions"] == 2 and m["valid"] == 10 and m["f1_valid"] == 1.0
    assert "test-key" not in json.dumps(detector.configuration)


def test_provider_requires_configuration(monkeypatch):
    for name in ("BROWSESAFE_MODEL_ENDPOINT", "BROWSESAFE_MODEL", "BROWSESAFE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="BROWSESAFE_MODEL_ENDPOINT"):
        bs.OpenAICompatible()
    monkeypatch.setenv("BROWSESAFE_MODEL_ENDPOINT", "https://api.example.com/v1/")
    monkeypatch.setenv("BROWSESAFE_MODEL", "m")
    provider = bs.OpenAICompatible()
    assert provider.url == "https://api.example.com/v1/chat/completions" and provider.model == "m"
