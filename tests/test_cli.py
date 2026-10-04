import json

from browsesafe_pp import cli


def test_info(source, capsys):
    cli.main(["info", "--source", str(source)])
    assert json.loads(capsys.readouterr().out)["pairs"] == 12


def test_evaluate_with_predict(source, tmp_path, capsys, monkeypatch):
    (tmp_path / "my_detector.py").write_text(
        "def predict(request):\n    return 'Assistant note' in request.content\n")
    monkeypatch.chdir(tmp_path)
    cli.main(["evaluate", "--source", str(source), "--predict", "my_detector:predict",
              "--n-pairs", "0", "--output", str(tmp_path / "out")])
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["f1_valid"] == 1.0 and (tmp_path / "out" / "metrics.json").exists()
