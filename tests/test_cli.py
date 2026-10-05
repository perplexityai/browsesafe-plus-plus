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


def test_evaluate_records_provenance(source, tmp_path, capsys, monkeypatch):
    (tmp_path / "my_detector.py").write_text("def predict(request):\n    return True\n")
    monkeypatch.chdir(tmp_path)
    cli.main(["evaluate", "--source", str(source), "--predict", "my_detector:predict", "--split", "test",
              "--n-pairs", "2", "--evaluator", "sii", "--run-id", "abc", "--output", str(tmp_path / "out")])
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert (manifest["partition"], manifest["evaluator"], manifest["run_id"]) == ("test", "sii", "abc")


def test_evaluate_defaults_to_train(source, tmp_path, capsys, monkeypatch):
    (tmp_path / "my_detector.py").write_text("def predict(request):\n    return True\n")
    monkeypatch.chdir(tmp_path)
    cli.main(["evaluate", "--source", str(source), "--predict", "my_detector:predict", "--n-pairs", "0",
              "--output", str(tmp_path / "out")])
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["partition"] == "train"
    assert json.loads(capsys.readouterr().out)["pairs"] == 6
