import pytest

from browsesafe_pp.testing import write_fixture


@pytest.fixture(scope="session")
def source(tmp_path_factory):
    return write_fixture(tmp_path_factory.mktemp("fixture"), n_pairs=12)


@pytest.fixture(autouse=True)
def hf_home(tmp_path_factory, monkeypatch):
    """Keep the screenshot cache (under the Hugging Face home) out of the user's real cache."""
    import huggingface_hub.constants

    root = tmp_path_factory.mktemp("hf-home")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HOME", str(root))
    return root
