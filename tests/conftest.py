import pytest

from browsesafe_pp.testing import write_fixture


@pytest.fixture(scope="session")
def source(tmp_path_factory):
    return write_fixture(tmp_path_factory.mktemp("fixture"), n_pairs=12)
