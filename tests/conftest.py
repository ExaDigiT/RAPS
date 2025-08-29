import pytest
import uuid


def pytest_addoption(parser):
    parser.addoption(
        "--runlong", action="store_true", default=False, help="Run long-running tests"
    )


def pytest_runtest_setup(item):
    if "long" in item.keywords and not item.config.getoption("--runlong"):
        # reason = f"Skipping {item.nodeid} because it requires --runlong"
        reason = "Skipping test because it requires --runlong"
        pytest.skip(reason)


@pytest.fixture
def random_id():
    return f"test-{str(uuid.uuid4())[:8]}"
