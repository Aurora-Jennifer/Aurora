import json
import os
import pathlib

import pytest

from core.data_sanity.trace import Trace


def pytest_addoption(parser):
    parser.addoption("--emit-traces", action="store_true", default=False)


@pytest.fixture
def trace(request):
    """Attach to tests that call validate(..., trace=trace). Emits only for datasanity paths."""
    t = Trace()
    yield t
    if not request.config.getoption("--emit-traces"):
        return
    fspath = str(getattr(request.node, "fspath", ""))
    if "tests/datasanity" not in fspath:
        return
    out = pathlib.Path("artifacts/traces")
    out.mkdir(parents=True, exist_ok=True)
    node = request.node.nodeid.replace(os.sep, "_").replace("::", "__")
    with open(out / f"{node}.json", "w") as f:
        json.dump({"nodeid": request.node.nodeid, "summary": t.summary()}, f)


