"""Shared pytest hooks of the test suite.

The temp folder of a test (``tmp_path``, and ``tmpdir``, which is built on
it) goes at the end of the test when the test passed. Many tests run
train.py, and each run saves about 6 checkpoint files of 1 to 10 MB. pytest
keeps every temp folder to the end of the session, so a full run of the
suite wrote GBs on the shared disk of elisa (11-11). A failed test keeps its
folder for the debug. The folders of ``tmp_path_factory`` (shared by the
tests of a module) stay to the end of the session, as before.

pytest 7.3 does the same with ``tmp_path_retention_policy = "failed"``; elisa
has pytest 6.2.
"""
import shutil

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Keep the report of each phase on the test item (``rep_call``)."""
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"rep_{report.when}", report)


@pytest.fixture(autouse=True)
def _drop_tmp_path_of_passed_test(request):
    """Delete the ``tmp_path`` folder of the test at its end when it passed.

    An autouse fixture is set up before the fixtures that the test asks for,
    so its teardown runs after theirs: no fixture reads the folder after it
    goes.
    """
    yield
    report = getattr(request.node, "rep_call", None)
    path = request.node.funcargs.get("tmp_path")
    if report is not None and report.passed and path is not None:
        shutil.rmtree(path, ignore_errors=True)
