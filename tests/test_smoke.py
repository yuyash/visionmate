import importlib


def test_import_package():
    assert importlib.import_module("deskmate") is not None


def test_truth():
    assert True
