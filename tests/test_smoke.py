"""Test khói: mọi package của dự án import được."""

import importlib

import pytest

PACKAGES = ["src", "src.games", "src.agents", "src.eval", "src.ui", "experiments"]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_importable(name: str) -> None:
    importlib.import_module(name)
