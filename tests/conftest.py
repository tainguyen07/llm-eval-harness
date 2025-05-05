"""Pytest configuration."""

import pytest

collect_ignore = []


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.integration)