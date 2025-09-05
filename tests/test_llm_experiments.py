"""Test suite for llm_experiments."""

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "llm_experiments",
        "llm_experiments.introspection_by_telephone",
        "llm_experiments.self_selective_amnesia",
    ],
)
def test_imports(module_name):
    """Test that the specified module can be imported."""
    module = __import__(module_name)
    assert module is not None
