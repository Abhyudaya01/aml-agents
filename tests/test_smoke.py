import importlib
import pkgutil

import pytest
import scripts
from pydantic import ValidationError

import aml
from aml.config import Settings


def _module_names(package) -> list[str]:
    return [m.name for m in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}.")]


@pytest.mark.parametrize("name", ["aml", "scripts", *_module_names(aml), *_module_names(scripts)])
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_expected_packages_present() -> None:
    expected = {"data", "alerts", "tools", "pii", "llm", "audit", "agents", "validate", "eval"}
    assert expected <= {n.removeprefix("aml.") for n in _module_names(aml)}


def test_settings_defaults() -> None:
    assert Settings(_env_file=None).llm_provider == "fake"


def test_settings_rejects_unknown_provider() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_provider="nope")
