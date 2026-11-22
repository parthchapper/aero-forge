import importlib

import pytest

from aeroforge import constants

DEPENDENCIES = [
    "aerosandbox",
    "cadquery",
    "ribs",
    "trimesh",
    "networkx",
    "numpy",
    "pydantic",
    "yaml",
    "PIL",
]


@pytest.mark.parametrize("module", DEPENDENCIES)
def test_dependency_imports(module: str) -> None:
    importlib.import_module(module)


def test_constants_table_loads_as_mapping() -> None:
    assert isinstance(constants.load_table("constants"), dict)


def test_missing_constant_raises_typed_error() -> None:
    with pytest.raises(constants.DataFileError):
        constants.constant("constants", "no_such_entry")


def test_missing_table_raises_typed_error() -> None:
    with pytest.raises(constants.DataFileError):
        constants.load_table("no_such_table")
