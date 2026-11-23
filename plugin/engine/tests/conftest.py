from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import yaml
from threadpoolctl import threadpool_limits

from aeroforge.brief import BRIEFS, brief_requirements, load_brief
from aeroforge.checks import CheckReport, evaluate
from aeroforge.constants import number
from aeroforge.derive import baseline_prepared, derive
from aeroforge.genome import decode, random_genomes
from aeroforge.models import Brief, Design, Requirements
from aeroforge.prepare import Prepared, prepare
from aeroforge.reference import BASELINE, encode, reference_genome

N_PROPERTY_GENOMES = 1000
N_SMOKE_GENOMES = 50  # the fast suite's share of every 1,000-genome property test (U1, user)
SMALL_BOXES_GENOMES = 400  # branch coverage (bodyless, wheels); the stub brief carries the 1,000
PROPERTY_SEED = 20261112
BALANCED_GENOMES = 6


@pytest.fixture(scope="session", autouse=True)
def blas_threads() -> Iterator[None]:
    with threadpool_limits(limits=int(number("blas_threads")), user_api="blas"):
        yield


@pytest.fixture(scope="session")
def req() -> Requirements:
    return brief_requirements("stub")


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(N_SMOKE_GENOMES, id="smoke"),
        pytest.param(N_PROPERTY_GENOMES, id="full", marks=pytest.mark.slow),
    ],
)
def population(request: pytest.FixtureRequest) -> int:
    """How many seeded random genomes a property test sees: the first 50 (fast suite) or all
    1,000 (slow). Same draw, so the smoke rows are the first rows of the full set."""
    return int(request.param)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """A slow test runs on the full population only: its smoke variant would need more genomes
    than 50 (it is skipped, the full variant is not)."""
    skip = pytest.mark.skip(reason="slow tests run on the full population")
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        if params.get("population") == N_SMOKE_GENOMES and item.get_closest_marker("slow"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def genomes(population: int) -> np.ndarray:
    return random_genomes(np.random.default_rng(PROPERTY_SEED), population)


LEGACY_GENES = Path(__file__).parent / "fixtures" / "baseline_pre_step10" / "genes_pre_step10.yaml"
GENES_0_3 = Path(__file__).parent / "fixtures" / "baseline_0.3" / "genes_0.3.yaml"
LEGACY_GENOMES = 200


def legacy_random_genomes(seed: int, n: int, table_path: Path = LEGACY_GENES) -> np.ndarray:
    """The first n random genomes the pre-step-10 table (71 genes) drew for seed, re-encoded into
    today's table (new genes at their defaults): the very designs the calibrated tests were tuned
    on, which the 86-gene table would otherwise silently replace with different ones."""
    table = yaml.safe_load(table_path.read_text(encoding="utf-8"))["genes"]
    u = np.random.default_rng(seed).random((n, len(table)))
    rows = []
    for row in u:
        physical: dict[str, object] = {}
        for gene, x in zip(table, row, strict=True):
            if gene["kind"] == "choice":
                choices = gene["choices"]
                physical[gene["name"]] = choices[min(int(x * len(choices)), len(choices) - 1)]
            else:
                physical[gene["name"]] = gene["lo"] + x * (gene["hi"] - gene["lo"])
        rows.append(encode(physical))
    return np.array(rows)


@pytest.fixture(scope="session")
def legacy_genomes() -> np.ndarray:
    return legacy_random_genomes(PROPERTY_SEED, LEGACY_GENOMES)


@pytest.fixture(scope="session")
def genomes_0_3(population: int) -> np.ndarray:
    """The 1,000 random genomes the 0.3 table (86 genes) drew for the property seed, re-encoded
    into today's table (the U1 genes at their defaults): the population calibrated 0.3 tests
    were tuned on, which the wider table would otherwise replace."""
    return legacy_random_genomes(PROPERTY_SEED, population, GENES_0_3)


@pytest.fixture(scope="session")
def designs(genomes: np.ndarray, req: Requirements) -> list[Design]:
    return [decode(g, req) for g in genomes]


@pytest.fixture(scope="session")
def full_designs(req: Requirements) -> list[Design]:
    """All 1,000 seeded random designs whatever the population param: for the few tests whose
    claim is statistical or an existence claim over the full draw (never cut to the smoke rows)."""
    full = random_genomes(np.random.default_rng(PROPERTY_SEED), N_PROPERTY_GENOMES)
    return [decode(g, req) for g in full]


@pytest.fixture(scope="session", params=BRIEFS)
def brief(request: pytest.FixtureRequest) -> Requirements:
    return brief_requirements(request.param)


@pytest.fixture(scope="session")
def decoded(
    brief: Requirements, req: Requirements, genomes: np.ndarray, designs: list[Design]
) -> list[Design]:
    if brief == req:
        return designs
    return [decode(g, brief) for g in genomes[:SMALL_BOXES_GENOMES]]


@pytest.fixture(scope="session")
def decoded_0_3(brief: Requirements, genomes_0_3: np.ndarray) -> list[Design]:
    """genomes_0_3 decoded for the parametrised brief (the small-boxes brief: first rows only)."""
    rows = genomes_0_3 if brief == brief_requirements("stub") else genomes_0_3[:SMALL_BOXES_GENOMES]
    return [decode(g, brief) for g in rows]


SAMPLE_BRIEF = Path(__file__).parent / "fixtures" / "brief_sample.yaml"


@pytest.fixture(scope="session")
def sample_brief() -> Brief:
    return load_brief(SAMPLE_BRIEF)


@pytest.fixture(scope="session")
def sample_req(sample_brief: Brief) -> Requirements:
    return derive(sample_brief)


@pytest.fixture(scope="session")
def conventional(sample_req: Requirements, sample_brief: Brief) -> Design:
    return baseline_prepared(sample_req, sample_brief.payload_mass_kg).design


@pytest.fixture(scope="session")
def conventional_t2(sample_req: Requirements, sample_brief: Brief) -> tuple[Design, CheckReport]:
    return evaluate(reference_genome(BASELINE), sample_brief, sample_req, 2)


@pytest.fixture(scope="session")
def conventional_tier2(conventional_t2: tuple[Design, CheckReport]) -> CheckReport:
    return conventional_t2[1]


@pytest.fixture(scope="session")
def sample_prepared(
    legacy_genomes: np.ndarray, sample_req: Requirements, sample_brief: Brief
) -> list[Prepared]:
    return [
        prepare(g, sample_req, sample_brief.payload_mass_kg)
        for g in legacy_genomes[:BALANCED_GENOMES]
    ]
