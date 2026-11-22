import numpy as np
import pytest

from aeroforge.genetable import GenomeError, genome_length
from aeroforge.genome import DecodeError, decode
from aeroforge.models import Requirements


def test_wrong_length_genome_raises_genome_error(req: Requirements) -> None:
    with pytest.raises(GenomeError):
        decode(np.full(genome_length() - 1, 0.5), req)


def test_out_of_range_genome_raises_genome_error(req: Requirements) -> None:
    with pytest.raises(GenomeError):
        decode(np.full(genome_length(), 1.5), req)


def test_brief_too_small_for_any_spar_raises_decode_error(
    genomes: np.ndarray, req: Requirements
) -> None:
    with pytest.raises(DecodeError):
        decode(genomes[0], req.model_copy(update={"lifting_area_min_m2": 1e-4}))
