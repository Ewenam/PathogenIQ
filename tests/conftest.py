"""Shared pytest fixtures for PathogenIQ tests."""
import pytest
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_dir():
    return FIXTURES


@pytest.fixture
def two_sample_dir(tmp_path):
    """Temp directory with two fixture .report files."""
    import shutil
    shutil.copy(FIXTURES / "sample_a.report", tmp_path / "sample_a.report")
    shutil.copy(FIXTURES / "sample_b.report", tmp_path / "sample_b.report")
    return tmp_path


@pytest.fixture
def sampleset(two_sample_dir):
    from pathogeniq.ingestion.reader import load_sample_directory
    return load_sample_directory(two_sample_dir, rank="G", pattern="*.report")
