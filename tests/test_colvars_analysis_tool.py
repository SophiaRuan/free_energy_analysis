"""Tests for ColvarsAnalyzer's replica-directory discovery."""
from free_energy_analysis.colvars_analysis_tool import ColvarsAnalyzer


def make_replica_dirs(tmp_path, n):
    for i in range(1, n + 1):
        (tmp_path / f"{i:02}_IDNR").mkdir()
    # a directory that should NOT be picked up
    (tmp_path / "results").mkdir()
    return tmp_path


def test_discovers_exactly_the_idnr_directories_present(tmp_path):
    make_replica_dirs(tmp_path, 10)
    analyzer = ColvarsAnalyzer(str(tmp_path), number_of_cv=3)
    assert len(analyzer.directories) == 10
    assert all(d.endswith("_IDNR") for d in analyzer.directories)


def test_discovers_a_different_replica_count_not_ten(tmp_path):
    make_replica_dirs(tmp_path, 4)
    analyzer = ColvarsAnalyzer(str(tmp_path), number_of_cv=3)
    assert len(analyzer.directories) == 4


def test_directories_are_sorted(tmp_path):
    make_replica_dirs(tmp_path, 12)
    analyzer = ColvarsAnalyzer(str(tmp_path), number_of_cv=3)
    assert analyzer.directories == sorted(analyzer.directories)
    # sorted() on these zero-padded names should put 02 before 10
    assert analyzer.directories.index(str(tmp_path / "02_IDNR")) < analyzer.directories.index(str(tmp_path / "10_IDNR"))


def test_pmf_and_colvar_files_match_directory_count(tmp_path):
    make_replica_dirs(tmp_path, 6)
    analyzer = ColvarsAnalyzer(str(tmp_path), number_of_cv=3)
    assert len(analyzer.pmf_files) == 6
    assert len(analyzer.colvar_files) == 6
