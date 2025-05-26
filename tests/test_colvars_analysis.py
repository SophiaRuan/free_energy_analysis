#!/usr/bin/env python

"""Tests for `free_energy_analysis.colvars_analysis_tool` package."""

import os
import pytest
import numpy as np
import matplotlib
from free_energy_analysis.colvars_analysis_tool import ColvarsAnalyzer


# ======== FIXTURES ========

@pytest.fixture
def mock_colvar_dir_1cv(tmp_path):
    base_dir = tmp_path
    for i in range(1, 3):
        replica_dir = base_dir / f"{i:02}_IDNR"
        replica_dir.mkdir()
        colvar_data = "0 1.0 0.1\n1 1.1 0.2\n2 1.2 0.3\n"
        pmf_data = "1.0 0.5\n1.1 0.3\n1.2 0.0\n"
        (replica_dir / "colvar.out.colvars.traj").write_text(colvar_data)
        (replica_dir / "colvar.out.pmf").write_text(pmf_data)
    return str(base_dir)


@pytest.fixture
def analyzer_1cv(mock_colvar_dir_1cv):
    return ColvarsAnalyzer(
        base_dir=mock_colvar_dir_1cv,
        number_of_cv=1,
        cv_labels=["CV1"],
        replica_range=range(1, 3)
    )

@pytest.fixture
def test_path_2cv():
    return "/Users/user/software_dev/free_energy_analysis/data/LiCl_0.5M_298K_2cv"

@pytest.fixture
def analyzer_2cv(test_path_2cv):
    return ColvarsAnalyzer(
        base_dir=test_path_2cv,
        number_of_cv=2,
        cv_labels=["Li_O_CN", "Li_Cl_CN"],
        replica_range=range(1,11)
    )

@pytest.fixture
def test_path_3cv():
    return "/Users/user/software_dev/free_energy_analysis/data/LiCl_0.5M_298K"

@pytest.fixture
def analyzer_3cv(test_path_3cv):
    return ColvarsAnalyzer(
        base_dir=test_path_3cv,
        number_of_cv=3,
        cv_labels=["Li_O_CN", "Li_Cl_CN", "Li_CN"],
        replica_range=range(1, 11)
    )

# ======== TESTS ========

def test_initialization(analyzer_1cv):
    assert len(analyzer_1cv.pmf_files) == 2
    assert os.path.exists(analyzer_1cv.pmf_files[0])
    assert os.path.exists(analyzer_1cv.colvar_files[0])


def test_read_data(analyzer_1cv):
    df = analyzer_1cv.read_data(analyzer_1cv.colvar_files[0])
    assert df.shape == (3, 3)
    assert np.allclose(df.iloc[:, 1].values, [1.0, 1.1, 1.2])


def test_plot_pmf_1cv(analyzer_1cv):
    analyzer_1cv.plot_pmf()
    assert os.path.exists("PMF_1CV.png")


def test_plot_cv_histogram(analyzer_1cv):
    analyzer_1cv.plot_cv_histogram_full()
    assert os.path.exists("CV_Histograms.png")

def test_plot_traj_overtime(analyzer_1cv):
    analyzer_1cv.plot_traj_overtime_all()
    assert os.path.exists("CV_trajectory.png")

def test_plot_pmf_2cv(analyzer_2cv):
    # Use non-interactive backend for headless testing
    analyzer_2cv.plot_pmf()
    assert os.path.exists("PMF_2CV.png")


def test_plot_pmf_3cv(analyzer_3cv):
    analyzer_3cv.plot_pmf()
    assert os.path.exists("PMF_3CV.png")
