====================
free_energy_analysis
====================


.. image:: https://img.shields.io/pypi/v/free_energy_analysis.svg
        :target: https://pypi.python.org/pypi/free_energy_analysis

.. image:: https://img.shields.io/travis/SophiaRuan/free_energy_analysis.svg
        :target: https://travis-ci.com/SophiaRuan/free_energy_analysis

.. image:: https://readthedocs.org/projects/free-energy-analysis/badge/?version=latest
        :target: https://free-energy-analysis.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Python Boilerplate contains all the boilerplate you need to create a Python package.


* Free software: MIT license
* Documentation: https://free-energy-analysis.readthedocs.io.


Features
--------

* TODO

Usage
-----

This package runs a three-stage free energy analysis pipeline over LAMMPS
metadynamics trajectories (``data/<system>/<NN_IDNR>/lammps.*.lammpstrj``):

1. ``colvars_analyzer_script.py`` — analyzes collective variables (CVs)
2. ``free_energy_analysis_script.py`` — computes the free energy surface
3. ``free_energy_correction_script.py`` — applies finite-size/concentration
   corrections

``scripts/analysis.sh`` runs all three in sequence and moves the resulting
plots, CSVs, pickles, and logs into ``data/<system>/results/``. Edit the
variables at the top of the script (``BASE_PATH``, ``LI_INDEX``,
``NUMBER_OF_CV``, radii, temperature, etc.) for your system before running.

**Linux — verified working (e.g. HPC clusters like SDSC Expanse)**

``environment.yml`` is a full conda-forge lockfile and can fail or hang on
memory-constrained machines — in particular, shared HPC login nodes often
cap each process's virtual memory (check with ``ulimit -v``), and
``conda``/``mamba`` extracting ~200 packages in one transaction can exceed
that cap even though the machine has plenty of free RAM overall. If
``conda env create -f environment.yml`` hangs, OOMs, or fails with
``CondaMemoryError`` / ``std::bad_alloc``, use this instead — it only asks
conda for a minimal Python environment and installs everything else with
``pip``, which is far lighter on memory:

.. code-block:: bash

   # 1. Get conda + mamba if you don't already have them (skip if you do)
   wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
   bash Miniforge3.sh -b -p ~/miniforge3
   source ~/miniforge3/etc/profile.d/conda.sh

   # 2. Minimal env: just Python 3.10 + pip (small transaction, won't hit memory caps)
   mamba create -n ele_machine python=3.10 pip -y
   conda activate ele_machine

   # 3. sea_urchin isn't on PyPI — clone and install it from source.
   #    Use the `release` branch: this repo's scripts import
   #    sea_urchin.sea_urchin.SeaUrchin, which lives at that path only on
   #    `release`; the `main` branch renamed the module to core.py.
   git clone https://gitlab.com/electrolyte-machine/sea_urchin.git ~/sea_urchin
   cd ~/sea_urchin && git checkout release
   pip install -e .

   # 4. Install this package (pulls typer, rich, scipy, matplotlib, pandas,
   #    MDAnalysis, pycolvars, solvation_analysis — see pyproject.toml)
   cd ~/free_energy_analysis
   pip install -e .

   # 5. Run the pipeline
   cd scripts
   chmod +x analysis.sh
   ./analysis.sh

If you already have a working conda/mamba install elsewhere and
``environment.yml`` solves fine for you, that route still works too —
just make sure the conda-forge build hashes resolve for your platform
(they're pinned to specific version strings, not exact builds, so this
should hold across Linux distros).

**Troubleshooting (Linux)**

* ``OSError: ... does not appear to be a valid lammpstrj file`` — the
  trajectory is likely truncated (e.g. the LAMMPS job was killed
  mid-write). Check whether the file ends mid-record; if so, trim it back
  to the last complete frame (look for the last complete
  ``ITEM: TIMESTEP`` block) rather than discarding the whole replica.
* ``free_energy_analysis_script.py: error: the following arguments are
  required: --skip_frames`` — set ``SKIP_FRAMES`` in ``analysis.sh`` to
  the number of initial trajectory frames to discard for equilibration.
  There's no universal default; it depends on your simulation's
  equilibration time.
* ``free_energy_analysis_script.py`` caches its clustering result at
  ``<base_path>/urchin_LiClOH_<nstrides>.pkl`` and skips recomputation if
  that file already exists — delete it before rerunning with a different
  ``SKIP_FRAMES`` or ``NSTRIDES``, or the run will silently reuse stale
  results.

**macOS (Apple) — not verified by us, best-effort guidance**

Same general approach as Linux should apply.
``conda env create -f environment.yml`` was originally authored on macOS
(Apple Silicon), so it's more likely to solve directly there than on
Linux; if it doesn't, fall back to the minimal-env + ``pip install -e .``
steps above (skip the Miniforge download if you already have
conda/mamba). Adjust the conda source path in ``analysis.sh`` to match
your local install (e.g. ``~/miniconda3/etc/profile.d/conda.sh``).

**Windows — not verified by us, best-effort guidance**

``analysis.sh`` is a bash script and won't run directly in PowerShell or
cmd.exe. Use **WSL2** (Windows Subsystem for Linux):

1. Install WSL2 with a Linux distro (e.g. Ubuntu):
   ``wsl --install`` in an administrator PowerShell.
2. Follow the Linux instructions above from within the WSL shell — WSL2
   is a real Linux environment, so the same steps and troubleshooting
   notes apply.

Git Bash can run the script syntactically, but conda activation and file
paths behave differently there, so WSL2 is recommended over Git Bash.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
