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

**Setup (all platforms)**

.. code-block:: bash

   conda env create -f environment.yml
   conda activate ele_machine
   cd scripts
   ./analysis.sh

**Linux**

Works natively as above. If ``analysis.sh`` isn't executable yet, run
``chmod +x scripts/analysis.sh`` first. The script sources
``~/miniforge3/etc/profile.d/conda.sh`` — update that path if your conda/
miniforge install lives elsewhere (e.g. ``~/miniconda3``).

**macOS (Apple)**

Same steps as Linux, using the system Terminal or iTerm. ``environment.yml``
no longer pins conda build hashes, so it should solve on both Apple Silicon
and Intel Macs, but if channel resolution fails, recreate the environment
manually with ``conda create -n ele_machine python=3.10`` and
``pip install -r requirements_dev.txt``. Adjust the conda source path in
``analysis.sh`` to match your local install (e.g.
``~/miniconda3/etc/profile.d/conda.sh``).

**Windows**

``analysis.sh`` is a bash script and won't run directly in PowerShell or
cmd.exe. Use **WSL2** (Windows Subsystem for Linux):

1. Install WSL2 with a Linux distro (e.g. Ubuntu):
   ``wsl --install`` in an administrator PowerShell.
2. Install Miniforge/Miniconda inside the WSL environment.
3. Follow the Linux instructions above from within the WSL shell.

Git Bash can run the script syntactically, but conda activation and file
paths behave differently there, so WSL2 is recommended for a native Linux
environment.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
