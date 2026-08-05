====================
free_energy_analysis
====================

Free-energy analysis toolkit for solvation microstates sampled from
metadynamics simulations of concentrated electrolytes. It implements the
free-energy analysis and finite-size correction stages of **SCOPE**
(Solvation Characterization via Optimized Probability Ensemble averaging),
the workflow introduced in:

    Xiaoxu Ruan, Fabrice Roncoroni, David Prendergast, and Tod A. Pascal,
    "Practical considerations for finite concentration molecular dynamics
    simulations," *J. Chem. Phys.* **165**, 044111 (2026).
    https://doi.org/10.1063/5.0322052

* Free software: MIT license
* Paper: https://doi.org/10.1063/5.0322052

What this repo does
--------------------

Understanding how ions organize in a concentrated electrolyte — whether
they stay fully hydrated, pair up, or aggregate toward precipitation —
requires more than averaged structural metrics like RDFs or coordination
numbers, which blur exactly the rare, high-concentration configurations
that control phase behavior. SCOPE addresses this by treating each
distinct solvation geometry around a tagged ion as a discrete
**microstate**, and reconstructing the true equilibrium probability (and
therefore free energy) of every microstate from biased metadynamics
trajectories.

This package covers the back half of that pipeline — everything from
"I have raw LAMMPS/Colvars trajectories from a metadynamics run" to
"here is the corrected free-energy ranking of solvation microstates."
Concretely, ``scripts/analysis.sh`` runs three stages in sequence:

1. **Colvars analysis** (``colvars_analyzer_script.py``) — sanity-checks
   the metadynamics sampling itself: plots the collective-variable (CV)
   trajectories, histograms, and potential-of-mean-force (PMF) surface
   across all replica walkers, so you can confirm the bias is actually
   exploring the intended coordination-number space before trusting
   anything downstream.
2. **Free energy analysis** (``free_energy_analysis_script.py``) —
   reconstructs solvation clusters around a tagged ion from the raw
   trajectory (via the `sea_urchin`_ clustering library), reweights the
   biased trajectory frames to recover unbiased microstate probabilities
   (Eq. 5 of the paper), and ranks each distinct cluster formula
   (e.g. ``Li[H2O]4``, ``LiCl[H2O]3``) by its probability-derived free
   energy (Eq. 4).
3. **Free energy correction** (``free_energy_correction_script.py``) —
   applies the water activity/availability-based chemical-potential
   correction (Eqs. 6–9) that accounts for the limited free-water
   reservoir in a finite simulation box, which would otherwise
   artificially stabilize oversized clusters. This is what lets the
   corrected free-energy spectrum line up with real solubility limits
   instead of a simulation-box artifact.

Who this is for
----------------

If you're studying speciation, ion pairing, or precipitation onset in
concentrated aqueous or non-aqueous electrolytes — not just LiCl, the
method generalizes to any system where a tagged solute's local
coordination environment is the quantity of interest — this package
gives you a working, reusable implementation of the reweighting and
finite-size correction math from the paper, rather than having to
re-derive Eqs. 4–9 from scratch. It expects metadynamics trajectories
biased on coordination-number collective variables (via LAMMPS' Colvars
module) as input; setting up and running those simulations themselves is
covered by the companion `solvation_spectra`_ repository (see below).

**Related repositories**

* `solvation_spectra`_ — the full, original SCOPE workflow this package
  is derived from: classical MD and metadynamics job submission scripts
  in addition to the same analysis stages. This is the repository cited
  in the paper's Data Availability statement (published as
  ``atlas-nano/solvation_spectra``). Use it if you need the simulation
  setup/submission side too, not just analysis of trajectories you
  already have.
* `sea_urchin`_ — the underlying library for extracting and clustering
  local atomic arrangements from MD trajectories (Roncoroni et al.,
  *Phys. Chem. Chem. Phys.* 2023), used internally by the free-energy
  analysis stage here.

.. _solvation_spectra: https://github.com/atlas-nano/solvation_spectra
.. _sea_urchin: https://gitlab.com/electrolyte-machine/sea_urchin

Usage
-----

Trajectories are expected at
``data/<system>/<NN_IDNR>/lammps.*.lammpstrj`` (one subdirectory per
metadynamics replica walker). ``scripts/analysis.sh`` runs the three
stages described above in sequence and moves the resulting plots, CSVs,
pickles, and logs into ``data/<system>/results/``. Edit the variables at
the top of the script (``BASE_PATH``, ``LI_INDEX``, ``NUMBER_OF_CV``,
radii, temperature, ``SKIP_FRAMES``, etc.) for your system before
running.

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

Citation
--------

If this code is useful for your research, please cite the paper it
implements:

.. code-block:: bibtex

   @article{ruan2026practical,
     title   = {Practical considerations for finite concentration
                molecular dynamics simulations},
     author  = {Ruan, Xiaoxu and Roncoroni, Fabrice and Prendergast, David
                and Pascal, Tod A.},
     journal = {The Journal of Chemical Physics},
     volume  = {165},
     pages   = {044111},
     year    = {2026},
     doi     = {10.1063/5.0322052}
   }

Contributors
------------

* **Xiaoxu Ruan** — original author; developed the SCOPE workflow and
  analysis scripts this package is built on, and led the accompanying
  paper (see Citation above).
* **Mary Zhao** (this fork) — editing and further refinement:
  environment setup, cross-platform documentation, and bug fixes.

  *All glory to God and our Lord Jesus Christ, without whom none of
  this would be possible.* — Mary Zhao

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
