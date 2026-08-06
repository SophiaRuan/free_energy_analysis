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

Fork of Xiaoxu Ruan's original repository
------------------------------------------

This is a fork of `Xiaoxu Ruan's original free_energy_analysis
repository`_ — the code exactly as used to produce the paper's results,
built around one hardcoded system (0.5 M LiCl at 298 K). This fork's goal
is different: turn that same pipeline into a tool other people can point
at their own electrolyte systems, not just reproduce the paper's LiCl run.

.. _Xiaoxu Ruan's original free_energy_analysis repository: https://github.com/SophiaRuan/free_energy_analysis

**What's been generalized so far**

* State-point parameters (concentration, temperature, radii, stride/skip
  counts) are environment-variable overrides on ``analysis.sh`` instead of
  hardcoded, and ``scripts/sweep_analysis.sh`` (new) batches a run across
  every concentration/temperature combination that has data on disk.
* The tagged ion's LAMMPS atom index (``LI_INDEX``) is auto-derived from
  ``colvar.lmp`` instead of requiring a hand-maintained, per-concentration
  lookup table that had to be kept in sync by hand.
* Solute/solvent species names and the LAMMPS numeric atom-type mapping
  used by the finite-size correction moved out of hardcoded Python
  (``"Li"``, ``"Cl"``, ``type 1``–``type 4``) and into
  ``configs/ele_machine.yaml``, loaded through a new ``config.py``. A new
  salt/solvent combination is a new config file, not a source-code edit —
  see ``configs/example_new_salt.yaml`` for a worked (placeholder-data)
  template.
* The water-activity/solubility fit used by the finite-size correction is
  now read per-system from that system's own config (an optional
  ``activity_fit`` section), instead of one shared table in Python code —
  so two different salts that happen to share a temperature can't
  silently collide and get the wrong numbers. Configs that omit it (like
  the default ``ele_machine.yaml``) fall back to this package's built-in
  LiCl(aq) fit; the code raises a clear error if a run needs a
  temperature that's configured nowhere.
* The number of metadynamics replica walkers is discovered from the
  ``*_IDNR`` directories actually on disk instead of being hardcoded to
  10 — a different walker count no longer requires a code change.
* CLI flags on ``free_energy_analysis_script.py`` /
  ``free_energy_correction_script.py`` were renamed to generic terms —
  ``--solute_index``, ``--water_o_radii``, ``--water_h_radii``,
  ``--anion_radii`` — instead of ``--Li_index``/``--O_radii``/
  ``--H_radii``/``--Cl_radii``. The old names still work as deprecated
  aliases (same ``dest``), so nothing already calling this package breaks.
* Fixed a real (not just cosmetic) bug: ``ColvarsAnalyzer``'s CV-histogram
  plot always titled itself ``"Li-O CN and Li-Cl CN Histogram"`` regardless
  of the ``cv_labels`` you passed it — so even a correctly-configured
  non-LiCl run produced a mislabeled plot. It now builds the title from
  the actual labels in use.
* Cross-platform install docs (verified Linux/HPC path, best-effort
  macOS/Windows guidance, a memory-constrained install path for capped HPC
  login nodes) and a portable conda-activation fallback that checks for
  the actual ``conda activate`` shell function rather than assuming one
  specific install path or that the binary alone being on ``PATH`` is
  sufficient.
* A real, passing pytest suite (27 tests) for the pieces above that don't
  require HPC trajectory data to test — the original test file imported a
  module that never existed in this package and had never actually run.

**What's intentionally not generalized (yet)**

* The water-activity/solubility *numbers* themselves are still only
  populated for LiCl(aq) at 283/298/313 K — the mechanism to supply a
  different salt's numbers now exists (see above), but the actual
  experimentally-derived values for another system have to come from you,
  not from this codebase.
* ``src/free_energy_analysis/solvation_analysis_tool.py`` is dead code
  (nothing imports it) and hasn't been touched either way.

See "Adapting to a different salt or solute" under Usage_ below for how to
point this at a non-LiCl system today.

Usage
-----

Trajectories are expected at
``data/<system>/<NN_IDNR>/lammps.*.lammpstrj`` (one subdirectory per
metadynamics replica walker). ``scripts/analysis.sh`` runs the three
stages described above in sequence and moves the resulting plots, CSVs,
pickles, and logs into ``data/<system>/results/``.

Every variable ``analysis.sh`` uses (``BASE_PATH``, ``CONC``, ``TEMP``,
``NUMBER_OF_CV``, radii, ``SKIP_FRAMES``, etc.) can be overridden from the
environment instead of editing the file, e.g.:

.. code-block:: bash

   CONC=2 TEMP=298 ./analysis.sh

``LI_INDEX`` (the LAMMPS index of the tagged/biased ion) does *not* need to
be set — it's auto-derived from the ``group1 { atomNumbers N }`` entry
already present in that system's own ``colvar.lmp``, which records the
same atom used to bias the metadynamics run. Set it explicitly only if you
need to override that.

To run several concentrations/temperatures in one go, use
``scripts/sweep_analysis.sh``, which calls ``analysis.sh`` once per
``data/LiCl_<CONC>M_<TEMP>K`` directory that actually exists, skipping the
rest.

**Adapting to a different salt or solute.** The parts of the pipeline
that depend on *which chemical system* you're analyzing — not just which
concentration/temperature — live in ``configs/ele_machine.yaml``, not in
the scripts: the solute reference atom, the coordination-environment
species, and the LAMMPS numeric atom-type mapping used to pick out water
vs. salt in the finite-size correction. To analyze a non-LiCl system, copy
that file (``configs/example_new_salt.yaml`` is a worked, clearly-marked
placeholder-data template to start from), edit its values for your
chemistry, and pass ``--config path/to/your_system.yaml`` to
``free_energy_analysis_script.py`` / ``free_energy_correction_script.py``
(or set it as the default in ``analysis.sh``) — no changes to ``src/`` are
needed for this part.

The water-activity/solubility fit used by the finite-size correction
(Sec. II.A.4 of the paper) can also be supplied per-system, via an
optional ``activity_fit`` section in that same config file — see the
commented-out example in ``configs/ele_machine.yaml``. Leave it out and
the correction falls back to this package's built-in LiCl(aq) fit
(283/298/313 K). Either way, these are **experimentally-measured**
numbers (e.g. from published water-activity tables), not something this
tool can derive from your simulation — a different salt or temperature
needs its own real data, and the code raises a clear ``ValueError``
naming what's missing if you run the correction stage without any fit
available for that temperature.

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
  ``<base_path>/urchin_<system_tag>_<nstrides>.pkl`` (``system_tag`` comes
  from the config file, ``LiClOH`` by default) and skips recomputation if
  that file already exists — delete it before rerunning with a different
  ``SKIP_FRAMES`` or ``NSTRIDES``, or the run will silently reuse stale
  results.

**macOS (Apple) — not verified by us, best-effort guidance**

Same general approach as Linux should apply.
``conda env create -f environment.yml`` was originally authored on macOS
(Apple Silicon), so it's more likely to solve directly there than on
Linux; if it doesn't, fall back to the minimal-env + ``pip install -e .``
steps above (skip the Miniforge download if you already have
conda/mamba). ``analysis.sh`` auto-detects and sources ``conda.sh`` from
common install locations (miniforge3, miniconda3, anaconda3, mambaforge
under your home directory); if yours lives elsewhere, source it yourself
before running the script.

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
