# Generalization notes (branch: `GeneralizationExp(withMary)`)

Working notes on turning this repo from "the exact code behind the LiCl(aq)
paper" into a tool other people can point at their own electrolyte systems.
Full change-by-change detail lives in `README.rst`'s "Fork of Xiaoxu Ruan's
original repository" section — this file is a shorter, mentor-facing
summary of what's done and, more importantly, what's an open question.

## Done

- **State points** (concentration, temperature, radii, replica count): now
  environment-variable/config overrides on `analysis.sh` instead of
  hardcoded to 0.5M/298K, plus `sweep_analysis.sh` to batch-run every
  state point that has data on disk.
- **Chemistry** (solute/solvent atom names, LAMMPS numeric atom types):
  moved out of hardcoded Python into a per-system `configs/*.yaml`, loaded
  through `config.py`. A different salt is a new config file, not a
  source edit.
- **Water-activity fit**: now supplied per-system via that system's own
  config (optional `activity_fit` section), instead of one shared table
  in Python — two salts sharing a temperature can no longer silently
  collide and get the wrong numbers.
- Two real correctness bugs found and fixed along the way (not just
  naming/config issues):
  - `ColvarsAnalyzer`'s CV-histogram plot always titled itself
    `"Li-O CN and Li-Cl CN Histogram"` regardless of the actual `cv_labels`
    passed in.
  - `sea_urchin`'s `reconstruct` setting hardcoded LAMMPS atom type `1`
    for "reconstruct whole water molecules." For a config where
    water-oxygen isn't type 1, `sea_urchin`'s internal lookup fails
    silently (bare `except`, no error) and skips molecule reconstruction
    entirely — wrong clusters, no indication anything was wrong.
- Real pytest suite (36 tests) — the original test file imported a module
  that never existed in this package.

Every change above was verified against the real `LiCl_0.5M_298K` dataset:
numerical output is byte-identical to the pre-change baseline wherever
nothing was supposed to change.

## Open question for you: solvents other than water

We made the finite-size correction's **atom-counting arithmetic**
solvent-agnostic — `solvent_atoms_per_molecule` (was hardcoded to 3, for
water = O+2H) and which atom to count as the solvent "anchor" in a cluster
formula (was hardcoded to literal `"O"`) are now both config-driven,
defaulting to water's own values.

**That is arithmetic only, and we didn't want it to look more generalized
than it actually is.** The correction itself is built on a water
chemical-potential model specifically — paper Eq. 6:

```
mu_w = mu_w0 + kT * ln(a_w)
```

Swapping in a different solvent needs one of:
1. Confirmation that the same functional form holds for that solvent, with
   new activity data substituted in (via each config's `activity_fit`
   section, following the same pattern as a new salt's data), **or**
2. Genuinely different correction theory, if the water-specific model
   doesn't transfer.

That's a domain-science judgment call, not something resolvable by
refactoring — flagging it here rather than guessing. Once there's guidance
on which of the two applies, the config/code plumbing to support it should
be straightforward to add on top of what's already here.
