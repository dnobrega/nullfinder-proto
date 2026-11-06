# Changelog

## 0.1.0 — first public version

Changes with respect to the original single-file `null_finder.py`.

### Fixed
- **Poincaré sign with decreasing axes.** The index was computed in index-space
  orientation and came out as `-sign(det J)` for an odd number of decreasing
  coordinate axes. It is now in physical orientation.
- **Scale dependence.** The absolute defaults `b_tol=1e-10` and `det_tol=1e-20`
  made results depend on the field units: with `B × 1e-8` no null was found,
  and with only `det_tol` removed 12 spurious ones appeared. Both defaults are
  now 0; the relative tolerances are kept. As a side effect, Newton residuals
  in typical data drop from ~1e-8 to ~1e-12 relative.
- **Poincaré accuracy.** The 12-triangle vertex formula returned 0 instead of
  ±1 in about 9 % of the cells containing a single null. Faces are now refined
  adaptively (agreement with `sign(det J)`: 99.99 % on a 128³ random field).
- **Pair resolution on anisotropic grids.** Distances were divided by
  `min(dx, dy, dz)`, so a pair 0.6 cells apart along a coarse axis could be
  flagged as resolved. Distances are now measured in fractional index space.
- **Missed roots near cell faces.** The fixed 27-point start pattern can miss
  the second root of a close pair lying near a cell face. Newton is now also
  started from the sub-boxes that survive the exclusion test.
- **Result schema.** The key set no longer depends on whether candidates or
  nulls were found; `cell` is always int64; `settings` always has the same
  keys.
- Charge-0 nulls are no longer used as "opposite" partners; they get the new
  flag `degenerate`.
- Spiral nulls whose spine eigenvalue has the same sign as the fan real part
  (a source/sink-like structure caused by divergence errors) are now
  `unknown`, consistent with the rule already applied to real eigenvalues
  (40 of 8402 nulls on the random test field).
- `fan_is_perpendicular` is False for unclassified nulls (was True) and True
  for potential nulls.

### Added
- Exact exclusion test by Bernstein subdivision (`exclusion_depth`, default 3):
  removes ~77 % of the sign-change candidates before Newton without losing
  roots, about 3× faster refinement.
- Subdivision-seeded Newton starts (`max_seed_starts`).
- Chunked refinement (`chunk_size`): memory no longer grows with the total
  number of candidates.
- Any memory layout is accepted without copying (C, Fortran, strided).
- `check_finite` (default True): NaN/Inf in the field raise `ValueError`.
- `pad_periodic` helper for periodic axes.
- `current_spine_angle_deg`, `nearest_opposite_distance_cells`,
  `opposite_pair_index`, `opposite_pair_distance_cells`, `n_degenerate`,
  `n_candidates_excluded`, `n_candidates_refined`, `null_type_name`.
- Test suite, README, packaging, CI (tests on Linux/macOS, ruff lint and format check).
- Ruff configuration in `pyproject.toml`; code formatted with `ruff format`.
- Benchmark notebook on the Olshevsky et al. (2020) analytic field
  (`benchmarks/`), with optional comparison to other implementations.
- Authors, `CITATION.cff`, how to cite and acknowledgements.

### Changed (API)
- Parameters after `x, y, z` are keyword-only.
- `nearest_opposite_distance_over_h` → `nearest_opposite_distance_cells`;
  `min_opposite_distance_over_h` → `min_opposite_distance_cells`;
  `opposite_pairs[...]["distance_over_h"]` → `["distance_cells"]`.
- Default `div_tol_rel` 0.01 → 0.1 (0.01 flagged 85 % of the nulls of an
  exactly solenoidal field).
- `resolution_name` is a fixed-width string array (no pickling needed).
- `refine_candidates_serial` / `refine_candidates_parallel` were merged into
  one kernel; serial mode runs it with one thread.
