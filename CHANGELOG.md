# Changelog

All notable changes to this project are documented here, in the style of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Each entry links
to the ADR that documents the full context, decision, and consequences.

## [Unreleased]

### Removed

- **Pared the shipped package down to model fitting, `group_de_wald_test`,
  and plotting.** Removed classification-EM group inference
  (`StippleModel(..., em_n_groups=...)` and every EM-reachable code path;
  `group_key` is now the sole grouping mode), `stipple.tools.capture_field_permutation_test`,
  `capture_field_consistency_test`, `group_de_permutation_test`,
  `estimate_group_capture_offsets` (and their private helper modules), a
  `BasisFitResult` dataclass with no remaining callers, dead
  basis-roughness-penalty/collapse functions, and EM-only soft-assignment
  log-likelihoods. `scikit-learn` dropped from dependencies (its only use
  was EM's `KMeans` init). Every remaining docstring was cut to argument
  documentation plus a one-line purpose; methodological rationale moved
  into the relevant ADR. Every removed tool's fuller research record
  remains in `stipple/experimental/` (gitignored) and in git history. See
  [ADR 0027](adr/0027-pare-package-down-to-fitting-wald-test-and-plotting.md).

### Added

- **New `stipple.tools.classify_spatial_variable_genes` tool**, called after
  a model is fitted and `group_de_wald_test` has been run. Ports
  `stipple/experimental/classification/plot-classification.R`'s two-stage
  gene classification (naive vs. fitted Pearson residuals, Moran's I
  before/after regressing out the estimated capture-rate field, combined
  with the Wald DE result) into a shipped, DataFrame-returning function,
  labeling every gene `"Null Gene"`, `"DVG"`, `"DVG + Field-aligned SVG"`,
  `"Field-aligned SVG"`, or `"Field-independent SVG"`. Preserves the R
  script's `case_when` cascade verbatim, including its quirks (a rounding
  asymmetry in the borderline check, `phi`-based DE never feeding the final
  label). Two deliberate deviations from R: no hex-grid aspect correction
  before the k-NN graph (`model.data.coords_raw` used as-is), and Moran's I
  via `squidpy` (new dependency) instead of R's `spdep`. Validated against
  a real reference run (`gene_classification_v2.csv`), reproducing the
  original `group_de_wald_test(model, delta=1.0, fdr_method="qvalue",
  propagate_capture_rate_uncertainty=False)` call exactly: 98.9% exact
  `classification_all` agreement across 1049 genes (`classification_de_wald_mu`
  agrees on 99.0%, `classification_de_wald_phi` on 100%). Every one of the
  10 `classification_de_wald_mu` disagreements is a gene whose minimum
  pairwise `mu_qvalue` sits within ~0.03 of the `0.05` DE cutoff --
  consistent with ordinary floating-point/Hessian-evaluation differences
  landing a near-boundary q-value on the other side of the cutoff, not a
  parameter or logic mismatch. The 11th mismatch is a single borderline
  Moran's I value that crosses the `[0.18, 0.2)` window differently between
  `squidpy` and `spdep`, consistent with the deliberate library swap above.
- **`plot_spatial_residuals`/`plot_residual_variogram` gained `compare_naive`**,
  overlaying/paneling residuals against a naive, spatially-unaware
  IID/moment-matched model (same plug-in mean/variance as
  `plot_count_distribution`'s "moment-matched" line, broadcast to every
  location) alongside the fitted model's own residuals -- a fitted-model
  variogram/residual map that looks no flatter than the naive one means
  the spatial smoothing isn't earning its keep. `plot_spatial_residuals`
  renders the two as side-by-side panels sharing one color scale;
  `plot_residual_variogram` overlays a second line styled like
  `plot_count_distribution`'s moment-matched line. New
  `stipple.plotting._utils._moment_matched_mean_variance`/`_naive_residuals`.

### Changed

- **`compute_group_expression_uncertainty`/`compute_group_expression_uncertainty_block`:
  minimum-effect-size test in log2 units, applied to `mu` and `phi`, tidy
  `DataFrame` return instead of a dict.** The pairwise test is now
  TREAT-style (H0: `|log2fc| <= delta` vs H1: `... > delta`, default
  `delta=1.0` -- a literal 2-fold change) rather than a null-at-zero test,
  and it now applies to `phi` as well as `mu` in `_block`. Both methods
  gained `delta` and `fdr_method` (`None`/`"bh"`/`"qvalue"`, adding
  `mu_qvalue`/`phi_qvalue` columns) parameters. Both now return one
  `pandas.DataFrame` row per `(gene, group, reference_group)` ordered
  pair, replacing the old `(G,K)`/`(G,K,K)` array dict.
  `compute_group_expression_uncertainty_block` returns a
  `(DataFrame, dict | None)` tuple -- the second element is
  `capture_rate_uncertainty_diagnostics` (fit-level, no gene dimension),
  `None` unless `propagate_capture_rate_uncertainty=True`.
  `compute_group_expression_uncertainty` is unaffected by that (it never
  had this diagnostic) and returns just the `DataFrame`. Pass
  `delta=0.0` to recover the previous null-at-zero test exactly (bit-for
  -bit, unit-tested). See
  [ADR 0022](adr/0022-treat-style-log2-delta-test-tidy-frame-for-group-expression-uncertainty.md).
- **Removed the upper clamp on `n_knots_target` (`MAX_KNOTS_TARGET`,
  previously 2500).** The same classification-EM experiment
  (`stipple/experimental/`) swept `M/n` up to 5 (`M` up to 5x `n`) at both
  `n=900` and production scale (`n=5000`) with no recovery degradation
  observed -- results plateau, they don't decline -- and no prohibitive
  cost (`n=5000`, `M/n=3`: ~3.3 min per 1500-iteration fit, ~1.9GB peak
  memory). A hardcoded safety cap was therefore only ever going to block a
  legitimate use, not prevent a real failure mode found in testing; a
  caller who needs a resource ceiling should size `n_knots_target`
  directly. The `MIN_KNOTS_RATIO` floor is unchanged. `stipple.model.MAX_KNOTS_TARGET`
  is no longer exported.
- **Raised the minimum `n_knots_target`/`n` ratio (`MIN_KNOTS_RATIO`) from
  0.3 to 0.5.** A classification-EM group-recovery experiment
  (`stipple/experimental/`) found that raising `M/n` from ~0.3 to 0.5
  meaningfully improved pointwise capture-rate recovery (R^2 ~0.42-0.49 ->
  ~0.67-0.69 at `n` in {900, 3000}, `G=100`), corroborating this module's
  own earlier finding that recovery depends on `M/n`, not `n_knots_target`'s
  absolute value. Fit time stays roughly flat across an 8x range in `M` at
  fixed `n`/`G` (already documented in `resolve_n_knots_target`), so this
  raises the default basis size without a meaningful cost tradeoff. The
  `n_knots_target` auto-raise warning text changed from "below 30%" to
  "below 50%" accordingly.
- **`plot_qq_residuals` now only computes Dunn-Smyth residuals and
  requires an explicit `gene`.** Its `type` parameter is gone (it always
  uses `"dunn_smyth"`, the only residual type that's exactly `N(0, 1)`
  under a correctly-specified model, making the QQ reference line a real
  test); `gene=None` (pooling residuals across genes before the
  randomized-quantile transform) now raises `ValueError`, since pooling
  is not the true distribution of the summed counts for NegBin
  likelihood and silently broke the `N(0, 1)` guarantee.
- **Restructured the import surface around subpackages, breaking every
  old import path.** `stipple/__init__.py` now exposes only `__version__`.
  `StippleModel` moved to `stipple.model`
  (`from stipple.model import StippleModel`); `basis.py`/`data.py`/
  `grid.py`/`priors.py` moved into `stipple/model/`
  (`stipple.model.basis`, etc.); `stipple.utils` is now a package holding
  only real shipped utilities (`benjamini_hochberg`,
  `compute_dunn_smyth_residuals`/`set_dunn_smyth_residuals`) --
  `simulate`/`validate_recovery`/`ValidationReport` moved out of the
  shipped package entirely, into `tests/helpers.py`. No compatibility
  shims. See
  [ADR 0011](adr/0011-package-restructure-and-picklable-grid-transform.md).
- **Fixed `StippleModel` failing to pickle.**
  `stipple.model.grid.to_grid_coordinates`'s `inverse_transform` return
  value was a closure (unpicklable); replaced with module-level
  `_SquareInverseTransform`/`_HexInverseTransform` classes. See
  [ADR 0011](adr/0011-package-restructure-and-picklable-grid-transform.md).
- **Renamed the project `spamm-vb` -> `stipple`** (`import stipple as stp`,
  `SPAMM` -> `StippleModel`, AnnData result keys `spamm_*` -> `stipple_*`,
  `adata.uns["spamm"]` -> `adata.uns["stipple"]`). See
  [ADR 0006](adr/0006-rename-spamm-vb-to-stipple.md).
- **Replaced the Gamma-prior/`exp(-Bw)` spatial-field parameterization with
  a logit-normal one** (`u = B @ w`, `w_k ~ Normal(0, sigma2)`, exact mass
  constraint, `w=0` is the flat-field null). Dropped
  `roughness_scale`/`alpha_w`/`beta_w`/`lambda_mass`; added a fixed
  (not estimated) `sigma2` constructor argument. See
  [ADR 0007](adr/0007-logit-normal-reparameterization-and-diagnostics-port.md).
- **Restructured the package**: `spamm/model.py`/`spamm/likelihood.py`
  -> `stipple/model/` package; new `stipple/diagnostics/` package. Each
  subfolder now follows a public-`utils.py`/private-`_utils.py`
  convention. See [ADR 0007](adr/0007-logit-normal-reparameterization-and-diagnostics-port.md).
- **`n_knots_target` is now auto-clamped**: raised to 30% of `n` (with a
  warning) if smaller, capped at `MAX_KNOTS_TARGET` (2500, with a warning)
  if larger. `coarse_resolution` is exempt from the lower bound. See
  [ADR 0007](adr/0007-logit-normal-reparameterization-and-diagnostics-port.md).
- **`StippleModel.__init__` gained `gene_batch_size`/`gene_batch_seed`**
  for gene minibatching during training (unbiased-in-expectation rescaled
  estimator). `None` (default) uses every gene, every iteration, unchanged
  from prior behavior. **Must be paired with a tightened
  `convergence_tol`** or it can silently degrade recovery -- see the
  Negative/trade-offs section of
  [ADR 0007](adr/0007-logit-normal-reparameterization-and-diagnostics-port.md).
- **The spatial field `u = B @ w` is now centered per group on the logit
  scale before a single global normalization converts it to a
  probability**, instead of applying one pooled normalization directly.
  Closes the dominant channel of a MAP-fit degeneracy where a real
  between-group difference in expression mass and a spurious between-group
  difference in average capture efficiency were both consistent with the
  same observed counts. Two alternatives were considered and rejected
  first: a soft per-group penalty (its gradient doesn't scale with gene
  count `G` while the competing data-driven pull on `w` does, the same
  failure shape already seen with `sigma2` -- see ADR 0005's history), and
  an exact per-group normalization of `p` itself (gives the "no true
  between-group capture difference" assumption the same certainty as the
  measured `known_capture_rate` constant, and ties a location's reported
  capture rate to its group label). See
  [ADR 0009](adr/0009-per-group-capture-rate-normalization.md).
- **`run_map_optimization`'s convergence check now evaluates the full
  (unbatched) log-joint** (`_StippleBasisModel.full_log_joint`), sampled
  every `convergence_window` iterations, instead of the per-iteration
  training objective. The training objective is noisy whenever gene
  minibatching is enabled and would otherwise rarely satisfy a tight
  tolerance by chance regardless of whether the fit has actually
  stabilized. When `gene_batch_size` is `None` the two objectives
  coincide, so this only changes the check's sampling cadence, not its
  meaning.

### Added

- `plot_count_distribution` now overlays a moment-matched distribution
  line (in addition to the existing fitted-mixture-PMF line): the
  observed counts' plug-in sample mean/variance fit to a single
  Poisson/NegBin distribution of the same family, as if the counts were
  IID draws from it. New `moment_match_line_kwargs` parameter controls
  its styling.

- `stipple.utils.compute_dunn_smyth_residuals`/`set_dunn_smyth_residuals`:
  exact randomized quantile (Dunn-Smyth) residuals, exactly `N(0, 1)`
  under a correctly-specified model, closing the gap
  [ADR 0008](adr/0008-defer-dunn-smyth-residuals-and-qq-plotting.md) left
  open. `plot_qq_residuals`/`plot_spatial_residuals` gained
  `type="dunn_smyth"` and a `random_state` parameter. See
  [ADR 0010](adr/0010-implement-dunn-smyth-residuals.md).
- `StippleModel.fit_anndata`: convenience wrapper chaining `fit` and
  `set_anndata_results` in one call.
- `benchmark/run_benchmark.py`: a `N x G x gene_batch_size` sweep tool
  (styled after `spammonod/benchmark/run_benchmark.py`) for recovery and
  timing validation at production scale.

### Removed

- `stipple.plotting.parameters` / `plot_group_expression` (unused,
  unexported).
- `spamm.lr_test` and its `gene_subset`/`gene_subset_seed`-based cost
  mitigation, replaced by `gene_batch_size`/`gene_batch_seed` (re-sampled
  every training iteration, not once up front).
- 65 GP-era experimental scripts/models (`spamm.kernel`-dependent or
  otherwise testing the pre-ADR-0005 GP `SPAMM` model) and 2 abandoned
  score-test prototype files (`logitnormal_score_test.py` and its driver),
  from `stipple/experimental/`. The remaining 15 kept files (`rbf_capture_*`,
  `logitnormal_*`, and their drivers) had their imports fixed to `stipple`.

### Changed

- **`stipple.diagnostics` renamed to `stipple.tools`.**
- **`StippleModel.compute_group_expression_uncertainty`/
  `compute_group_expression_uncertainty_block` merged into one function,
  `stipple.tools.group_de_wald_test(model, ...)`**, moved out of
  `StippleModel` into `stipple.tools` (takes an already-fitted model
  instead of being a method) -- the block variant was
  a strict superset of the scalar one (joint per-gene Fisher block over
  `mu`+`phi`, not one parameter at a time; falls back to the scalar
  computation internally when a gene's Fisher block isn't positive
  definite), so keeping both was redundant. Every caller now gets
  `(result, capture_rate_diagnostics)` (previously only the `_block`
  variant's shape); `result` gains `mu_ci_lower`/`mu_ci_upper` unconditionally
  (previously scalar-only).

### Removed

- `stipple.diagnostics.parametric_bootstrap` (and its dedicated test file).
  Its own docstring named two reasons for existing alongside the
  Fisher/Wald test: no `phi_gk` support there (since closed, see
  `compute_group_expression_uncertainty_block`) and normality-free
  robustness -- but every SE-calibration ADR that actually needed a
  trustworthy cross-check (0009's Amendment, 0017's Amendment, 0021)
  reached for `group_stencil_test` instead, never this. See
  [ADR 0013](adr/0013-parametric-bootstrap-learning-rate.md) for its
  history.

## Prior history

### [ADR 0005] Replace the GP spatial field with a radial-basis-function capture model

Replaced a sparse variational Gaussian process (`SpammGPModel`) with a
deterministic radial-basis-function expansion, jointly MAP-fit via Adam --
no GP, no KL/quadrature machinery. Introduced the first pytest suite.
See [ADR 0005](adr/0005-replace-gp-spatial-field-with-radial-basis-capture-model.md).
