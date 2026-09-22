# stipple

Spatial mixed-model MAP estimation with a radial-basis-function spatial
field.

`StippleModel` fits per-location RNA capture efficiency for spatial
transcriptomics data. A deterministic radial-basis-function (RBF)
expansion models the capture-rate field (`u = B @ w`,
`w_k ~ Normal(0, sigma2)`, `sigma2` fixed not estimated) shared across the
entire tissue; sample groups (e.g. cell types or regions) don't get their
own spatial field, but `u` is centered *per group* on the logit scale
before a single global normalization converts it to a probability
(`p_i = known_capture_rate * expit(u_tilde_i) / mean_i(expit(u_tilde))`),
so that a between-group difference in total expression can't be laundered
into a spurious between-group difference in average capture efficiency —
see
[`adr/0009-per-group-capture-rate-normalization.md`](adr/0009-per-group-capture-rate-normalization.md). Fitting is plain MAP estimation via full-batch Adam —
there is no variational distribution anywhere in this package (see
[`adr/0006-rename-spamm-vb-to-stipple.md`](adr/0006-rename-spamm-vb-to-stipple.md)
for why the project used to be called `spamm-vb`). See
[`adr/0005-replace-gp-spatial-field-with-radial-basis-capture-model.md`](adr/0005-replace-gp-spatial-field-with-radial-basis-capture-model.md)
for why the basis-function approach replaced an earlier Gaussian-process
model, and
[`adr/0007-logit-normal-reparameterization-and-diagnostics-port.md`](adr/0007-logit-normal-reparameterization-and-diagnostics-port.md)
for the current logit-normal parameterization, package structure, and
diagnostic. See [`CHANGELOG.md`](CHANGELOG.md) for a running summary of
changes alongside each ADR.

## Setup

```bash
conda env create -f environment.yml
conda activate spamm-vb  # env name unchanged by the stipple rename, see ADR 0006
pip install -e .
```

This environment's `KMP_DUPLICATE_LIB_OK=TRUE` conda env variable (already
configured) works around a known macOS conflict between conda's
`llvm-openmp` and torch's bundled `libomp` — no action needed.

## Input format

- `adata.X`: `n_locations x n_genes` non-negative integer counts, dense or
  sparse.
- `adata.obsm['spatial']`: `n_locations x 2` coordinates forming a regular
  hexagonal or square grid (see `stipple.model.grid.check_grid`).
- `adata.obs[group_key]`: a categorical column (no missing values)
  assigning each location to one of `K` groups.

## Minimal working example

```python
from stipple.model import StippleModel

# adata: AnnData with adata.X (n x G counts), adata.obsm['spatial'] (n x 2),
# and adata.obs['group'] (categorical) -- see "Input format" above. This
# repo's own tests build synthetic data with tests/helpers.py's simulate().
model = StippleModel(adata, group_key="group", known_capture_rate=0.3, verbose=True)
model.fit(n_iterations=1000)

p_hat = model.get_capture_rates()
model.summary()
```

## The spatial basis

`StippleModel`'s constructor builds the spatial field from one or two
knot-grid *resolutions*:

```python
model = StippleModel(
    adata, group_key="group", known_capture_rate=0.3,
    sigma2=1.0,  # fixed prior variance on each basis weight w_k ~ Normal(0, sigma2)
    fine_resolution={"n_knots_target": 100, "radius_overlap_factor": 1.5, "basis_kind": "bisquare"},
    coarse_resolution=None,  # omit for a single-resolution basis (the default)
)
```

- `fine_resolution` / `coarse_resolution`: each a dict with
  `n_knots_target` (required), `radius_overlap_factor` (default `1.5`),
  and `basis_kind` (`"bisquare"` [default], `"gaussian"`, or `"wendland"`
  — compactly-supported radial kernels, see `stipple.model.basis`), or `None` to
  omit that resolution entirely. At least one must be given; omitting both
  defaults to a single `fine_resolution` (`n_knots_target=100`). Passing
  both builds a two-resolution basis (a fine grid for local structure plus
  a coarse grid for broad trends); passing only `coarse_resolution` fits a
  single coarse-only basis.
- `n_knots_target` is scale-aware: recovery of the capture-rate field
  depends on the ratio of basis functions to locations, not
  `n_knots_target`'s absolute value, so a value below 50% of `n` is raised
  to that floor (with a warning); there is no upper clamp. This only
  applies to `fine_resolution` (or a single resolution) —
  `coarse_resolution` is exempt from the lower bound, since it's meant to
  stay deliberately sparse regardless of `n`. See ADR 0007.
- `sigma2`: fixed prior variance for every basis weight — not estimated
  from data. `1.0` is a reasonable default, validated at a properly-sized
  basis, not independently tuned per dataset.
- `gene_batch_size`: if set (and `< G`), each training iteration uses a
  random subset of this many genes instead of all `G`, reducing
  per-iteration cost at large `G`. `None` (default) uses every gene, every
  iteration. See ADR 0007 for the recovery/timing trade-off.

`get_basis_params()` returns the resolved configuration and fitted-weight
summary after fitting (every per-resolution field is a list, fine first).

## Tools (stipple.tools)

`stipple.tools.group_de_wald_test(model, ...)` — see "Output methods"
below. Takes an already-fitted `StippleModel` and returns
`(pandas.DataFrame, dict | None)`.

## Output methods

- `get_capture_rates()` — estimated capture rate `p_hat_i` for every
  location. No posterior standard deviation is returned: `w` is a
  deterministic MAP point estimate, not a variationally-distributed
  quantity, so there is no analogous posterior uncertainty to report.
- `get_group_expression()` — estimated mean expression `mu_hat_gk`
  (`G x K`).
- `get_dispersion()` — estimated NegBin dispersion `phi_hat_gk` (`G x K`;
  only available with `likelihood="negbinomial"`).
- `get_residuals(type="pearson"|"deviance")` — residuals at the MAP fit
  (`G x n`).
- `get_basis_params()` — resolved basis configuration and fitted-weight
  summary (see above).
- `stipple.tools.group_de_wald_test(model, delta=1.0, propagate_capture_rate_uncertainty=False, fdr_method=None)`
  — Fisher-information standard errors and confidence intervals for
  `mu_hat_gk` and (for `likelihood="negbinomial"`) `phi_hat_gk`, plus a
  pairwise TREAT-style minimum-effect-size test in **log2** units: H0:
  `|log2fc| <= delta` vs H1: `|log2fc| > delta`, for both `mu` and `phi`.
  `delta=1.0` (the default) is a literal 2-fold-change threshold; pass
  `delta=0.0` to recover the plain null-at-zero two-sided Wald test.
  Returns `(result, capture_rate_diagnostics)`: `result` is a tidy
  `pandas.DataFrame`, one row per `(gene, group, reference_group)` ordered
  pair (`mu_se_log2`/`mu_se`/`mu_ci_lower`/`mu_ci_upper`/`mu_log2_fold_change`/
  `mu_wald_stat`/`mu_pvalue`, and the `phi_*`/`cov_log_mu_log_phi` columns
  for `likelihood="negbinomial"`); p-values are uncorrected by default —
  pass `fdr_method="bh"` or `"qvalue"` for an added `mu_qvalue`/
  `phi_qvalue` column. `capture_rate_diagnostics` is `None` unless
  `propagate_capture_rate_uncertainty=True`, in which case it's a dict
  (`group_pair_covariance`, `retained_variance_fraction`, `m_star`;
  fit-level, no gene dimension, so it's a second return value rather than
  a DataFrame column).
- `summary()` — model spec, known vs. estimated mean capture rate, basis
  configuration, final objective, convergence status, and runtime.
- `write_obs(adata)` / `write_obsm(adata)` / `write_varm(adata)` /
  `write_uns(adata)` — write fit results back into an `AnnData` in place:
  `adata.obs["stipple_p_hat"]` (length `n`);
  `adata.obsm["stipple_fitted_mean"]`/`"stipple_fitted_variance"` (`n x G`,
  per-location per-gene, see `get_fitted_values()`);
  `adata.varm["stipple_mu_gk"]`/`"stipple_phi_gk"` (`G x K`); scalar/short
  model parameters under `adata.uns["stipple"]` (no length-`n` arrays).
  `set_anndata_results(adata, copy=False)` runs all four — scanpy
  convention: `copy=False` (default) mutates `adata` in place and
  returns `None`; `copy=True` returns an updated copy, leaving `adata`
  untouched.

## Package layout

The top-level `stipple` package exposes nothing but `__version__` --
import from the subpackage you need:

- `stipple.model` — `StippleModel` (the fitting entry point,
  `from stipple.model import StippleModel`), plus the MAP-fitting
  machinery (`run_map_optimization`, `resolve_n_knots_target`) and the
  AnnData-preprocessing/basis-construction modules it's built from:
  - `stipple/model/basis.py` — radial basis construction (knot placement,
    basis evaluation/normalization, single- and multi-resolution)
  - `stipple/model/grid.py` — spatial grid detection and coordinate
    transforms
  - `stipple/model/priors.py` — plug-in gene-level estimates and
    hyperpriors
  - `stipple/model/data.py` — AnnData extraction, validation, and
    preprocessing

  The underlying `torch.nn.Module` (`stipple.model.model._StippleBasisModel`)
  is an internal implementation detail shared with `stipple.tools`,
  not a second model users choose between — the basis-function approach
  is the only one this package implements.
- `stipple.tools` — `group_de_wald_test` (see above)
- `stipple.utils` — standalone post-fit utilities that read/write a
  fitted `adata` directly (`import stipple.utils`), e.g.
  `compute_dunn_smyth_residuals`/`set_dunn_smyth_residuals`
- `stipple.plotting` — diagnostic and result plots
  (`import stipple.plotting`)

Each subpackage follows the same convention: public helpers live in
the package's own `__init__.py`, private/internal helpers in `_utils.py`.
`simulate()`/`validate_recovery()` (synthetic-data generation and
recovery validation) are test-only and live in `tests/helpers.py`, not in
the shipped package.

## Benchmarking

`benchmark/run_benchmark.py` sweeps `N x G x gene_batch_size` (plus seeds),
fitting `StippleModel` on simulated data and recording recovery metrics
and wall-clock timing to a CSV — used to validate that gene minibatching
doesn't hurt recovery and to project production-scale runtime (see ADR
0007). Run `python benchmark/run_benchmark.py --help` for the full sweep
grid and CLI options.

## Testing

```bash
pip install -e ".[test]"
pytest
```
