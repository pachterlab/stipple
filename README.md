# stipple

Spatial mixed-model MAP estimation with a spatial field.

`StippleModel` fits per-location RNA capture efficiency for spatial
transcriptomics data. A deterministic basis-function 
expansion models the capture-rate field (`u = B @ w`,
`w_k ~ Normal(0, sigma2)`, `sigma2` fixed not estimated) shared across the
entire tissue; sample groups (e.g. cell types or regions) don't get their
own spatial field, but `u` is centered *per group* on the logit scale
before a single global normalization converts it to a probability
(`p_i = known_capture_rate * expit(u_tilde_i) / mean_i(expit(u_tilde))`),
so that a between-group difference in total expression can't be laundered
into a spurious between-group difference in average capture efficiency. 

## Setup

```bash
conda env create -f environment.yml
conda activate stipple
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
  
## Citation