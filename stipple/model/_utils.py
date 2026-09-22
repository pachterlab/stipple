"""Private helpers for :mod:`stipple.model` -- not part of the public API.

Currently empty: the Gamma/``exp(-Bw)`` parameterization's private helpers
(``initial_basis_weight``, ``default_beta_w``, ``build_penalty_tensor``)
were dropped along with that parameterization -- the logit-normal model
needs none of them (``w`` initializes to exactly zero, there is no
``beta_w``, and no roughness penalty is applied). Reserved for future
model-internal helpers that shouldn't be part of the public API.
"""
from __future__ import annotations
