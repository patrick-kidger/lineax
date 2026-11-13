# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import equinox as eqx
import equinox.internal as eqxi
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array

from ._custom_types import sentinel
from ._operator import (
    AbstractLinearOperator,
    IdentityLinearOperator,
    in_dtype,
    TangentLinearOperator,
    trace,
)
from ._solve import AbstractDirectLinearSolver, invert
from ._solver import AutoLinearSolver
from ._solver.circulant import Circulant
from ._solver.normal import Normal
from ._solver.triangular import Triangular
from ._solver.tridiagonal import Tridiagonal


def _det_sign_error_msg(
    solver: "AbstractDirectLinearSolver | Normal",
    operator: AbstractLinearOperator,
) -> str:
    if isinstance(solver, Normal):
        return (
            "`lx.determinant` with `Normal`: sign of the determinant is not "
            "recoverable — the gram matrix construction destroys sign information. "
            "To recover the sign of a full-rank rectangular matrix's "
            "pseudodeterminant, use `lx.QR()`."
        )
    return (
        f"`lx.determinant` with `{type(solver).__name__}`: sign of the determinant "
        "is not available from this solver's factorisation. "
        "For Hermitian matrices (including rank-deficient ones) use `lx.HEVD()`, "
        "which returns the pseudodeterminant sign. "
        "For non-Hermitian rank-deficient matrices, lineax does not support "
        "pseudodeterminant sign recovery; use `jnp.linalg.eig` directly."
    )


def _jvp_through_state(
    solver: "AbstractDirectLinearSolver | Normal",
    operator: AbstractLinearOperator,
) -> bool:
    """Whether to differentiate this solver's own `slogdet` through its state, rather
    than use the solve-based `trace(A^-1 dA)` rule.

    These are the solvers that *benefit* from being differentiated through, not merely
    the ones that can be: `LU` differentiates perfectly well, but differentiating its
    factorisation costs what the solves it would save cost anyway (A100, float64,
    within 12% either way at n = 256 and n = 1024). For these three the determinant
    comes straight from the operator's entries -- a three-term minor recurrence, an
    FFT, or a product of the diagonal -- so differentiating it is O(n), or O(n log n)
    for `Circulant` and O(n**2) for `Triangular`, whose operator is a dense matrix in
    the first place. Against the solve-based rule, which for these would have to reach
    entries of `A^-1` that no amount of structural dispatch makes cheap: the band of a
    tridiagonal inverse needs the inverse. Only add a solver here whose `init` and
    `slogdet` are pure JAX: `QR` would raise `NotImplementedError` for `geqrf`.

    `Diagonal` is deliberately *not* here: `trace(A^-1 dA)` dispatches structurally --
    `invert` propagates the diagonal tag and the tangent operator inherits it, so the
    whole rule is one solve and an elementwise product, measured at 0.75x the cost of
    differentiating `Diagonal.slogdet` itself. The solve also masks the same
    (near-)zero entries the pseudodeterminant drops, so `well_posed=False` gradients
    are exact zeros for the masked entries in reverse mode too, where differentiating
    the masked `log` produced `nan`.

    `slogdet` consults this to decide whether to hand the state to `_slogdet`
    differentiably, and `_slogdet_jvp` to decide whether to use that. They have to
    agree: the rule would otherwise differentiate a state that was stopped.
    """
    # `AutoLinearSolver` forwards `slogdet` to whatever it selected, so look through it.
    if isinstance(solver, AutoLinearSolver):
        solver = solver.select_solver(operator)
    return isinstance(solver, Triangular | Tridiagonal | Circulant)


@eqx.filter_custom_jvp
def _slogdet(operator, solver, options, state):
    return solver.slogdet(state, options)


@_slogdet.def_jvp
def _slogdet_jvp(primals, tangents):
    operator, solver, options, state = primals
    t_operator, _, _, t_state = tangents

    if _jvp_through_state(solver, operator):
        # `slogdet` left the state differentiable for exactly this, so the
        # factorisation is differentiated where it was built, once. Rebuilding it here
        # instead would leave two of them in the jaxpr and lean on the compiler to
        # notice they are the same.
        #
        # `eqx.filter_jvp`, not `jax.jvp`: a state may carry non-array leaves --
        # `PackedStructures` among them -- which `jax.jvp` rejects outright, and
        # `filter_custom_jvp` hands us `None` tangents for them, which it also cannot
        # consume.
        return eqx.filter_jvp(
            lambda s: solver.slogdet(s, options), (state,), (t_state,)
        )

    sign, lad = solver.slogdet(state, options)

    # d(lad)/dA = trace(A† dA), where A† is the pseudoinverse.
    # operator is the only differentiable argument, so t_operator is always present.
    #
    # `invert` shares this primal's factorised state, and composing it with the
    # tangent operator lets `lx.trace` dispatch on structure: `invert` propagates the
    # inversion-closed tags and the tangent operator inherits its primal's, so for a
    # diagonal operator this is one solve and an elementwise product. With no
    # structure, `diagonal(ComposedLinearOperator)` falls back to one solve per
    # column of the materialised tangent -- the same work as a hand-rolled loop.
    #
    # `throw=True` mirrors `linear_solve`'s own JVP rule (see `_linear_solve_jvp`): a
    # failed tangent solve has nowhere to pipe an error result, so we surface it
    # loudly rather than silently returning a `nan` gradient. Pseudoinverse solvers
    # (SVD, HEVD, ...) never raise here, so the pseudodeterminant path is unchanged.
    # This applies to this solve-based path only: the solvers handled above
    # differentiate their own `slogdet`, and so return a non-finite gradient for a
    # singular operator -- where `d log|det A|` genuinely does not exist -- rather
    # than raising.
    dA = TangentLinearOperator(operator, t_operator)
    inverse = invert(operator, solver, options=options, state=state, throw=True)
    lad_dot = trace(inverse @ dA)

    if jnp.issubdtype(lad_dot.dtype, jnp.complexfloating):
        # For complex A: sign carries the imaginary part of the trace
        sign_dot = (lad_dot - jnp.real(lad_dot).astype(lad_dot.dtype)) * sign
        lad_dot = jnp.real(lad_dot)
    else:
        sign_dot = jnp.zeros_like(sign)

    return (sign, lad), (sign_dot, lad_dot)


def slogdet(
    operator: AbstractLinearOperator,
    solver: "AbstractDirectLinearSolver | Normal" = AutoLinearSolver(well_posed=True),
    *,
    options: dict[str, Any] | None = None,
    state: Any = sentinel,
) -> tuple[Array, Array]:
    """Compute `(sign, log|det(operator)|)` using the given direct solver.

    Follows the same convention as `numpy.linalg.slogdet`.

    **Arguments:**

    - `operator`: a linear operator.
    - `solver`: an [`lineax.AbstractDirectLinearSolver`][] or [`lineax.Normal`][].
        Defaults to [`lineax.AutoLinearSolver`][]`(well_posed=True)`, matching
        [`lineax.linear_solve`][].
    - `options`: any extra options to pass to the solver.
    - `state`: if provided, use this pre-computed factorised state instead of
        calling `solver.init`. Allows multiple determinant computations to share
        the same factorisation. This should be the result of calling
        [`lineax.AbstractLinearSolver.init`][] on the same `operator`; when
        differentiating, it should be built from `operator` inside the
        differentiated computation.

    **Returns:**

    A 2-tuple of `(sign, logabsdet)`. `sign` is `nan` when the solver cannot
    recover it cheaply (e.g. [`lineax.SVD`][] on a full-rank square matrix, or
    [`lineax.Normal`][]).
    """
    if not isinstance(solver, AbstractDirectLinearSolver | Normal):
        raise TypeError(
            "`lx.slogdet` requires a direct solver: an "
            "`lx.AbstractDirectLinearSolver`, or `lx.Normal` wrapping one. Got "
            f"`{type(solver).__name__}`, which has no factorisation to compute a "
            "determinant from."
        )
    if options is None:
        options = {}
    if isinstance(operator, IdentityLinearOperator):
        dtype = in_dtype(operator)
        return jnp.ones((), dtype=dtype), jnp.zeros((), dtype=dtype)
    # For the solvers whose own `slogdet` we differentiate, the state is the thing
    # being differentiated, so it has to reach `_slogdet` with its tangent intact --
    # both the `stop_gradient` and the `nondifferentiable` guard below would sever it.
    # For every other solver the state is a factorisation that the generic rule only
    # solves against, and differentiating it is a mistake we would rather catch.
    through_state = _jvp_through_state(solver, operator)
    if state is sentinel:
        if through_state:
            state = solver.init(operator, options)
        else:
            dynamic_op, static_op = eqx.partition(operator, eqx.is_array)
            stopped_op = eqx.combine(lax.stop_gradient(dynamic_op), static_op)
            state = solver.init(stopped_op, options)
    if not through_state:
        dynamic_state, static_state = eqx.partition(state, eqx.is_array)
        state = eqx.combine(lax.stop_gradient(dynamic_state), static_state)
        state = eqxi.nondifferentiable(state, name="`lx.slogdet` state")
    return _slogdet(operator, solver, options, state)


def determinant(
    operator: AbstractLinearOperator,
    solver: "AbstractDirectLinearSolver | Normal" = AutoLinearSolver(well_posed=True),
    *,
    options: dict[str, Any] | None = None,
    state: Any = sentinel,
    throw: bool = True,
) -> Array:
    """Compute det(operator) using the given direct solver.

    **Arguments:**

    - `operator`: a linear operator.
    - `solver`: an [`lineax.AbstractDirectLinearSolver`][] or [`lineax.Normal`][].
        Defaults to [`lineax.AutoLinearSolver`][]`(well_posed=True)`, matching
        [`lineax.linear_solve`][].
    - `options`: any extra options to pass to the solver.
    - `state`: if provided, use this pre-computed factorised state instead of
        calling `solver.init`. Allows multiple determinant computations to share
        the same factorisation. This should be the result of calling
        [`lineax.AbstractLinearSolver.init`][] on the same `operator`; when
        differentiating, it should be built from `operator` inside the
        differentiated computation.
    - `throw`: if `True` (the default), raise an error when the sign of the
        determinant is not available (e.g. when using [`lineax.Normal`][] or
        [`lineax.SVD`][] on a full-rank square matrix). If `False`, a `nan`
        result is returned silently.

    **Returns:**

    A scalar array equal to the determinant.
    """
    if options is None:
        options = {}
    sign, lad = slogdet(operator, solver, options=options, state=state)
    # `lad` is always real; cast `exp(lad)` to `sign`'s dtype so the product is
    # well-typed under strict dtype promotion when `sign` is complex.
    det = sign * jnp.exp(lad).astype(sign.dtype)
    if throw:
        msg = _det_sign_error_msg(solver, operator)
        det = eqx.error_if(det, jnp.isnan(sign), msg)
    return det
