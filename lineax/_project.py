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

"""Projection operators for variable projection and separable nonlinear least squares.

This module provides efficient projection operators P = A A^† with optimized
gradients following Golub & Pereyra (1973) for use in variable projection methods.
"""

from typing import Any

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.core
import jax.interpreters.ad as ad
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, PyTree

from ._custom_types import sentinel
from ._misc import to_shapedarray
from ._operator import (
    AbstractLinearOperator,
    conj,
    FunctionLinearOperator,
    IdentityLinearOperator,
    linearise,
    TangentLinearOperator,
)
from ._solve import (
    AbstractLinearSolver,
    AutoLinearSolver,
    check_rank_compat,
    linear_solve,
    projection_mv,
)
from ._tags import MaxRankTag, project_tags, tags_from_checks


# The solver-specific projection fast path `projection_mv` lives in `._solve`, with
# per-solver registrations in the solver modules (`_solver/svd.py`, `qr.py`,
# `normal.py`); it is shared with the `linear_solve` JVP's row-space projection.


# ============================================================================
# Public API
# ============================================================================


def project(
    operator: AbstractLinearOperator,
    solver: AbstractLinearSolver = AutoLinearSolver(well_posed=None),
    *,
    options: dict[str, Any] | None = None,
    state: PyTree[Any] = sentinel,
    throw: bool = True,
) -> AbstractLinearOperator:
    r"""Returns a [`lineax.FunctionLinearOperator`][] representing the orthogonal
    projection ``P = A A^†`` onto the range (column space) of `operator`.

    `project(A).mv(v)` is equivalent to `A.mv(linear_solve(A, v, solver).value)`: it
    projects `v` onto `range(A)`. The result is idempotent (`P.mv(P.mv(v)) == P.mv(v)`)
    and Hermitian.

    Like [`lineax.invert`][], the solver state (factorisation) is computed once and
    reused across every mv. The returned operator provides the efficient
    [Golub-Pereyra](https://doi.org/10.1137/0710036) gradient used to perform
    [variable projection](https://www.cs.umd.edu/users/oleary/software/varpro.pdf) in
    separable nonlinear least-squares problems; see the
    [worked example](../examples/variable_projection.ipynb).

    **Arguments:**

    - `operator`: the linear operator `A` to project onto. May be any shape. (Raw
        arrays are not accepted; wrap them as `lineax.MatrixLinearOperator(matrix)`.)
    - `solver`: the linear solver to use. Defaults to
        `AutoLinearSolver(well_posed=None)`. For rank-deficient `A`, use
        `AutoLinearSolver(well_posed=False)` or another appropriate solver.
    - `options`: additional options passed to the solver. Defaults to `None`.
    - `state`: as [`lineax.invert`][]. Defaults to being initialised with gradients
        stopped through the operator.
    - `throw`: as [`lineax.linear_solve`][]. Defaults to `True`.

    **Returns:**

    A [`lineax.FunctionLinearOperator`][] whose `mv` applies `P @ v`. When `A` has full
    row rank the projector is the identity, returned as a
    [`lineax.IdentityLinearOperator`][].
    """
    if eqx.is_array(operator):
        raise ValueError(
            "`project(operator=...)` should be an `AbstractLinearOperator`, not a "
            "raw JAX array. If you are trying to pass a matrix then this should be "
            "passed as `lineax.MatrixLinearOperator(matrix)`."
        )

    m = operator.out_size()
    n = operator.in_size()

    if m == 0 or n == 0:
        raise ValueError(
            f"operator cannot have zero dimensions, got out_size={m}, in_size={n}"
        )

    # raise if operator is rank-deficient and solver assumes full rank
    check_rank_compat(solver, operator)

    if solver.assume_full_rank() and m <= n:
        # square/wide matrix with full row rank: P = A A^†  = I
        return IdentityLinearOperator(operator.out_structure())

    if options is None:
        options = {}
    if state == sentinel:
        dynamic_operator, static_operator = eqx.partition(operator, eqx.is_array)
        stopped_operator = eqx.combine(
            lax.stop_gradient(dynamic_operator), static_operator
        )
        state = solver.init(stopped_operator, options)

    dynamic_state, static_state = eqx.partition(state, eqx.is_array)
    dynamic_state = lax.stop_gradient(dynamic_state)
    state = eqx.combine(dynamic_state, static_state)
    options = eqxi.nondifferentiable(options, name="`lineax.project(..., options=...)`")
    solver = eqxi.nondifferentiable(solver, name="`lineax.project(..., solver=...)`")

    def mv(vector: PyTree[ArrayLike]) -> PyTree[Array]:
        (out,) = eqxi.filter_primitive_bind(
            _projection_mv_p, operator, state, vector, options, solver, throw
        )
        return out

    # `tags_from_checks` drops MaxRankTag for the full rank case.
    # Add the cheap supplemental MaxRankTag(min(m, n)) to compensatee.
    # The min-rule in `project_tags` keeps whichever bound is tighter.
    in_tags = tags_from_checks(operator) | {MaxRankTag(min(m, n))}
    tags = project_tags(in_tags)
    return FunctionLinearOperator(mv, operator.out_structure(), tags)


# ============================================================================
# Primitive Implementation
# ============================================================================


@eqxi.filter_primitive_def
def _projection_mv_impl(
    Phi: AbstractLinearOperator,
    state: Any,
    v: PyTree[ArrayLike],
    options: dict[str, Any],
    solver: AbstractLinearSolver,
    throw: bool,
) -> tuple[PyTree[Array]]:
    """Implementation rule for the projection_mv primitive (base case): P @ v."""
    # Solver-specific fast path (e.g. O'Leary 1990 for SVD: P @ v = U U^H v)
    out = projection_mv(solver, state, v, options)
    if out is not NotImplemented:
        return (out,)

    c = linear_solve(Phi, v, solver, state=state, options=options, throw=throw).value
    return (Phi.mv(c),)


@eqxi.filter_primitive_def
def _projection_mv_abstract_eval(
    Phi: Any,
    state: Any,
    v: Any,
    options: Any,
    solver: Any,
    throw: Any,
) -> tuple[PyTree[jax.core.ShapedArray]]:
    """Abstract evaluation - returns the output shape/dtype PyTree."""
    # P = Φ Φ^† maps out-space to out-space, so P @ v has the same PyTree structure,
    # shapes and dtypes as `v` (which lives in the out-structure). Mirroring
    # `linear_solve`: get the shape/dtype PyTree via `filter_eval_shape`, then convert
    # to `ShapedArray` (the primitive's abstract-value type). (`Phi.out_structure()`
    # is unusable here -- the operator's arrays are abstract.)
    del Phi, state, options, solver, throw
    struct = eqx.filter_eval_shape(lambda x: x, v)
    return (jtu.tree_map(to_shapedarray, struct),)


def _is_none(x: Any) -> bool:
    return x is None


@eqxi.filter_primitive_jvp
def _projection_mv_jvp(
    primals: tuple[
        AbstractLinearOperator,
        Any,
        PyTree[ArrayLike],
        dict[str, Any],
        AbstractLinearSolver,
        bool,
    ],
    tangents: tuple[Any, Any, Any, Any, Any, Any],
) -> tuple[tuple[PyTree[Array]], tuple[PyTree[Array]]]:
    """JVP rule using the Golub-Pereyra efficient formulation.

    For P @ v where P = Φ Φ^†:
    d(P @ v) = P @ dv + A + B

    where:
      A = (I - P) @ (dΦ @ c)
      B = (Φ^†)^T @ (dΦ^H @ r)

    **Tangent handling**: dPhi can be:
    - AbstractLinearOperator (when Phi is MatrixLinearOperator)
    - ndarray (when materializing gradients)
    - Zero tangent (no gradient w.r.t. Phi)
    """
    Phi, state, v, options, solver, throw = primals
    dPhi, dstate, dv, doptions, dsolver, dthrow = tangents
    # dstate, doptions, dsolver, dthrow should be None (non-differentiable)
    del dstate, doptions, dsolver, dthrow

    # Forward pass: compute c = Φ^† v explicitly (the tangent terms below need it).
    c = linear_solve(Phi, v, solver, state=state, options=options, throw=throw).value
    Pv = Phi.mv(c)

    # dv term
    if any(t is not None for t in jtu.tree_leaves(dv, is_leaf=_is_none)):
        dv = jtu.tree_map(eqxi.materialise_zeros, v, dv, is_leaf=_is_none)
        (P_dv,) = eqxi.filter_primitive_bind(
            _projection_mv_p, Phi, state, dv, options, solver, throw
        )
    else:
        P_dv = jtu.tree_map(jnp.zeros_like, Pv)

    if all(t is None for t in jtu.tree_leaves(dPhi, is_leaf=_is_none)):
        # No gradient w.r.t. Phi
        return (Pv,), (P_dv,)

    dPhi_op = TangentLinearOperator(Phi, dPhi)
    dPhi_op = linearise(dPhi_op)  # Optimize for matvecs

    # Operator tangent path
    dPhi_c = dPhi_op.mv(c)

    # A term: (I - P) @ (dΦ @ c)
    (P_dPhi_c,) = eqxi.filter_primitive_bind(
        _projection_mv_p, Phi, state, dPhi_c, options, solver, throw
    )
    A = (dPhi_c**ω - P_dPhi_c**ω).ω

    # Full Golub-Pereyra: B term = (Φ^†)^H (dΦ^H r), where r = (I-P) v is the residual.
    # Under the Kaufman approximation (1975, https://doi.org/10.1007/BF01932995)
    # this term is dropped as an optimisation which is valid near an optimum as r -> 0.
    # In lineax, we do not offer the Kaufman approximation as option to simplify the API
    # and avoid a footgun due to inaccurate gradient when the residual is large.
    r = (v**ω - Pv**ω).ω
    dPhi_H_r = conj(dPhi_op).T.mv(r)  # dΦ^H r
    # (Φ^†)^H = (Φ^H)^†
    transposed_state, transposed_options = solver.transpose(state, options)
    conj_transposed_state, conj_transposed_options = solver.conj(
        transposed_state, transposed_options
    )
    B = linear_solve(
        conj(Phi).T,
        dPhi_H_r,
        solver,
        state=conj_transposed_state,
        options=conj_transposed_options,
        throw=throw,
    ).value
    return (Pv,), ((P_dv**ω + A**ω + B**ω).ω,)


def _is_undefined(x: Any) -> bool:
    return isinstance(x, ad.UndefinedPrimal)


def _keep_undefined(v: Any, ct: Any) -> Any:
    if _is_undefined(v):
        return ct
    else:
        return None


@eqxi.filter_primitive_transpose(materialise_zeros=True)  # pyright: ignore
def _projection_mv_transpose(
    inputs: tuple[Any, Any, Any, Any, Any, Any],
    cts_out: tuple[PyTree[Array]],
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Transpose rule for VJP w.r.t. the vector input.

    The map ``v -> P v`` is linear, so its transpose applies ``P^T``. The
    projection ``P = Φ Φ^†`` is Hermitian (P^H = P), hence:
      * real Φ: P^T = P, so ct_v = P @ ct_out.
      * complex Φ: P^T = conj(P), so ct_v = conj(P @ conj(ct_out)).
    The complex branch is the conjugate of the real one; on the real path
    `iscomplexobj` is False so no conjugation is emitted.

    The cotangent w.r.t. Phi is left to the JVP rule (Phi enters nonlinearly),
    so ct_Phi is None here.
    """
    (ct_out,) = cts_out
    Phi, state, v, options, solver, throw = inputs

    if any(jnp.iscomplexobj(x) for x in jtu.tree_leaves(ct_out)):
        (pc,) = eqxi.filter_primitive_bind(
            _projection_mv_p,
            Phi,
            state,
            jtu.tree_map(jnp.conj, ct_out),
            options,
            solver,
            throw,
        )
        ct_v = jtu.tree_map(jnp.conj, pc)
    else:
        (ct_v,) = eqxi.filter_primitive_bind(
            _projection_mv_p, Phi, state, ct_out, options, solver, throw
        )
    ct_v = jtu.tree_map(_keep_undefined, v, ct_v, is_leaf=_is_undefined)

    # Φ enters nonlinearly, so its cotangent is handled by the JVP rule (None
    # here). The remaining inputs are non-differentiable.
    ct_Phi = jtu.tree_map(lambda _: None, Phi)
    ct_state = jtu.tree_map(lambda _: None, state)
    ct_options = jtu.tree_map(lambda _: None, options)
    ct_solver = jtu.tree_map(lambda _: None, solver)
    ct_throw = None

    return ct_Phi, ct_state, ct_v, ct_options, ct_solver, ct_throw


# Create the primitive using create_vprim
# This automatically handles batching by vmapping the rules
_projection_mv_p = eqxi.create_vprim(
    "projection_mv",
    _projection_mv_impl,
    _projection_mv_abstract_eval,
    _projection_mv_jvp,
    _projection_mv_transpose,
)
