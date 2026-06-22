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

from copy import copy
from typing import Any, TypeVar

import equinox.internal as eqxi
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._operator import (
    AbstractLinearOperator,
    conj,
    linearise,
    materialise,
    TaggedLinearOperator,
)
from .._solution import RESULTS
from .._tags import positive_semidefinite_tag
from .base import AbstractDirectLinearSolver, AbstractLinearSolver


_InnerSolverState = TypeVar("_InnerSolverState")


def normal_preconditioner_and_y0(options: dict[str, Any], tall: bool):
    preconditioner = options.get("preconditioner")
    y0 = options.get("y0")
    inner_options = copy(options)
    del options
    if preconditioner is not None:
        preconditioner = linearise(preconditioner)
        if tall:
            inner_options["preconditioner"] = TaggedLinearOperator(
                preconditioner @ conj(preconditioner.transpose()),
                positive_semidefinite_tag,
            )
        else:
            inner_options["preconditioner"] = TaggedLinearOperator(
                conj(preconditioner.transpose()) @ preconditioner,
                positive_semidefinite_tag,
            )
            if y0 is not None:
                inner_options["y0"] = conj(preconditioner.transpose()).mv(y0)
    return inner_options


class Normal(
    AbstractLinearSolver[
        tuple[_InnerSolverState, eqxi.Static, AbstractLinearOperator, dict[str, Any]]
    ]
):
    """Wrapper for an inner solver of positive (semi)definite systems. The
    wrapped solver handles possibly nonsquare systems $Ax = b$ by applying the
    inner solver to the normal equations

    $A^* A x = A^* b$

    if $m \\ge n$, otherwise

    $A A^* y = b$,

    where $x = A^* y$.

    If the inner solver solves systems with positive definite $A$, the wrapped
    solver solves systems with full rank $A$.

    If the inner solver solves systems with positive semidefinite $A$, the
    wrapped solver solves systems with arbitrary, possibly rank deficient, $A$.

    Note that this squares the condition number, so applying this method to an
    iterative inner solver may result in slow convergence and high sensitivity
    to roundoff error. In this case it may be advantageous to choose an
    appropriate preconditioner or initial solution guess for the problem.

    This wrapper adjusts the following `options` before passing to the inner
    operator (as passed to `lx.linear_solve(..., options=...)`).

    - `preconditioner`: A [`lineax.AbstractLinearOperator`][] to be used as
        preconditioner. Defaults to [`lineax.IdentityLinearOperator`][]. This
        should be an approximation of the (pseudo)inverse of $A$. When passed
        to the inner solver, the preconditioner $M$ is replaced by $M M^*$ and
        $M^* M$ in the first and second versions of the normal equations,
        respectively.

    - `y0`: An initial estimate of the solution of the linear system $Ax = b$.
        Defaults to all zeros. In the second version of the normal equations,
        $y_0$ is replaced with $M^* y_0$, where $M$ is the given outer
        preconditioner.

    !!! Info

        Good choices of inner solvers are the direct [`lineax.Cholesky`][] and
        the iterative [`lineax.CG`][].

    """

    inner_solver: AbstractLinearSolver[_InnerSolverState]

    def init(self, operator, options):
        tall = operator.out_size() >= operator.in_size()
        # Direct solvers materialise the operator; materialise first to avoid
        # computing (op^H @ op).as_matrix() twice via the two branches.
        # For iterative solvers we only linearise to avoid eager materialisation.
        lin_op = (
            materialise(operator)
            if is_direct(self.inner_solver)
            else linearise(operator)
        )
        if tall:
            inner_operator = conj(lin_op.transpose()) @ lin_op
        else:
            inner_operator = lin_op @ conj(lin_op.transpose())
        inner_operator = TaggedLinearOperator(inner_operator, positive_semidefinite_tag)
        inner_options = normal_preconditioner_and_y0(options, tall)
        inner_state = self.inner_solver.init(inner_operator, inner_options)
        operator_conj_transpose = conj(lin_op.transpose())
        return inner_state, eqxi.Static(tall), operator_conj_transpose, inner_options

    def compute(
        self,
        state: tuple[
            _InnerSolverState, eqxi.Static, AbstractLinearOperator, dict[str, Any]
        ],
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        inner_state, tall, operator_conj_transpose, inner_options = state
        tall = tall.value
        del state, options
        if tall:
            vector = operator_conj_transpose.mv(vector)
        solution, result, extra_stats = self.inner_solver.compute(
            inner_state, vector, inner_options
        )
        if not tall:
            solution = operator_conj_transpose.mv(solution)
        return solution, result, extra_stats

    def transpose(
        self,
        state: tuple[
            _InnerSolverState, eqxi.Static, AbstractLinearOperator, dict[str, Any]
        ],
        options: dict[str, Any],
    ):
        inner_state, tall, operator_conj_transpose, inner_options = state
        inner_state_conj, inner_options = self.inner_solver.conj(
            inner_state, inner_options
        )
        state_transpose = (
            inner_state_conj,
            eqxi.Static(not tall.value),
            operator_conj_transpose.transpose(),
            inner_options,
        )
        return state_transpose, options

    def conj(
        self,
        state: tuple[
            _InnerSolverState, eqxi.Static, AbstractLinearOperator, dict[str, Any]
        ],
        options: dict[str, Any],
    ):
        inner_state, tall, operator_conj_transpose, inner_options = state
        inner_state_conj, inner_options = self.inner_solver.conj(
            inner_state, inner_options
        )
        state_conj = (
            inner_state_conj,
            tall,
            conj(operator_conj_transpose),
            inner_options,
        )
        return state_conj, options

    def assume_full_rank(self):
        return self.inner_solver.assume_full_rank()

    def slogdet(
        self,
        state: tuple[
            _InnerSolverState, eqxi.Static, AbstractLinearOperator, dict[str, Any]
        ],
        options: dict[str, Any],
    ) -> tuple[Array, Array]:
        if not is_direct(self.inner_solver):
            raise TypeError(
                f"`Normal.slogdet` requires a direct inner solver, "
                f"got {type(self.inner_solver).__name__}. "
                f"Use a direct solver such as `lx.Cholesky()`."
            )
        inner_state, _, _, inner_options = state
        # log|det(A^H A)| = 2 * log|det(A)| for tall A (m >= n)
        # log|det(A A^H)| = 2 * log|det(A)| for wide A (m < n)
        # so log|det(A)| = 0.5 * log|det(normal_operator)|
        # The gram matrix construction destroys sign information, so sign is nan.
        _, inner_lad = self.inner_solver.slogdet(inner_state, inner_options)  # pyright: ignore[reportAttributeAccessIssue]
        lad = 0.5 * inner_lad
        sign = jnp.full((), jnp.nan, dtype=lad.dtype)
        return sign, lad


Normal.__init__.__doc__ = """**Arguments:**

- `inner_solver`: The solver to wrap. It should support solving positive
  definite systems or positive semidefinite systems
"""


def is_direct(solver: AbstractLinearSolver) -> bool:
    """Returns `True` if `solver` is a direct solver that supports `slogdet`.

    Direct solvers (e.g. [`lineax.LU`][], [`lineax.Cholesky`][],
    [`lineax.SVD`][], [`lineax.Triangular`][], [`lineax.Diagonal`][],
    [`lineax.Tridiagonal`][]) materialise the operator and can compute
    determinants from their factored state.

    [`lineax.Normal`][] with a direct inner solver also satisfies this check.
    """
    if isinstance(solver, Normal):
        return is_direct(solver.inner_solver)
    return isinstance(solver, AbstractDirectLinearSolver)
