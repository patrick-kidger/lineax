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

from jaxtyping import Array, PyTree

from .._operator import (
    conj,
    TaggedLinearOperator,
)
from .._solution import RESULTS
from .._solve import AbstractLinearOperator, AbstractLinearSolver
from .._tags import positive_semidefinite_tag


_InnerSolverState = TypeVar("_InnerSolverState")


def normal_preconditioner_and_y0(options: dict[str, Any], tall: bool):
    preconditioner = options.get("preconditioner")
    y0 = options.get("y0")
    inner_options = copy(options)
    del options
    if preconditioner is not None:
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
    if preconditioner is not None and y0 is not None and not tall:
        inner_options["y0"] = conj(preconditioner.transpose()).mv(y0)
    return inner_options


class Normal(
    # AbstractLinearSolver[_NormalState]
    AbstractLinearSolver[
        tuple[_InnerSolverState, bool, AbstractLinearOperator, dict[str, Any]]
    ]
):
    """Wrapper for an inner solver of positive (semi)definite systems. The
    wrapped solver handles possible nonsquare systems $Ax = b$ by applying the
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
        $y0$ is replaced with $M^* y0$, where $M$ is the given outer
        preconditioner.

    !!! Info

        Good choices of inner solvers are the direct [`lineax.Cholesky`][] and
        the iterative [`lineax.CG`][].

    """

    inner_solver: AbstractLinearSolver[_InnerSolverState]

    def init(self, operator, options):
        tall = operator.out_size() >= operator.in_size()
        if tall:
            inner_operator = conj(operator.transpose()) @ operator
        else:
            inner_operator = operator @ conj(operator.transpose())
        inner_operator = TaggedLinearOperator(inner_operator, positive_semidefinite_tag)
        inner_options = normal_preconditioner_and_y0(options, tall)
        inner_state = self.inner_solver.init(inner_operator, inner_options)
        operator_conj_transpose = conj(operator.transpose())
        return inner_state, tall, operator_conj_transpose, inner_options

    def compute(
        self,
        state: tuple[_InnerSolverState, bool, AbstractLinearOperator, dict[str, Any]],
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        inner_state, tall, operator_conj_transpose, inner_options = state
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
        state: tuple[_InnerSolverState, bool, AbstractLinearOperator, dict[str, Any]],
        options: dict[str, Any],
    ):
        inner_state, tall, operator_conj_transpose, inner_options = state
        inner_state_conj, inner_options = self.inner_solver.conj(
            inner_state, inner_options
        )
        state_transpose = (
            inner_state_conj,
            not tall,
            operator_conj_transpose.transpose(),
            inner_options,
        )
        return state_transpose, options

    def conj(
        self,
        state: tuple[_InnerSolverState, bool, AbstractLinearOperator, dict[str, Any]],
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

    def allow_dependent_columns(self, operator):
        rows = operator.out_size()
        columns = operator.in_size()
        if rows > columns:  # strictly tall
            inner_operator = conj(operator.transpose()) @ operator
            inner_solver_assumes_constant_row_or_col_space = (
                not self.inner_solver.allow_dependent_columns(inner_operator)
                or not self.inner_solver.allow_dependent_rows(inner_operator)
            )
            # inner_operator has constant row space if and only if it has
            # constant column space if and only if operator has constant column
            # space (these spaces are all equal). Thus an assumption by the
            # inner solver that either the row or column space of
            # inner_operator is constant means it is safe for the outer solver
            # to assume here operator has constant column space.
            return not inner_solver_assumes_constant_row_or_col_space
        elif rows < columns:  # strictly wide
            return True
        else:  # square
            # Handle the square case separately, as we may be applying the
            # inner solver to either version of the normal equations according
            # to if the outer solver state has been transposed. To be safe, we
            # take the more permissive set of assumptions between these
            # possibilities.
            inner_operator1 = conj(operator.transpose()) @ operator
            inner_solver_assumes_constant_row_or_col_space1 = (
                not self.inner_solver.allow_dependent_columns(inner_operator1)
                or not self.inner_solver.allow_dependent_rows(inner_operator1)
            )
            inner_operator2 = operator @ conj(operator.transpose())
            inner_solver_assumes_constant_row_or_col_space2 = (
                not self.inner_solver.allow_dependent_columns(inner_operator2)
                or not self.inner_solver.allow_dependent_rows(inner_operator2)
            )
            inner_solver_always_assumes_constant_row_or_col_space = (
                inner_solver_assumes_constant_row_or_col_space1
                and inner_solver_assumes_constant_row_or_col_space2
            )
            return not inner_solver_always_assumes_constant_row_or_col_space

    def allow_dependent_rows(self, operator):
        rows = operator.out_size()
        columns = operator.in_size()
        if rows > columns:  # strictly tall
            return True
        elif rows < columns:  # strictly wide
            inner_operator = operator @ conj(operator.transpose())
            inner_solver_assumes_constant_row_or_col_space = (
                not self.inner_solver.allow_dependent_columns(inner_operator)
                or not self.inner_solver.allow_dependent_rows(inner_operator)
            )
            # inner_operator has constant row space if and only if it has
            # constant column space if and only if operator has constant row
            # space (these spaces are all equal). Thus an assumption by the
            # inner solver that either the row or column space of
            # inner_operator is constant means it is safe for the outer solver
            # to assume here operator has constant row space.
            return not inner_solver_assumes_constant_row_or_col_space
        else:  # square
            # See the remark in allow_dependent_columns for why we handle the
            # square case separately.
            inner_operator1 = conj(operator.transpose()) @ operator
            inner_solver_assumes_constant_row_or_col_space1 = (
                not self.inner_solver.allow_dependent_columns(inner_operator1)
                or not self.inner_solver.allow_dependent_rows(inner_operator1)
            )
            inner_operator2 = operator @ conj(operator.transpose())
            inner_solver_assumes_constant_row_or_col_space2 = (
                not self.inner_solver.allow_dependent_columns(inner_operator2)
                or not self.inner_solver.allow_dependent_rows(inner_operator2)
            )
            inner_solver_always_assumes_constant_row_or_col_space = (
                inner_solver_assumes_constant_row_or_col_space1
                and inner_solver_assumes_constant_row_or_col_space2
            )
            return not inner_solver_always_assumes_constant_row_or_col_space


Normal.__init__.__doc__ = """**Arguments:**

- `inner_solver`: The solver to wrap. It should support solving positive
  definite systems or positive semidefinite systems
"""
