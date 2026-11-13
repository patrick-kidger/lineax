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

import abc
from typing import Any, Generic, TypeVar

import equinox as eqx
from jaxtyping import Array, PyTree

from .._operator import AbstractLinearOperator
from .._solution import RESULTS


_SolverState = TypeVar("_SolverState")


class AbstractLinearSolver(eqx.Module, Generic[_SolverState]):
    """Abstract base class for all linear solvers."""

    @abc.abstractmethod
    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _SolverState:
        """Do any initial computation on just the `operator`.

        For example, an LU solver would compute the LU decomposition of the operator
        (and this does not require knowing the vector yet).

        It is common to need to solve the linear system `Ax=b` multiple times in
        succession, with the same operator `A` and multiple vectors `b`. This method
        improves efficiency by making it possible to re-use the computation performed
        on just the operator.

        !!! Example

            ```python
            operator = lx.MatrixLinearOperator(...)
            vector1 = ...
            vector2 = ...
            solver = lx.LU()
            state = solver.init(operator, options={})
            solution1 = lx.linear_solve(operator, vector1, solver, state=state)
            solution2 = lx.linear_solve(operator, vector2, solver, state=state)
            ```

        **Arguments:**

        - `operator`: a linear operator.
        - `options`: a dictionary of any extra options that the solver may wish to
            accept.

        **Returns:**

        A PyTree of arbitrary Python objects.
        """

    @abc.abstractmethod
    def compute(
        self, state: _SolverState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        """Solves a linear system.

        **Arguments:**

        - `state`: as returned from [`lineax.AbstractLinearSolver.init`][].
        - `vector`: the vector to solve against.
        - `options`: a dictionary of any extra options that the solver may wish to
            accept. For example, [`lineax.CG`][] accepts a `preconditioner` option.

        **Returns:**

        A 3-tuple of:

        - The solution to the linear system.
        - An integer indicating the success or failure of the solve. This is an integer
            which may be converted to a human-readable error message via
            `lx.RESULTS[...]`.
        - A dictionary of an extra statistics about the solve, e.g. the number of steps
            taken.
        """

    @abc.abstractmethod
    def transpose(
        self, state: _SolverState, options: dict[str, Any]
    ) -> tuple[_SolverState, dict[str, Any]]:
        """Transposes the result of [`lineax.AbstractLinearSolver.init`][].

        That is, it should be the case that
        ```python
        state_transpose, _ = solver.transpose(solver.init(operator, options), options)
        state_transpose2 = solver.init(operator.T, options)
        ```
        must be identical to each other.

        It is relatively common (in particular when differentiating through a linear
        solve) to need to solve both `Ax = b` and `A^T x = b`. This method makes it
        possible to avoid computing both `solver.init(operator)` and
        `solver.init(operator.T)` if one can be cheaply computed from the other.

        **Arguments:**

        - `state`: as returned from `solver.init`.
        - `options`: any extra options that were passed to `solve.init`.

        **Returns:**

        A 2-tuple of:

        - The state of the transposed operator.
        - The options for the transposed operator.
        """

    @abc.abstractmethod
    def conj(
        self, state: _SolverState, options: dict[str, Any]
    ) -> tuple[_SolverState, dict[str, Any]]:
        """Conjugate the result of [`lineax.AbstractLinearSolver.init`][].

        That is, it should be the case that
        ```python
        state_conj, _ = solver.conj(solver.init(operator, options), options)
        state_conj2 = solver.init(conj(operator), options)
        ```
        must be identical to each other.

        **Arguments:**

        - `state`: as returned from `solver.init`.
        - `options`: any extra options that were passed to `solve.init`.

        **Returns:**

        A 2-tuple of:

        - The state of the conjugated operator.
        - The options for the conjugated operator.
        """

    @abc.abstractmethod
    def assume_full_rank(self) -> bool:
        """Does this solver assume that all operators are full rank?

        When `False`, a more expensive backward pass is needed to account for
        the extra generality. In a custom linear solver, it is always safe to
        return False.

        **Arguments:**

        Nothing.

        **Returns:**

        Either `True` or `False`.
        """


class AbstractDirectLinearSolver(AbstractLinearSolver[_SolverState]):
    """Abstract base class for direct linear solvers.

    Direct solvers materialise the operator (as a matrix or factorisation) and
    can therefore expose the (log absolute) determinant from their factored state
    without any additional linear solves.
    """

    @abc.abstractmethod
    def slogdet(
        self, state: _SolverState, options: dict[str, Any]
    ) -> tuple[Array, Array]:
        """Compute `(sign, log|det(operator)|)` from the factored state.

        Follows the same convention as `numpy.linalg.slogdet`.

        **Arguments:**

        - `state`: as returned from [`lineax.AbstractLinearSolver.init`][].
        - `options`: any extra options that were passed to `solver.init`.

        **Returns:**

        A 2-tuple of `(sign, logabsdet)`.  `sign` is `nan` when it cannot be
        recovered from the factorisation (e.g. gram-matrix solvers such as
        [`lineax.Normal`][], or [`lineax.SVD`][]).
        """
