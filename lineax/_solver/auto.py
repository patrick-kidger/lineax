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

from typing import Any, TypeAlias

from jaxtyping import Array, PyTree

from .._operator import (
    AbstractLinearOperator,
    is_circulant,
    is_diagonal,
    is_hermitian,
    is_lower_triangular,
    is_semidefinite,
    is_tridiagonal,
    is_upper_triangular,
)
from .._solution import RESULTS
from .base import AbstractLinearSolver
from .cholesky import Cholesky
from .circulant import Circulant
from .diagonal import Diagonal
from .hevd import HEVD
from .lu import LU
from .qr import QR
from .svd import SVD
from .triangular import Triangular
from .tridiagonal import Tridiagonal


_AutoLinearSolverState: TypeAlias = tuple[AbstractLinearSolver, Any]


class AutoLinearSolver(AbstractLinearSolver[_AutoLinearSolverState]):
    """Automatically determines a good linear solver based on the structure of the
    operator.

    - If `well_posed=True`:
        - If the operator is diagonal, then use [`lineax.Diagonal`][].
        - If the operator is tridiagonal, then use [`lineax.Tridiagonal`][].
        - If the operator is circulant, then use [`lineax.Circulant`][].
        - If the operator is triangular, then use [`lineax.Triangular`][].
        - If the matrix is positive or negative (semi-)definite, then use
            [`lineax.Cholesky`][].
        - Else use [`lineax.LU`][].

    This is a good choice if you want to be certain that an error is raised for
    ill-posed systems.

    - If `well_posed=False`:
        - If the operator is diagonal, then use [`lineax.Diagonal`][].
        - If the operator is circulant, then use [`lineax.Circulant`][].
        - If the operator is Hermitian, then use [`lineax.HEVD`][].
        - Else use [`lineax.SVD`][].

    This is a good choice if you want to be certain that you can handle ill-posed
    systems.

    - If `well_posed=None`:
        - If the operator is non-square, then use [`lineax.QR`][].
        - If the operator is diagonal, then use [`lineax.Diagonal`][].
        - If the operator is tridiagonal, then use [`lineax.Tridiagonal`][].
        - If the operator is circulant, then use [`lineax.Circulant`][].
        - If the operator is triangular, then use [`lineax.Triangular`][].
        - If the matrix is positive or negative (semi-)definite, then use
            [`lineax.Cholesky`][].
        - Else, use [`lineax.LU`][].

    This is a good choice if your primary concern is computational efficiency. It will
    handle ill-posed systems as long as it is not computationally expensive to do so.
    """

    well_posed: bool | None

    def _select_solver(self, operator: AbstractLinearOperator) -> AbstractLinearSolver:
        if self.well_posed is True:
            if operator.in_size() != operator.out_size():
                raise ValueError(
                    "Cannot use `AutoLinearSolver(well_posed=True)` with a non-square "
                    "operator. If you are trying solve a least-squares problem then "
                    "you should pass `solver=AutoLinearSolver(well_posed=False)`. By "
                    "default `lineax.linear_solve` assumes that the operator is "
                    "square and nonsingular."
                )
            if is_diagonal(operator):
                solver = Diagonal(well_posed=True)
            elif is_tridiagonal(operator):
                solver = Tridiagonal()
            elif is_circulant(operator):
                solver = Circulant(well_posed=True)
            elif is_lower_triangular(operator) or is_upper_triangular(operator):
                solver = Triangular()
            elif is_semidefinite(operator):
                solver = Cholesky()
            else:
                solver = LU()
        elif self.well_posed is False:
            if is_diagonal(operator):
                solver = Diagonal()
            elif is_circulant(operator):
                # An FFT-based solve is cheaper than any dense decomposition, so this
                # takes priority over the Hermitian case below.
                solver = Circulant()
            elif is_hermitian(operator):
                # A Hermitian eigendecomposition is cheaper than a general SVD, and
                # handles ill-posed Hermitian systems via the same pseudoinverse.
                solver = HEVD()
            else:
                # TODO: use rank-revealing QR instead.
                solver = SVD()
        elif self.well_posed is None:
            if operator.in_size() != operator.out_size():
                solver = QR()
            elif is_diagonal(operator):
                solver = Diagonal()
            elif is_tridiagonal(operator):
                solver = Tridiagonal()
            elif is_circulant(operator):
                solver = Circulant()
            elif is_lower_triangular(operator) or is_upper_triangular(operator):
                solver = Triangular()
            elif is_semidefinite(operator):
                solver = Cholesky()
            else:
                solver = LU()
        else:
            raise ValueError(f"Invalid value `well_posed={self.well_posed}`.")
        return solver

    def select_solver(self, operator: AbstractLinearOperator) -> AbstractLinearSolver:
        """Check which solver that [`lineax.AutoLinearSolver`][] will dispatch to.

        **Arguments:**

        - `operator`: a linear operator.

        **Returns:**

        The linear solver that will be used.
        """
        return self._select_solver(operator)

    def init(self, operator, options) -> _AutoLinearSolverState:
        solver = self._select_solver(operator)
        return solver, solver.init(operator, options)

    def compute(
        self,
        state: _AutoLinearSolverState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        solver, state = state
        solution, result, _ = solver.compute(state, vector, options)
        return solution, result, {}

    def transpose(self, state: _AutoLinearSolverState, options: dict[str, Any]):
        solver, state = state
        transpose_state, transpose_options = solver.transpose(state, options)
        transpose_state = (solver, transpose_state)
        return transpose_state, transpose_options

    def conj(self, state: _AutoLinearSolverState, options: dict[str, Any]):
        solver, state = state
        conj_state, conj_options = solver.conj(state, options)
        conj_state = (solver, conj_state)
        return conj_state, conj_options

    def assume_full_rank(self):
        return self.well_posed is not False


AutoLinearSolver.__init__.__doc__ = """**Arguments:**

- `well_posed`: whether to only handle well-posed systems or not, as discussed above.
"""
