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
from typing_extensions import TypeAlias

import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, PyTree

from .._solution import RESULTS
from .._solve import AbstractLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_QRState: TypeAlias = tuple[tuple[Array, Array], bool, PackedStructures]


class QR(AbstractLinearSolver, strict=True):
    """QR solver for linear systems.

    This solver can handle non-square operators.

    This is usually the preferred solver when dealing with non-square operators.

    !!! info

        Note that whilst this does handle non-square operators, it still can only
        handle full-rank operators.

        This is because JAX does not currently support a rank-revealing/pivoted QR
        decomposition, see [issue #12897](https://github.com/google/jax/issues/12897).

        For such use cases, switch to [`lineax.SVD`][] instead.
    """

    def init(self, operator, options):
        del options
        matrix = operator.as_matrix()
        m, n = matrix.shape
        transpose = n > m
        if transpose:
            matrix = matrix.T
        qr = jnp.linalg.qr(matrix, mode="reduced")  # pyright: ignore
        packed_structures = pack_structures(operator)
        return qr, transpose, packed_structures

    def compute(
        self,
        state: _QRState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        (q, r), transpose, packed_structures = state
        del state, options
        vector = ravel_vector(vector, packed_structures)
        if transpose:
            # Minimal norm solution if underdetermined.
            solution = q.conj() @ jsp.linalg.solve_triangular(
                r, vector, trans="T", unit_diagonal=False
            )
        else:
            # Least squares solution if overdetermined.
            solution = jsp.linalg.solve_triangular(
                r, q.T.conj() @ vector, trans="N", unit_diagonal=False
            )
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _QRState, options: dict[str, Any]):
        (q, r), transpose, structures = state
        transposed_packed_structures = transpose_packed_structures(structures)
        transpose_state = (q, r), not transpose, transposed_packed_structures
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _QRState, options: dict[str, Any]):
        (q, r), transpose, structures = state
        conj_state = (
            (q.conj(), r.conj()),
            transpose,
            structures,
        )
        conj_options = {}
        return conj_state, conj_options

    def allow_dependent_columns(self, operator):
        rows = operator.out_size()
        columns = operator.in_size()
        # We're able to pull an efficiency trick here.
        #
        # As we don't use a rank-revealing implementation, then we always require that
        # the operator have full rank.
        #
        # So if we have columns <= rows, then we know that all our columns are linearly
        # independent. We can return `False` and get a computationally cheaper jvp rule.
        return columns > rows

    def allow_dependent_rows(self, operator):
        rows = operator.out_size()
        columns = operator.in_size()
        return rows > columns


QR.__init__.__doc__ = """**Arguments:**

Nothing.
"""
