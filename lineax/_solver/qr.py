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

import equinox.internal as eqxi
import jax.lax.linalg as jll
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


_QRState: TypeAlias = tuple[tuple[Array, Array], eqxi.Static, PackedStructures]


class QR(AbstractLinearSolver):
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
        h, taus = jnp.linalg.qr(matrix, mode="raw")  # pyright: ignore
        a = h.mT
        packed_structures = pack_structures(operator)
        return (a, taus), eqxi.Static(transpose), packed_structures

    def compute(
        self,
        state: _QRState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        (a, taus), transpose, packed_structures = state
        transpose = transpose.value
        del state, options
        vector = ravel_vector(vector, packed_structures)
        n_full, n_min = a.shape
        r = a[:n_min]
        if transpose:
            # Minimal norm solution if underdetermined: x = Q.conj() @ R^{-T} @ b.
            # Use Q.conj() @ z = (z^T @ Q^H)^T to avoid explicit `conj` calls,
            # and pad `y` along the row axis to absorb the discarded columns of Q.
            y = jsp.linalg.solve_triangular(r, vector, trans="T", unit_diagonal=False)
            zeros = jnp.zeros((1, n_full - n_min), dtype=y.dtype)
            y_pad = jnp.concatenate([y[None, :], zeros], axis=1)
            solution = jll.ormqr(a, taus, y_pad, left=False, transpose=True)[0]
        else:
            # Least squares solution if overdetermined.
            qHv = jll.ormqr(a, taus, vector[:, None], transpose=True)[:n_min, 0]
            solution = jsp.linalg.solve_triangular(
                r, qHv, trans="N", unit_diagonal=False
            )
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _QRState, options: dict[str, Any]):
        (a, taus), transpose, structures = state
        transposed_packed_structures = transpose_packed_structures(structures)
        transpose_state = (
            (a, taus),
            eqxi.Static(not transpose.value),
            transposed_packed_structures,
        )
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _QRState, options: dict[str, Any]):
        (a, taus), transpose, structures = state
        conj_state = (
            (a.conj(), taus.conj()),
            transpose,
            structures,
        )
        conj_options = {}
        return conj_state, conj_options

    def assume_full_rank(self):
        return True


QR.__init__.__doc__ = """**Arguments:**

Nothing.
"""
