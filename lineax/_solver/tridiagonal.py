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

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._operator import AbstractLinearOperator, is_tridiagonal, tridiagonal
from .._solution import RESULTS
from .base import AbstractDirectLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_TridiagonalState: TypeAlias = tuple[tuple[Array, Array, Array], PackedStructures]


class Tridiagonal(AbstractDirectLinearSolver[_TridiagonalState]):
    """Tridiagonal solver for linear systems, uses the LAPACK/cusparse implementation
    of Gaussian elimination with partial pivotting (which increases stability).
    ."""

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`Tridiagonal` may only be used for linear solves with square matrices"
            )
        if not is_tridiagonal(operator):
            raise ValueError(
                "`Tridiagonal` may only be used for linear solves with tridiagonal "
                "matrices"
            )
        return tridiagonal(operator), pack_structures(operator)

    def compute(
        self,
        state: _TridiagonalState,
        vector,
        options,
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        (diagonal, lower_diagonal, upper_diagonal), packed_structures = state
        del state, options
        vector = ravel_vector(vector, packed_structures)

        solution = lax.linalg.tridiagonal_solve(
            jnp.append(0.0, lower_diagonal),
            diagonal,
            jnp.append(upper_diagonal, 0.0),
            vector[:, None],
        ).flatten()

        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _TridiagonalState, options: dict[str, Any]):
        (diagonal, lower_diagonal, upper_diagonal), packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_diagonals = (diagonal, upper_diagonal, lower_diagonal)
        transpose_state = (transpose_diagonals, transposed_packed_structures)
        return transpose_state, options

    def conj(self, state: _TridiagonalState, options: dict[str, Any]):
        (diagonal, lower_diagonal, upper_diagonal), packed_structures = state
        conj_diagonals = (diagonal.conj(), lower_diagonal.conj(), upper_diagonal.conj())
        conj_state = (conj_diagonals, packed_structures)
        return conj_state, options

    def slogdet(self, state: _TridiagonalState, options: dict[str, Any]) -> tuple[Array, Array]:
        del options
        (diagonal, lower_diagonal, upper_diagonal), _ = state
        n = diagonal.shape[0]

        def step(pivot_prev, i):
            pivot = diagonal[i] - lower_diagonal[i - 1] * upper_diagonal[i - 1] / pivot_prev
            return pivot, pivot

        pivot0 = diagonal[0]
        _, pivots_rest = lax.scan(step, pivot0, jnp.arange(1, n))
        pivots = jnp.concatenate([pivot0[None], pivots_rest])
        sign = jnp.prod(jnp.sign(pivots))
        lad = jnp.sum(jnp.log(jnp.abs(pivots)))
        return sign, lad

    def assume_full_rank(self):
        return True


Tridiagonal.__init__.__doc__ = """**Arguments:**

Nothing.
"""
