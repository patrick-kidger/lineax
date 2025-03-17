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

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._operator import AbstractLinearOperator, is_tridiagonal, tridiagonal
from .._solution import RESULTS
from .._solve import AbstractLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_TridiagonalState: TypeAlias = tuple[tuple[Array, Array, Array], PackedStructures]


class Tridiagonal(AbstractLinearSolver[_TridiagonalState], strict=True):
    """Tridiagonal solver for linear systems, using the Thomas algorithm."""

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

        #
        # notation from: https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        # _p indicates prime, ie. `d_p` is the variable name for d' on wikipedia
        #

        size = len(diagonal)

        def thomas_scan(prev_cd_carry, bd):
            c_p, d_p, step = prev_cd_carry
            # the index of `a` doesn't matter at step 0 as
            # we won't use it at all. Same for `c` at final step
            a_index = jnp.where(step > 0, step - 1, 0)
            c_index = jnp.where(step < size, step, 0)

            b, d = bd
            a, c = lower_diagonal[a_index], upper_diagonal[c_index]
            denom = b - a * c_p
            new_d_p = (d - a * d_p) / denom
            new_c_p = c / denom
            return (new_c_p, new_d_p, step + 1), (new_c_p, new_d_p)

        def backsub(prev_x_carry, cd_p):
            x_prev, step = prev_x_carry
            c_p, d_p = cd_p
            x_new = d_p - c_p * x_prev
            return (x_new, step + 1), x_new

        # not a dummy init! 0 is the proper value for all of these
        init_thomas = (0, 0, 0)
        init_backsub = (0, 0)
        diag_vec = (diagonal, vector)
        _, cd_p = lax.scan(thomas_scan, init_thomas, diag_vec, unroll=32)
        _, solution = lax.scan(backsub, init_backsub, cd_p, reverse=True, unroll=32)

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

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False


Tridiagonal.__init__.__doc__ = """**Arguments:**

Nothing.
"""
