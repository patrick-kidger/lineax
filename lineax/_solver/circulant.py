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

import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._operator import AbstractLinearOperator, circulant_column, is_circulant
from .._solution import RESULTS
from .base import AbstractLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_CirculantState: TypeAlias = tuple[tuple[Array, bool, int], PackedStructures]


class Circulant(AbstractLinearSolver[_CirculantState]):
    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _CirculantState:
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`Circulant` may only be used for linear solves with square matrices"
            )
        if not is_circulant(operator):
            raise ValueError(
                "`Circulant` may only be used for linear solves with circulant matrices"
            )
        column = circulant_column(operator)
        is_complex = jnp.iscomplexobj(column)
        if is_complex:
            fft_column = jnp.fft.fft(column)
        else:
            fft_column = jnp.fft.rfft(column)
        return (fft_column, is_complex, len(column)), pack_structures(operator)

    def compute(
        self,
        state: _CirculantState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        (fft_column, is_complex, n), packed_structures = state
        del state, options
        vector = ravel_vector(vector, packed_structures)
        if is_complex:
            fft_fn = jnp.fft.fft
            ifft_fn = jnp.fft.ifft
        else:
            fft_fn = jnp.fft.rfft
            ifft_fn = lambda x: jnp.fft.irfft(x, n=n)
        vector_fft = fft_fn(vector)
        solution = ifft_fn(vector_fft / fft_column)
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _CirculantState, options: dict[str, Any]):
        del options
        (column_fft, is_complex, n), packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        # Transposing reverses the column, `c[(-k) % n]`, and reversal negates the
        # frequency index: `λ_k -> λ_{-k}`. `rfft` keeps only half the spectrum, on
        # which that reindexing acts as conjugation.
        if is_complex:
            transpose_freq = jnp.concatenate([column_fft[:1], jnp.flip(column_fft[1:])])
        else:
            transpose_freq = jnp.conjugate(column_fft)
        transpose_state = (
            (transpose_freq, is_complex, n),
            transposed_packed_structures,
        )
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _CirculantState, options: dict[str, Any]):
        del options
        (column_fft, is_complex, n), packed_structures = state
        # Conjugating the column conjugates the eigenvalues and, as in `transpose`,
        # negates the frequency index. A real column is its own conjugate.
        if is_complex:
            conj_freq = jnp.conjugate(column_fft)
            conj_freq = jnp.concatenate([conj_freq[:1], jnp.flip(conj_freq[1:])])
            conj_state = ((conj_freq, is_complex, n), packed_structures)
        else:
            conj_state = state
        conj_options = {}
        return conj_state, conj_options

    def assume_full_rank(self):
        return True
