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

import functools as ft
from typing import Any, TypeAlias

import equinox.internal as eqxi
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._misc import resolve_rcond
from .._operator import AbstractLinearOperator, first_column, is_circulant
from .._solution import RESULTS
from .base import AbstractLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_CirculantState: TypeAlias = tuple[tuple[Array, eqxi.Static[bool]], PackedStructures]


class Circulant(AbstractLinearSolver[_CirculantState]):
    """Circulant solver for linear systems.

    Requires that the operator be circulant. Then $Ax = b$ is solved by dividing by the
    eigenvalues of $A$, which are the FFT of its first column.

    This solver can handle singular operators (i.e. zero eigenvalues).
    """

    well_posed: bool = False
    rcond: float | None = None

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
        column = first_column(operator)
        is_complex = jnp.iscomplexobj(column)
        if is_complex:
            eigenvalues = jnp.fft.fft(column)
        else:
            eigenvalues = jnp.fft.rfft(column)
        return (eigenvalues, eqxi.Static(is_complex)), pack_structures(operator)

    def compute(
        self,
        state: _CirculantState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        (eigenvalues, is_complex), packed_structures = state
        del state, options
        vector = ravel_vector(vector, packed_structures)
        if is_complex.value:
            fft_fn = jnp.fft.fft
            ifft_fn = jnp.fft.ifft
        else:
            fft_fn = jnp.fft.rfft
            ifft_fn = ft.partial(jnp.fft.irfft, n=vector.shape[0])
        vector_fft = fft_fn(vector)

        if not self.well_posed:
            (size,) = eigenvalues.shape
            rcond = resolve_rcond(self.rcond, size, size, eigenvalues.dtype)
            abs_eig = jnp.abs(eigenvalues)
            eigenvalues = jnp.where(
                abs_eig > rcond * jnp.max(abs_eig), eigenvalues, jnp.inf
            )  # pyright: ignore

        solution = ifft_fn(vector_fft / eigenvalues)
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _CirculantState, options: dict[str, Any]):
        del options
        (eigenvalues, is_complex), packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        # Transposing reverses the column, `c[(-k) % n]`, and reversal negates the
        # frequency index: `λ_k -> λ_{-k}`. `rfft` keeps only half the spectrum, on
        # which that reindexing acts as conjugation.
        if is_complex.value:
            transpose_freq = jnp.concatenate(
                [eigenvalues[:1], jnp.flip(eigenvalues[1:])]
            )
        else:
            transpose_freq = jnp.conjugate(eigenvalues)
        transpose_state = (
            (transpose_freq, is_complex),
            transposed_packed_structures,
        )
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _CirculantState, options: dict[str, Any]):
        del options
        (eigenvalues, is_complex), packed_structures = state
        # Conjugating the column conjugates the eigenvalues and, as in `transpose`,
        # negates the frequency index. A real column is its own conjugate.
        if is_complex.value:
            conj_eig = jnp.conjugate(eigenvalues)
            conj_eig = jnp.concatenate([conj_eig[:1], jnp.flip(conj_eig[1:])])
            conj_state = ((conj_eig, is_complex), packed_structures)
        else:
            conj_state = state
        conj_options = {}
        return conj_state, conj_options

    def assume_full_rank(self):
        return self.well_posed


Circulant.__init__.__doc__ = """**Arguments**:

- `well_posed`: if `False`, then singular operators are accepted, and the pseudoinverse
    solution is returned. If `True` then passing a singular operator will cause an error
    to be raised instead.
- `rcond`: the cutoff for handling zero eigenvalues. Defaults to machine precision times
    `N`, where `N` is the input (or output) size of the operator. Only used if
    `well_posed=False`
"""
