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
import math
from typing import Any, TypeAlias

import equinox.internal as eqxi
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from .._misc import cyclic_reverse, resolve_rcond
from .._operator import AbstractLinearOperator, first_column, is_circulant
from .._solution import RESULTS
from .base import AbstractDirectLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_CirculantState: TypeAlias = tuple[tuple[Array, eqxi.Static[bool]], PackedStructures]


class Circulant(AbstractDirectLinearSolver[_CirculantState]):
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
            ifft_fn = ft.partial(jnp.fft.irfft, n=len(vector))
        vector_fft = fft_fn(vector)

        if not self.well_posed:
            size = len(vector)
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
            transpose_freq = cyclic_reverse(eigenvalues)
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
            conj_eig = cyclic_reverse(jnp.conjugate(eigenvalues))
            conj_state = ((conj_eig, is_complex), packed_structures)
        else:
            conj_state = state
        conj_options = {}
        return conj_state, conj_options

    def slogdet(
        self, state: _CirculantState, options: dict[str, Any]
    ) -> tuple[Array, Array]:
        del options
        (eigenvalues, is_complex), packed_structures = state
        # A circulant matrix is diagonalised by the DFT, so its determinant is the
        # product of its eigenvalues (the FFT of the first column).
        leaves, treedef = packed_structures.value
        out_structure, _ = jtu.tree_unflatten(treedef, leaves)
        n = sum(math.prod(x.shape) for x in jtu.tree_leaves(out_structure))
        abs_eig = jnp.abs(eigenvalues)
        if self.well_posed:
            mask = jnp.ones(abs_eig.shape, dtype=bool)
        else:
            # Match `compute`: drop (near-)zero eigenvalues to return the
            # pseudodeterminant. Magnitudes are conjugate-symmetric, so each pair is
            # masked together and the completion below stays consistent.
            rcond = resolve_rcond(self.rcond, n, n, eigenvalues.dtype)
            threshold = jnp.array(rcond, dtype=abs_eig.dtype) * jnp.max(abs_eig)
            mask = abs_eig > threshold
        log_abs = jnp.where(mask, jnp.log(jnp.where(mask, abs_eig, 1.0)), 0.0)
        if is_complex.value:
            # `fft` stores all `n` eigenvalues, each with multiplicity one.
            safe_abs = jnp.where(mask, abs_eig, 1.0).astype(eigenvalues.dtype)
            unit = jnp.where(mask, eigenvalues / safe_abs, 1.0)
            sign = jnp.prod(unit)
            lad = jnp.sum(log_abs)
        else:
            # `rfft` stores the non-redundant half of a conjugate-symmetric spectrum.
            # The DC term (index 0) and, when `n` is even, the Nyquist term (index
            # `n // 2`) are real and unpaired; every other stored eigenvalue pairs with
            # its conjugate, contributing `|lambda|**2` (real, positive) to the
            # determinant. So paired terms count double in `lad` and never affect sign.
            m = eigenvalues.shape[0]
            mult = jnp.full((m,), 2.0).at[0].set(1.0)
            if n % 2 == 0:
                mult = mult.at[m - 1].set(1.0)
            lad = jnp.sum(mult * log_abs)
            real_sign = jnp.sign(eigenvalues.real)
            sign = jnp.where(mask[0], real_sign[0], 1.0)
            if n % 2 == 0:
                sign = sign * jnp.where(mask[m - 1], real_sign[m - 1], 1.0)
        return sign, lad

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
