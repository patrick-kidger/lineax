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

# Number of raw recurrence steps between renormalisations in `Tridiagonal.slogdet`.
# Bigger blocks amortise the renormalisation over more steps; smaller blocks let the
# minors decay further before a block underflows. A block underflows once the minors
# decay past float range within it, i.e. roughly when R * K > 308 for an operator
# whose entries span 10**-R, so this directly sets the tolerated grading:
#
#     K = 16 -> R ~ 19     K = 8 -> R ~ 38     K = 4 -> R ~ 77     K = 2 -> R ~ 154
#
# 4 is the smallest value that still beats LAPACK `gttrf` on CPU at every size
# (1.1-1.5x unbatched, 2.2-3.0x batched). Going to 2 buys more grading tolerance but
# only reaches parity with `gttrf` unbatched, and is *less* safe rather than more: its
# per-block decay lands inside the denormal band, so instead of underflowing cleanly
# to -inf it returns a silently wrong answer (~5e-4 relative) for R in 120..150.
_SLOGDET_BLOCK = 4


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

    def slogdet(
        self, state: _TridiagonalState, options: dict[str, Any]
    ) -> tuple[Array, Array]:
        del options
        (diagonal, lower_diagonal, upper_diagonal), _ = state
        n = diagonal.shape[0]
        dtype = diagonal.dtype
        real_dtype = jnp.finfo(dtype).dtype
        block = _SLOGDET_BLOCK

        # Three-term recurrence on the leading principal minors
        # D_i = det(A[:i+1, :i+1]):
        #
        #     D_i = d_i D_{i-1} - l_{i-1} u_{i-1} D_{i-2},    D_{-1} = 1, D_0 = d_0
        #
        # with det(A) = D_{n-1}. This is division-free, unlike the LU-pivot recurrence
        # p_i = d_i - l_{i-1} u_{i-1} / p_{i-1}, which divides by zero as soon as any
        # *leading* minor is singular -- even when A itself is perfectly invertible
        # (e.g. [[0, 1], [1, 0]]).
        coupling = lower_diagonal * upper_diagonal

        # The minors grow or decay by roughly one entry-magnitude per step, so for an
        # operator whose entries are far from unit scale they leave float range fast.
        # Normalise the whole operator first, using det(A) = sigma**n det(A / sigma):
        #
        #     logabsdet(A) = n log(sigma) + logabsdet(A / sigma)
        #
        # `coupling` is degree 2 in the entries, hence the sigma**2. Choosing sigma as
        # a power of two makes the rescale exact, and choosing it to bound both scaled
        # arrays by 1 bounds the growth to |D_i| <= |D_{i-1}| + |D_{i-2}|, i.e. at most
        # 2**block per block -- so the recurrence can no longer overflow at all. This
        # is a single reduction, outside the serial loop.
        magnitude = jnp.maximum(
            jnp.max(jnp.abs(diagonal)),
            jnp.sqrt(jnp.max(jnp.abs(coupling), initial=jnp.zeros((), real_dtype))),
        )
        _, exponent = jnp.frexp(magnitude)
        sigma_inv = jnp.ldexp(jnp.ones((), real_dtype), -exponent)
        diagonal = diagonal * sigma_inv.astype(dtype)
        coupling = coupling * (sigma_inv * sigma_inv).astype(dtype)

        def unit_step(carry, args):
            prev, curr = carry
            d_i, coupling_i = args
            return (curr, d_i * curr - coupling_i * prev), None

        def block_step(carry, args):
            prev, curr = carry
            (prev, curr), _ = lax.scan(unit_step, (prev, curr), args, unroll=block)
            scale = jnp.maximum(jnp.abs(prev), jnp.abs(curr))
            # A singular A drives both minors to exactly zero. Dividing by `scale`
            # would turn that into `nan`; holding the pair at zero instead lets the
            # `log(scale)` sum absorb the `-inf` and yields `(sign, lad) = (0, -inf)`,
            # matching `jnp.linalg.slogdet` on a singular input.
            nonzero = jnp.where(scale == 0, 1.0, scale)
            # Reciprocal-and-multiply rather than two divides: measurably faster, and
            # `nonzero >= ` the larger minor keeps the reciprocal in range.
            inv_scale = (1.0 / nonzero).astype(dtype)
            # The scales are emitted rather than accumulated so that their logs happen
            # in one vectorised pass instead of once per step of the serial loop.
            return (prev * inv_scale, curr * inv_scale), scale

        # Pad the tail to a whole number of blocks with identity steps
        # (d=1, coupling=0), which map D_i -> D_{i-1} and so leave the value alone.
        pad = (-(n - 1)) % block
        diagonal_rest = jnp.concatenate([diagonal[1:], jnp.ones((pad,), dtype=dtype)])
        coupling_rest = jnp.concatenate([coupling, jnp.zeros((pad,), dtype=dtype)])
        num_blocks = (n - 1 + pad) // block

        # Normalise the initial pair (D_{-1}, D_0) = (1, d_0) the same way.
        scale0 = jnp.maximum(jnp.abs(diagonal[0]), 1.0)
        init = (
            jnp.ones((), dtype=dtype) / scale0.astype(dtype),
            diagonal[0] / scale0.astype(dtype),
        )
        (_, det), scales = lax.scan(
            block_step,
            init,
            (
                diagonal_rest.reshape(num_blocks, block),
                coupling_rest.reshape(num_blocks, block),
            ),
        )

        sign = jnp.sign(det)
        lad = (
            n * exponent.astype(real_dtype) * jnp.log(jnp.array(2.0, real_dtype))
            + jnp.log(scale0)
            + jnp.sum(jnp.log(scales))
            + jnp.log(jnp.abs(det))
        )
        return sign, lad

    def assume_full_rank(self):
        return True


Tridiagonal.__init__.__doc__ = """**Arguments:**

Nothing.
"""
