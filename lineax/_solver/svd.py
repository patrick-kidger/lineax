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

import equinox as eqx
import equinox.internal as eqxi
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, PyTree

from .._misc import resolve_rcond
from .._operator import AbstractLinearOperator, max_rank
from .._solution import RESULTS
from .base import AbstractDirectLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


# The static integer is the number of leading singular components that `compute`
# uses: the operator's declared rank bound. The full decomposition is kept, so that
# the trailing components (e.g. a null space) remain recoverable.
_SVDState: TypeAlias = tuple[
    tuple[Array, Array, Array], eqxi.Static[int], PackedStructures
]


def _retained(state: _SVDState) -> tuple[Array, Array, Array]:
    (u, s, vt), rank_bound, _ = state
    r = rank_bound.value
    return u[:, :r], s[:r], vt[:r, :]


class SVD(AbstractDirectLinearSolver[_SVDState]):
    """SVD solver for linear systems.

    This solver can handle any operator, even nonsquare or singular ones. In these
    cases it will return the pseudoinverse solution to the linear system.

    Equivalent to `scipy.linalg.lstsq`.

    If the operator is Hermitian (e.g. real-symmetric, or positive/negative
    semidefinite) then [`lineax.HEVD`][] is usually faster, as a Hermitian
    eigendecomposition is cheaper than a general SVD.
    """

    rcond: float | None = None

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        del options
        u, s, vt = jsp.linalg.svd(operator.as_matrix(), full_matrices=False)
        # If the operator is known to have rank at most `r`, the trailing
        # singular values are mathematically zero, so `compute` statically truncates to
        # the leading `r` components. The state itself is not truncated, as that would
        # discard the trailing components irrecoverably; if they are never used then
        # JAX eliminates them as dead code.
        r = max_rank(operator)
        if r < s.shape[0]:
            # `compute` masks out `s_i <= rcond * s[0]`, so dropping the tail is
            # lossless iff it all sits below that floor (using the same rcond).
            # Otherwise the `max_rank` claim is false and truncating would change
            # the solution. `s` is descending, so testing the largest discarded
            # value `s[r]` certifies the tail.
            m, n = u.shape[0], vt.shape[1]
            # s.size > 0 since r < size
            rcond = resolve_rcond(self.rcond, n, m, s.dtype) * s[0]
            s = eqx.error_if(
                s,
                s[r] > rcond,
                "lineax.SVD: the operator was declared (via a `MaxRankTag`, or by "
                f"composition rules) to have rank at most {r}, but it has a singular "
                "value above the rcond threshold beyond that rank. Truncating to the "
                "declared rank would change the solution, so the rank claim appears to "
                "be incorrect. Remove/loosen the rank tag, increase `rcond` if you "
                "intend a low-rank approximation, or set `EQX_ON_ERROR=off` to skip "
                "this check.",
            )
        packed_structures = pack_structures(operator)
        return (u, s, vt), eqxi.Static(r), packed_structures

    def compute(
        self,
        state: _SVDState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        _, _, packed_structures = state
        u, s, vt = _retained(state)
        vector = ravel_vector(vector, packed_structures)
        m, _ = u.shape
        _, n = vt.shape
        rcond = resolve_rcond(self.rcond, n, m, s.dtype)
        rcond = jnp.array(rcond, dtype=s.dtype)
        if s.size > 0:
            rcond = rcond * s[0]
        # Not >=, or this fails with a matrix of all-zeros.
        mask = s > rcond
        rank = mask.sum()
        safe_s = jnp.where(mask, s, 1)
        s_inv = jnp.where(mask, jnp.array(1.0) / safe_s, 0).astype(u.dtype)
        uTb = jnp.matmul(u.conj().T, vector, precision=lax.Precision.HIGHEST)
        solution = jnp.matmul(vt.conj().T, s_inv * uTb, precision=lax.Precision.HIGHEST)
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {"rank": rank}

    def transpose(self, state: _SVDState, options: dict[str, Any]):
        del options
        (u, s, vt), rank_bound, packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_state = (vt.T, s, u.T), rank_bound, transposed_packed_structures
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _SVDState, options: dict[str, Any]):
        del options
        (u, s, vt), rank_bound, packed_structures = state
        conj_state = (u.conj(), s, vt.conj()), rank_bound, packed_structures
        conj_options = {}
        return conj_state, conj_options

    def slogdet(self, state: _SVDState, options: dict[str, Any]) -> tuple[Array, Array]:
        del options
        u, s, vt = _retained(state)
        m, _ = u.shape
        _, n = vt.shape
        rcond = resolve_rcond(self.rcond, n, m, s.dtype)
        rcond_arr = jnp.array(rcond, dtype=s.dtype)
        if s.size > 0:
            threshold = rcond_arr * s[0]
        else:
            threshold = rcond_arr
        mask = s > threshold
        # Log-pseudodeterminant: sum of logs of non-zero singular values only.
        # Zero singular values (below threshold) contribute 0 via log(1) = 0.
        # For full-rank operators this equals the true logabsdet.
        safe_s = jnp.where(mask, s, 1.0)
        lad = jnp.sum(jnp.log(safe_s))
        # Sign is not recoverable from SVD alone:
        #   full-rank square: needs sign(det(U)) * sign(det(V^T)), O(n^3) extra work
        #   full-rank rectangular: future work via QR Householder vectors
        #   rank-deficient: needs an eigensolver for pseudodeterminant sign
        sign = jnp.full((), jnp.nan, dtype=s.dtype)
        return sign, lad

    def assume_full_rank(self):
        return False


SVD.__init__.__doc__ = """**Arguments**:

- `rcond`: the cutoff for handling zero entries on the diagonal. Defaults to machine
    precision times `max(N, M)`, where `(N, M)` is the shape of the operator. (I.e.
    `N` is the output size and `M` is the input size.)
"""
