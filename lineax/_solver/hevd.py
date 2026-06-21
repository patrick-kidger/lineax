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
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._misc import resolve_rcond
from .._operator import (
    AbstractLinearOperator,
    is_hermitian,
    is_negative_semidefinite,
    is_positive_semidefinite,
    max_rank,
)
from .._solution import RESULTS
from .base import AbstractDirectLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    unravel_solution,
)


_HEVDState: TypeAlias = tuple[tuple[Array, Array], PackedStructures]


class HEVD(AbstractDirectLinearSolver[_HEVDState]):
    """Eigenvalue decomposition solver for Hermitian linear systems.

    The operator must be square and Hermitian (self-adjoint), but need not be
    definite or even nonsingular: in the singular case this solver returns
    the pseudoinverse solution. This is the optimised Hermitian analogue of
    [`lineax.SVD`][].
    """

    rcond: float | None = None

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        del options
        if not is_hermitian(operator):
            raise ValueError(
                "`HEVD` may only be used for linear solves with Hermitian matrices."
            )
        # `jnp.linalg.eigh` returns eigenvalues in ascending (signed) order
        w, v = jnp.linalg.eigh(operator.as_matrix())
        r = max_rank(operator)
        if r < w.shape[0]:
            # The operator is declared to have rank at most `r`, so all but the `r`
            # largest-magnitude eigenvalues are mathematically zero. Statically drop
            # them to shrink the matmuls (and storage) in `compute`.
            m = v.shape[0]
            rcond = resolve_rcond(self.rcond, m, m, w.dtype) * jnp.max(jnp.abs(w))
            if is_positive_semidefinite(operator):
                # Eigenvalues are >= 0, so in ascending order the `r` largest are a
                # contiguous trailing slice (cheaper than a reordering gather).
                w, v, dropped = w[m - r :], v[:, m - r :], w[: m - r]
            elif is_negative_semidefinite(operator):
                # Eigenvalues are <= 0, so the `r` largest in magnitude are a
                # contiguous leading slice.
                w, v, dropped = w[:r], v[:, :r], w[r:]
            else:
                # Indefinite: the small-magnitude eigenvalues sit in the interior of
                # the spectrum, so no contiguous slice works. Reorder by descending
                # magnitude (an O(n^2) gather, dominated by the O(n^3) eigensolve)
                # and take the leading `r`.
                order = jnp.argsort(jnp.abs(w))[::-1]
                w, v = w[order], v[:, order]
                w, v, dropped = w[:r], v[:, :r], w[r:]
            # `compute` masks out `|w_i| <= rcond * max|w|`, so dropping these is
            # lossless iff they all sit below that floor. Otherwise the `max_rank`
            # claim is false (truncation would change the solution), so error out.
            # Checking the largest discarded magnitude also catches a mistagged
            # PSD/NSD operator whose true large eigenvalues sit on the dropped side.
            w = eqx.error_if(
                w,
                jnp.max(jnp.abs(dropped)) > rcond,
                "lineax.HEVD: the operator was declared (via a `MaxRankTag`, or by "
                f"composition rules) to have rank at most {r}, but it has an "
                "eigenvalue above the rcond threshold beyond that rank. Truncating to "
                "the declared rank would change the solution, so the rank claim "
                "appears to be incorrect. Remove/loosen the rank tag, increase "
                "`rcond` if you intend a low-rank approximation, or set "
                "`EQX_ON_ERROR=off` to skip this check.",
            )
        packed_structures = pack_structures(operator)
        return (w, v), packed_structures

    def compute(
        self,
        state: _HEVDState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        (w, v), packed_structures = state
        vector = ravel_vector(vector, packed_structures)
        m = v.shape[0]
        rcond = resolve_rcond(self.rcond, m, m, w.dtype)
        rcond = jnp.array(rcond, dtype=w.dtype)
        abs_w = jnp.abs(w)
        if w.size > 0:
            # Eigenvalues are signed and not magnitude-sorted; scale by max |w|.
            rcond = rcond * jnp.max(abs_w)
        # Not >=, or this fails with a matrix of all-zeros.
        mask = abs_w > rcond
        rank = mask.sum()
        safe_w = jnp.where(mask, w, 1)
        w_inv = jnp.where(mask, jnp.array(1.0) / safe_w, 0).astype(v.dtype)
        vHb = jnp.matmul(v.conj().T, vector, precision=lax.Precision.HIGHEST)
        solution = jnp.matmul(v, w_inv * vHb, precision=lax.Precision.HIGHEST)
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {"rank": rank}

    def transpose(self, state: _HEVDState, options: dict[str, Any]):
        del options
        (w, v), packed_structures = state
        # `A` is Hermitian, so `A^T = conj(A)` (with `w` real). The structure is
        # square symmetric, so the packed structures are unchanged.
        transpose_state = (w, v.conj()), packed_structures
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _HEVDState, options: dict[str, Any]):
        del options
        (w, v), packed_structures = state
        # `A` is Hermitian, so `conj(A) = conj(V) diag(w) conj(V)^H` (with `w` real).
        conj_state = (w, v.conj()), packed_structures
        conj_options = {}
        return conj_state, conj_options

    def slogdet(self, state: _HEVDState, options: dict[str, Any]) -> tuple[Array, Array]:
        del options
        (w, v), _ = state
        m = v.shape[0]
        rcond = resolve_rcond(self.rcond, m, m, w.dtype)
        abs_w = jnp.abs(w)
        if w.size > 0:
            threshold = jnp.array(rcond, dtype=w.dtype) * jnp.max(abs_w)
        else:
            threshold = jnp.array(rcond, dtype=w.dtype)
        mask = abs_w > threshold
        safe_w = jnp.where(mask, w, 1.0)
        sign = jnp.prod(jnp.sign(safe_w)).real.astype(w.real.dtype)
        lad = jnp.sum(jnp.where(mask, jnp.log(abs_w), 0.0))
        return sign, lad

    def assume_full_rank(self):
        return False


HEVD.__init__.__doc__ = """**Arguments**:

- `rcond`: the cutoff for handling zero entries on the diagonal. Defaults to machine
    precision times `N`, where `(N, N)` is the shape of the operator.
"""
