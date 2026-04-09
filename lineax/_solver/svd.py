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
import jax.scipy as jsp
from jaxtyping import Array, PyTree

from .._misc import resolve_rcond
from .._operator import AbstractLinearOperator
from .._solution import RESULTS
from .._solve import _gram_inverse_mv, _row_space_projection, AbstractLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_SVDState: TypeAlias = tuple[tuple[Array, Array, Array], PackedStructures]


class SVD(AbstractLinearSolver[_SVDState]):
    """SVD solver for linear systems.

    This solver can handle any operator, even nonsquare or singular ones. In these
    cases it will return the pseudoinverse solution to the linear system.

    Equivalent to `scipy.linalg.lstsq`.
    """

    rcond: float | None = None

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        del options
        svd = jsp.linalg.svd(operator.as_matrix(), full_matrices=False)
        packed_structures = pack_structures(operator)
        return svd, packed_structures

    def _singular_mask(self, s, n, m):
        rcond = resolve_rcond(self.rcond, n, m, s.dtype)
        rcond = jnp.array(rcond, dtype=s.dtype)
        if s.size > 0:
            rcond = rcond * s[0]
        # Not >=, or this fails with a matrix of all-zeros.
        return s > rcond

    def compute(
        self,
        state: _SVDState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        (u, s, vt), packed_structures = state
        vector = ravel_vector(vector, packed_structures)
        m, _ = u.shape
        _, n = vt.shape
        mask = self._singular_mask(s, n, m)
        rank = mask.sum()
        safe_s = jnp.where(mask, s, 1)
        s_inv = jnp.where(mask, jnp.array(1.0) / safe_s, 0).astype(u.dtype)
        uTb = jnp.matmul(u.conj().T, vector, precision=lax.Precision.HIGHEST)
        solution = jnp.matmul(vt.conj().T, s_inv * uTb, precision=lax.Precision.HIGHEST)
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {"rank": rank}

    def transpose(self, state: _SVDState, options: dict[str, Any]):
        del options
        (u, s, vt), packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_state = (vt.T, s, u.T), transposed_packed_structures
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _SVDState, options: dict[str, Any]):
        del options
        (u, s, vt), packed_structures = state
        conj_state = (u.conj(), s, vt.conj()), packed_structures
        conj_options = {}
        return conj_state, conj_options

    def assume_full_rank(self):
        return False


SVD.__init__.__doc__ = """**Arguments**:

- `rcond`: the cutoff for handling zero entries on the diagonal. Defaults to machine
    precision times `max(N, M)`, where `(N, M)` is the shape of the operator. (I.e.
    `N` is the output size and `M` is the input size.)
"""


@_gram_inverse_mv.register(SVD)
def _(solver: SVD, state: _SVDState, vector):
    (u, s, vt), packed_structures = state
    m, n = u.shape[0], vt.shape[1]
    transposed_ps = transpose_packed_structures(packed_structures)
    w = ravel_vector(vector, transposed_ps)
    mask = solver._singular_mask(s, n, m)
    safe_s = jnp.where(mask, s, 1)
    # (A^H A)^{-1} v = VΣ⁻²V^H v.  U^H U = I cancels entirely.
    s_inv_sq = jnp.where(mask, 1.0 / safe_s**2, 0).astype(vt.dtype)
    vt_w = jnp.matmul(vt, w, precision=lax.Precision.HIGHEST)
    result = jnp.matmul(vt.conj().T, s_inv_sq * vt_w, precision=lax.Precision.HIGHEST)
    return unravel_solution(result, packed_structures)


@_row_space_projection.register(SVD)
def _(solver: SVD, state: _SVDState, vector):
    (u, s, vt), packed_structures = state
    m, n = u.shape[0], vt.shape[1]
    transposed_ps = transpose_packed_structures(packed_structures)
    w = ravel_vector(vector, transposed_ps)
    mask = solver._singular_mask(s, n, m)
    # A^†A v = VV^H v (restricted to row space).  Σ cancels entirely.
    vt_w = jnp.matmul(vt, w, precision=lax.Precision.HIGHEST)
    masked_vt_w = jnp.where(mask, vt_w, 0)
    result = jnp.matmul(vt.conj().T, masked_vt_w, precision=lax.Precision.HIGHEST)
    return unravel_solution(result, packed_structures)
