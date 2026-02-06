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
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, PyTree

from .._misc import resolve_rcond
from .._solution import RESULTS
from .._solve import AbstractLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_QRState: TypeAlias = tuple[
    tuple[Array, Array] | tuple[Array, Array, Array], eqxi.Static, PackedStructures
]


def _rank_mask(q, r, rcond):
    """Return (diag_r, is_valid, is_full_rank) for a pivoted QR factor."""
    m, *_ = q.shape
    k, *_ = r.shape
    tol = resolve_rcond(rcond, k, m, r.dtype) * jnp.abs(r[0, 0])
    diag_r = jnp.diag(r)
    is_valid = jnp.abs(diag_r) > jnp.asarray(tol)
    return diag_r, is_valid, jnp.all(is_valid)


def _compute_pivoted(q, r, jpvt, vector, rcond):
    diag_r, is_valid, is_full_rank = _rank_mask(q, r, rcond)
    k, *_ = r.shape
    c = q.T.conj() @ vector
    inv_perm = jnp.argsort(jpvt)

    def _full_rank(_):
        y = jsp.linalg.solve_triangular(r, c, unit_diagonal=False)
        return y[inv_perm]

    def _rank_deficient(_):
        null_cols = ~is_valid
        diag_idx = jnp.arange(k)
        r_safe = r.at[diag_idx, diag_idx].set(
            jnp.where(is_valid, diag_r, jnp.ones(k, dtype=r.dtype))
        )
        c_masked = c * is_valid.astype(r.dtype)
        R_valid = r * is_valid[:, None].astype(r.dtype)
        R_null = R_valid * null_cols[None, :].astype(r.dtype)

        combined = jnp.concatenate([jnp.expand_dims(c_masked, axis=1), R_null], axis=1)
        sol = jsp.linalg.solve_triangular(r_safe, combined, unit_diagonal=False)
        w = sol[:, 0]
        F = sol[:, 1:]

        G = jnp.eye(k, dtype=r.dtype) + F.T.conj() @ F
        c_chol, lower = jsp.linalg.cho_factor(G)
        z = jsp.linalg.cho_solve((c_chol, lower), F.T.conj() @ w)
        y = w - F @ z + z
        return y[inv_perm]

    return jax.lax.cond(is_full_rank, _full_rank, _rank_deficient, None)


def _compute_pivoted_transpose(q, r, jpvt, vector, rcond):
    _, is_valid, is_full_rank = _rank_mask(q, r, rcond)
    c_perm = vector[jpvt]

    def _full_rank(_):
        v = jsp.linalg.solve_triangular(r, c_perm, trans="T", unit_diagonal=False)
        return q.conj() @ v

    def _rank_deficient(_):
        R_valid = r * is_valid[:, None].astype(r.dtype)
        G = R_valid.conj() @ R_valid.T + jnp.diag((~is_valid).astype(r.dtype))
        rhs = R_valid.conj() @ c_perm
        c_chol, lower = jsp.linalg.cho_factor(G)
        v = jsp.linalg.cho_solve((c_chol, lower), rhs)
        return q.conj() @ v

    return jax.lax.cond(is_full_rank, _full_rank, _rank_deficient, None)


class QR(AbstractLinearSolver):
    """QR solver for linear systems.

    This solver can handle non-square operators.

    This is usually the preferred solver when dealing with non-square operators.

    !!! info

        When `pivoting=False` (the default), this solver can only handle full-rank
        operators. For rank-deficient operators, use `pivoting=True` or switch to
        [`lineax.SVD`][] instead.

    """

    pivoting: bool = False
    rcond: float | None = None

    def init(self, operator, options):
        del options
        matrix = operator.as_matrix()
        m, n = matrix.shape
        transpose = n > m
        if transpose:
            matrix = matrix.T
        qr = jsp.linalg.qr(matrix, mode="economic", pivoting=self.pivoting)
        packed_structures = pack_structures(operator)
        return qr, eqxi.Static(transpose), packed_structures

    def compute(
        self,
        state: _QRState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        (q, r, *p), transpose, packed_structures = state
        transpose = transpose.value
        del state, options
        vector = ravel_vector(vector, packed_structures)
        if transpose:
            if self.pivoting:
                solution = _compute_pivoted_transpose(q, r, p[0], vector, self.rcond)
            else:
                solution = q.conj() @ jsp.linalg.solve_triangular(
                    r,
                    vector,
                    trans="T",
                    unit_diagonal=False,
                )
        else:
            if self.pivoting:
                solution = _compute_pivoted(q, r, p[0], vector, self.rcond)
            else:
                solution = jsp.linalg.solve_triangular(
                    r, q.T.conj() @ vector, trans="N", unit_diagonal=False
                )
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _QRState, options: dict[str, Any]):
        (q, r, *p), transpose, structures = state
        transposed_packed_structures = transpose_packed_structures(structures)
        transpose_state = (
            (q, r, *p),
            eqxi.Static(not transpose.value),
            transposed_packed_structures,
        )
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _QRState, options: dict[str, Any]):
        (q, r, *p), transpose, structures = state
        conj_state = (
            (q.conj(), r.conj(), *p),
            transpose,
            structures,
        )
        conj_options = {}
        return conj_state, conj_options

    def assume_full_rank(self):
        return not self.pivoting


QR.__init__.__doc__ = """**Arguments:**

- `pivoting`: If `True`, use column-pivoted QR decomposition for improved numerical
    stability. The pivoted decomposition satisfies `A[:, P] = Q R` where the diagonal
    of `R` is non-increasing. Defaults to `False`.
- `rcond`: the cutoff for determining numerical rank when `pivoting=True`. Diagonal
    entries of `R` smaller than `rcond * |R[0, 0]|` are treated as zero. Defaults to
    machine precision times `max(N, M)`, where `(N, M)` is the shape of the operator.
    Ignored when `pivoting=False`.
"""
