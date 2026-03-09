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


_QRPState: TypeAlias = tuple[
    tuple[Array, Array, Array, Array] | tuple[Array, Array, Array],
    eqxi.Static,
    PackedStructures,
]


class QRP(AbstractLinearSolver):
    """QR with column pivoting solver for linear systems.

    This solver can handle arbitrary operators, including non-square and rank
    deficient ones. In this case it will return the pseudoinverse solution to
    the linear system.

    The solver operates by default in a mode where the rank is determined
    dynamically. In order to make jax transformations possible, the computation
    s structured in such a way as to work simultaneously for all possible
    ranks, but this comes at a cost. In order to preserve efficiency under
    jax's model, this solver optionally allows the user to provide the rank of
    the operator statically, through the `rank_defect` parameter. In static
    rank mode and if `rcond` is provided, guaranteed errors are emitted if
    the actual rank is different than the statically asserted rank.

    """

    rank_defect: int | None = None
    rcond: float | None = None

    def init(self, operator, options):
        del options
        packed_structures = pack_structures(operator)
        matrix = operator.as_matrix()
        m, n = matrix.shape
        transpose = n > m
        if transpose:
            matrix = matrix.T
        q, r, p = jsp.linalg.qr(matrix, mode="economic", pivoting=True)
        if self.rank_defect is not None:
            if self.rcond is not None and r.size > 0:
                r = eqx.error_if(
                    r,
                    (jnp.abs(r.diagonal()) < self.rcond * jnp.abs(r[0, 0])).sum()
                    != self.rank_defect,
                    "QRP: rcond and rank_defect both provided and operator is not "
                    "the asserted rank",
                )
            if self.rank_defect > 0:
                r = r[: -self.rank_defect]
                q = q[:, : -self.rank_defect]
        else:
            rcond = resolve_rcond(self.rcond, n, m, r.dtype)
            rcond = jnp.array(rcond, dtype=r.real.dtype)
            if r.size > 0:
                rcond = rcond * jnp.abs(r[0, 0])
            mask = jnp.abs(r.diagonal()) > rcond
            r = jnp.where(mask[:, None], r, 0.0)
        if self.rank_defect == 0:
            return (q, r, p), eqxi.Static(transpose), packed_structures
        else:
            # Complete orthogonal factorization case (see lapack sgelsy
            # documentation)

            # In this case we must eliminate to the right of the r x r triangle
            # with orthogonal transformations on columns. jax currently doesn't
            # expose the trapezoidal orthogonal elimination (eg lapack stzrzf,
            # needed for the implementation of sgelsy). We work around this by
            # not exploiting the upper trapezoidal property and instead doing a
            # second unpivoted qr.

            # In the dynamic rank case, we are forced to work at the same time
            # on here on the negligible bottom part of r, but this does not
            # interfere with the result of the top part since we are doing
            # column operations and the bottom is assumed negligible.
            z, t = jnp.linalg.qr(r.T, mode="reduced")
            t = t.T
            z = z.T
            return (q, t, z, p), eqxi.Static(transpose), packed_structures

    def compute(
        self,
        state: _QRPState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        factorization, transpose, packed_structures = state
        transpose = transpose.value
        del state, options
        vector = ravel_vector(vector, packed_structures)
        info = {}

        if self.rank_defect == 0:
            q, r, p = factorization  # pyright: ignore
            if transpose:
                # Minimal norm solution if underdetermined.
                solution = q.conj() @ jsp.linalg.solve_triangular(
                    r, vector[p], trans="T", unit_diagonal=False
                )
            else:
                # Least squares solution if overdetermined.
                solution = jsp.linalg.solve_triangular(
                    r, q.T.conj() @ vector, trans="N", unit_diagonal=False
                )
                solution = solution.at[p].set(solution)
        else:
            # complete orthogonal factorization case
            q, t, z, p = factorization  # pyright: ignore
            if self.rank_defect is not None:
                if transpose:
                    solution = q.conj() @ jsp.linalg.solve_triangular(
                        t,
                        z.conj() @ vector[p],
                        trans="T",
                        unit_diagonal=False,
                        lower=True,
                    )
                else:
                    solution = z.T.conj() @ jsp.linalg.solve_triangular(
                        t,
                        q.T.conj() @ vector,
                        trans="N",
                        unit_diagonal=False,
                        lower=True,
                    )
                    solution = solution.at[p].set(solution)
            else:
                mask = jnp.abs(t.diagonal()) > 0.0
                rank = mask.sum()
                info["rank"] = rank

                # Avoid the creation of NaN and inf in values which will
                # later be discarded
                t += jnp.diag(jnp.where(mask, 0.0, 1.0))

                if transpose:
                    solution = q.conj() @ jnp.where(
                        mask,
                        jsp.linalg.solve_triangular(
                            t,
                            z.conj() @ vector[p],
                            trans="T",
                            unit_diagonal=False,
                            lower=True,
                        ),
                        0,
                    )
                else:
                    solution = z.T.conj() @ jnp.where(
                        mask,
                        jsp.linalg.solve_triangular(
                            t,
                            q.T.conj() @ vector,
                            trans="N",
                            unit_diagonal=False,
                            lower=True,
                        ),
                        0,
                    )
                    solution = solution.at[p].set(solution)

        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, info

    def transpose(self, state: _QRPState, options: dict[str, Any]):
        factorization, transpose, structures = state
        transposed_packed_structures = transpose_packed_structures(structures)
        transpose_state = (
            factorization,
            eqxi.Static(not transpose.value),
            transposed_packed_structures,
        )
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _QRPState, options: dict[str, Any]):
        factorization, transpose, structures = state
        conj_factorization = tuple(f.conj() for f in factorization[:-1]) + (
            factorization[-1],
        )
        conj_state = (
            conj_factorization,
            transpose,
            structures,
        )
        conj_options = {}
        return conj_state, conj_options

    def assume_full_rank(self):
        return self.rank_defect == 0


QRP.__init__.__doc__ = """**Arguments:**

- `rank_defect`: If set, the rank of the operator is statically assumed to be
  `min(N,M) - rank_defect`, where `(N,M)` is the shape of the operator.
 
    If not set, the solver is in dynamic rank mode.

- `rcond`: The threshold for determining rank. Matrices will be
    considered to have rank at most $r$ roughly when the ratio of the $r+1$st
    and first singular values is at most `rcond`.

    In dynamic rank mode (see `rank_defect`), `rcond` defaults to machine
    precision times `max(N, M)`, where `(N, M)` is the shape of the operator.

    In static rank mode and if `rcond` is provided, an error will be emitted if
    the dynamically determined rank doesn't match the statically asserted rank.
"""
