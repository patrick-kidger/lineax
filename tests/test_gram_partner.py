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

"""Tests for the private gram-partner optimisation used in `_linear_solve_jvp`.

Every non-well-posed solver whose factorisation cheaply yields `(AᴴA)⁺` has a
"gram partner": a `(gram_solver, gram_state)` pair, reusing the existing
factorisation, such that solving `AᴴA` with it computes `(AᴴA)⁺`. The JVP uses
this to collapse a nested pair of adjoint solves into a single gram solve.

These tests assert the defining property directly -- that the partner really does
compute `(AᴴA)⁺` -- rather than leaving it to be checked implicitly (and only
partially) by the JVP suites. In particular the JVP suites exercise the QR partner
not at all: QR is registered full-rank/square-only, whereas its partner is reached
only for *tall* operators. The `_has_gram_partner` gate below is keyed on the same
predicate the JVP uses, so any newly added solver that opts into the fast path is
covered here automatically -- no separate registration to remember.
"""

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest
from lineax._solve import _gram_partner, _has_gram_partner

from .helpers import (
    construct_matrix,
    make_matrix_operator,
    solvers_tags_pseudoinverse,
    tree_allclose,
)


def _assert_gram_partner(operator, solver, getkey, dtype):
    """If `solver` has a gram partner for `operator`, check it computes `(AᴴA)⁺`."""
    state = solver.init(operator, options={})
    if not _has_gram_partner(solver, state):
        return False
    gram_operator = lx.TaggedLinearOperator(
        operator.H @ operator, lx.positive_semidefinite_tag
    )
    gram_solver, gram_state = _gram_partner(solver, gram_operator, state)
    v = jr.normal(getkey(), (operator.in_size(),), dtype=dtype)
    got = lx.linear_solve(gram_operator, v, gram_solver, state=gram_state).value
    matrix = operator.as_matrix()
    expected = jnp.linalg.pinv(matrix.conj().T @ matrix) @ v  # pyright: ignore
    assert tree_allclose(got, expected, atol=1e-4, rtol=1e-4)
    return True


@pytest.mark.parametrize("solver, tags, pseudoinverse", solvers_tags_pseudoinverse)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_gram_partner_square(solver, tags, pseudoinverse, getkey, dtype):
    del pseudoinverse
    (matrix,) = construct_matrix(getkey, solver, tags, dtype=dtype)
    operator = make_matrix_operator(getkey, matrix, tags)
    # Solvers without a gram partner (Triangular, LU, Cholesky, CG, ...) are skipped
    # via the `_has_gram_partner` gate inside the helper.
    _assert_gram_partner(operator, solver, getkey, dtype)


@pytest.mark.parametrize("solver", (lx.QR(), lx.SVD()))
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_gram_partner_tall(solver, getkey, dtype):
    # Tall is the shape for which the QR partner is actually reached in the JVP, and
    # which the square-only standard suites never test.
    matrix = jr.normal(getkey(), (5, 3), dtype=dtype)
    operator = lx.MatrixLinearOperator(matrix)
    used = _assert_gram_partner(operator, solver, getkey, dtype)
    assert used  # both QR and SVD must expose a gram partner when tall
