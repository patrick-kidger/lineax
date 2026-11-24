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
"gram partner": a solver, state, and options, reusing the existing
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

import jax
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
    gram_solver, gram_state, gram_options = _gram_partner(solver, gram_operator, state)
    v = jr.normal(getkey(), (operator.in_size(),), dtype=dtype)
    got = lx.linear_solve(
        gram_operator, v, gram_solver, state=gram_state, options=gram_options
    ).value
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


@pytest.mark.parametrize("complex_matrix", (False, True), ids=("real", "complex"))
def test_normal_gram_jvp_reuses_preconditioner(complex_matrix):
    if complex_matrix:
        # The rotation gives non-real Gram entries using exactly representable values.
        rotation = jnp.array([[1 + 1j, 1 - 1j, 0], [1 - 1j, 1 + 1j, 0], [0, 0, 2]]) / 2
    else:
        rotation = jnp.eye(3)
    scales = jnp.array([1.0, 4.0, 16.0], dtype=rotation.dtype)
    rotation_h = rotation.conj().T
    matrix = jnp.pad(jnp.diag(scales) @ rotation, ((0, 1), (0, 0)))
    preconditioner = lx.MatrixLinearOperator(
        jnp.pad(rotation_h @ jnp.diag(1 / scales), ((0, 0), (0, 1)))
    )
    rhs = jnp.array([1.0, 4.0, 16.0, 1.0], dtype=matrix.dtype)
    solver = lx.Normal(lx.CG(rtol=1e-6, atol=1e-6, max_steps=2))

    def solve(a):
        return lx.linear_solve(
            lx.MatrixLinearOperator(a),
            rhs,
            solver,
            options={"preconditioner": preconditioner},
        ).value

    # Changing just the zero row leaves the Gram matrix stationary and changes
    # Aᴴb by the adjoint of the rotated row. Without preconditioning, the Gram
    # solve runs out of steps.
    direction = jnp.zeros_like(matrix).at[-1].set(jnp.sum(rotation, axis=0))
    value, tangent = jax.jvp(solve, (matrix,), (direction,))
    assert tree_allclose(value, rotation_h @ jnp.ones_like(scales))
    assert tree_allclose(tangent, rotation_h @ (1 / scales**2))


def test_normal_gram_jvp_ignores_primal_initial_guess():
    matrix = jnp.pad(jnp.diag(jnp.array([1.0, 4.0, 16.0])), ((0, 1), (0, 0)))
    rhs = jnp.array([1.0, 0.0, 0.0, 1.0])
    y0 = jnp.array([1.0, 0.0, 0.0])
    solver = lx.Normal(lx.CG(rtol=1e-6, atol=1e-6, max_steps=2))

    def solve(a):
        return lx.linear_solve(
            lx.MatrixLinearOperator(a), rhs, solver, options={"y0": y0}
        ).value

    # y0 solves the primal system. The Gram solve converges within the step
    # limit from zero, but hits the limit if it inherits y0.
    direction = jnp.zeros_like(matrix).at[3, 1].set(1.0)
    value, tangent = jax.jvp(solve, (matrix,), (direction,))
    assert tree_allclose(value, y0)
    assert tree_allclose(tangent, jnp.array([0.0, 1 / 16, 0.0]))
