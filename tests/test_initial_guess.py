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

"""Tests for differentiating a linear solve given an initial guess `y0`.

Only the primal solve starts from `y0`: the tangent and cotangent solves have other
right-hand sides. For a singular operator LSMR keeps the null-space component of `y0`,
returning `A⁺b + (I - A⁺A) y0`, and the derivatives are those of that function.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from .helpers import tree_allclose


_tol = 1e-12


def _rank_deficient(getkey, symmetric):
    """A rank-3 5x5 matrix and a tangent direction (symmetric if the matrix is)."""
    left, _ = jnp.linalg.qr(jr.normal(getkey(), (5, 5)))
    right = left if symmetric else jnp.linalg.qr(jr.normal(getkey(), (5, 5)))[0]
    matrix = left @ jnp.diag(jnp.array([3.0, 2.0, 1.0, 0.0, 0.0])) @ right.T
    tangent = jr.normal(getkey(), (5, 5))
    if symmetric:
        tangent = tangent + tangent.T
    return matrix, tangent


def _exact(matrix, vector, y0):
    """What LSMR started from `y0` returns."""
    pinv = jnp.linalg.pinv(matrix)
    return pinv @ vector + y0 - pinv @ (matrix @ y0)


def _lsmr_solve(matrix, vector, y0, tags=()):
    operator = lx.MatrixLinearOperator(matrix, tags)
    solver = lx.LSMR(rtol=_tol, atol=_tol)
    return lx.linear_solve(operator, vector, solver, options={"y0": y0}).value


@pytest.mark.parametrize("symmetric", (True, False))
def test_jvp_singular(symmetric, getkey):
    # `y0` has components in both the row space and the null space. The null space
    # moves with the operator, so even a constant `y0` contributes to the derivative.
    # (A symmetric operator takes the JVP's Hermitian shortcut.)
    matrix, t_matrix = _rank_deficient(getkey, symmetric)
    vector, t_vector = jr.normal(getkey(), (5,)), jr.normal(getkey(), (5,))
    y0 = jr.normal(getkey(), (5,))
    tags = lx.symmetric_tag if symmetric else ()

    def solve(matrix, vector):
        return _lsmr_solve(matrix, vector, y0, tags)

    assert tree_allclose(solve(matrix, vector), _exact(matrix, vector, y0), atol=1e-10)
    # The derivatives are linear in the tangents...
    _, linear = jax.linearize(solve, matrix, vector)
    t_out = linear(t_matrix, t_vector)
    assert tree_allclose(linear(0 * t_matrix, 0 * t_vector), jnp.zeros(5), atol=1e-12)
    assert tree_allclose(linear(2 * t_matrix, 2 * t_vector), 2 * t_out, rtol=1e-10)
    # ...and exactly those of what the solve returns.
    _, t_expected = jax.jvp(
        lambda m, v: _exact(m, v, y0), (matrix, vector), (t_matrix, t_vector)
    )
    assert tree_allclose(t_out, t_expected, rtol=1e-8, atol=1e-10)


def test_grad_singular(getkey):
    matrix, _ = _rank_deficient(getkey, symmetric=False)
    vector = jr.normal(getkey(), (5,))
    y0 = jr.normal(getkey(), (5,))

    def solve(matrix, vector):
        return _lsmr_solve(matrix, vector, y0)

    def exact(matrix, vector):
        return _exact(matrix, vector, y0)

    grad = jax.grad(lambda m, v: jnp.sum(solve(m, v) ** 3), argnums=(0, 1))
    grad_expected = jax.grad(lambda m, v: jnp.sum(exact(m, v) ** 3), argnums=(0, 1))
    assert tree_allclose(
        grad(matrix, vector), grad_expected(matrix, vector), rtol=1e-8, atol=1e-10
    )
    jacfwd = jax.jacfwd(solve, argnums=(0, 1))(matrix, vector)
    jacrev = jax.jacrev(solve, argnums=(0, 1))(matrix, vector)
    assert tree_allclose(jacfwd, jacrev, rtol=1e-8, atol=1e-10)


def test_direct_solver_ignores_y0(getkey):
    # A direct solver ignores `y0`: the solution and its derivatives are those of `A⁺b`.
    matrix, t_matrix = _rank_deficient(getkey, symmetric=False)
    vector = jr.normal(getkey(), (5,))
    y0 = jr.normal(getkey(), (5,))

    def solve(matrix):
        operator = lx.MatrixLinearOperator(matrix)
        return lx.linear_solve(operator, vector, lx.SVD(), options={"y0": y0}).value

    def exact(matrix):
        return jnp.linalg.pinv(matrix) @ vector

    _, t_out = jax.jvp(solve, (matrix,), (t_matrix,))
    _, t_expected = jax.jvp(exact, (matrix,), (t_matrix,))
    assert tree_allclose(t_out, t_expected, rtol=1e-8, atol=1e-10)
    assert tree_allclose(
        jax.jacrev(solve)(matrix), jax.jacrev(exact)(matrix), rtol=1e-8, atol=1e-10
    )


def test_y0_not_differentiable(getkey):
    matrix = jr.normal(getkey(), (4, 4))
    operator = lx.MatrixLinearOperator(matrix)
    vector = jr.normal(getkey(), (4,))
    solver = lx.LSMR(rtol=_tol, atol=_tol)

    def solve(y0):
        return lx.linear_solve(operator, vector, solver, options={"y0": y0}).value

    with pytest.raises(Exception, match="cannot be autodifferentiated"):
        jax.jvp(solve, (jnp.zeros(4),), (jnp.ones(4),))


_full_rank_cases = (
    # (solver, tags, shape)
    (lx.CG(rtol=_tol, atol=_tol), lx.positive_semidefinite_tag, (4, 4)),
    (lx.GMRES(rtol=_tol, atol=_tol), (), (4, 4)),
    (lx.BiCGStab(rtol=_tol, atol=_tol), (), (4, 4)),
    (lx.LSMR(rtol=_tol, atol=_tol), (), (4, 4)),
    (lx.Normal(lx.CG(rtol=_tol, atol=_tol)), (), (6, 4)),
)


@pytest.mark.parametrize("solver, tags, shape", _full_rank_cases)
def test_derivatives_independent_of_y0(solver, tags, shape, getkey):
    # Without a null space, `y0` changes only where the iterations start.
    matrix = jr.normal(getkey(), shape)
    if tags == lx.positive_semidefinite_tag:
        matrix = matrix @ matrix.T + jnp.eye(shape[0])
    vector = jr.normal(getkey(), (shape[0],))
    t_matrix = jr.normal(getkey(), shape)
    y0 = jr.normal(getkey(), (shape[1],))

    def derivatives(options):
        def solve(matrix):
            operator = lx.MatrixLinearOperator(matrix, tags)
            return lx.linear_solve(operator, vector, solver, options=options).value

        _, t_out = jax.jvp(solve, (matrix,), (t_matrix,))
        grad = jax.grad(lambda m: jnp.sum(solve(m) ** 2))(matrix)
        return t_out, grad

    assert tree_allclose(derivatives({"y0": y0}), derivatives({}), rtol=1e-8)


def test_normal_y0_is_per_solve(getkey):
    # `Normal` used to fix `y0` in its state at `init`. It is now read at each solve.
    matrix = jr.normal(getkey(), (6, 4))
    operator = lx.MatrixLinearOperator(matrix)
    solution = jr.normal(getkey(), (4,))
    vector = matrix @ solution
    solver = lx.Normal(lx.CG(rtol=_tol, atol=_tol, max_steps=1))
    state = solver.init(operator, {})
    # Starting from the solution, a single step is enough.
    out = lx.linear_solve(
        operator, vector, solver, state=state, options={"y0": solution}, throw=False
    )
    assert tree_allclose(out.value, solution, rtol=1e-10)
    # Whereas from zero it is not.
    out = lx.linear_solve(operator, vector, solver, state=state, throw=False)
    assert not tree_allclose(out.value, solution, rtol=1e-3)


def test_normal_transpose_and_conj_drop_y0(getkey):
    operator = lx.MatrixLinearOperator(jr.normal(getkey(), (6, 4)))
    solver = lx.Normal(lx.CG(rtol=_tol, atol=_tol))
    options = {"y0": jnp.zeros(4)}
    state = solver.init(operator, options)
    _, transpose_options = solver.transpose(state, options)
    _, conj_options = solver.conj(state, options)
    assert "y0" not in transpose_options
    assert "y0" not in conj_options


def test_wide_normal_preconditioned_y0(getkey):
    # A wide `Normal` maps `y0` through the preconditioner into its inner problem. Its
    # solution `Aᴴy` lies in the row space, so `y0` does not affect it.
    matrix = jr.normal(getkey(), (4, 6))
    vector = jr.normal(getkey(), (4,))
    t_matrix = jr.normal(getkey(), (4, 6))
    approx_pinv = jnp.linalg.pinv(matrix) + 0.01 * jr.normal(getkey(), (6, 4))
    preconditioner = lx.MatrixLinearOperator(approx_pinv)
    y0 = jr.normal(getkey(), (6,))
    solver = lx.Normal(lx.CG(rtol=_tol, atol=_tol))

    def derivatives(options):
        def solve(matrix):
            operator = lx.MatrixLinearOperator(matrix)
            return lx.linear_solve(operator, vector, solver, options=options).value

        out, t_out = jax.jvp(solve, (matrix,), (t_matrix,))
        grad = jax.grad(lambda m: jnp.sum(solve(m) ** 2))(matrix)
        return out, t_out, grad

    with_y0 = derivatives({"preconditioner": preconditioner, "y0": y0})
    without_y0 = derivatives({"preconditioner": preconditioner})
    assert tree_allclose(with_y0[0], jnp.linalg.pinv(matrix) @ vector, rtol=1e-8)
    assert tree_allclose(with_y0, without_y0, rtol=1e-8)
