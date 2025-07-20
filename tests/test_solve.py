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

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from .helpers import construct_poisson_matrix, tree_allclose


def test_gmres_large_dense(getkey):
    if jax.config.jax_enable_x64:  # pyright: ignore
        tol = 1e-10
    else:
        tol = 1e-4
    solver = lx.GMRES(atol=tol, rtol=tol, restart=100)

    matrix = jr.normal(getkey(), (100, 100))
    operator = lx.MatrixLinearOperator(matrix)
    true_x = jr.normal(getkey(), (100,))
    b = matrix @ true_x

    lx_soln = lx.linear_solve(operator, b, solver).value

    assert tree_allclose(lx_soln, true_x, atol=tol, rtol=tol)


def test_nontrivial_pytree_operator():
    x = [[1, 5.0], [jnp.array(-2), jnp.array(-2.0)]]
    y = [3, 4]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct)
    out = lx.linear_solve(operator, y).value
    true_out = [jnp.array(-3.25), jnp.array(1.25)]
    assert tree_allclose(out, true_out)


def test_nontrivial_diagonal_operator():
    x = (8.0, jnp.array([1, 2, 3]), {"a": jnp.array([4, 5]), "b": 6})
    y = (4.0, jnp.array([7, 8, 9]), {"a": jnp.array([2, 10]), "b": 12})
    operator = lx.DiagonalLinearOperator(x)
    out = lx.linear_solve(operator, y).value
    true_out = (
        jnp.array(0.5),
        jnp.array([7.0, 4.0, 3.0]),
        {"a": jnp.array([0.5, 2.0]), "b": jnp.array(2.0)},
    )
    assert tree_allclose(out, true_out)


@pytest.mark.parametrize("solver", (lx.LU(), lx.QR(), lx.SVD()))
def test_mixed_dtypes(solver):
    f32 = lambda x: jnp.array(x, dtype=jnp.float32)
    f64 = lambda x: jnp.array(x, dtype=jnp.float64)
    x = [[f32(1), f64(5)], [f32(-2), f64(-2)]]
    y = [f64(3), f64(4)]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct)
    out = lx.linear_solve(operator, y, solver=solver).value
    true_out = [f32(-3.25), f64(1.25)]
    assert tree_allclose(out, true_out)


@pytest.mark.parametrize("solver", (lx.LU(), lx.QR(), lx.SVD()))
def test_mixed_dtypes_complex(solver):
    c64 = lambda x: jnp.array(x, dtype=jnp.complex64)
    c128 = lambda x: jnp.array(x, dtype=jnp.complex128)
    x = [[c64(1), c128(5.0j)], [c64(2.0j), c128(-2)]]
    y = [c128(3), c128(4)]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct)
    out = lx.linear_solve(operator, y, solver=solver).value
    true_out = [c64(-0.75 - 2.5j), c128(0.5 - 0.75j)]
    assert tree_allclose(out, true_out)


@pytest.mark.parametrize("solver", (lx.LU(), lx.QR(), lx.SVD()))
def test_mixed_dtypes_complex_real(solver):
    f64 = lambda x: jnp.array(x, dtype=jnp.float64)
    c128 = lambda x: jnp.array(x, dtype=jnp.complex128)
    x = [[f64(1), c128(-5.0j)], [f64(2.0), c128(-2j)]]
    y = [c128(3), c128(4)]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct)
    out = lx.linear_solve(operator, y, solver=solver).value
    true_out = [f64(1.75), c128(0.25j)]
    assert tree_allclose(out, true_out)


def test_mixed_dtypes_triangular():
    f32 = lambda x: jnp.array(x, dtype=jnp.float32)
    f64 = lambda x: jnp.array(x, dtype=jnp.float64)
    x = [[f32(1), f64(0)], [f32(-2), f64(-2)]]
    y = [f64(3), f64(4)]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct, lx.lower_triangular_tag)
    out = lx.linear_solve(operator, y, solver=lx.Triangular()).value
    true_out = [f32(3), f64(-5)]
    assert tree_allclose(out, true_out)


def test_mixed_dtypes_complex_triangular():
    c64 = lambda x: jnp.array(x, dtype=jnp.complex64)
    c128 = lambda x: jnp.array(x, dtype=jnp.complex128)
    x = [[c64(1), c128(0)], [c64(2.0j), c128(-2)]]
    y = [c128(3), c128(4)]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct, lx.lower_triangular_tag)
    out = lx.linear_solve(operator, y, solver=lx.Triangular()).value
    true_out = [c64(3), c128(-2 + 3.0j)]
    assert tree_allclose(out, true_out)


def test_mixed_dtypes_complex_real_triangular():
    f64 = lambda x: jnp.array(x, dtype=jnp.float64)
    c128 = lambda x: jnp.array(x, dtype=jnp.complex128)
    x = [[f64(1), c128(0)], [f64(2.0), c128(2j)]]
    y = [c128(3), c128(4)]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct, lx.lower_triangular_tag)
    out = lx.linear_solve(operator, y, solver=lx.Triangular()).value
    true_out = [f64(3), c128(1j)]
    assert tree_allclose(out, true_out)


def test_ad_closure_function_linear_operator(getkey):
    def f(x, z):
        def fn(y):
            return x * y

        op = lx.FunctionLinearOperator(fn, jax.eval_shape(lambda: z))
        sol = lx.linear_solve(op, z).value
        return jnp.sum(sol), sol

    x = jr.normal(getkey(), (3,))
    x = jnp.where(jnp.abs(x) < 1e-6, 0.7, x)
    z = jr.normal(getkey(), (3,))
    grad, sol = jax.grad(f, has_aux=True)(x, z)
    assert tree_allclose(grad, -z / (x**2))
    assert tree_allclose(sol, z / x)


def test_grad_vmap_symbolic_cotangent():
    def f(x):
        return x[0], x[1]

    @jax.vmap
    def to_vmap(x):
        op = lx.FunctionLinearOperator(f, jax.eval_shape(lambda: x))
        sol = lx.linear_solve(op, x)
        return sol.value[0]

    @jax.grad
    def to_grad(x):
        return jnp.sum(to_vmap(x))

    x = (jnp.arange(3.0), jnp.arange(3.0))
    to_grad(x)


@pytest.mark.parametrize(
    "solver",
    (
        lx.CG(0.0, 0.0, max_steps=2),
        lx.NormalCG(0.0, 0.0, max_steps=2),
        lx.BiCGStab(0.0, 0.0, max_steps=2),
        lx.GMRES(0.0, 0.0, max_steps=2),
        lx.LSMR(0.0, 0.0, max_steps=2),
    ),
)
def test_iterative_solver_max_steps_only(solver):
    """Iterative solvers should work with max_steps only (no Equinox errors)."""
    SIZE = 100

    poisson_matrix = construct_poisson_matrix(SIZE)
    poisson_operator = lx.MatrixLinearOperator(
        poisson_matrix, tags=(lx.negative_semidefinite_tag, lx.symmetric_tag)
    )
    rhs = jax.random.normal(jax.random.key(0), (SIZE,))

    lx.linear_solve(poisson_operator, rhs, solver)


def test_nonfinite_input():
    operator = lx.DiagonalLinearOperator((1.0, 1.0))
    vector = (1.0, jnp.inf)
    sol = lx.linear_solve(operator, vector, throw=False)
    assert sol.result == lx.RESULTS.nonfinite_input

    vector = (1.0, jnp.nan)
    sol = lx.linear_solve(operator, vector, throw=False)
    assert sol.result == lx.RESULTS.nonfinite_input

    vector = (jnp.nan, jnp.inf)
    sol = lx.linear_solve(operator, vector, throw=False)
    assert sol.result == lx.RESULTS.nonfinite_input
