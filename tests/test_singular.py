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

import functools as ft

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from .helpers import (
    construct_singular_matrix,
    finite_difference_jvp,
    make_jac_operator,
    make_matrix_operator,
    ops,
    params,
    tol,
    tree_allclose,
)


@pytest.mark.parametrize("make_operator,solver,tags", params(only_pseudo=True))
@pytest.mark.parametrize("ops", ops)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_small_singular(make_operator, solver, tags, ops, getkey, dtype):
    if jax.config.jax_enable_x64:  # pyright: ignore
        tol = 1e-10
    else:
        tol = 1e-4
    (matrix,) = construct_singular_matrix(getkey, solver, tags, dtype=dtype)
    operator = make_operator(getkey, matrix, tags)
    operator, matrix = ops(operator, matrix)
    assert tree_allclose(operator.as_matrix(), matrix, rtol=tol, atol=tol)
    out_size, in_size = matrix.shape
    true_x = jr.normal(getkey(), (in_size,), dtype=dtype)
    b = matrix @ true_x
    x = lx.linear_solve(operator, b, solver=solver, throw=False).value
    jax_x, *_ = jnp.linalg.lstsq(matrix, b)  # pyright: ignore
    assert tree_allclose(x, jax_x, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_bicgstab_breakdown(getkey, dtype):
    if jax.config.jax_enable_x64:  # pyright: ignore
        tol = 1e-10
    else:
        tol = 1e-4
    solver = lx.GMRES(atol=tol, rtol=tol, restart=2)

    matrix = jr.normal(jr.PRNGKey(0), (100, 100), dtype=dtype)
    true_x = jr.normal(jr.PRNGKey(0), (100,), dtype=dtype)
    b = matrix @ true_x
    operator = lx.MatrixLinearOperator(matrix)

    # result != 0 implies lineax reported failure
    lx_soln = lx.linear_solve(operator, b, solver, throw=False)

    assert jnp.all(lx_soln.result != lx.RESULTS.successful)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_gmres_stagnation_or_breakdown(getkey, dtype):
    if jax.config.jax_enable_x64:  # pyright: ignore
        tol = 1e-10
    else:
        tol = 1e-4
    solver = lx.GMRES(atol=tol, rtol=tol, restart=2)

    matrix = jnp.array(
        [
            [0.15892892, 0.05884365, -0.60427412, 0.1891916],
            [-1.5484863, 0.93608822, 1.94888868, 1.37069667],
            [0.62687318, -0.13996738, -0.6824359, 0.30975754],
            [-0.67428635, 1.52372255, -0.88277754, 0.69633816],
        ],
        dtype=dtype,
    )
    true_x = jnp.array([0.51383273, 1.72983427, -0.43251078, -1.11764668], dtype=dtype)
    b = matrix @ true_x
    operator = lx.MatrixLinearOperator(matrix)

    # result != 0 implies lineax reported failure
    lx_soln = lx.linear_solve(operator, b, solver, throw=False)

    assert jnp.all(lx_soln.result != lx.RESULTS.successful)


@pytest.mark.parametrize(
    "solver",
    (
        lx.AutoLinearSolver(well_posed=None),
        lx.QR(),
        lx.SVD(),
        lx.LSMR(atol=tol, rtol=tol),
        lx.Normal(lx.Cholesky()),
        lx.Normal(lx.SVD()),
    ),
)
def test_nonsquare_pytree_operator1(solver):
    x = [[1, 5.0, jnp.array(-1.0)], [jnp.array(-2), jnp.array(-2.0), 3.0]]
    y = [3.0, 4]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct)
    out = lx.linear_solve(operator, y, solver=solver).value
    matrix = jnp.array([[1.0, 5.0, -1.0], [-2.0, -2.0, 3.0]])
    true_out, _, _, _ = jnp.linalg.lstsq(matrix, jnp.array(y))  # pyright: ignore
    true_out = [true_out[0], true_out[1], true_out[2]]
    assert tree_allclose(out, true_out)


@pytest.mark.parametrize(
    "solver",
    (
        lx.AutoLinearSolver(well_posed=None),
        lx.QR(),
        lx.SVD(),
        lx.LSMR(atol=tol, rtol=tol),
        lx.Normal(lx.Cholesky()),
        lx.Normal(lx.SVD()),
    ),
)
def test_nonsquare_pytree_operator2(solver):
    x = [[1, jnp.array(-2)], [5.0, jnp.array(-2.0)], [jnp.array(-1.0), 3.0]]
    y = [3.0, 4, 5.0]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct)
    out = lx.linear_solve(operator, y, solver=solver).value
    matrix = jnp.array([[1.0, -2.0], [5.0, -2.0], [-1.0, 3.0]])
    true_out, _, _, _ = jnp.linalg.lstsq(matrix, jnp.array(y))  # pyright: ignore
    true_out = [true_out[0], true_out[1]]
    assert tree_allclose(out, true_out)


@pytest.mark.parametrize(
    "solver",
    (
        lx.AutoLinearSolver(well_posed=None),
        lx.QR(),
        lx.SVD(),
        lx.Normal(lx.Cholesky()),
        lx.Normal(lx.SVD()),
    ),
)
@pytest.mark.parametrize("full_rank", (True, False))
@pytest.mark.parametrize("jvp", (False, True))
@pytest.mark.parametrize("wide", (False, True))
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_nonsquare_mat_vec(solver, full_rank, jvp, wide, dtype, getkey):
    if wide:
        out_size = 3
        in_size = 6
    else:
        out_size = 6
        in_size = 3
    matrix = jr.normal(getkey(), (out_size, in_size), dtype=dtype)
    if not full_rank:
        if solver.assume_full_rank():
            # There is nothing to test.
            return
        # nontrivial rank 2 sparsity pattern
        matrix = matrix.at[1:, 1:].set(0)
    vector = jr.normal(getkey(), (out_size,), dtype=dtype)
    lx_solve = lambda mat, vec: lx.linear_solve(
        lx.MatrixLinearOperator(mat), vec, solver
    ).value
    jnp_solve = lambda mat, vec: jnp.linalg.lstsq(mat, vec)[0]  # pyright: ignore
    if jvp:
        lx_solve = eqx.filter_jit(ft.partial(eqx.filter_jvp, lx_solve))
        jnp_solve = eqx.filter_jit(ft.partial(finite_difference_jvp, jnp_solve))
        t_matrix = jr.normal(getkey(), (out_size, in_size), dtype=dtype)
        if not full_rank:
            # t_matrix must be chosen tangent to the manifold of rank 2
            # matrices at matrix. A simple way to achieve this is to make the
            # same restriction as we did to matrix
            t_matrix = t_matrix.at[1:, 1:].set(0)
        t_vector = jr.normal(getkey(), (out_size,), dtype=dtype)
        args = ((matrix, vector), (t_matrix, t_vector))
    else:
        args = (matrix, vector)
    x = lx_solve(*args)  # pyright: ignore
    true_x = jnp_solve(*args)
    assert tree_allclose(x, true_x, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "solver",
    (
        lx.AutoLinearSolver(well_posed=None),
        lx.QR(),
        lx.SVD(),
        lx.Normal(lx.Cholesky()),
        lx.Normal(lx.SVD()),
    ),
)
@pytest.mark.parametrize("full_rank", (True, False))
@pytest.mark.parametrize("jvp", (False, True))
@pytest.mark.parametrize("wide", (False, True))
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_nonsquare_vec(solver, full_rank, jvp, wide, dtype, getkey):
    if wide:
        out_size = 3
        in_size = 6
    else:
        out_size = 6
        in_size = 3
    matrix = jr.normal(getkey(), (out_size, in_size), dtype=dtype)
    if not full_rank:
        if solver.assume_full_rank():
            # There is nothing to test.
            return
        # nontrivial rank 2 sparsity pattern
        matrix = matrix.at[1:, 1:].set(0)
    vector = jr.normal(getkey(), (out_size,), dtype=dtype)
    lx_solve = lambda vec: lx.linear_solve(
        lx.MatrixLinearOperator(matrix), vec, solver
    ).value
    jnp_solve = lambda vec: jnp.linalg.lstsq(matrix, vec)[0]  # pyright: ignore
    if jvp:
        lx_solve = eqx.filter_jit(ft.partial(eqx.filter_jvp, lx_solve))
        jnp_solve = eqx.filter_jit(ft.partial(finite_difference_jvp, jnp_solve))
        t_vector = jr.normal(getkey(), (out_size,), dtype=dtype)
        args = ((vector,), (t_vector,))
    else:
        args = (vector,)
    x = lx_solve(*args)  # pyright: ignore
    true_x = jnp_solve(*args)
    assert tree_allclose(x, true_x, atol=1e-4, rtol=1e-4)


_iterative_solvers = (
    (lx.CG(rtol=tol, atol=tol), lx.positive_semidefinite_tag),
    (lx.CG(rtol=tol, atol=tol, max_steps=512), lx.negative_semidefinite_tag),
    (lx.GMRES(rtol=tol, atol=tol), ()),
    (lx.BiCGStab(rtol=tol, atol=tol), ()),
)


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("solver, tags", _iterative_solvers)
@pytest.mark.parametrize("use_state", (False, True))
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_iterative_singular(getkey, solver, tags, use_state, make_operator, dtype):
    (matrix,) = construct_singular_matrix(getkey, solver, tags)
    operator = make_operator(getkey, matrix, tags)

    out_size, _ = matrix.shape
    vec = jr.normal(getkey(), (out_size,), dtype=dtype)

    if use_state:
        state = solver.init(operator, options={})
        linear_solve = ft.partial(lx.linear_solve, state=state)
    else:
        linear_solve = lx.linear_solve

    with pytest.raises(Exception):
        linear_solve(operator, vec, solver)
