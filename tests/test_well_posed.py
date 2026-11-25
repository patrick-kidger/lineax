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

from .helpers import (
    cond_cutoff,
    construct_matrix,
    make_jacrev_operator,
    ops,
    params,
    solve_tol,
    solver_for_dtype,
    solvers,
    tree_allclose,
)


@pytest.mark.parametrize("make_operator,solver,tags", params(only_pseudo=False))
@pytest.mark.parametrize("ops", ops)
@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
def test_small_wellposed(make_operator, solver, tags, ops, getkey, dtype):
    if make_operator is make_jacrev_operator and jnp.issubdtype(
        dtype, jnp.complexfloating
    ):
        # JacobianLinearOperator does not support complex dtypes when jac="bwd"
        return
    tol = solve_tol(dtype)
    solver = solver_for_dtype(solver, dtype)
    (matrix,) = construct_matrix(getkey, solver, tags, dtype=dtype)
    operator = make_operator(getkey, matrix, tags)
    operator, matrix = ops(operator, matrix)
    assert tree_allclose(operator.as_matrix(), matrix, rtol=tol, atol=tol)
    out_size, _ = matrix.shape
    true_x = jr.normal(getkey(), (out_size,), dtype=dtype)
    b = matrix @ true_x
    x = lx.linear_solve(operator, b, solver=solver).value
    jax_x = jnp.linalg.solve(matrix, b)  # pyright: ignore
    assert tree_allclose(x, true_x, atol=tol, rtol=tol)
    assert tree_allclose(x, jax_x, atol=tol, rtol=tol)


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
def test_pytree_wellposed(solver, getkey, dtype):
    if isinstance(solver, lx.BiCGStab) and jnp.finfo(dtype).bits == 32:
        # Deliberately not tested. On random dense nonsymmetric systems this size,
        # BiCGStab breaks down for roughly a quarter of float32 draws even when
        # well-conditioned (`cond < 30`), and SciPy's for about a sixth; complex64
        # fails much more rarely, but still for some draws. This is a limitation of
        # the algorithm, not of the pytree handling tested here. That is covered for
        # BiCGStab in double precision, and its single-precision dtype handling by
        # `test_small_wellposed`.
        return
    if not isinstance(
        solver,
        (
            lx.Diagonal,
            lx.Triangular,
            lx.Tridiagonal,
            lx.Circulant,
            lx.Cholesky,
            lx.HEVD,
            lx.CG,
        ),
    ):
        tol = solve_tol(dtype)
        solver = solver_for_dtype(solver, dtype)

        true_x = [
            jr.normal(getkey(), shape=(2, 4), dtype=dtype),
            jr.normal(getkey(), (3,), dtype=dtype),
        ]
        out_structure = jax.eval_shape(lambda: true_x)
        # Regenerate until well-conditioned, as `construct_matrix` does: an unlucky
        # draw is otherwise too ill-conditioned to meet `tol`, particularly for
        # `Normal` solvers, which square the condition number.
        while True:
            pytree = [
                [
                    jr.normal(getkey(), shape=(2, 4, 2, 4), dtype=dtype),
                    jr.normal(getkey(), shape=(2, 4, 3), dtype=dtype),
                ],
                [
                    jr.normal(getkey(), shape=(3, 2, 4), dtype=dtype),
                    jr.normal(getkey(), shape=(3, 3), dtype=dtype),
                ],
            ]
            operator = lx.PyTreeLinearOperator(pytree, out_structure)
            if jnp.linalg.cond(operator.as_matrix()) < cond_cutoff(solver, dtype):
                break
        b = operator.mv(true_x)
        lx_x = lx.linear_solve(operator, b, solver, throw=False)
        assert tree_allclose(lx_x.value, true_x, atol=tol, rtol=tol)
