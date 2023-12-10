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
    construct_matrix,
    ops,
    params,
    solvers,
    tree_allclose,
)


@pytest.mark.parametrize("make_operator,solver,tags", params(only_pseudo=False))
@pytest.mark.parametrize("ops", ops)
@pytest.mark.parametrize("dtype", (jnp.float64,))
def test_small_wellposed(make_operator, solver, tags, ops, getkey, dtype):
    if jax.config.jax_enable_x64:  # pyright: ignore
        tol = 1e-10
    else:
        tol = 1e-4
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
def test_pytree_wellposed(solver, getkey):
    if not isinstance(
        solver,
        (lx.Diagonal, lx.Triangular, lx.Tridiagonal, lx.Cholesky, lx.CG, lx.NormalCG),
    ):
        if jax.config.jax_enable_x64:  # pyright: ignore
            tol = 1e-10
        else:
            tol = 1e-4

        true_x = [jr.normal(getkey(), shape=(2, 4)), jr.normal(getkey(), (3,))]
        pytree = [
            [
                jr.normal(getkey(), shape=(2, 4, 2, 4)),
                jr.normal(getkey(), shape=(2, 4, 3)),
            ],
            [
                jr.normal(getkey(), shape=(3, 2, 4)),
                jr.normal(getkey(), shape=(3, 3)),
            ],
        ]
        out_structure = jax.eval_shape(lambda: true_x)

        operator = lx.PyTreeLinearOperator(pytree, out_structure)
        b = operator.mv(true_x)
        lx_x = lx.linear_solve(operator, b, solver, throw=False)
        assert tree_allclose(lx_x.value, true_x, atol=tol, rtol=tol)
