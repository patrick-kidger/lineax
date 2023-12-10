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
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import lineax as lx


sys.path.append("../tests")
from helpers import getkey, shaped_allclose  # pyright: ignore


jax.config.update("jax_enable_x64", True)


def make_problem(mat_size: int, *, key):
    mat = jr.normal(key, (mat_size, mat_size))
    true_x = jr.normal(key, (mat_size,))
    b = mat @ true_x
    op = lx.MatrixLinearOperator(mat)
    return mat, op, b, true_x


def benchmark_jax(mat_size: int, *, key):
    mat, _, b, true_x = make_problem(mat_size, key=key)

    solve_with_jax = ft.partial(
        jsp.sparse.linalg.gmres, tol=1e-5, solve_method="batched"
    )
    gmres_jit = jax.jit(solve_with_jax)
    jax_soln, info = gmres_jit(mat, b)

    # info == 0.0 implies that the solve has succeeded.
    returned_failed = jnp.all(info != 0.0)
    actually_failed = not shaped_allclose(jax_soln, true_x, atol=1e-4, rtol=1e-4)

    assert actually_failed

    captured_failure = returned_failed & actually_failed
    return captured_failure


def benchmark_lx(mat_size: int, *, key):
    _, op, b, true_x = make_problem(mat_size, key=key)

    lx_soln = lx.linear_solve(op, b, lx.GMRES(atol=1e-5, rtol=1e-5), throw=False)

    returned_failed = jnp.all(lx_soln.result != lx.RESULTS.successful)
    actually_failed = not shaped_allclose(lx_soln.value, true_x, atol=1e-4, rtol=1e-4)

    assert actually_failed

    captured_failure = returned_failed & actually_failed
    return captured_failure


lx_failed_safely = 0
jax_failed_safely = 0

for _ in range(100):
    key = getkey()
    jax_captured_failure = benchmark_jax(100, key=key)
    lx_captured_failure = benchmark_lx(100, key=key)

    jax_failed_safely = jax_failed_safely + jax_captured_failure
    lx_failed_safely = lx_failed_safely + lx_captured_failure

print(f"JAX failed safely {jax_failed_safely} out of 100 times")
print(f"Lineax failed safely {lx_failed_safely} out of 100 times")
