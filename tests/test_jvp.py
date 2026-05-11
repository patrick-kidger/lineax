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
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from .helpers import (
    all_test_cases,
    construct_matrix,
    has_tag,
    jnp_lstsq,
    make_jac_operator,
    make_matrix_operator,
    tree_allclose,
)


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("solver, tags, rows, cols, full_rank", all_test_cases())
@pytest.mark.parametrize("use_state", (True, False))
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_jvp(
    getkey,
    make_operator,
    solver,
    tags,
    rows,
    cols,
    full_rank,
    use_state,
    dtype,
):
    t_tags = (None,) * len(tags) if isinstance(tags, tuple) else None

    matrix, t_matrix = construct_matrix(
        getkey,
        solver,
        tags,
        num=2,
        dtype=dtype,
        rows=rows,
        cols=cols,
        full_rank=full_rank,
    )

    out_size, _ = matrix.shape
    vec = jr.normal(getkey(), (out_size,), dtype=dtype)
    t_vec = jr.normal(getkey(), (out_size,), dtype=dtype)

    if has_tag(tags, lx.unit_diagonal_tag):
        # For all the other tags, A + εB with A, B \in {matrices satisfying the tag}
        # still satisfies the tag itself.
        # This is the exception.
        t_matrix = t_matrix.at[
            jnp.arange(min(rows, cols)), jnp.arange(min(rows, cols))
        ].set(0)

    make_op = ft.partial(make_operator, getkey)
    operator, t_operator = eqx.filter_jvp(make_op, (matrix, tags), (t_matrix, t_tags))

    if use_state:
        state = solver.init(operator, options={})
        linear_solve = ft.partial(lx.linear_solve, state=state)
    else:
        linear_solve = lx.linear_solve

    solve_vec_only = lambda v: linear_solve(operator, v, solver).value
    solve_op_only = lambda op: linear_solve(op, vec, solver).value
    solve_op_vec = lambda op, v: linear_solve(op, v, solver).value

    vec_out, t_vec_out = eqx.filter_jvp(solve_vec_only, (vec,), (t_vec,))
    op_out, t_op_out = eqx.filter_jvp(solve_op_only, (operator,), (t_operator,))
    op_vec_out, t_op_vec_out = eqx.filter_jvp(
        solve_op_vec,
        (operator, vec),
        (t_operator, t_vec),
    )
    r = min(rows, cols) if full_rank else min(rows, cols) - 1
    expected_op_out, t_expected_op_out = eqx.filter_jvp(
        lambda op: jnp_lstsq(op, vec, r),
        (matrix,),
        (t_matrix,),
    )
    expected_op_vec_out, t_expected_op_vec_out = eqx.filter_jvp(
        lambda A, v: jnp_lstsq(A, v, r),
        (matrix, vec),
        (t_matrix, t_vec),
    )

    pinv_matrix = jnp.linalg.pinv(matrix)  # pyright: ignore
    expected_vec_out = pinv_matrix @ vec
    assert tree_allclose(vec_out, expected_vec_out)
    assert tree_allclose(op_out, expected_op_out)
    assert tree_allclose(op_vec_out, expected_op_vec_out)

    t_expected_vec_out = pinv_matrix @ t_vec
    assert tree_allclose(matrix @ t_vec_out, matrix @ t_expected_vec_out, rtol=1e-3)
    assert tree_allclose(t_op_out, t_expected_op_out, rtol=1e-3)
    assert tree_allclose(t_op_vec_out, t_expected_op_vec_out, rtol=1e-3)
