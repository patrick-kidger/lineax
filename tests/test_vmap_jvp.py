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
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from .helpers import (
    all_test_cases,
    construct_matrix,
    jnp_lstsq,
    make_jac_operator,
    make_matrix_operator,
    tree_allclose,
)


# Workaround for https://github.com/jax-ml/jax/issues/27201
@pytest.fixture(autouse=True)
def _clear_cache():
    eqx.clear_caches()


@pytest.mark.parametrize("solver, tags, rows, cols, full_rank", all_test_cases())
@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("use_state", (True, False))
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_vmap_jvp(
    getkey,
    solver,
    tags,
    rows,
    cols,
    full_rank,
    make_operator,
    use_state,
    dtype,
):
    t_tags = (None,) * len(tags) if isinstance(tags, tuple) else None
    matrix, t_matrix = construct_matrix(
        getkey,
        solver,
        tags,
        num=2,
        rows=rows,
        cols=cols,
        full_rank=full_rank,
        dtype=dtype,
    )

    make_op = ft.partial(make_operator, getkey)
    operator, t_operator = eqx.filter_jvp(make_op, (matrix, tags), (t_matrix, t_tags))

    out_size, _ = matrix.shape
    vec = jr.normal(getkey(), (out_size,), dtype=dtype)
    t_vec = jr.normal(getkey(), (out_size,), dtype=dtype)

    if use_state:

        def linear_solve1(operator, vector):
            op_dynamic, op_static = eqx.partition(operator, eqx.is_inexact_array)
            stopped_operator = eqx.combine(lax.stop_gradient(op_dynamic), op_static)
            state = solver.init(stopped_operator, options={})

            return lx.linear_solve(operator, vector, state=state, solver=solver)

    else:
        linear_solve1 = ft.partial(lx.linear_solve, solver=solver)

    if full_rank and rows == cols:
        jnp_solve1 = jnp.linalg.solve
    else:
        r = min(rows, cols) if full_rank else min(rows, cols) - 1
        jnp_solve1 = lambda mat, vec: jnp_lstsq(mat, vec, r)

    for mode in ("vec", "op", "op_vec"):
        if "op" in mode:
            axis_size = 10
            out_axes = eqx.if_array(0)
        else:
            axis_size = None
            out_axes = None

        def _make():
            matrix, t_matrix = construct_matrix(
                getkey,
                solver,
                tags,
                num=2,
                rows=rows,
                cols=cols,
                full_rank=full_rank,
                dtype=dtype,
            )
            make_op = ft.partial(make_operator, getkey)
            operator, t_operator = eqx.filter_jvp(
                make_op, (matrix, tags), (t_matrix, t_tags)
            )
            return matrix, t_matrix, operator, t_operator

        matrix, t_matrix, operator, t_operator = eqx.filter_vmap(
            _make, axis_size=axis_size, out_axes=out_axes
        )()

        if "op" in mode:
            _, out_size, _ = matrix.shape
        else:
            out_size, _ = matrix.shape

        if "vec" in mode:
            vec = jr.normal(getkey(), (10, out_size), dtype=dtype)
            t_vec = jr.normal(getkey(), (10, out_size), dtype=dtype)
        else:
            vec = jr.normal(getkey(), (out_size,), dtype=dtype)
            t_vec = jr.normal(getkey(), (out_size,), dtype=dtype)

        if mode == "op":
            linear_solve2 = lambda op: linear_solve1(op, vector=vec)
            jnp_solve2 = lambda mat: jnp_solve1(mat, vec)
        elif mode == "vec":
            linear_solve2 = lambda vector: linear_solve1(operator, vector)
            jnp_solve2 = lambda vector: jnp_solve1(matrix, vector)
        elif mode == "op_vec":
            linear_solve2 = linear_solve1
            jnp_solve2 = jnp_solve1
        else:
            assert False
        for jvp_first in (True, False):
            if jvp_first:
                linear_solve3 = ft.partial(eqx.filter_jvp, linear_solve2)
            else:
                linear_solve3 = linear_solve2
            linear_solve3 = eqx.filter_vmap(linear_solve3)
            if not jvp_first:
                linear_solve3 = ft.partial(eqx.filter_jvp, linear_solve3)
            linear_solve3 = eqx.filter_jit(linear_solve3)
            jnp_solve3 = ft.partial(eqx.filter_jvp, jnp_solve2)
            jnp_solve3 = eqx.filter_vmap(jnp_solve3)
            jnp_solve3 = eqx.filter_jit(jnp_solve3)
            if mode == "op":
                out, t_out = linear_solve3((operator,), (t_operator,))
                true_out, true_t_out = jnp_solve3((matrix,), (t_matrix,))
            elif mode == "vec":
                out, t_out = linear_solve3((vec,), (t_vec,))
                true_out, true_t_out = jnp_solve3((vec,), (t_vec,))
            elif mode == "op_vec":
                out, t_out = linear_solve3((operator, vec), (t_operator, t_vec))
                true_out, true_t_out = jnp_solve3((matrix, vec), (t_matrix, t_vec))
            else:
                assert False
            assert tree_allclose(out.value, true_out, atol=1e-4)
            assert tree_allclose(t_out.value, true_t_out, atol=1e-4)
