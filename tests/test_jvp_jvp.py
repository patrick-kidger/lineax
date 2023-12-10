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
    construct_matrix,
    construct_singular_matrix,
    make_jac_operator,
    make_matrix_operator,
    solvers_tags_pseudoinverse,
    tree_allclose,
)


@pytest.mark.parametrize("solver, tags, pseudoinverse", solvers_tags_pseudoinverse)
@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("use_state", (True, False))
@pytest.mark.parametrize(
    "make_matrix",
    (
        construct_matrix,
        construct_singular_matrix,
    ),
)
@pytest.mark.parametrize("dtype", (jnp.float64,))
def test_jvp_jvp(
    getkey, solver, tags, pseudoinverse, make_operator, use_state, make_matrix, dtype
):
    t_tags = (None,) * len(tags) if isinstance(tags, tuple) else None
    if (make_matrix is construct_matrix) or pseudoinverse:
        matrix, t_matrix, tt_matrix, tt_t_matrix = construct_matrix(
            getkey, solver, tags, num=4, dtype=dtype
        )

        make_op = ft.partial(make_operator, getkey)
        t_make_operator = lambda p, t_p: eqx.filter_jvp(
            make_op, (p, tags), (t_p, t_tags)
        )
        tt_make_operator = lambda p, t_p, tt_p, tt_t_p: eqx.filter_jvp(
            t_make_operator, (p, t_p), (tt_p, tt_t_p)
        )
        (operator, t_operator), (tt_operator, tt_t_operator) = tt_make_operator(
            matrix, t_matrix, tt_matrix, tt_t_matrix
        )

        out_size, _ = matrix.shape
        vec = jr.normal(getkey(), (out_size,), dtype=dtype)
        t_vec = jr.normal(getkey(), (out_size,), dtype=dtype)
        tt_vec = jr.normal(getkey(), (out_size,), dtype=dtype)
        tt_t_vec = jr.normal(getkey(), (out_size,), dtype=dtype)

        if use_state:

            def linear_solve1(operator, vector):
                state = solver.init(operator, options={})
                state_dynamic, state_static = eqx.partition(state, eqx.is_inexact_array)
                state_dynamic = lax.stop_gradient(state_dynamic)
                state = eqx.combine(state_dynamic, state_static)

                sol = lx.linear_solve(operator, vector, state=state, solver=solver)
                return sol.value

        else:

            def linear_solve1(operator, vector):
                sol = lx.linear_solve(operator, vector, solver=solver)
                return sol.value

        if pseudoinverse:
            jnp_solve1 = lambda mat, vec: jnp.linalg.lstsq(mat, vec)[0]  # pyright: ignore
        else:
            jnp_solve1 = jnp.linalg.solve  # pyright: ignore

        linear_solve2 = ft.partial(eqx.filter_jvp, linear_solve1)
        jnp_solve2 = ft.partial(eqx.filter_jvp, jnp_solve1)

        def _make_primal_tangents(mode):
            lx_args = ([], [], operator, t_operator, tt_operator, tt_t_operator)
            jnp_args = ([], [], matrix, t_matrix, tt_matrix, tt_t_matrix)
            for primals, ttangents, op, t_op, tt_op, tt_t_op in (lx_args, jnp_args):
                if "op" in mode:
                    primals.append(op)
                    ttangents.append(tt_op)
                if "vec" in mode:
                    primals.append(vec)
                    ttangents.append(tt_vec)
                if "t_op" in mode:
                    primals.append(t_op)
                    ttangents.append(tt_t_op)
                if "t_vec" in mode:
                    primals.append(t_vec)
                    ttangents.append(tt_t_vec)
            lx_out = tuple(lx_args[0]), tuple(lx_args[1])
            jnp_out = tuple(jnp_args[0]), tuple(jnp_args[1])
            return lx_out, jnp_out

        modes = (
            {"op"},
            {"vec"},
            {"t_op"},
            {"t_vec"},
            {"op", "vec"},
            {"op", "t_op"},
            {"op", "t_vec"},
            {"vec", "t_op"},
            {"vec", "t_vec"},
            {"op", "vec", "t_op"},
            {"op", "vec", "t_vec"},
            {"vec", "t_op", "t_vec"},
            {"op", "vec", "t_op", "t_vec"},
        )
        for mode in modes:
            if mode == {"op"}:
                linear_solve3 = lambda op: linear_solve2((op, vec), (t_operator, t_vec))
                jnp_solve3 = lambda mat: jnp_solve2((mat, vec), (t_matrix, t_vec))
            elif mode == {"vec"}:
                linear_solve3 = lambda v: linear_solve2(
                    (operator, v), (t_operator, t_vec)
                )
                jnp_solve3 = lambda v: jnp_solve2((matrix, v), (t_matrix, t_vec))
            elif mode == {"op", "vec"}:
                linear_solve3 = lambda op, v: linear_solve2(
                    (op, v), (t_operator, t_vec)
                )
                jnp_solve3 = lambda mat, v: jnp_solve2((mat, v), (t_matrix, t_vec))
            elif mode == {"t_op"}:
                linear_solve3 = lambda t_op: linear_solve2(
                    (operator, vec), (t_op, t_vec)
                )
                jnp_solve3 = lambda t_mat: jnp_solve2((matrix, vec), (t_mat, t_vec))
            elif mode == {"t_vec"}:
                linear_solve3 = lambda t_v: linear_solve2(
                    (operator, vec), (t_operator, t_v)
                )
                jnp_solve3 = lambda t_v: jnp_solve2((matrix, vec), (t_matrix, t_v))
            elif mode == {"op", "vec"}:
                linear_solve3 = lambda op, v: linear_solve2(
                    (op, v), (t_operator, t_vec)
                )
                jnp_solve3 = lambda mat, v: jnp_solve2((mat, v), (t_matrix, t_vec))
            elif mode == {"op", "t_op"}:
                linear_solve3 = lambda op, t_op: linear_solve2((op, vec), (t_op, t_vec))
                jnp_solve3 = lambda mat, t_mat: jnp_solve2((mat, vec), (t_mat, t_vec))
            elif mode == {"op", "t_vec"}:
                linear_solve3 = lambda op, t_v: linear_solve2(
                    (op, vec), (t_operator, t_v)
                )
                jnp_solve3 = lambda mat, t_v: jnp_solve2((mat, vec), (t_matrix, t_v))
            elif mode == {"vec", "t_op"}:
                linear_solve3 = lambda v, t_op: linear_solve2(
                    (operator, v), (t_op, t_vec)
                )
                jnp_solve3 = lambda v, t_mat: jnp_solve2((matrix, v), (t_mat, t_vec))
            elif mode == {"vec", "t_vec"}:
                linear_solve3 = lambda v, t_v: linear_solve2(
                    (operator, v), (t_operator, t_v)
                )
                jnp_solve3 = lambda v, t_v: jnp_solve2((matrix, v), (t_matrix, t_v))
            elif mode == {"op", "vec", "t_op"}:
                linear_solve3 = lambda op, v, t_op: linear_solve2(
                    (op, v), (t_op, t_vec)
                )
                jnp_solve3 = lambda mat, v, t_mat: jnp_solve2((mat, v), (t_mat, t_vec))
            elif mode == {"op", "vec", "t_vec"}:
                linear_solve3 = lambda op, v, t_v: linear_solve2(
                    (op, v), (t_operator, t_v)
                )
                jnp_solve3 = lambda mat, v, t_v: jnp_solve2((mat, v), (t_matrix, t_v))
            elif mode == {"vec", "t_op", "t_vec"}:
                linear_solve3 = lambda v, t_op, t_v: linear_solve2(
                    (operator, v), (t_op, t_v)
                )
                jnp_solve3 = lambda v, t_mat, t_v: jnp_solve2((matrix, v), (t_mat, t_v))
            elif mode == {"op", "vec", "t_op", "t_vec"}:
                linear_solve3 = lambda op, v, t_op, t_v: linear_solve2(
                    (op, v), (t_op, t_v)
                )
                jnp_solve3 = lambda mat, v, t_mat, t_v: jnp_solve2(
                    (mat, v), (t_mat, t_v)
                )
            else:
                assert False

            linear_solve3 = ft.partial(eqx.filter_jvp, linear_solve3)
            linear_solve3 = eqx.filter_jit(linear_solve3)
            jnp_solve3 = ft.partial(eqx.filter_jvp, jnp_solve3)
            jnp_solve3 = eqx.filter_jit(jnp_solve3)

            (primal, tangent), (jnp_primal, jnp_tangent) = _make_primal_tangents(mode)
            (out, t_out), (minus_out, tt_out) = linear_solve3(primal, tangent)
            (true_out, true_t_out), (minus_true_out, true_tt_out) = jnp_solve3(
                jnp_primal, jnp_tangent
            )

            assert tree_allclose(out, true_out, atol=1e-4)
            assert tree_allclose(t_out, true_t_out, atol=1e-4)
            assert tree_allclose(tt_out, true_tt_out, atol=1e-4)
            assert tree_allclose(minus_out, minus_true_out, atol=1e-4)
