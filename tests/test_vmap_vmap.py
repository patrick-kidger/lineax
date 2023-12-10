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
import equinox.internal as eqxi
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


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("solver, tags, pseudoinverse", solvers_tags_pseudoinverse)
@pytest.mark.parametrize("use_state", (True, False))
@pytest.mark.parametrize(
    "make_matrix",
    (
        construct_matrix,
        construct_singular_matrix,
    ),
)
@pytest.mark.parametrize("dtype", (jnp.float64,))
def test_vmap_vmap(
    getkey, make_operator, solver, tags, pseudoinverse, use_state, make_matrix, dtype
):
    if (make_matrix is construct_matrix) or pseudoinverse:
        # combinations with nontrivial application across both vmaps
        axes = [
            (eqx.if_array(0), eqx.if_array(0), None, None),
            (None, None, 0, 0),
            (eqx.if_array(0), eqx.if_array(0), None, 0),
            (eqx.if_array(0), eqx.if_array(0), 0, 0),
            (None, eqx.if_array(0), 0, 0),
        ]

        for vmap2_op, vmap1_op, vmap2_vec, vmap1_vec in axes:
            if vmap1_op is not None:
                axis_size1 = 10
                out_axis1 = eqx.if_array(0)
            else:
                axis_size1 = None
                out_axis1 = None

            if vmap2_op is not None:
                axis_size2 = 10
                out_axis2 = eqx.if_array(0)
            else:
                axis_size2 = None
                out_axis2 = None

            (matrix,) = eqx.filter_vmap(
                eqx.filter_vmap(
                    lambda getkey, solver, tags: make_matrix(
                        getkey, solver, tags, dtype=dtype
                    ),
                    axis_size=axis_size1,
                    out_axes=out_axis1,
                ),
                axis_size=axis_size2,
                out_axes=out_axis2,
            )(getkey, solver, tags)

            if vmap1_op is not None:
                if vmap2_op is not None:
                    _, _, out_size, _ = matrix.shape
                else:
                    _, out_size, _ = matrix.shape
            else:
                out_size, _ = matrix.shape

            if vmap1_vec is None:
                vec = jr.normal(getkey(), (out_size,), dtype=dtype)
            elif (vmap1_vec is not None) and (vmap2_vec is None):
                vec = jr.normal(getkey(), (10, out_size), dtype=dtype)
            else:
                vec = jr.normal(getkey(), (10, 10, out_size), dtype=dtype)

            make_op = ft.partial(make_operator, getkey)
            operator = eqx.filter_vmap(
                eqx.filter_vmap(
                    make_op,
                    in_axes=vmap1_op,
                    out_axes=out_axis1,
                ),
                in_axes=vmap2_op,
                out_axes=out_axis2,
            )(matrix, tags)

            if use_state:

                def linear_solve(operator, vector):
                    state = solver.init(operator, options={})
                    return lx.linear_solve(operator, vector, state=state, solver=solver)

            else:

                def linear_solve(operator, vector):
                    return lx.linear_solve(operator, vector, solver)

            as_matrix_vmapped = eqx.filter_vmap(
                eqx.filter_vmap(
                    lambda x: x.as_matrix(),
                    in_axes=vmap1_op,
                    out_axes=eqxi.if_mapped(0),
                ),
                in_axes=vmap2_op,
                out_axes=eqxi.if_mapped(0),
            )(operator)

            vmap1_axes = (vmap1_op, vmap1_vec)
            vmap2_axes = (vmap2_op, vmap2_vec)

            result = eqx.filter_vmap(
                eqx.filter_vmap(linear_solve, in_axes=vmap1_axes), in_axes=vmap2_axes
            )(operator, vec).value

            solve_with = lambda x: eqx.filter_vmap(
                eqx.filter_vmap(x, in_axes=vmap1_axes), in_axes=vmap2_axes
            )(as_matrix_vmapped, vec)

            if make_matrix is construct_singular_matrix:
                true_result, _, _, _ = solve_with(jnp.linalg.lstsq)  # pyright: ignore
            else:
                true_result = solve_with(jnp.linalg.solve)  # pyright: ignore
            assert tree_allclose(result, true_result, rtol=1e-3)
