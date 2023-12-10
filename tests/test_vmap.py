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

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import lineax as lx

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
def test_vmap(
    getkey, make_operator, solver, tags, pseudoinverse, use_state, make_matrix, dtype
):
    if (make_matrix is construct_matrix) or pseudoinverse:

        def wrap_solve(matrix, vector):
            operator = make_operator(getkey, matrix, tags)
            if use_state:
                state = solver.init(operator, options={})
                return lx.linear_solve(operator, vector, solver, state=state).value
            else:
                return lx.linear_solve(operator, vector, solver).value

        for op_axis, vec_axis in (
            (None, 0),
            (eqx.if_array(0), None),
            (eqx.if_array(0), 0),
        ):
            if op_axis is None:
                axis_size = None
                out_axes = None
            else:
                axis_size = 10
                out_axes = eqx.if_array(0)

            (matrix,) = eqx.filter_vmap(
                lambda getkey, solver, tags: make_matrix(
                    getkey, solver, tags, dtype=dtype
                ),
                axis_size=axis_size,
                out_axes=out_axes,
            )(getkey, solver, tags)
            out_dim = matrix.shape[-2]

            if vec_axis is None:
                vec = jr.normal(getkey(), (out_dim,), dtype=dtype)
            else:
                vec = jr.normal(getkey(), (10, out_dim), dtype=dtype)

            jax_result, _, _, _ = eqx.filter_vmap(
                jnp.linalg.lstsq, in_axes=(op_axis, vec_axis)  # pyright: ignore
            )(matrix, vec)
            lx_result = eqx.filter_vmap(wrap_solve, in_axes=(op_axis, vec_axis))(
                matrix, vec
            )
            assert tree_allclose(lx_result, jax_result)
