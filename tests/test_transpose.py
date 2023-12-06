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
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import lineax as lx

from .helpers import construct_matrix, params, shaped_allclose


class TestTranspose:
    @pytest.fixture(scope="class")
    def assert_transpose_fixture(_):
        @eqx.filter_jit
        def solve_transpose(operator, out_vec, in_vec, solver):
            return jax.linear_transpose(
                lambda v: lx.linear_solve(operator, v, solver).value, out_vec
            )(in_vec)

        def assert_transpose(operator, out_vec, in_vec, solver):
            (out,) = solve_transpose(operator, out_vec, in_vec, solver)
            true_out = lx.linear_solve(operator.T, in_vec, solver).value
            assert shaped_allclose(out, true_out)

        return assert_transpose

    @pytest.mark.parametrize("make_operator,solver,tags", params(only_pseudo=False))
    @pytest.mark.parametrize("dtype", (jnp.float64,))
    def test_transpose(
        _, make_operator, solver, tags, assert_transpose_fixture, dtype, getkey
    ):
        (matrix,) = construct_matrix(getkey, solver, tags, dtype=dtype)
        operator = make_operator(getkey, matrix, tags)
        out_size, in_size = matrix.shape
        out_vec = jr.normal(getkey(), (out_size,), dtype=dtype)
        in_vec = jr.normal(getkey(), (in_size,), dtype=dtype)
        solver = lx.AutoLinearSolver(well_posed=True)
        assert_transpose_fixture(operator, out_vec, in_vec, solver)

    def test_pytree_transpose(_, assert_transpose_fixture):  # pyright: ignore
        a = jnp.array
        pytree = [[a(1), a(2), a(3)], [a(4), a(5), a(6)]]
        output_structure = jax.eval_shape(lambda: [1, 2])
        operator = lx.PyTreeLinearOperator(pytree, output_structure)
        out_vec = [a(1.0), a(2.0)]
        in_vec = [a(1.0), 2.0, 3.0]
        solver = lx.AutoLinearSolver(well_posed=False)
        assert_transpose_fixture(operator, out_vec, in_vec, solver)
