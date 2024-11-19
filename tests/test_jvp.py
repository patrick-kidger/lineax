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
from lineax._misc import complex_to_real_dtype

from .helpers import (
    construct_matrix,
    construct_singular_matrix,
    finite_difference_jvp,
    has_tag,
    make_jac_operator,
    make_matrix_operator,
    make_real_function_operator,
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
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_jvp(
    getkey, solver, tags, pseudoinverse, make_operator, use_state, make_matrix, dtype
):
    t_tags = (None,) * len(tags) if isinstance(tags, tuple) else None

    if (make_matrix is construct_matrix) or pseudoinverse:
        matrix, t_matrix = make_matrix(getkey, solver, tags, num=2, dtype=dtype)

        out_size, _ = matrix.shape
        vec = jr.normal(getkey(), (out_size,), dtype=dtype)
        t_vec = jr.normal(getkey(), (out_size,), dtype=dtype)

        if has_tag(tags, lx.unit_diagonal_tag):
            # For all the other tags, A + εB with A, B \in {matrices satisfying the tag}
            # still satisfies the tag itself.
            # This is the exception.
            t_matrix.at[jnp.arange(3), jnp.arange(3)].set(0)

        make_op = ft.partial(make_operator, getkey)
        operator, t_operator = eqx.filter_jvp(
            make_op, (matrix, tags), (t_matrix, t_tags)
        )

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
        (expected_op_out, *_), (t_expected_op_out, *_) = eqx.filter_jvp(
            lambda op: jnp.linalg.lstsq(op, vec),  # pyright: ignore
            (matrix,),
            (t_matrix,),
        )
        (expected_op_vec_out, *_), (t_expected_op_vec_out, *_) = eqx.filter_jvp(
            jnp.linalg.lstsq,
            (matrix, vec),
            (t_matrix, t_vec),  # pyright: ignore
        )

        # Work around JAX issue #14868.
        if jnp.any(jnp.isnan(t_expected_op_out)):
            _, (t_expected_op_out, *_) = finite_difference_jvp(
                lambda op: jnp.linalg.lstsq(op, vec),  # pyright: ignore
                (matrix,),
                (t_matrix,),
            )
        if jnp.any(jnp.isnan(t_expected_op_vec_out)):
            _, (t_expected_op_vec_out, *_) = finite_difference_jvp(
                jnp.linalg.lstsq,
                (matrix, vec),
                (t_matrix, t_vec),  # pyright: ignore
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


@pytest.mark.parametrize(
    "solver, tags, pseudoinverse",
    [stp for stp in solvers_tags_pseudoinverse if stp[-1]],
)  # only pseudoinverse
@pytest.mark.parametrize("use_state", (True, False))
@pytest.mark.parametrize(
    "make_matrix",
    (
        construct_matrix,
        construct_singular_matrix,
    ),
)
def test_jvp_c_to_r(getkey, solver, tags, pseudoinverse, use_state, make_matrix):
    t_tags = (None,) * len(tags) if isinstance(tags, tuple) else None

    matrix, t_matrix = make_matrix(getkey, solver, tags, num=2, dtype=jnp.complex128)

    out_size, in_size = matrix.shape
    out_dtype = complex_to_real_dtype(matrix.dtype)
    vec = jr.normal(getkey(), (out_size,), dtype=out_dtype)
    t_vec = jr.normal(getkey(), (out_size,), dtype=out_dtype)

    if has_tag(tags, lx.unit_diagonal_tag):
        # For all the other tags, A + εB with A, B \in {matrices satisfying the tag}
        # still satisfies the tag itself.
        # This is the exception.
        t_matrix.at[jnp.arange(3), jnp.arange(3)].set(0)

    make_op = ft.partial(make_real_function_operator, getkey)
    operator, t_operator = eqx.filter_jvp(make_op, (matrix, tags), (t_matrix, t_tags))

    if use_state:
        state = solver.init(operator, options={})
        linear_solve = ft.partial(lx.linear_solve, state=state)
    else:
        linear_solve = lx.linear_solve

    solve_vec_only = lambda v: linear_solve(operator, v, solver).value
    vec_out, t_vec_out = eqx.filter_jvp(solve_vec_only, (vec,), (t_vec,))

    solve_op_only = lambda op: linear_solve(op, vec, solver).value
    solve_op_vec = lambda op, v: linear_solve(op, v, solver).value

    op_out, t_op_out = eqx.filter_jvp(solve_op_only, (operator,), (t_operator,))
    op_vec_out, t_op_vec_out = eqx.filter_jvp(
        solve_op_vec,
        (operator, vec),
        (t_operator, t_vec),
    )
    (expected_op_out, *_), (t_expected_op_out, *_) = eqx.filter_jvp(
        lambda op: jnp.linalg.lstsq(
            jnp.concatenate([jnp.real(op), -jnp.imag(op)], axis=-1), vec
        ),  # pyright: ignore
        (matrix,),
        (t_matrix,),
    )
    (expected_op_vec_out, *_), (t_expected_op_vec_out, *_) = eqx.filter_jvp(
        lambda op, v: jnp.linalg.lstsq(
            jnp.concatenate([jnp.real(op), -jnp.imag(op)], axis=-1), v
        ),
        (matrix, vec),
        (t_matrix, t_vec),  # pyright: ignore
    )

    # Work around JAX issue #14868.
    if jnp.any(jnp.isnan(t_expected_op_out)):
        _, (t_expected_op_out, *_) = finite_difference_jvp(
            lambda op: jnp.linalg.lstsq(
                jnp.concatenate([jnp.real(op), -jnp.imag(op)], axis=-1), vec
            ),  # pyright: ignore
            (matrix,),
            (t_matrix,),
        )
    if jnp.any(jnp.isnan(t_expected_op_vec_out)):
        _, (t_expected_op_vec_out, *_) = finite_difference_jvp(
            lambda op, v: jnp.linalg.lstsq(
                jnp.concatenate([jnp.real(op), -jnp.imag(op)], axis=-1), v
            ),
            (matrix, vec),
            (t_matrix, t_vec),  # pyright: ignore
        )

    real_mat = jnp.concatenate([jnp.real(matrix), -jnp.imag(matrix)], axis=-1)
    pinv_matrix = jnp.linalg.pinv(real_mat)  # pyright: ignore
    expected_vec_out = pinv_matrix @ vec
    with jax.numpy_dtype_promotion("standard"):
        expected_complex_vec_out = (
            expected_vec_out[:in_size] + 1.0j * expected_vec_out[in_size:]
        )
        expected_complex_op_out = (
            expected_op_out[:in_size] + 1.0j * expected_op_out[in_size:]
        )
        expected_complex_op_vec_out = (
            expected_op_vec_out[:in_size] + 1.0j * expected_op_vec_out[in_size:]
        )

    assert tree_allclose(vec_out, expected_complex_vec_out)
    assert tree_allclose(op_out, expected_complex_op_out)
    assert tree_allclose(op_vec_out, expected_complex_op_vec_out)

    t_expected_vec_out = pinv_matrix @ t_vec

    with jax.numpy_dtype_promotion("standard"):
        t_expected_complex_vec_out = (
            t_expected_vec_out[:in_size] + 1.0j * t_expected_vec_out[in_size:]
        )
        t_expected_complex_op_out = (
            t_expected_op_out[:in_size] + 1.0j * t_expected_op_out[in_size:]
        )

        t_expected_complex_op_vec_out = (
            t_expected_op_vec_out[:in_size] + 1.0j * t_expected_op_vec_out[in_size:]
        )
    assert tree_allclose(
        matrix @ t_vec_out, matrix @ t_expected_complex_vec_out, rtol=1e-3
    )
    assert tree_allclose(t_op_out, t_expected_complex_op_out, rtol=1e-3)
    assert tree_allclose(t_op_vec_out, t_expected_complex_op_vec_out, rtol=1e-3)
