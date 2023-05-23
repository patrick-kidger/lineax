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

import contextlib
import functools as ft
import random
from typing import cast

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.core
import jax.flatten_util as jfu
import jax.interpreters.partial_eval as pe
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from jaxtyping import PyTree

import lineax as lx

from .helpers import (
    construct_matrix,
    finite_difference_jvp,
    has_tag,
    make_diagonal_operator,
    make_jac_operator,
    make_matrix_operator,
    make_operators,
    shaped_allclose,
)


if jax.config.jax_enable_x64:  # pyright: ignore
    tol = 1e-12
else:
    tol = 1e-6
_solvers_tags_pseudoinverse = [
    (lx.AutoLinearSolver(well_posed=True), (), False),
    (lx.AutoLinearSolver(well_posed=False), (), True),
    (lx.Triangular(), lx.lower_triangular_tag, False),
    (lx.Triangular(), lx.upper_triangular_tag, False),
    (lx.Triangular(), (lx.lower_triangular_tag, lx.unit_diagonal_tag), False),
    (lx.Triangular(), (lx.upper_triangular_tag, lx.unit_diagonal_tag), False),
    (lx.Diagonal(), lx.diagonal_tag, False),
    (lx.Diagonal(), (lx.diagonal_tag, lx.unit_diagonal_tag), False),
    (lx.Tridiagonal(), lx.tridiagonal_tag, False),
    (lx.LU(), (), False),
    (lx.QR(), (), False),
    (lx.SVD(), (), True),
    (lx.BiCGStab(rtol=tol, atol=tol), (), False),
    (lx.GMRES(rtol=tol, atol=tol), (), False),
    (lx.NormalCG(rtol=tol, atol=tol), (), False),
    (lx.CG(rtol=tol, atol=tol), lx.positive_semidefinite_tag, False),
    (lx.CG(rtol=tol, atol=tol), lx.negative_semidefinite_tag, False),
    (lx.NormalCG(rtol=tol, atol=tol), lx.negative_semidefinite_tag, False),
    (lx.Cholesky(), lx.positive_semidefinite_tag, False),
    (lx.Cholesky(), lx.negative_semidefinite_tag, False),
]
_solvers_tags = [(a, b) for a, b, _ in _solvers_tags_pseudoinverse]
_solvers = [a for a, _, _ in _solvers_tags_pseudoinverse]
_pseudosolvers_tags = [(a, b) for a, b, c in _solvers_tags_pseudoinverse if c]


def _transpose(operator, matrix):
    return operator.T, matrix.T


def _linearise(operator, matrix):
    return lx.linearise(operator), matrix


def _materialise(operator, matrix):
    return lx.materialise(operator), matrix


_ops = (lambda x, y: (x, y), _transpose, _linearise, _materialise)


def _params(only_pseudo):
    for make_operator in make_operators:
        for solver, tags, pseudoinverse in _solvers_tags_pseudoinverse:
            if only_pseudo and not pseudoinverse:
                continue
            if make_operator is make_diagonal_operator and not has_tag(
                tags, lx.diagonal_tag
            ):
                continue
            yield make_operator, solver, tags


def _construct_singular_matrix(getkey, solver, tags, num=1):
    matrices = construct_matrix(getkey, solver, tags, num)
    if isinstance(solver, (lx.Diagonal, lx.CG, lx.BiCGStab, lx.GMRES)):
        return tuple(matrix.at[0, :].set(0) for matrix in matrices)
    else:
        version = random.choice([0, 1, 2])
        if version == 0:
            return tuple(matrix.at[0, :].set(0) for matrix in matrices)
        elif version == 1:
            return tuple(matrix[1:, :] for matrix in matrices)
        else:
            return tuple(matrix[:, 1:] for matrix in matrices)


@pytest.mark.parametrize("make_operator,solver,tags", _params(only_pseudo=False))
@pytest.mark.parametrize("ops", _ops)
def test_small_wellposed(make_operator, solver, tags, ops, getkey):
    if jax.config.jax_enable_x64:  # pyright: ignore
        tol = 1e-10
    else:
        tol = 1e-4
    (matrix,) = construct_matrix(getkey, solver, tags)
    operator = make_operator(matrix, tags)
    operator, matrix = ops(operator, matrix)
    assert shaped_allclose(operator.as_matrix(), matrix, rtol=tol, atol=tol)
    out_size, _ = matrix.shape
    true_x = jr.normal(getkey(), (out_size,))
    b = matrix @ true_x
    x = lx.linear_solve(operator, b, solver=solver).value
    jax_x = jnp.linalg.solve(matrix, b)
    assert shaped_allclose(x, true_x, atol=tol, rtol=tol)
    assert shaped_allclose(x, jax_x, atol=tol, rtol=tol)


@pytest.mark.parametrize("solver", _solvers)
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
        assert shaped_allclose(lx_x.value, true_x, atol=tol, rtol=tol)


@pytest.mark.parametrize("make_operator,solver,tags", _params(only_pseudo=True))
@pytest.mark.parametrize("ops", _ops)
def test_small_singular(make_operator, solver, tags, ops, getkey):
    if jax.config.jax_enable_x64:  # pyright: ignore
        tol = 1e-10
    else:
        tol = 1e-4
    (matrix,) = _construct_singular_matrix(getkey, solver, tags)
    operator = make_operator(matrix, tags)
    operator, matrix = ops(operator, matrix)
    assert shaped_allclose(operator.as_matrix(), matrix, rtol=tol, atol=tol)
    out_size, in_size = matrix.shape
    true_x = jr.normal(getkey(), (in_size,))
    b = matrix @ true_x
    x = lx.linear_solve(operator, b, solver=solver, throw=False).value
    jax_x, *_ = jnp.linalg.lstsq(matrix, b)
    assert shaped_allclose(x, jax_x, atol=tol, rtol=tol)


def test_bicgstab_breakdown(getkey):
    if jax.config.jax_enable_x64:  # pyright: ignore
        tol = 1e-10
    else:
        tol = 1e-4
    solver = lx.GMRES(atol=tol, rtol=tol, restart=2)

    matrix = jr.normal(jr.PRNGKey(0), (100, 100))
    true_x = jr.normal(jr.PRNGKey(0), (100,))
    b = matrix @ true_x
    operator = lx.MatrixLinearOperator(matrix)

    # result != 0 implies lineax reported failure
    lx_soln = lx.linear_solve(operator, b, solver, throw=False)

    assert jnp.all(lx_soln.result != lx.RESULTS.successful)


def test_gmres_stagnation_or_breakdown(getkey):
    if jax.config.jax_enable_x64:  # pyright: ignore
        tol = 1e-10
    else:
        tol = 1e-4
    solver = lx.GMRES(atol=tol, rtol=tol, restart=2)

    matrix = jnp.array(
        [
            [0.15892892, 0.05884365, -0.60427412, 0.1891916],
            [-1.5484863, 0.93608822, 1.94888868, 1.37069667],
            [0.62687318, -0.13996738, -0.6824359, 0.30975754],
            [-0.67428635, 1.52372255, -0.88277754, 0.69633816],
        ]
    )
    true_x = jnp.array([0.51383273, 1.72983427, -0.43251078, -1.11764668])
    b = matrix @ true_x
    operator = lx.MatrixLinearOperator(matrix)

    # result != 0 implies lineax reported failure
    lx_soln = lx.linear_solve(operator, b, solver, throw=False)

    assert jnp.all(lx_soln.result != lx.RESULTS.successful)


def test_gmres_large_dense(getkey):
    if jax.config.jax_enable_x64:  # pyright: ignore
        tol = 1e-10
    else:
        tol = 1e-4
    solver = lx.GMRES(atol=tol, rtol=tol, restart=100)

    matrix = jr.normal(getkey(), (100, 100))
    operator = lx.MatrixLinearOperator(matrix)
    true_x = jr.normal(getkey(), (100,))
    b = matrix @ true_x

    lx_soln = lx.linear_solve(operator, b, solver).value

    assert shaped_allclose(lx_soln, true_x, atol=tol, rtol=tol)


def test_nontrivial_pytree_operator():
    x = [[1, 5.0], [jnp.array(-2), jnp.array(-2.0)]]
    y = [3, 4]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct)
    out = lx.linear_solve(operator, y).value
    true_out = [jnp.array(-3.25), jnp.array(1.25)]
    assert shaped_allclose(out, true_out)


@pytest.mark.parametrize("solver", (lx.LU(), lx.QR(), lx.SVD()))
def test_mixed_dtypes(solver):
    f32 = lambda x: jnp.array(x, dtype=jnp.float32)
    f64 = lambda x: jnp.array(x, dtype=jnp.float64)
    x = [[f32(1), f64(5)], [f32(-2), f64(-2)]]
    y = [f64(3), f64(4)]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct)
    out = lx.linear_solve(operator, y, solver=solver).value
    true_out = [f32(-3.25), f64(1.25)]
    assert shaped_allclose(out, true_out)


def test_mixed_dtypes_triangular():
    f32 = lambda x: jnp.array(x, dtype=jnp.float32)
    f64 = lambda x: jnp.array(x, dtype=jnp.float64)
    x = [[f32(1), f64(0)], [f32(-2), f64(-2)]]
    y = [f64(3), f64(4)]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct, lx.lower_triangular_tag)
    out = lx.linear_solve(operator, y, solver=lx.Triangular()).value
    true_out = [f32(3), f64(-5)]
    assert shaped_allclose(out, true_out)


@pytest.mark.parametrize(
    "solver", (lx.AutoLinearSolver(well_posed=None), lx.QR(), lx.SVD())
)
def test_nonsquare_pytree_operator1(solver):
    x = [[1, 5.0, jnp.array(-1.0)], [jnp.array(-2), jnp.array(-2.0), 3.0]]
    y = [3.0, 4]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct)
    out = lx.linear_solve(operator, y, solver=solver).value
    matrix = jnp.array([[1.0, 5.0, -1.0], [-2.0, -2.0, 3.0]])
    true_out, _, _, _ = jnp.linalg.lstsq(matrix, jnp.array(y))
    true_out = [true_out[0], true_out[1], true_out[2]]
    assert shaped_allclose(out, true_out)


@pytest.mark.parametrize(
    "solver", (lx.AutoLinearSolver(well_posed=None), lx.QR(), lx.SVD())
)
def test_nonsquare_pytree_operator2(solver):
    x = [[1, jnp.array(-2)], [5.0, jnp.array(-2.0)], [jnp.array(-1.0), 3.0]]
    y = [3.0, 4, 5.0]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct)
    out = lx.linear_solve(operator, y, solver=solver).value
    matrix = jnp.array([[1.0, -2.0], [5.0, -2.0], [-1.0, 3.0]])
    true_out, _, _, _ = jnp.linalg.lstsq(matrix, jnp.array(y))
    true_out = [true_out[0], true_out[1]]
    assert shaped_allclose(out, true_out)


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("solver, tags, pseudoinverse", _solvers_tags_pseudoinverse)
@pytest.mark.parametrize("use_state", (True, False))
@pytest.mark.parametrize(
    "make_matrix",
    (
        construct_matrix,
        _construct_singular_matrix,
    ),
)
def test_vmap(
    getkey, make_operator, solver, tags, pseudoinverse, use_state, make_matrix
):
    if (make_matrix is construct_matrix) or pseudoinverse:

        def wrap_solve(matrix, vector):
            operator = make_operator(matrix, tags)
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
                make_matrix, axis_size=axis_size, out_axes=out_axes
            )(getkey, solver, tags)
            out_dim = matrix.shape[-2]

            if vec_axis is None:
                vec = jr.normal(getkey(), (out_dim,))
            else:
                vec = jr.normal(getkey(), (10, out_dim))

            jax_result, _, _, _ = eqx.filter_vmap(
                jnp.linalg.lstsq, in_axes=(op_axis, vec_axis)
            )(matrix, vec)
            lx_result = eqx.filter_vmap(wrap_solve, in_axes=(op_axis, vec_axis))(
                matrix, vec
            )
            assert shaped_allclose(lx_result, jax_result)


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("solver, tags, pseudoinverse", _solvers_tags_pseudoinverse)
@pytest.mark.parametrize("use_state", (True, False))
@pytest.mark.parametrize(
    "make_matrix",
    (
        construct_matrix,
        _construct_singular_matrix,
    ),
)
def test_jvp(
    getkey, solver, tags, pseudoinverse, make_operator, use_state, make_matrix
):
    t_tags = (None,) * len(tags) if isinstance(tags, tuple) else None

    if (make_matrix is construct_matrix) or pseudoinverse:
        matrix, t_matrix = make_matrix(getkey, solver, tags, num=2)

        out_size, _ = matrix.shape
        vec = jr.normal(getkey(), (out_size,))
        t_vec = jr.normal(getkey(), (out_size,))

        if has_tag(tags, lx.unit_diagonal_tag):
            # For all the other tags, A + εB with A, B \in {matrices satisfying the tag}
            # still satisfies the tag itself.
            # This is the exception.
            t_matrix.at[jnp.arange(3), jnp.arange(3)].set(0)

        operator, t_operator = eqx.filter_jvp(
            make_operator, (matrix, tags), (t_matrix, t_tags)
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
            lambda op: jnp.linalg.lstsq(op, vec), (matrix,), (t_matrix,)
        )
        (expected_op_vec_out, *_), (t_expected_op_vec_out, *_) = eqx.filter_jvp(
            jnp.linalg.lstsq, (matrix, vec), (t_matrix, t_vec)
        )

        # Work around JAX issue #14868.
        if jnp.any(jnp.isnan(t_expected_op_out)):
            _, (t_expected_op_out, *_) = finite_difference_jvp(
                lambda op: jnp.linalg.lstsq(op, vec), (matrix,), (t_matrix,)
            )
        if jnp.any(jnp.isnan(t_expected_op_vec_out)):
            _, (t_expected_op_vec_out, *_) = finite_difference_jvp(
                jnp.linalg.lstsq, (matrix, vec), (t_matrix, t_vec)
            )

        pinv_matrix = jnp.linalg.pinv(matrix)
        expected_vec_out = pinv_matrix @ vec
        assert shaped_allclose(vec_out, expected_vec_out)
        assert shaped_allclose(op_out, expected_op_out)
        assert shaped_allclose(op_vec_out, expected_op_vec_out)

        t_expected_vec_out = pinv_matrix @ t_vec
        assert shaped_allclose(
            matrix @ t_vec_out, matrix @ t_expected_vec_out, rtol=1e-3
        )
        assert shaped_allclose(t_op_out, t_expected_op_out, rtol=1e-3)
        assert shaped_allclose(t_op_vec_out, t_expected_op_vec_out, rtol=1e-3)


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("solver, tags, pseudoinverse", _solvers_tags_pseudoinverse)
@pytest.mark.parametrize("use_state", (True, False))
@pytest.mark.parametrize(
    "make_matrix",
    (
        construct_matrix,
        _construct_singular_matrix,
    ),
)
def test_vmap_vmap(
    getkey, make_operator, solver, tags, pseudoinverse, use_state, make_matrix
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
                eqx.filter_vmap(make_matrix, axis_size=axis_size1, out_axes=out_axis1),
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
                vec = jr.normal(getkey(), (out_size,))
            elif (vmap1_vec is not None) and (vmap2_vec is None):
                vec = jr.normal(getkey(), (10, out_size))
            else:
                vec = jr.normal(getkey(), (10, 10, out_size))

            operator = eqx.filter_vmap(
                eqx.filter_vmap(
                    make_operator,
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

            if make_matrix is _construct_singular_matrix:
                true_result, _, _, _ = solve_with(jnp.linalg.lstsq)
            else:
                true_result = solve_with(jnp.linalg.solve)
            assert shaped_allclose(result, true_result, rtol=1e-3)


@pytest.mark.parametrize("solver, tags, pseudoinverse", _solvers_tags_pseudoinverse)
@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("use_state", (True, False))
@pytest.mark.parametrize(
    "make_matrix",
    (
        construct_matrix,
        _construct_singular_matrix,
    ),
)
def test_vmap_jvp(
    getkey, solver, tags, make_operator, pseudoinverse, use_state, make_matrix
):
    if (make_matrix is construct_matrix) or pseudoinverse:
        t_tags = (None,) * len(tags) if isinstance(tags, tuple) else None
        if pseudoinverse:
            jnp_solve1 = lambda mat, vec: jnp.linalg.lstsq(mat, vec)[0]
        else:
            jnp_solve1 = jnp.linalg.solve
        if use_state:

            def linear_solve1(operator, vector):
                state = solver.init(operator, options={})
                state_dynamic, state_static = eqx.partition(state, eqx.is_inexact_array)
                state_dynamic = lax.stop_gradient(state_dynamic)
                state = eqx.combine(state_dynamic, state_static)

                return lx.linear_solve(operator, vector, state=state, solver=solver)

        else:
            linear_solve1 = ft.partial(lx.linear_solve, solver=solver)

        for mode in ("vec", "op", "op_vec"):
            if "op" in mode:
                axis_size = 10
                out_axes = eqx.if_array(0)
            else:
                axis_size = None
                out_axes = None

            def _make():
                matrix, t_matrix = make_matrix(getkey, solver, tags, num=2)
                operator, t_operator = eqx.filter_jvp(
                    make_operator, (matrix, tags), (t_matrix, t_tags)
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
                vec = jr.normal(getkey(), (10, out_size))
                t_vec = jr.normal(getkey(), (10, out_size))
            else:
                vec = jr.normal(getkey(), (out_size,))
                t_vec = jr.normal(getkey(), (out_size,))

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
                assert shaped_allclose(out.value, true_out, atol=1e-4)
                assert shaped_allclose(t_out.value, true_t_out, atol=1e-4)


@pytest.mark.parametrize("solver, tags, pseudoinverse", _solvers_tags_pseudoinverse)
@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("use_state", (True, False))
@pytest.mark.parametrize(
    "make_matrix",
    (
        construct_matrix,
        _construct_singular_matrix,
    ),
)
def test_jvp_jvp(
    getkey, solver, tags, pseudoinverse, make_operator, use_state, make_matrix
):
    t_tags = (None,) * len(tags) if isinstance(tags, tuple) else None
    if (make_matrix is construct_matrix) or pseudoinverse:
        matrix, t_matrix, tt_matrix, tt_t_matrix = construct_matrix(
            getkey, solver, tags, num=4
        )

        t_make_operator = lambda p, t_p: eqx.filter_jvp(
            make_operator, (p, tags), (t_p, t_tags)
        )
        tt_make_operator = lambda p, t_p, tt_p, tt_t_p: eqx.filter_jvp(
            t_make_operator, (p, t_p), (tt_p, tt_t_p)
        )
        (operator, t_operator), (tt_operator, tt_t_operator) = tt_make_operator(
            matrix, t_matrix, tt_matrix, tt_t_matrix
        )

        out_size, _ = matrix.shape
        vec = jr.normal(getkey(), (out_size,))
        t_vec = jr.normal(getkey(), (out_size,))
        tt_vec = jr.normal(getkey(), (out_size,))
        tt_t_vec = jr.normal(getkey(), (out_size,))

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
            jnp_solve1 = lambda mat, vec: jnp.linalg.lstsq(mat, vec)[0]
        else:
            jnp_solve1 = jnp.linalg.solve

        linear_solve2 = ft.partial(eqx.filter_jvp, linear_solve1)
        jnp_solve2 = ft.partial(eqx.filter_jvp, jnp_solve1)

        def _make_primal_tangents(mode):
            lx_args = ([], [], operator, t_operator, tt_operator, tt_t_operator)
            jnp_args = ([], [], matrix, t_matrix, tt_matrix, tt_t_matrix)
            for (primals, ttangents, op, t_op, tt_op, tt_t_op) in (lx_args, jnp_args):
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

            assert shaped_allclose(out, true_out, atol=1e-4)
            assert shaped_allclose(t_out, true_t_out, atol=1e-4)
            assert shaped_allclose(tt_out, true_tt_out, atol=1e-4)
            assert shaped_allclose(minus_out, minus_true_out, atol=1e-4)


@pytest.mark.parametrize("full_rank", (True, False))
@pytest.mark.parametrize("jvp", (False, True))
@pytest.mark.parametrize("wide", (False, True))
def test_qr_nonsquare_mat_vec(full_rank, jvp, wide, getkey):
    if wide:
        out_size = 3
        in_size = 6
    else:
        out_size = 6
        in_size = 3
    matrix = jr.normal(getkey(), (out_size, in_size))
    if full_rank:
        context = contextlib.nullcontext()
    else:
        context = pytest.raises(Exception)
        if wide:
            matrix = matrix.at[:, 2:].set(0)
        else:
            matrix = matrix.at[2:, :].set(0)
    vector = jr.normal(getkey(), (out_size,))
    lx_solve = lambda mat, vec: lx.linear_solve(
        lx.MatrixLinearOperator(mat), vec, lx.QR()
    ).value
    jnp_solve = lambda mat, vec: jnp.linalg.lstsq(mat, vec)[0]
    if jvp:
        lx_solve = eqx.filter_jit(ft.partial(eqx.filter_jvp, lx_solve))
        jnp_solve = eqx.filter_jit(ft.partial(finite_difference_jvp, jnp_solve))
        t_matrix = jr.normal(getkey(), (out_size, in_size))
        t_vector = jr.normal(getkey(), (out_size,))
        args = ((matrix, vector), (t_matrix, t_vector))
    else:
        args = (matrix, vector)
    with context:
        x = lx_solve(*args)  # pyright: ignore
    if full_rank:
        true_x = jnp_solve(*args)
        assert shaped_allclose(x, true_x, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("full_rank", (True, False))
@pytest.mark.parametrize("jvp", (False, True))
@pytest.mark.parametrize("wide", (False, True))
def test_qr_nonsquare_vec(full_rank, jvp, wide, getkey):
    if wide:
        out_size = 3
        in_size = 6
    else:
        out_size = 6
        in_size = 3
    matrix = jr.normal(getkey(), (out_size, in_size))
    if full_rank:
        context = contextlib.nullcontext()
    else:
        context = pytest.raises(Exception)
        if wide:
            matrix = matrix.at[:, 2:].set(0)
        else:
            matrix = matrix.at[2:, :].set(0)
    vector = jr.normal(getkey(), (out_size,))
    lx_solve = lambda vec: lx.linear_solve(
        lx.MatrixLinearOperator(matrix), vec, lx.QR()
    ).value
    jnp_solve = lambda vec: jnp.linalg.lstsq(matrix, vec)[0]
    if jvp:
        lx_solve = eqx.filter_jit(ft.partial(eqx.filter_jvp, lx_solve))
        jnp_solve = eqx.filter_jit(ft.partial(finite_difference_jvp, jnp_solve))
        t_vector = jr.normal(getkey(), (out_size,))
        args = ((vector,), (t_vector,))
    else:
        args = (vector,)
    with context:
        x = lx_solve(*args)  # pyright: ignore
    if full_rank:
        true_x = jnp_solve(*args)
        assert shaped_allclose(x, true_x, atol=1e-4, rtol=1e-4)


_cg_solvers = (
    (lx.CG(rtol=tol, atol=tol), lx.positive_semidefinite_tag),
    (lx.CG(rtol=tol, atol=tol, max_steps=512), lx.negative_semidefinite_tag),
    (lx.GMRES(rtol=tol, atol=tol), ()),
    (lx.BiCGStab(rtol=tol, atol=tol), ()),
)


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("solver, tags", _cg_solvers)
@pytest.mark.parametrize("use_state", (False, True))
def test_cg_inf(getkey, solver, tags, use_state, make_operator):
    (matrix,) = _construct_singular_matrix(getkey, solver, tags)
    operator = make_operator(matrix, tags)

    out_size, _ = matrix.shape
    vec = jr.normal(getkey(), (out_size,))

    if use_state:
        state = solver.init(operator, options={})
        linear_solve = ft.partial(lx.linear_solve, state=state)
    else:
        linear_solve = lx.linear_solve

    with pytest.raises(Exception):
        linear_solve(operator, vec, solver)


# Like `jax.linear_transpose`, but performs DCE first. See JAX issue #15660.
def _dce_linear_transpose(fn, *primals):
    jaxpr, struct = cast(
        tuple[jax.core.ClosedJaxpr, PyTree[jax.ShapeDtypeStruct]],
        jax.make_jaxpr(fn, return_shape=True)(*primals),
    )
    dce_jaxpr, used_inputs = pe.dce_jaxpr(jaxpr.jaxpr, [True] * len(jaxpr.out_avals))
    dce_fn = jax.core.jaxpr_as_fun(jax.core.ClosedJaxpr(dce_jaxpr, jaxpr.consts))
    flat_primals, treedef_primals = jtu.tree_flatten(primals)
    assert len(flat_primals) == len(used_inputs)
    dce_primals = [p for p, x in zip(flat_primals, used_inputs) if x]
    dce_fn_transpose = jax.linear_transpose(dce_fn, *dce_primals)  # pyright: ignore

    def fn_transpose(cotangents_out):
        flat_cotangents_out, treedef_contangents_out = jtu.tree_flatten(cotangents_out)
        treedef_struct = jtu.tree_structure(struct)
        assert treedef_struct == treedef_contangents_out
        del treedef_struct, treedef_contangents_out
        flat_cotangents = dce_fn_transpose(flat_cotangents_out)
        return jtu.tree_unflatten(treedef_primals, flat_cotangents)

    return fn_transpose


def _test_transpose(operator, out_vec, in_vec, solver):
    # throw=False is needed, as otherwise error-checking is a nonlinearity.
    solve_transpose = _dce_linear_transpose(
        lambda v: lx.linear_solve(operator, v, solver, throw=False).value, out_vec
    )
    (out,) = solve_transpose(in_vec)
    true_out = lx.linear_solve(operator.T, in_vec, solver).value
    assert shaped_allclose(out, true_out)


@pytest.mark.parametrize("make_operator,solver,tags", _params(only_pseudo=False))
def test_transpose(make_operator, solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    operator = make_operator(matrix, tags)
    out_size, in_size = matrix.shape
    out_vec = jr.normal(getkey(), (out_size,))
    in_vec = jr.normal(getkey(), (in_size,))
    solver = lx.AutoLinearSolver(well_posed=True)
    _test_transpose(operator, out_vec, in_vec, solver)


def test_pytree_transpose():
    a = jnp.array
    pytree = [[a(1), a(2), a(3)], [a(4), a(5), a(6)]]
    output_structure = jax.eval_shape(lambda: [1, 2])
    operator = lx.PyTreeLinearOperator(pytree, output_structure)
    out_vec = [a(1.0), a(2.0)]
    in_vec = [a(1.0), 2.0, 3.0]
    solver = lx.AutoLinearSolver(well_posed=False)
    _test_transpose(operator, out_vec, in_vec, solver)


def _test_vec_grad(operator, vec, solver):
    @jax.grad
    def run(vec):
        out = lx.linear_solve(operator, vec, solver).value
        return jnp.sum(jfu.ravel_pytree(out)[0])

    @jax.grad
    def true_run(vec):
        out = jnp.linalg.pinv(operator.as_matrix()) @ jfu.ravel_pytree(vec)[0]
        return jnp.sum(out)

    out = run(vec)
    true_out = true_run(vec)
    assert shaped_allclose(out, true_out)


def _test_both_grad(operator, vec, solver):
    @eqx.filter_grad
    def run(operator__vec):
        operator, vec = operator__vec
        out = lx.linear_solve(operator, vec, solver).value
        return jnp.sum(jfu.ravel_pytree(out)[0])

    @eqx.filter_grad
    def true_run(operator__vec):
        operator, vec = operator__vec
        out = jnp.linalg.pinv(operator.as_matrix()) @ jfu.ravel_pytree(vec)[0]
        return jnp.sum(out)

    out = run((operator, vec))
    true_out = true_run((operator, vec))
    assert shaped_allclose(out, true_out)


@pytest.mark.parametrize("make_operator,solver,tags", _params(only_pseudo=False))
def test_grad(make_operator, solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    operator = make_operator(matrix, tags)
    out_size, _ = matrix.shape
    vec = jr.normal(getkey(), (out_size,))
    _test_vec_grad(operator, vec, solver)
    _test_both_grad(operator, vec, solver)


def test_pytree_grad(getkey):
    a = jnp.array
    pytree = [[a(1), a(2), a(3)], [a(4), a(5), a(6)]]
    output_structure = jax.eval_shape(lambda: [1, 2])
    operator = lx.PyTreeLinearOperator(pytree, output_structure)
    vec = [a(1.0), a(2.0)]
    solver = lx.AutoLinearSolver(well_posed=False)
    _test_vec_grad(operator, vec, solver)
    _test_both_grad(operator, vec, solver)


def test_ad_closure_function_linear_operator(getkey):
    def f(x, z):
        def fn(y):
            return x * y

        op = lx.FunctionLinearOperator(fn, jax.eval_shape(lambda: z))
        sol = lx.linear_solve(op, z).value
        return jnp.sum(sol), sol

    x = jr.normal(getkey(), (3,))
    x = jnp.where(jnp.abs(x) < 1e-6, 0.7, x)
    z = jr.normal(getkey(), (3,))
    grad, sol = jax.grad(f, has_aux=True)(x, z)
    assert shaped_allclose(grad, -z / (x**2))
    assert shaped_allclose(sol, z / x)
