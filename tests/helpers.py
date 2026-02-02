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
import math

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax as lx
import numpy as np
from equinox.internal import ω


@ft.cache
def _construct_matrix_impl(getkey, cond_cutoff, tags, size, dtype, i):
    del i  # used to break the cache
    while True:
        matrix = jr.normal(getkey(), (size, size), dtype=dtype)
        if has_tag(tags, lx.diagonal_tag):
            matrix = jnp.diag(jnp.diag(matrix))
        if has_tag(tags, lx.symmetric_tag):
            matrix = matrix + matrix.T
        if has_tag(tags, lx.lower_triangular_tag):
            matrix = jnp.tril(matrix)
        if has_tag(tags, lx.upper_triangular_tag):
            matrix = jnp.triu(matrix)
        if has_tag(tags, lx.unit_diagonal_tag):
            matrix = matrix.at[jnp.arange(size), jnp.arange(size)].set(1)
        if has_tag(tags, lx.tridiagonal_tag):
            diagonal = jnp.diag(jnp.diag(matrix))
            upper_diagonal = jnp.diag(jnp.diag(matrix, k=1), k=1)
            lower_diagonal = jnp.diag(jnp.diag(matrix, k=-1), k=-1)
            matrix = lower_diagonal + diagonal + upper_diagonal
        if has_tag(tags, lx.positive_semidefinite_tag):
            matrix = matrix @ matrix.T.conj()
        if has_tag(tags, lx.negative_semidefinite_tag):
            matrix = -matrix @ matrix.T.conj()
        if eqxi.unvmap_all(jnp.linalg.cond(matrix) < cond_cutoff):  # pyright: ignore
            break
    return matrix


def construct_matrix(getkey, solver, tags, num=1, *, size=3, dtype=jnp.float64):
    if isinstance(solver, lx.Normal):
        cond_cutoff = math.sqrt(1000)
    elif isinstance(solver, lx.LSMR):
        cond_cutoff = 10  # it's not doing super well for some reason
    else:
        cond_cutoff = 1000
    return tuple(
        _construct_matrix_impl(getkey, cond_cutoff, tags, size, dtype, i)
        for i in range(num)
    )


def construct_singular_matrix(getkey, solver, tags, num=1, dtype=jnp.float64):
    matrices = construct_matrix(getkey, solver, tags, num, dtype=dtype)
    if isinstance(solver, (lx.Diagonal, lx.CG, lx.BiCGStab, lx.GMRES)):
        return tuple(matrix.at[0, :].set(0) for matrix in matrices)
    else:
        version = jr.choice(getkey(), np.array([0, 1, 2]))
        if version == 0:
            return tuple(matrix.at[0, :].set(0) for matrix in matrices)
        elif version == 1:
            return tuple(matrix[1:, :] for matrix in matrices)
        else:
            return tuple(matrix[:, 1:] for matrix in matrices)


def construct_poisson_matrix(size, dtype=jnp.float64):
    matrix = (
        -2 * jnp.diag(jnp.ones(size, dtype=dtype))
        + jnp.diag(jnp.ones(size - 1, dtype=dtype), 1)
        + jnp.diag(jnp.ones(size - 1, dtype=dtype), -1)
    )
    return matrix


if jax.config.jax_enable_x64:  # pyright: ignore
    tol = 1e-12
else:
    tol = 1e-6
solvers_tags_pseudoinverse = [
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
    (lx.CG(rtol=tol, atol=tol), lx.positive_semidefinite_tag, False),
    (lx.CG(rtol=tol, atol=tol), lx.negative_semidefinite_tag, False),
    (lx.Normal(lx.CG(rtol=tol, atol=tol)), (), False),
    (lx.LSMR(atol=tol, rtol=tol), (), True),
    (lx.Cholesky(), lx.positive_semidefinite_tag, False),
    (lx.Cholesky(), lx.negative_semidefinite_tag, False),
    (lx.Normal(lx.Cholesky()), (), False),
]
solvers_tags = [(a, b) for a, b, _ in solvers_tags_pseudoinverse]
solvers = [a for a, _, _ in solvers_tags_pseudoinverse]
pseudosolvers_tags = [(a, b) for a, b, c in solvers_tags_pseudoinverse if c]


def _transpose(operator, matrix):
    return operator.T, matrix.T


def _linearise(operator, matrix):
    return lx.linearise(operator), matrix


def _materialise(operator, matrix):
    return lx.materialise(operator), matrix


ops = (lambda x, y: (x, y), _transpose, _linearise, _materialise)


def params(only_pseudo):
    for make_operator in make_operators:
        for solver, tags, pseudoinverse in solvers_tags_pseudoinverse:
            if only_pseudo and not pseudoinverse:
                continue
            if (
                make_operator is make_trivial_diagonal_operator
                and tags != lx.diagonal_tag
            ):
                continue
            if make_operator is make_identity_operator and tags != lx.unit_diagonal_tag:
                continue
            if (
                make_operator is make_tridiagonal_operator
                and tags != lx.tridiagonal_tag
            ):
                continue
            yield make_operator, solver, tags


def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8):
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


def has_tag(tags, tag):
    return tag is tags or (isinstance(tags, tuple) and tag in tags)


make_operators = []


def _operators_append(x):
    make_operators.append(x)
    return x


@_operators_append
def make_matrix_operator(getkey, matrix, tags):
    return lx.MatrixLinearOperator(matrix, tags)


@_operators_append
def make_trivial_pytree_operator(getkey, matrix, tags):
    out_size, _ = matrix.shape
    struct = jax.ShapeDtypeStruct((out_size,), matrix.dtype)
    return lx.PyTreeLinearOperator(matrix, struct, tags)


@_operators_append
def make_function_operator(getkey, matrix, tags):
    fn = lambda x: matrix @ x
    _, in_size = matrix.shape
    in_struct = jax.ShapeDtypeStruct((in_size,), matrix.dtype)
    return lx.FunctionLinearOperator(fn, in_struct, tags)


@_operators_append
def make_jac_operator(getkey, matrix, tags):
    out_size, in_size = matrix.shape
    x = jr.normal(getkey(), (in_size,), dtype=matrix.dtype)
    a = jr.normal(getkey(), (out_size,), dtype=matrix.dtype)
    b = jr.normal(getkey(), (out_size, in_size), dtype=matrix.dtype)
    c = jr.normal(getkey(), (out_size, in_size), dtype=matrix.dtype)
    fn_tmp = lambda x, _: a + b @ x + c @ x**2
    jac = jax.jacfwd(fn_tmp, holomorphic=jnp.iscomplexobj(x))(x, None)
    diff = matrix - jac
    fn = lambda x, _: a + (b + diff) @ x + c @ x**2
    return lx.JacobianLinearOperator(fn, x, None, tags)


@_operators_append
def make_jacfwd_operator(getkey, matrix, tags):
    out_size, in_size = matrix.shape
    x = jr.normal(getkey(), (in_size,), dtype=matrix.dtype)
    a = jr.normal(getkey(), (out_size,), dtype=matrix.dtype)
    b = jr.normal(getkey(), (out_size, in_size), dtype=matrix.dtype)
    c = jr.normal(getkey(), (out_size, in_size), dtype=matrix.dtype)
    fn_tmp = lambda x, _: a + b @ x + c @ x**2
    jac = jax.jacfwd(fn_tmp, holomorphic=jnp.iscomplexobj(x))(x, None)
    diff = matrix - jac
    fn = lambda x, _: a + (b + diff) @ x + c @ x**2
    return lx.JacobianLinearOperator(fn, x, None, tags, jac="fwd")


@_operators_append
def make_jacrev_operator(getkey, matrix, tags):
    """JacobianLinearOperator with jac='bwd' using a custom_vjp function.

    This uses custom_vjp so that forward-mode autodiff is NOT available,
    which tests that jac='bwd' works correctly without relying on JVP.
    """
    out_size, in_size = matrix.shape
    x = jr.normal(getkey(), (in_size,), dtype=matrix.dtype)
    a = jr.normal(getkey(), (out_size,), dtype=matrix.dtype)
    b = jr.normal(getkey(), (out_size, in_size), dtype=matrix.dtype)
    c = jr.normal(getkey(), (out_size, in_size), dtype=matrix.dtype)
    fn_tmp = lambda x, _: a + b @ x + c @ x**2
    jac = jax.jacfwd(fn_tmp, holomorphic=jnp.iscomplexobj(x))(x, None)
    diff = matrix - jac

    # Use custom_vjp to define a function that only has reverse-mode autodiff
    @jax.custom_vjp
    def custom_fn(x):
        return a + (b + diff) @ x + c @ x**2

    def custom_fn_fwd(x):
        return custom_fn(x), x

    def custom_fn_bwd(x, g):
        # Jacobian is: (b + diff) + 2 * c * x
        # VJP is: g @ J = g @ ((b + diff) + 2 * c * x)
        # So J.T @ g =
        return ((b + diff).T @ g + 2 * (c.T @ g) * x,)

    custom_fn.defvjp(custom_fn_fwd, custom_fn_bwd)

    fn = lambda x, _: custom_fn(x)
    return lx.JacobianLinearOperator(fn, x, None, tags, jac="bwd")


@_operators_append
def make_trivial_diagonal_operator(getkey, matrix, tags):
    assert tags == lx.diagonal_tag
    diag = jnp.diag(matrix)
    return lx.DiagonalLinearOperator(diag)


@_operators_append
def make_identity_operator(getkey, matrix, tags):
    in_struct = jax.ShapeDtypeStruct((matrix.shape[-1],), matrix.dtype)
    return lx.IdentityLinearOperator(input_structure=in_struct)


@_operators_append
def make_tridiagonal_operator(getkey, matrix, tags):
    diag1 = jnp.diag(matrix)
    if tags == lx.tridiagonal_tag:
        diag2 = jnp.diag(matrix, k=-1)
        diag3 = jnp.diag(matrix, k=1)
        return lx.TridiagonalLinearOperator(diag1, diag2, diag3)
    elif tags == lx.diagonal_tag:
        diag2 = diag3 = jnp.zeros(matrix.shape[0] - 1)
        return lx.TaggedLinearOperator(
            lx.TridiagonalLinearOperator(diag1, diag2, diag3), lx.diagonal_tag
        )
    elif tags == lx.symmetric_tag:
        diag2 = diag3 = jnp.diag(matrix, k=1)
        return lx.TaggedLinearOperator(
            lx.TridiagonalLinearOperator(diag1, diag2, diag3), lx.symmetric_tag
        )
    else:
        assert False, tags


@_operators_append
def make_add_operator(getkey, matrix, tags):
    matrix1 = 0.7 * matrix
    matrix2 = 0.3 * matrix
    operator = make_matrix_operator(getkey, matrix1, ()) + make_function_operator(
        getkey, matrix2, ()
    )
    return lx.TaggedLinearOperator(operator, tags)


@_operators_append
def make_mul_operator(getkey, matrix, tags):
    operator = make_jac_operator(getkey, 0.7 * matrix, ()) / 0.7
    return lx.TaggedLinearOperator(operator, tags)


@_operators_append
def make_composed_operator(getkey, matrix, tags):
    _, size = matrix.shape
    diag = jr.normal(getkey(), (size,), dtype=matrix.dtype)
    diag = jnp.where(jnp.abs(diag) < 0.05, 0.8, diag)
    operator1 = make_trivial_pytree_operator(getkey, matrix / diag[None], ())
    operator2 = lx.DiagonalLinearOperator(diag)
    return lx.TaggedLinearOperator(operator1 @ operator2, tags)


# Slightly sketchy approach to finite differences, in that this is pulled out of
# Numerical Recipes.
# I also don't know of a handling of the JVP case off the top of my head -- although
# I'm sure it exists somewhere -- so I'm improvising a little here. (In particular
# removing the usual "(x + h) - x" denominator.)
def finite_difference_jvp(fn, primals, tangents):
    out = fn(*primals)
    # Choose ε to trade-off truncation error and floating-point rounding error.
    max_leaves = [jnp.max(jnp.abs(p)) for p in jtu.tree_leaves(primals)] + [1]
    scale = jnp.max(jnp.stack(max_leaves))
    ε = np.sqrt(np.finfo(np.float64).eps) * scale
    with jax.numpy_dtype_promotion("standard"):
        primals_ε = (ω(primals) + ε * ω(tangents)).ω
        out_ε = fn(*primals_ε)
        tangents_out = jtu.tree_map(lambda x, y: (x - y) / ε, out_ε, out)
    return out, tangents_out


def jvp_jvp_impl(
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
