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
import operator
import random

import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω

import lineax as lx


def getkey():
    return jr.PRNGKey(random.randint(0, 2**31 - 1))


@ft.lru_cache(maxsize=None)
def _construct_matrix_impl(getkey, cond_cutoff, tags, size, dtype):
    while True:
        matrix = jr.normal(getkey(), (size, size), dtype=dtype)
        if has_tag(tags, lx.diagonal_tag):
            matrix = jnp.diag(jnp.diag(matrix))
        if has_tag(tags, lx.self_adjoint_tag):
            matrix = matrix + matrix.T.conj()
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
            matrix = matrix @ matrix.T
        if has_tag(tags, lx.negative_semidefinite_tag):
            matrix = -matrix @ matrix.T
        if eqxi.unvmap_all(jnp.linalg.cond(matrix) < cond_cutoff):  # pyright: ignore
            break
    return matrix


def construct_matrix(getkey, solver, tags, num=1, *, size=3, dtype=jnp.float64):
    if isinstance(solver, lx.NormalCG):
        cond_cutoff = math.sqrt(1000)
    else:
        cond_cutoff = 1000
    return tuple(
        _construct_matrix_impl(getkey, cond_cutoff, tags, size, dtype)
        for _ in range(num)
    )


def construct_singular_matrix(getkey, solver, tags, num=1, dtype=jnp.float64):
    matrices = construct_matrix(getkey, solver, tags, num, dtype=dtype)
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
    (lx.NormalCG(rtol=tol, atol=tol), (), False),
    (lx.CG(rtol=tol, atol=tol), lx.positive_semidefinite_tag, False),
    (lx.CG(rtol=tol, atol=tol), lx.negative_semidefinite_tag, False),
    (lx.NormalCG(rtol=tol, atol=tol), lx.negative_semidefinite_tag, False),
    (lx.Cholesky(), lx.positive_semidefinite_tag, False),
    (lx.Cholesky(), lx.negative_semidefinite_tag, False),
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
            if make_operator is make_diagonal_operator and tags != lx.diagonal_tag:
                continue
            if (
                make_operator is make_tridiagonal_operator
                and tags != lx.tridiagonal_tag
            ):
                continue
            yield make_operator, solver, tags


def _shaped_allclose(x, y, **kwargs):
    if type(x) is not type(y):
        return False
    if isinstance(x, jax.Array):
        x = np.asarray(x)
        y = np.asarray(y)
    if isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.inexact):
            return (
                x.shape == y.shape
                and x.dtype == y.dtype
                and np.allclose(x, y, **kwargs)
            )
        else:
            return x.shape == y.shape and x.dtype == y.dtype and np.all(x == y)
    elif isinstance(x, jax.ShapeDtypeStruct):
        assert x.shape == y.shape and x.dtype == y.dtype
    else:
        return x == y


def shaped_allclose(x, y, **kwargs):
    """As `jnp.allclose`, except:
    - It also supports PyTree arguments.
    - It mandates that shapes match as well (no broadcasting)
    """
    same_structure = jtu.tree_structure(x) == jtu.tree_structure(y)
    allclose = ft.partial(_shaped_allclose, **kwargs)
    return same_structure and jtu.tree_reduce(
        operator.and_, jtu.tree_map(allclose, x, y), True
    )


def has_tag(tags, tag):
    return tag is tags or (isinstance(tags, tuple) and tag in tags)


make_operators = []


def _operators_append(x):
    make_operators.append(x)
    return x


@_operators_append
def make_matrix_operator(matrix, tags):
    return lx.MatrixLinearOperator(matrix, tags)


@_operators_append
def make_trivial_pytree_operator(matrix, tags):
    out_size, _ = matrix.shape
    struct = jax.ShapeDtypeStruct((out_size,), matrix.dtype)
    return lx.PyTreeLinearOperator(matrix, struct, tags)


@_operators_append
def make_function_operator(matrix, tags):
    fn = lambda x: matrix @ x
    _, in_size = matrix.shape
    in_struct = jax.ShapeDtypeStruct((in_size,), matrix.dtype)
    return lx.FunctionLinearOperator(fn, in_struct, tags)


@_operators_append
def make_jac_operator(matrix, tags):
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
def make_diagonal_operator(matrix, tags):
    assert tags == lx.diagonal_tag
    diag = jnp.diag(matrix)
    return lx.DiagonalLinearOperator(diag)


@_operators_append
def make_tridiagonal_operator(matrix, tags):
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
def make_add_operator(matrix, tags):
    matrix1 = 0.7 * matrix
    matrix2 = 0.3 * matrix
    operator = make_matrix_operator(matrix1, ()) + make_function_operator(matrix2, ())
    return lx.TaggedLinearOperator(operator, tags)


@_operators_append
def make_mul_operator(matrix, tags):
    operator = make_jac_operator(0.7 * matrix, ()) / 0.7
    return lx.TaggedLinearOperator(operator, tags)


@_operators_append
def make_composed_operator(matrix, tags):
    _, size = matrix.shape
    diag = jr.normal(getkey(), (size,), dtype=matrix.dtype)
    diag = jnp.where(jnp.abs(diag) < 0.05, 0.8, diag)
    operator1 = make_trivial_pytree_operator(matrix / diag, ())
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
    primals_ε = (ω(primals) + ε * ω(tangents)).ω
    out_ε = fn(*primals_ε)
    tangents_out = jtu.tree_map(lambda x, y: (x - y) / ε, out_ε, out)
    return out, tangents_out
