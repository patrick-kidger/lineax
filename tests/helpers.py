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
def _construct_matrix_impl(
    getkey,
    tags,
    rows: int,
    cols: int,
    full_rank: bool,
    dtype,
    cond_cutoff: float | None,
    i: int,
):
    if i > 0:  # we are giving a tangent vector
        # handle the tags special cased for fast mv by _try_sparse_materialise,
        # as tangent vectors must also have the tags to be mv'd correctly.

        # It is otherwise not necessary that tangent vectors (and higher order)
        # have the same tags as the primal unless we have optimized mv to
        # exploit the structure of the tags. Furthermore, we pick tangent
        # directions tangent to the constant rank locus at the primal.
        t_matrix = jr.normal(getkey(), (rows, cols), dtype=dtype)
        if not full_rank:
            matrix = _construct_matrix_impl(
                getkey, tags, rows, cols, full_rank, dtype, cond_cutoff, 0
            )
            r = min(rows, cols) - 1
            t_matrix = jax.jvp(lambda A: Ahat(A, r), (matrix,), (t_matrix,))[1]
        if has_tag(tags, lx.diagonal_tag):
            t_matrix = jnp.tril(jnp.triu(t_matrix))
            assert full_rank
        if has_tag(tags, lx.tridiagonal_tag):
            t_matrix = jnp.tril(jnp.triu(t_matrix, k=-1), k=1)
            assert full_rank
        return t_matrix
    assert full_rank or cond_cutoff is None, (
        "Cannot construct rank deficient matrix with bounded condition number"
    )
    while True:
        matrix = jr.normal(getkey(), (rows, cols), dtype=dtype)
        if isinstance(tags, tuple) and len(tags) == 0 and not full_rank:
            u, s, vh = jnp.linalg.svd(matrix, full_matrices=False)
            s = s.at[-1].set(0).astype(matrix.dtype)
            matrix = (u * s[None, :]) @ vh
        if has_tag(tags, lx.diagonal_tag):
            matrix = jnp.tril(jnp.triu(matrix))
            if not full_rank:
                matrix = matrix.at[0, :].set(0)
        if has_tag(tags, lx.lower_triangular_tag):
            matrix = jnp.tril(matrix)
            if not full_rank:
                matrix = matrix.at[:, 0].set(0)
        if has_tag(tags, lx.upper_triangular_tag):
            matrix = jnp.triu(matrix)
            if not full_rank:
                matrix = matrix.at[0, :].set(0)
        if has_tag(tags, lx.unit_diagonal_tag):
            matrix = matrix.at[
                jnp.arange(min(rows, cols)), jnp.arange(min(rows, cols))
            ].set(1)
            if not full_rank:
                raise NotImplementedError(
                    "Rank deficient matrix with unit diagonal not implemented."
                )
        if has_tag(tags, lx.tridiagonal_tag):
            matrix = jnp.tril(jnp.triu(matrix, k=-1), k=1)
            if not full_rank:
                matrix = matrix.at[0, :].set(0)
        if has_tag(tags, lx.symmetric_tag):
            assert rows == cols, "Symmetric matrix must be square"
            matrix = matrix + matrix.T
            if not full_rank:
                matrix = matrix.at[0, :].set(0).at[:, 0].set(0)
        if has_tag(tags, lx.positive_semidefinite_tag):
            assert rows == cols, "Positive semidefinite matrix must be square"
            u, s, _ = jnp.linalg.svd(matrix)
            if not full_rank:
                s = s.at[-1].set(0)
            matrix = (u * s[None, :].astype(matrix.dtype)) @ u.conj().T
        if has_tag(tags, lx.negative_semidefinite_tag):
            assert rows == cols, "Negative semidefinite matrix must be square"
            u, s, _ = jnp.linalg.svd(matrix)
            if not full_rank:
                s = s.at[-1].set(0)
            matrix = -(u * s[None, :].astype(matrix.dtype)) @ u.conj().T

        if cond_cutoff is None or eqxi.unvmap_all(
            jnp.linalg.cond(matrix) < cond_cutoff
        ):
            break
    return matrix


def construct_matrix(
    getkey,
    solver,
    tags,
    num=1,
    *,
    rows=3,
    cols=3,
    full_rank=True,
    dtype=jnp.float64,
):
    """Construct test matrices with specified shape and rank properties.

    Args:
        getkey: Random key generator
        solver: Solver instance (used to determine condition cutoff)
        tags: Matrix tags
        num: number of examples, the first is the primal and the rest are
            tangent vectors to primal
        rows: Number of rows (default 3)
        cols: Number of columns (default 3)
        full_rank: If True, construct full-rank matrix. If False, construct
            rank-deficient matrix with rank min(rows,cols)-1
        dtype: Data type
    """
    if isinstance(solver, lx.Normal):
        cond_cutoff = math.sqrt(1000)
    else:
        cond_cutoff = 1000
    return tuple(
        _construct_matrix_impl(
            getkey,
            tags,
            rows,
            cols,
            full_rank,
            dtype,
            cond_cutoff if full_rank else None,
            i,
        )
        for i in range(num)
    )


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


# Each entry: (solver, tags, nonsquare, rank_deficient)
# - nonsquare: True if solver supports nonsquare (rectangular) matrices
# - rank_deficient: True if solver supports rank-deficient matrices
solver_features = [
    (lx.AutoLinearSolver(well_posed=True), (), False, False),
    (lx.AutoLinearSolver(well_posed=None), (), True, False),
    (lx.AutoLinearSolver(well_posed=False), (), True, True),
    (lx.Triangular(), lx.lower_triangular_tag, False, False),
    (lx.Triangular(), lx.upper_triangular_tag, False, False),
    (
        lx.Triangular(),
        (lx.lower_triangular_tag, lx.unit_diagonal_tag),
        False,
        False,
    ),
    (
        lx.Triangular(),
        (lx.upper_triangular_tag, lx.unit_diagonal_tag),
        False,
        False,
    ),
    (lx.Diagonal(), lx.diagonal_tag, False, False),
    (lx.Diagonal(), (lx.diagonal_tag, lx.unit_diagonal_tag), False, False),
    (lx.Tridiagonal(), lx.tridiagonal_tag, False, False),
    (lx.LU(), (), False, False),
    (lx.QR(), (), True, False),
    (lx.SVD(), (), True, True),
    (lx.BiCGStab(rtol=tol, atol=tol), (), False, False),
    (lx.GMRES(rtol=tol, atol=tol), (), False, False),
    (lx.CG(rtol=tol, atol=tol), lx.positive_semidefinite_tag, False, False),
    (lx.CG(rtol=tol, atol=tol), lx.negative_semidefinite_tag, False, False),
    (lx.Normal(lx.CG(rtol=tol, atol=tol)), (), True, False),
    (lx.LSMR(atol=tol, rtol=tol), (), True, True),
    (lx.Cholesky(), lx.positive_semidefinite_tag, False, False),
    (lx.Cholesky(), lx.negative_semidefinite_tag, False, False),
    (lx.Normal(lx.Cholesky()), (), True, False),
]


def all_test_cases():
    for solver, tags, nonsquare, rank_deficient in solver_features:
        for rows, cols in [(3, 3), (2, 3), (3, 2)] if nonsquare else [(3, 3)]:
            # One need not restrict to square below. Do just for test speed
            for fullrank in (
                [True, False] if rank_deficient and rows == cols else [True]
            ):
                yield (solver, tags, rows, cols, fullrank)


def _transpose(operator, matrix):
    return operator.T, matrix.T


def _linearise(operator, matrix):
    return lx.linearise(operator), matrix


def _materialise(operator, matrix):
    return lx.materialise(operator), matrix


ops = (lambda x, y: (x, y), _transpose, _linearise, _materialise)


def params(only_rank_deficient):
    for make_operator in make_operators:
        for solver, tags, nonsquare, rank_deficient in solver_features:
            if only_rank_deficient and not rank_deficient:
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


# Nearest rank r factorization via Eckart-Young-Mirsky + IFT.
# Computes B, C so that Ahat = B @ C where the inner dimension is r,
# C[:,:r] = I_r, and |A - Ahat|_F is smallest.
# The JVP is defined via the implicit function theorem applied to the
# stationarity conditions of the Frobenius-norm minimization on the free
# variables of C (B obtained by linear least squares).
# This is smooth in a neighborhood of any A where s_r is separated from
# s_{r+1} and A[:,:r] is full rank
def Ahat_factor(A, r):
    @jax.custom_jvp
    def Cfree(A):
        # Use the Eckart-Young-Mirsky theorem and normalize. We must not
        # differentiate through this, so we define custom_jvp
        u, s, vh = jnp.linalg.svd(A, full_matrices=False)
        C = vh[:r, :]
        Cfree = jnp.linalg.solve(C[:, :r], C[:, r:])
        return Cfree

    @Cfree.defjvp
    def Cfree_jvp(primals, tangents):
        # obtain derivatives by the implicit function theorem applied to the
        # conditions defining Cfree. Everything here must be differentiable
        # in order for higher derivatives to be defined: Cfree is obtained
        # recursively and everything else is obtained with autodifferentiable
        # functions.
        (A,) = primals
        (A_t,) = tangents
        Cf = Cfree(A)
        a, b = Cf.shape

        # Cf is defined by this quantity being minimized in the L2 sense.
        def minimized(A, Cfree):
            C = jnp.hstack([jnp.eye(r, dtype=A.dtype), Cfree])
            # B = lstsq(C.T, A.T).T
            B = jnp.linalg.solve(C.conj() @ C.T, C.conj() @ A.T).T
            E = A - B @ C
            return jnp.real(jnp.sum(E.conj() * E))

        # equations obtained from critical points of the minimized function
        def stationary_eq(A, Cfree):
            return jax.grad(minimized, argnums=[1])(A, Cfree)[0]

        # implicit function theorem
        eq_t = jax.jvp(lambda a: stationary_eq(a, Cf), (A,), (A_t,))[1]
        if jnp.issubdtype(A.dtype, jnp.complexfloating):
            toreal = lambda z: jnp.stack([jnp.real(z), jnp.imag(z)])
            tocx = lambda x: jax.lax.complex(x[0], x[1])
            JCf = jax.jacfwd(lambda cfree: toreal(stationary_eq(A, tocx(cfree))))(
                toreal(Cf)
            )
            Cf_t = -tocx(
                jnp.linalg.solve(
                    JCf.reshape(2 * a * b, 2 * a * b), toreal(eq_t).reshape(2 * a * b)
                ).reshape(2, a, b)
            )
        else:
            JCf = jax.jacfwd(lambda cfree: stationary_eq(A, cfree))(Cf)
            Cf_t = -jnp.linalg.solve(
                JCf.reshape(a * b, a * b), eq_t.reshape(a * b)
            ).reshape(a, b)

        return Cf, Cf_t

    # if A is wide, we transpose so our IFT solve through Cfree is as small as possible
    transpose = A.shape[0] < A.shape[1]
    if transpose:
        A = A.T
    Cf = Cfree(A)
    C = jnp.hstack([jnp.eye(r, dtype=A.dtype), Cf])
    # B = lstsq(C.T, A.T).T
    B = jnp.linalg.solve(C.conj() @ C.T, C.conj() @ A.T).T
    if transpose:
        return C.T, B.T
    else:
        return B, C


# Nearest rank r matrix to A, smoothly defined in a neighborhood of any A for
# which s_r is separated from s_{r+1}
def Ahat(A, r):
    if r == min(A.shape):
        return A
    else:
        B, C = Ahat_factor(A, r)
        return B @ C


# jnp.linalg.lstsq with correct derivatives at rank-deficient points.
# Explicitly, we pass r and define the following function infinitely
# differentiably: Pick the nearest rank r matrix Ahat to A in the Frobenius
# This is the most principled choice of smooth extension out of the rank r
# locus and agrees with the classical derivative formula for the pseudoinverse
# implemented by lineax.
def jnp_lstsq(A, v, r):
    if r == min(A.shape):
        if A.shape[0] == A.shape[1]:
            return jnp.linalg.solve(A, v)
        elif A.shape[0] > A.shape[1]:
            return jnp.linalg.solve(A.conj().T @ A, A.conj().T @ v)
        else:
            return A.conj().T @ jnp.linalg.solve(A @ A.conj().T, v)
    else:
        # We compute lstsq(B @ C, v) as lstsq(C, lstsq(B, v)), but use solve with
        # the normal equations to avoid implicitly differentiating through an svd
        # solve
        B, C = Ahat_factor(A, r)
        y = jnp.linalg.solve(B.conj().T @ B, B.conj().T @ v)
        x = C.conj().T @ jnp.linalg.solve(C @ C.conj().T, y)
        return x


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
    getkey,
    solver,
    tags,
    make_operator,
    use_state,
    rows,
    cols,
    full_rank,
    dtype,
):
    t_tags = (None,) * len(tags) if isinstance(tags, tuple) else None
    matrix, t_matrix, tt_matrix, tt_t_matrix = construct_matrix(
        getkey,
        solver,
        tags,
        num=4,
        dtype=dtype,
        rows=rows,
        cols=cols,
        full_rank=full_rank,
    )

    make_op = ft.partial(make_operator, getkey)
    t_make_operator = lambda p, t_p: eqx.filter_jvp(make_op, (p, tags), (t_p, t_tags))
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
            op_dynamic, op_static = eqx.partition(operator, eqx.is_inexact_array)
            stopped_operator = eqx.combine(lax.stop_gradient(op_dynamic), op_static)
            state = solver.init(stopped_operator, options={})

            sol = lx.linear_solve(operator, vector, state=state, solver=solver)
            return sol.value

    else:

        def linear_solve1(operator, vector):
            sol = lx.linear_solve(operator, vector, solver=solver)
            return sol.value

    if full_rank and rows == cols:
        jnp_solve1 = jnp.linalg.solve  # pyright: ignore
    else:
        r = min(rows, cols) if full_rank else min(rows, cols) - 1
        jnp_solve1 = lambda mat, vec: jnp_lstsq(mat, vec, r)

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
            linear_solve3 = lambda v: linear_solve2((operator, v), (t_operator, t_vec))
            jnp_solve3 = lambda v: jnp_solve2((matrix, v), (t_matrix, t_vec))
        elif mode == {"op", "vec"}:
            linear_solve3 = lambda op, v: linear_solve2((op, v), (t_operator, t_vec))
            jnp_solve3 = lambda mat, v: jnp_solve2((mat, v), (t_matrix, t_vec))
        elif mode == {"t_op"}:
            linear_solve3 = lambda t_op: linear_solve2((operator, vec), (t_op, t_vec))
            jnp_solve3 = lambda t_mat: jnp_solve2((matrix, vec), (t_mat, t_vec))
        elif mode == {"t_vec"}:
            linear_solve3 = lambda t_v: linear_solve2(
                (operator, vec), (t_operator, t_v)
            )
            jnp_solve3 = lambda t_v: jnp_solve2((matrix, vec), (t_matrix, t_v))
        elif mode == {"op", "t_op"}:
            linear_solve3 = lambda op, t_op: linear_solve2((op, vec), (t_op, t_vec))
            jnp_solve3 = lambda mat, t_mat: jnp_solve2((mat, vec), (t_mat, t_vec))
        elif mode == {"op", "t_vec"}:
            linear_solve3 = lambda op, t_v: linear_solve2((op, vec), (t_operator, t_v))
            jnp_solve3 = lambda mat, t_v: jnp_solve2((mat, vec), (t_matrix, t_v))
        elif mode == {"vec", "t_op"}:
            linear_solve3 = lambda v, t_op: linear_solve2((operator, v), (t_op, t_vec))
            jnp_solve3 = lambda v, t_mat: jnp_solve2((matrix, v), (t_mat, t_vec))
        elif mode == {"vec", "t_vec"}:
            linear_solve3 = lambda v, t_v: linear_solve2(
                (operator, v), (t_operator, t_v)
            )
            jnp_solve3 = lambda v, t_v: jnp_solve2((matrix, v), (t_matrix, t_v))
        elif mode == {"op", "vec", "t_op"}:
            linear_solve3 = lambda op, v, t_op: linear_solve2((op, v), (t_op, t_vec))
            jnp_solve3 = lambda mat, v, t_mat: jnp_solve2((mat, v), (t_mat, t_vec))
        elif mode == {"op", "vec", "t_vec"}:
            linear_solve3 = lambda op, v, t_v: linear_solve2((op, v), (t_operator, t_v))
            jnp_solve3 = lambda mat, v, t_v: jnp_solve2((mat, v), (t_matrix, t_v))
        elif mode == {"vec", "t_op", "t_vec"}:
            linear_solve3 = lambda v, t_op, t_v: linear_solve2(
                (operator, v), (t_op, t_v)
            )
            jnp_solve3 = lambda v, t_mat, t_v: jnp_solve2((matrix, v), (t_mat, t_v))
        elif mode == {"op", "vec", "t_op", "t_vec"}:
            linear_solve3 = lambda op, v, t_op, t_v: linear_solve2((op, v), (t_op, t_v))
            jnp_solve3 = lambda mat, v, t_mat, t_v: jnp_solve2((mat, v), (t_mat, t_v))
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
