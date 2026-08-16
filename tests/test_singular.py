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

from .helpers import (
    construct_singular_matrix,
    finite_difference_jvp,
    ops,
    params,
    tol,
    tree_allclose,
)


@pytest.mark.parametrize("make_operator,solver,tags", params(only_pseudo=True))
@pytest.mark.parametrize("ops", ops)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_small_singular(make_operator, solver, tags, ops, getkey, dtype):
    if jax.config.jax_enable_x64:  # pyright: ignore
        tol = 1e-10
    else:
        tol = 1e-4
    (matrix,) = construct_singular_matrix(getkey, solver, tags, dtype=dtype)
    operator = make_operator(getkey, matrix, tags)
    operator, matrix = ops(operator, matrix)
    assert tree_allclose(operator.as_matrix(), matrix, rtol=tol, atol=tol)
    out_size, in_size = matrix.shape
    true_x = jr.normal(getkey(), (in_size,), dtype=dtype)
    b = matrix @ true_x
    x = lx.linear_solve(operator, b, solver=solver, throw=False).value
    jax_x, *_ = jnp.linalg.lstsq(matrix, b)  # pyright: ignore
    assert tree_allclose(x, jax_x, atol=tol, rtol=tol)


# `construct_singular_matrix` has no way to build a singular *circulant* matrix (its
# `zero` method clears the leading row, which the circulant construction then
# overwrites), so `Circulant` is excluded from the parametrised singular tests above --
# including their JVP coverage. Build one directly instead, by zeroing an eigenvalue.
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_circulant_singular_jvp(getkey, dtype):
    size = 6
    column = jr.normal(getkey(), (size,), dtype=dtype)
    # Zero eigenvalue 2, so that the operator is singular but still circulant.
    if jnp.iscomplexobj(column):
        eigenvalues = jnp.fft.fft(column).at[2].set(0)
        column = jnp.fft.ifft(eigenvalues)
    else:
        # `rfft`/`irfft` keep the spectrum conjugate-symmetric, so the column stays
        # real.
        eigenvalues = jnp.fft.rfft(column).at[2].set(0)
        column = jnp.fft.irfft(eigenvalues, n=size)

    def circulant(column, vector):
        operator = lx.CirculantLinearOperator(column)
        return lx.linear_solve(operator, vector, solver=lx.Circulant()).value

    def dense(column, vector):
        # The pseudoinverse solution, via a solver with no circulant structure to
        # exploit (and, in particular, no gram partner of its own to shortcut the JVP).
        matrix = lx.CirculantLinearOperator(column).as_matrix()
        return lx.linear_solve(
            lx.MatrixLinearOperator(matrix), vector, solver=lx.SVD()
        ).value

    vector = jr.normal(getkey(), (size,), dtype=dtype)
    t_column = jr.normal(getkey(), (size,), dtype=dtype)
    t_vector = jr.normal(getkey(), (size,), dtype=dtype)

    x, t_x = eqx.filter_jvp(circulant, (column, vector), (t_column, t_vector))
    true_x, true_t_x = eqx.filter_jvp(dense, (column, vector), (t_column, t_vector))
    assert tree_allclose(x, true_x, atol=tol, rtol=tol)
    # The JVP takes the `_gram_partner` path, as `Circulant.assume_full_rank()` is
    # `False`: the gram matrix `AᴴA` is itself circulant, with eigenvalues `|λ|²`.
    assert tree_allclose(t_x, true_t_x, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_bicgstab_breakdown(getkey, dtype):
    if jax.config.jax_enable_x64:  # pyright: ignore
        tol = 1e-10
    else:
        tol = 1e-4
    solver = lx.GMRES(atol=tol, rtol=tol, restart=2)

    matrix = jr.normal(jr.PRNGKey(0), (100, 100), dtype=dtype)
    true_x = jr.normal(jr.PRNGKey(0), (100,), dtype=dtype)
    b = matrix @ true_x
    operator = lx.MatrixLinearOperator(matrix)

    # result != 0 implies lineax reported failure
    lx_soln = lx.linear_solve(operator, b, solver, throw=False)

    assert jnp.all(lx_soln.result != lx.RESULTS.successful)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_gmres_stagnation_or_breakdown(getkey, dtype):
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
        ],
        dtype=dtype,
    )
    true_x = jnp.array([0.51383273, 1.72983427, -0.43251078, -1.11764668], dtype=dtype)
    b = matrix @ true_x
    operator = lx.MatrixLinearOperator(matrix)

    # result != 0 implies lineax reported failure
    lx_soln = lx.linear_solve(operator, b, solver, throw=False)

    assert jnp.all(lx_soln.result != lx.RESULTS.successful)


@pytest.mark.parametrize(
    "solver",
    (
        lx.AutoLinearSolver(well_posed=None),
        lx.QR(),
        lx.SVD(),
        lx.LSMR(atol=tol, rtol=tol),
        lx.Normal(lx.Cholesky()),
        lx.Normal(lx.SVD()),
    ),
)
def test_nonsquare_pytree_operator1(solver):
    x = [[1, 5.0, jnp.array(-1.0)], [jnp.array(-2), jnp.array(-2.0), 3.0]]
    y = [3.0, 4]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct)
    out = lx.linear_solve(operator, y, solver=solver).value
    matrix = jnp.array([[1.0, 5.0, -1.0], [-2.0, -2.0, 3.0]])
    true_out, _, _, _ = jnp.linalg.lstsq(matrix, jnp.array(y))  # pyright: ignore
    true_out = [true_out[0], true_out[1], true_out[2]]
    assert tree_allclose(out, true_out)


@pytest.mark.parametrize(
    "solver",
    (
        lx.AutoLinearSolver(well_posed=None),
        lx.QR(),
        lx.SVD(),
        lx.LSMR(atol=tol, rtol=tol),
        lx.Normal(lx.Cholesky()),
        lx.Normal(lx.SVD()),
    ),
)
def test_nonsquare_pytree_operator2(solver):
    x = [[1, jnp.array(-2)], [5.0, jnp.array(-2.0)], [jnp.array(-1.0), 3.0]]
    y = [3.0, 4, 5.0]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct)
    out = lx.linear_solve(operator, y, solver=solver).value
    matrix = jnp.array([[1.0, -2.0], [5.0, -2.0], [-1.0, 3.0]])
    true_out, _, _, _ = jnp.linalg.lstsq(matrix, jnp.array(y))  # pyright: ignore
    true_out = [true_out[0], true_out[1]]
    assert tree_allclose(out, true_out)


@pytest.mark.parametrize(
    "solver",
    (
        lx.AutoLinearSolver(well_posed=None),
        lx.QR(),
        lx.SVD(),
        lx.Normal(lx.Cholesky()),
        lx.Normal(lx.SVD()),
    ),
)
@pytest.mark.parametrize("full_rank", (True, False))
@pytest.mark.parametrize("jvp", (False, True))
@pytest.mark.parametrize("wide", (False, True))
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_nonsquare_mat_vec(solver, full_rank, jvp, wide, dtype, getkey):
    if wide:
        out_size = 3
        in_size = 6
    else:
        out_size = 6
        in_size = 3
    matrix = jr.normal(getkey(), (out_size, in_size), dtype=dtype)
    if not full_rank:
        if solver.assume_full_rank():
            # There is nothing to test.
            return
        # nontrivial rank 2 sparsity pattern
        matrix = matrix.at[1:, 1:].set(0)
    vector = jr.normal(getkey(), (out_size,), dtype=dtype)
    lx_solve = lambda mat, vec: lx.linear_solve(
        lx.MatrixLinearOperator(mat), vec, solver
    ).value
    jnp_solve = lambda mat, vec: jnp.linalg.lstsq(mat, vec)[0]  # pyright: ignore
    if jvp:
        lx_solve = eqx.filter_jit(ft.partial(eqx.filter_jvp, lx_solve))
        jnp_solve = eqx.filter_jit(ft.partial(finite_difference_jvp, jnp_solve))
        t_matrix = jr.normal(getkey(), (out_size, in_size), dtype=dtype)
        if not full_rank:
            # t_matrix must be chosen tangent to the manifold of rank 2
            # matrices at matrix. A simple way to achieve this is to make the
            # same restriction as we did to matrix
            t_matrix = t_matrix.at[1:, 1:].set(0)
        t_vector = jr.normal(getkey(), (out_size,), dtype=dtype)
        args = ((matrix, vector), (t_matrix, t_vector))
    else:
        args = (matrix, vector)
    x = lx_solve(*args)  # pyright: ignore
    true_x = jnp_solve(*args)
    assert tree_allclose(x, true_x, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "solver",
    (
        lx.AutoLinearSolver(well_posed=None),
        lx.QR(),
        lx.SVD(),
        lx.Normal(lx.Cholesky()),
        lx.Normal(lx.SVD()),
    ),
)
@pytest.mark.parametrize("full_rank", (True, False))
@pytest.mark.parametrize("jvp", (False, True))
@pytest.mark.parametrize("wide", (False, True))
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_nonsquare_vec(solver, full_rank, jvp, wide, dtype, getkey):
    if wide:
        out_size = 3
        in_size = 6
    else:
        out_size = 6
        in_size = 3
    matrix = jr.normal(getkey(), (out_size, in_size), dtype=dtype)
    if not full_rank:
        if solver.assume_full_rank():
            # There is nothing to test.
            return
        # nontrivial rank 2 sparsity pattern
        matrix = matrix.at[1:, 1:].set(0)
    vector = jr.normal(getkey(), (out_size,), dtype=dtype)
    lx_solve = lambda vec: lx.linear_solve(
        lx.MatrixLinearOperator(matrix), vec, solver
    ).value
    jnp_solve = lambda vec: jnp.linalg.lstsq(matrix, vec)[0]  # pyright: ignore
    if jvp:
        lx_solve = eqx.filter_jit(ft.partial(eqx.filter_jvp, lx_solve))
        jnp_solve = eqx.filter_jit(ft.partial(finite_difference_jvp, jnp_solve))
        t_vector = jr.normal(getkey(), (out_size,), dtype=dtype)
        args = ((vector,), (t_vector,))
    else:
        args = (vector,)
    x = lx_solve(*args)  # pyright: ignore
    true_x = jnp_solve(*args)
    assert tree_allclose(x, true_x, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_circulant_singular(getkey, dtype):
    # A column summing to zero gives a zero eigenvalue at the zero frequency, so the
    # operator is singular but still circulant.
    column = jnp.array([1.0, -1.0, 2.0, -2.0], dtype=dtype)
    operator = lx.CirculantLinearOperator(column)
    matrix = operator.as_matrix()
    vec = jr.normal(getkey(), (4,), dtype=dtype)

    # The DFT diagonalises, so zeroing the vanishing eigenvalue is exactly the
    # pseudoinverse.
    expected = jnp.linalg.pinv(matrix) @ vec
    assert tree_allclose(lx.linear_solve(operator, vec, lx.Circulant()).value, expected)
    auto = lx.AutoLinearSolver(well_posed=False)
    assert tree_allclose(lx.linear_solve(operator, vec, auto).value, expected)

    # `well_posed=True` promises nonsingularity, so the zero eigenvalue is not filtered
    # and the solve is reported as failing rather than silently returning a
    # pseudoinverse solution.
    for solver in (lx.Circulant(well_posed=True), lx.AutoLinearSolver(well_posed=True)):
        sol = lx.linear_solve(operator, vec, solver, throw=False)
        assert sol.result != lx.RESULTS.successful
        with pytest.raises(Exception):
            lx.linear_solve(operator, vec, solver)


def test_circulant_singular_rcond_size():
    # `rfft` returns `size // 2 + 1` eigenvalues, but `rcond` resolves from the size of
    # the matrix. Real dtypes only, as `fft` returns all `size` of them.
    size = 8
    # The zero and Nyquist bins must be real for `irfft` to round-trip.
    tail = jnp.array(
        [1.0 + 2.0j, -0.5 + 0.3j, 0.7 - 1.1j, 2.0 + 0.0j], dtype=jnp.complex128
    )
    eps = jnp.finfo(jnp.float64).eps
    max_abs = jnp.max(jnp.abs(tail))
    # Midway between the two thresholds, so only the correct one filters it.
    tiny = 13 * eps * max_abs
    eigenvalues = jnp.concatenate([tiny.astype(jnp.complex128)[None], tail])

    column = jnp.fft.irfft(eigenvalues, n=size)
    assert tree_allclose(jnp.fft.rfft(column), eigenvalues)
    # Bracket the round-tripped eigenvalue rather than the ideal `tiny`: `irfft`/`rfft`
    # perturbs it by ~`eps * max_abs`, and the `tree_allclose` above is far too loose to
    # notice at this magnitude. These are the two candidate thresholds, so the assert
    # pins the test's discriminating power rather than assuming it.
    realised = jnp.abs(jnp.fft.rfft(column))
    max_realised = jnp.max(realised)
    assert 2 * eps * realised.size * max_realised < realised[0]
    assert realised[0] < 2 * eps * size * max_realised
    operator = lx.CirculantLinearOperator(column)
    # A nonzero mean gives a component along the near-null zero-frequency eigenvector.
    vec = jnp.linspace(0.5, 2.0, size, dtype=jnp.float64)

    # Filtering `tiny` matches zeroing it, and for a zero the pseudoinverse is exact.
    zeroed = jnp.concatenate([jnp.zeros((1,), jnp.complex128), tail])
    matrix = lx.CirculantLinearOperator(jnp.fft.irfft(zeroed, n=size)).as_matrix()
    expected = jnp.linalg.pinv(matrix) @ vec

    solution = lx.linear_solve(operator, vec, lx.Circulant(well_posed=False)).value
    assert tree_allclose(solution, expected)
    # Keeping `tiny` would blow the solution up by fourteen orders of magnitude.
    assert jnp.max(jnp.abs(solution)) < 1e3


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_circulant_auto_dispatch(dtype):
    column = jnp.array([1.0, -1.0, 2.0, -2.0], dtype=dtype)
    operator = lx.CirculantLinearOperator(column)
    for well_posed in (True, False, None):
        solver = lx.AutoLinearSolver(well_posed=well_posed).select_solver(operator)
        assert isinstance(solver, lx.Circulant), (well_posed, solver)
        assert solver.well_posed is (well_posed is True)


# No test for iterative solvers on singular operators: Krylov methods have no rank
# detection and no defined behaviour on rank-deficient systems, so there is nothing to
# assert (their failure reporting is covered by test_bicgstab_breakdown and
# test_gmres_stagnation_or_breakdown above).
