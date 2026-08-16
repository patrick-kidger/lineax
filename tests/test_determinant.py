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
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from .helpers import (
    construct_matrix,
    make_jac_operator,
    make_matrix_operator,
)


# ----------------------------------------------------------------------------
# Square determinant and slogdet: correctness vs jnp.linalg
# Parametrised over operator type to exercise both direct matrix storage
# and the as_matrix() materialisation path (JacobianLinearOperator).
# ----------------------------------------------------------------------------

SQUARE_DET_CASES = [
    (lx.LU(), ()),
    (lx.QR(), ()),
    (lx.Cholesky(), lx.positive_semidefinite_tag),
    (lx.Cholesky(), lx.negative_semidefinite_tag),
    (lx.Triangular(), lx.lower_triangular_tag),
    (lx.Triangular(), lx.upper_triangular_tag),
    (lx.Diagonal(well_posed=True), lx.diagonal_tag),
    (lx.Diagonal(well_posed=False), lx.diagonal_tag),
    (lx.Tridiagonal(), lx.tridiagonal_tag),
    (lx.HEVD(), lx.symmetric_tag),
    (lx.Circulant(well_posed=True), lx.circulant_tag),
    (lx.Circulant(well_posed=False), lx.circulant_tag),
    (lx.AutoLinearSolver(well_posed=True), ()),
    (lx.AutoLinearSolver(well_posed=None), ()),
]

# Complex analogue. `symmetric_tag` would build a complex-*symmetric* (non-Hermitian)
# matrix, which `HEVD` rejects, so the Hermitian solvers use `hermitian_tag` /
# `positive_semidefinite_tag` instead. These exercise the complex sign paths -- most
# notably the `QR` complex-Householder sign and the `Circulant` FFT determinant -- which
# the real cases cannot reach.
COMPLEX_DET_CASES = [
    (lx.LU(), ()),
    (lx.QR(), ()),
    (lx.Cholesky(), lx.positive_semidefinite_tag),
    (lx.Triangular(), lx.lower_triangular_tag),
    (lx.Triangular(), lx.upper_triangular_tag),
    (lx.Diagonal(well_posed=True), lx.diagonal_tag),
    (lx.Diagonal(well_posed=False), lx.diagonal_tag),
    (lx.Tridiagonal(), lx.tridiagonal_tag),
    (lx.HEVD(), lx.hermitian_tag),
    (lx.Circulant(well_posed=True), lx.circulant_tag),
    (lx.Circulant(well_posed=False), lx.circulant_tag),
    (lx.AutoLinearSolver(well_posed=True), ()),
    (lx.AutoLinearSolver(well_posed=None), ()),
]


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("solver,tags", SQUARE_DET_CASES)
def test_determinant_square(make_operator, solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    op = make_operator(getkey, matrix, tags)
    det = lx.determinant(op, solver, throw=False)
    expected = jnp.linalg.det(matrix)
    assert jnp.allclose(det, expected, atol=1e-10), f"got {det}, expected {expected}"


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("solver,tags", SQUARE_DET_CASES)
def test_slogdet_square(make_operator, solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    op = make_operator(getkey, matrix, tags)
    sign, lad = lx.slogdet(op, solver)
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(lad, ref_lad, atol=1e-10), f"lad: {lad} vs {ref_lad}"
    if not jnp.isnan(sign):
        assert jnp.allclose(sign, ref_sign, atol=1e-10), f"sign: {sign} vs {ref_sign}"


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("solver,tags", COMPLEX_DET_CASES)
def test_determinant_square_complex(make_operator, solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags, dtype=jnp.complex128)
    op = make_operator(getkey, matrix, tags)
    det = lx.determinant(op, solver, throw=False)
    expected = jnp.linalg.det(matrix)
    assert jnp.allclose(det, expected, atol=1e-10), f"got {det}, expected {expected}"


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("solver,tags", COMPLEX_DET_CASES)
def test_slogdet_square_complex(make_operator, solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags, dtype=jnp.complex128)
    op = make_operator(getkey, matrix, tags)
    sign, lad = lx.slogdet(op, solver)
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(lad, ref_lad, atol=1e-10), f"lad: {lad} vs {ref_lad}"
    if not jnp.isnan(sign):
        assert jnp.allclose(sign, ref_sign, atol=1e-10), f"sign: {sign} vs {ref_sign}"


# ----------------------------------------------------------------------------
# Normal(Cholesky): sign=nan, lad = sum(log(singular values)) for rectangular A
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(5, 3), (3, 5)])
def test_normal_cholesky_slogdet_rectangular(shape, getkey):
    A = jr.normal(getkey(), shape, dtype=jnp.float64)
    op = lx.MatrixLinearOperator(A)
    sign, lad = lx.slogdet(op, lx.Normal(lx.Cholesky()))
    assert jnp.isnan(sign)
    s = jnp.linalg.svd(A, compute_uv=False)
    assert jnp.allclose(lad, jnp.sum(jnp.log(s)), atol=1e-8)


# ----------------------------------------------------------------------------
# sign=nan: throw kwarg
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver,tags",
    [
        (lx.SVD(), ()),
        (lx.Normal(lx.Cholesky()), lx.positive_semidefinite_tag),
    ],
)
def test_slogdet_sign_is_nan(solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    op = lx.MatrixLinearOperator(matrix, tags)
    sign, _ = lx.slogdet(op, solver)
    assert jnp.isnan(sign)


@pytest.mark.parametrize(
    "solver,tags",
    [
        (lx.SVD(), ()),
        (lx.Normal(lx.Cholesky()), lx.positive_semidefinite_tag),
    ],
)
def test_determinant_throw_true_raises(solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    op = lx.MatrixLinearOperator(matrix, tags)
    with pytest.raises(Exception):
        lx.determinant(op, solver, throw=True)


@pytest.mark.parametrize(
    "solver,tags",
    [
        (lx.SVD(), ()),
        (lx.Normal(lx.Cholesky()), lx.positive_semidefinite_tag),
    ],
)
def test_determinant_throw_false_nan(solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    op = lx.MatrixLinearOperator(matrix, tags)
    det = lx.determinant(op, solver, throw=False)
    assert jnp.isnan(det)


# ----------------------------------------------------------------------------
# SVD slogdet: log-pseudodeterminant
# ----------------------------------------------------------------------------


def test_svd_slogdet_lad_fullrank(getkey):
    (matrix,) = construct_matrix(getkey, lx.SVD(), ())
    op = lx.MatrixLinearOperator(matrix)
    _, lad = lx.slogdet(op, lx.SVD())
    _, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(lad, ref_lad, atol=1e-10)


def test_diagonal_slogdet_rankdeficient(getkey):
    """Diagonal(well_posed=False): zero entry excluded from pseudodeterminant."""
    diag = jr.normal(getkey(), (4,), dtype=jnp.float64)
    diag = diag.at[1].set(0.0)
    op = lx.DiagonalLinearOperator(diag)
    sign, lad = lx.slogdet(op, lx.Diagonal(well_posed=False))
    nonzero = diag[jnp.abs(diag) > 1e-10]
    assert jnp.allclose(sign, jnp.prod(jnp.sign(nonzero)).real, atol=1e-10)
    assert jnp.allclose(lad, jnp.sum(jnp.log(jnp.abs(nonzero))), atol=1e-10)


def test_svd_slogdet_lad_rankdeficient(getkey):
    """Rank-deficient: lad = sum of log(nonzero singular values)."""
    matrix = jr.normal(getkey(), (3, 3), dtype=jnp.float64)
    matrix = matrix.at[0, :].set(0)
    op = lx.MatrixLinearOperator(matrix)
    _, lad = lx.slogdet(op, lx.SVD())
    s = jnp.linalg.svd(matrix, compute_uv=False)
    s_nonzero = s[s > 1e-10]
    assert jnp.allclose(lad, jnp.sum(jnp.log(s_nonzero)), atol=1e-8)


# ----------------------------------------------------------------------------
# QR rectangular: lad, sign=±1, sign vs explicit full-QR
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 3), (3, 4)])
def test_qr_rectangular_lad(shape, getkey):
    A = jr.normal(getkey(), shape, dtype=jnp.float64)
    op = lx.MatrixLinearOperator(A)
    _, lad = lx.slogdet(op, lx.QR())
    _, ref_lad = jnp.linalg.slogdet(A.T @ A if shape[0] >= shape[1] else A @ A.T)
    assert jnp.allclose(lad, 0.5 * ref_lad, atol=1e-10)


@pytest.mark.parametrize("shape", [(4, 3), (3, 4)])
def test_qr_rectangular_sign_is_pm1(shape, getkey):
    A = jr.normal(getkey(), shape, dtype=jnp.float64)
    op = lx.MatrixLinearOperator(A)
    sign, _ = lx.slogdet(op, lx.QR())
    assert not jnp.isnan(sign)
    assert jnp.allclose(jnp.abs(sign), 1.0, atol=1e-10)


@pytest.mark.parametrize("shape", [(4, 3), (3, 4)])
def test_qr_rectangular_sign_vs_full_qr(shape, getkey):
    """sign matches sign(det(Q_full)) * prod(sign(diag(R))) via jnp.linalg.qr."""
    A = jr.normal(getkey(), shape, dtype=jnp.float64)
    op = lx.MatrixLinearOperator(A)
    sign, _ = lx.slogdet(op, lx.QR())

    # lineax QR decomposes A directly (tall) or A^T (wide)
    B = A if A.shape[0] >= A.shape[1] else A.T
    Q_full, R_full = jnp.linalg.qr(B, mode="complete")
    n = min(B.shape)
    R_sq = R_full[:n, :n]
    sign_ref = (
        jnp.sign(jnp.linalg.det(Q_full)) * jnp.prod(jnp.sign(jnp.diag(R_sq)))
    ).astype(jnp.float64)
    assert jnp.allclose(sign, sign_ref, atol=1e-10), f"sign {sign} vs ref {sign_ref}"


# ----------------------------------------------------------------------------
# JVP and grad: validated against jax.jvp/grad of jnp.linalg.slogdet.
# Parametrised over operator type: make_jac_operator exercises the AD path
# through TangentLinearOperator.as_matrix() for a JacobianLinearOperator.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize(
    "solver,tags,use_state",
    [
        (lx.LU(), (), False),
        (lx.LU(), (), True),
        (lx.QR(), (), False),
    ],
)
def test_slogdet_jvp_lad(make_operator, solver, tags, use_state, getkey):
    (matrix, t_matrix) = construct_matrix(getkey, solver, tags, num=2)

    def lad_lx(mat):
        op = make_operator(getkey, mat, tags)
        if use_state:
            op_dyn, op_st = eqx.partition(op, eqx.is_inexact_array)
            op_stopped = eqx.combine(lax.stop_gradient(op_dyn), op_st)
            state = solver.init(op_stopped, {})
            _, lad = lx.slogdet(op, solver, state=state)
        else:
            _, lad = lx.slogdet(op, solver)
        return lad

    def lad_jax(mat):
        return jnp.linalg.slogdet(mat)[1]

    _, lad_dot_lx = jax.jvp(lad_lx, (matrix,), (t_matrix,))
    _, lad_dot_jax = jax.jvp(lad_jax, (matrix,), (t_matrix,))
    assert jnp.allclose(lad_dot_lx, lad_dot_jax, atol=1e-8), (
        f"lad_dot {lad_dot_lx} vs jax {lad_dot_jax}"
    )


@pytest.mark.parametrize("solver", (lx.LU(), lx.QR()))
def test_slogdet_jvp_complex(solver, getkey):
    # For complex `A`, `log det A = log|det A| + i*arg(det A)`, so the tangent of the
    # complex `sign = det/|det|` is non-trivial. This exercises the `sign_dot` branch of
    # the custom JVP (dormant for real inputs) against `jnp.linalg.slogdet`.
    matrix = jr.normal(getkey(), (3, 3), dtype=jnp.complex128)
    t_matrix = jr.normal(getkey(), (3, 3), dtype=jnp.complex128)

    def slogdet_lx(mat):
        return lx.slogdet(lx.MatrixLinearOperator(mat), solver)

    def slogdet_jax(mat):
        return jnp.linalg.slogdet(mat)

    (s_lx, l_lx), (sd_lx, ld_lx) = jax.jvp(slogdet_lx, (matrix,), (t_matrix,))
    (s_jax, l_jax), (sd_jax, ld_jax) = jax.jvp(slogdet_jax, (matrix,), (t_matrix,))
    assert jnp.allclose(l_lx, l_jax, atol=1e-8), f"lad {l_lx} vs {l_jax}"
    assert jnp.allclose(ld_lx, ld_jax, atol=1e-8), f"lad_dot {ld_lx} vs {ld_jax}"
    assert jnp.allclose(s_lx, s_jax, atol=1e-8), f"sign {s_lx} vs {s_jax}"
    assert jnp.allclose(sd_lx, sd_jax, atol=1e-8), f"sign_dot {sd_lx} vs {sd_jax}"


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize(
    "solver,tags",
    [
        (lx.LU(), ()),
        (lx.QR(), ()),
    ],
)
def test_slogdet_grad(make_operator, solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)

    def lad_lx(mat):
        op = make_operator(getkey, mat, tags)
        return lx.slogdet(op, solver)[1]

    def lad_jax(mat):
        return jnp.linalg.slogdet(mat)[1]

    grad_lx = jax.grad(lad_lx)(matrix)
    grad_jax = jax.grad(lad_jax)(matrix)
    assert jnp.allclose(grad_lx, grad_jax, atol=1e-8), (
        f"max diff {jnp.max(jnp.abs(grad_lx - grad_jax))}"
    )


def test_slogdet_grad_singular_pseudodet(getkey):
    # A rank-deficient Hermitian operator: differentiating the log-pseudodeterminant
    # via HEVD (a pseudoinverse solver) must succeed without raising, even though the
    # JVP's internal tangent solves use `throw=True`. The derivative is `trace(A⁺ dA)`,
    # which is finite despite `A` being singular.
    n, r = 5, 3
    factor = jr.normal(getkey(), (n, r), dtype=jnp.float64)
    matrix = factor @ factor.T  # symmetric PSD, rank r < n -> singular

    def lad_lx(mat):
        op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
        return lx.slogdet(op, lx.HEVD())[1]

    grad = jax.grad(lad_lx)(matrix)
    assert jnp.all(jnp.isfinite(grad)), grad


# ----------------------------------------------------------------------------
# Second-order AD: JVP of JVP, compared against jnp.linalg.slogdet
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver,tags",
    [
        (lx.LU(), ()),
        (lx.QR(), ()),
    ],
)
def test_slogdet_jvp_jvp(solver, tags, getkey):
    (matrix, t1, t2) = construct_matrix(getkey, solver, tags, num=3)

    def lad_lx(mat):
        return lx.slogdet(lx.MatrixLinearOperator(mat), solver)[1]

    def lad_jax(mat):
        return jnp.linalg.slogdet(mat)[1]

    inner_lx = lambda m: jax.jvp(lad_lx, (m,), (t1,))[1]
    inner_jax = lambda m: jax.jvp(lad_jax, (m,), (t1,))[1]

    _, dot2_lx = jax.jvp(inner_lx, (matrix,), (t2,))
    _, dot2_jax = jax.jvp(inner_jax, (matrix,), (t2,))
    assert jnp.allclose(dot2_lx, dot2_jax, atol=1e-6), (
        f"jvp_jvp {dot2_lx} vs jax {dot2_jax}"
    )
