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


from collections.abc import Callable

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax as lx
import lineax._determinant as _determinant
import pytest
from lineax._solver.tridiagonal import _slogdet_associative_reduce, _slogdet_scan

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


def test_default_solver(getkey):
    # `determinant`/`slogdet` default to `AutoLinearSolver(well_posed=True)`, matching
    # `linear_solve`, so a solver need not be passed explicitly.
    (matrix,) = construct_matrix(getkey, lx.LU(), ())
    op = lx.MatrixLinearOperator(matrix)
    assert jnp.allclose(lx.determinant(op), jnp.linalg.det(matrix), atol=1e-10)
    sign, lad = lx.slogdet(op)
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(sign, ref_sign, atol=1e-10)
    assert jnp.allclose(lad, ref_lad, atol=1e-10)


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


def test_slogdet_rejects_iterative_solver(getkey):
    """A bare iterative solver fails up front with a pointed message.

    `Normal` checks its own inner solver, but an iterative solver passed directly
    would otherwise sail through `init` and only die on the missing `slogdet`
    attribute.
    """
    matrix = construct_matrix(getkey, lx.LU(), ())[0]
    op = lx.MatrixLinearOperator(matrix)
    # Under the test suite's jaxtyping/beartype import hook, the annotation itself
    # rejects the call; in a plain run, the isinstance check in `slogdet` does. Both
    # raise `TypeError`, with different messages.
    with pytest.raises(TypeError, match="requires a direct solver|Expected type"):
        lx.slogdet(op, lx.GMRES(rtol=1e-6, atol=1e-6))  # pyright: ignore


def test_slogdet_unit_diagonal_dtype():
    """`Diagonal` on a unit-diagonal operator: `sign` takes the operator's dtype.

    The determinant is one whatever the dtype, but a complex operator should get a
    complex `sign`, as every other path arranges.
    """
    for dtype in (jnp.float64, jnp.complex128):
        structure = jax.ShapeDtypeStruct((3,), dtype)
        operator = lx.FunctionLinearOperator(
            lambda x: x, structure, tags=(lx.diagonal_tag, lx.unit_diagonal_tag)
        )
        sign, lad = lx.slogdet(operator, lx.Diagonal(well_posed=True))
        assert sign.dtype == dtype
        assert lad.dtype == jnp.finfo(dtype).dtype
        assert jnp.allclose(sign, 1.0)
        assert jnp.allclose(lad, 0.0)


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
# Tridiagonal slogdet: the division-free minor recurrence
# ----------------------------------------------------------------------------


def test_tridiagonal_slogdet_singular_leading_minor():
    """An invertible operator whose leading principal minor is singular.

    `d[0] == 0` makes the LU-pivot recurrence divide by zero on the very first
    step; the minor recurrence is division-free and handles it.
    """
    matrix = jnp.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 2.0]], dtype=jnp.float64
    )
    op = lx.MatrixLinearOperator(matrix, lx.tridiagonal_tag)
    sign, lad = lx.slogdet(op, lx.Tridiagonal())
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(sign, ref_sign, atol=1e-10), f"sign: {sign} vs {ref_sign}"
    assert jnp.allclose(lad, ref_lad, atol=1e-10), f"lad: {lad} vs {ref_lad}"


@pytest.mark.parametrize("n", [1, 2, 3, 16, 17, 33, 256])
def test_tridiagonal_slogdet_block_boundaries(n, getkey):
    """Sizes either side of the renormalisation block length, plus n=1."""
    key = getkey()
    diagonal = jr.normal(key, (n,), dtype=jnp.float64) + 4.0
    off = jr.normal(getkey(), (2, max(n - 1, 0)), dtype=jnp.float64) * 0.5
    matrix = (
        jnp.diag(diagonal) + jnp.diag(off[0], -1) + jnp.diag(off[1], 1)
        if n > 1
        else jnp.diag(diagonal)
    )
    op = lx.MatrixLinearOperator(matrix, lx.tridiagonal_tag)
    sign, lad = lx.slogdet(op, lx.Tridiagonal())
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(sign, ref_sign, atol=1e-10)
    assert jnp.allclose(lad, ref_lad, rtol=1e-10)


def test_tridiagonal_slogdet_no_overflow():
    """A large well-conditioned operator: the raw minors would overflow float64.

    det of this operator is ~exp(1400), so an unscaled three-term recurrence
    returns `inf`. The per-block renormalisation keeps `lad` finite and exact.
    """
    n = 1024
    diagonal = jnp.full((n,), 4.0, dtype=jnp.float64)
    off = jnp.full((n - 1,), 0.5, dtype=jnp.float64)
    matrix = jnp.diag(diagonal) + jnp.diag(off, -1) + jnp.diag(off, 1)
    op = lx.MatrixLinearOperator(matrix, lx.tridiagonal_tag)
    _, lad = lx.slogdet(op, lx.Tridiagonal())
    _, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.isfinite(lad)
    assert jnp.allclose(lad, ref_lad, rtol=1e-10)


@pytest.mark.parametrize("scale", [1e-90, 1e-20, 1e20, 1e90])
def test_tridiagonal_slogdet_badly_scaled(scale, getkey):
    """Uniformly scaled operators: the up-front power-of-two prescale handles these.

    Without it the minors leave float range inside a single block, since they grow or
    decay by roughly one entry-magnitude per step. The extremes here are near the limit
    of what the prescale can express: it needs an exponent for `sigma`, so it gives up
    in the outermost binade at each end (see `_prescale`). The moderate scales are the
    ones a caller is likely to have, and they pass unaided -- the per-block
    renormalisation alone survives about 1e+-60.
    """
    n = 128
    diagonal = (jr.normal(getkey(), (n,), dtype=jnp.float64) + 4.0) * scale
    off = jr.normal(getkey(), (2, n - 1), dtype=jnp.float64) * 0.5 * scale
    matrix = jnp.diag(diagonal) + jnp.diag(off[0], -1) + jnp.diag(off[1], 1)
    op = lx.MatrixLinearOperator(matrix, lx.tridiagonal_tag)
    sign, lad = lx.slogdet(op, lx.Tridiagonal())
    # `jnp.linalg.slogdet` is itself fine here; only `det` would overflow.
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.isfinite(lad)
    assert jnp.allclose(sign, ref_sign, atol=1e-10)
    assert jnp.allclose(lad, ref_lad, rtol=1e-12)


@pytest.mark.parametrize("span", [12.0, 40.0])
def test_tridiagonal_slogdet_graded(span, getkey):
    """Entries spanning 10**-span within one operator.

    The renormalisation block length caps this: a block underflows once the minors
    decay past float range within it. At `_SLOGDET_BLOCK = 4` the measured limit is
    `span ~ 63`, so 40 has comfortable margin -- but raising the block length to 8
    would drop the limit to 38 and fail this case. See the block comment beside
    `_SLOGDET_BLOCK`, which also records the silently-wrong band just above the limit.
    """
    n = 128
    grade = 10.0 ** (-span * jnp.arange(n, dtype=jnp.float64) / n)
    diagonal = (jr.normal(getkey(), (n,), dtype=jnp.float64) + 4.0) * grade
    off = jr.normal(getkey(), (2, n - 1), dtype=jnp.float64) * 0.5 * grade[None, :-1]
    matrix = jnp.diag(diagonal) + jnp.diag(off[0], -1) + jnp.diag(off[1], 1)
    op = lx.MatrixLinearOperator(matrix, lx.tridiagonal_tag)
    sign, lad = lx.slogdet(op, lx.Tridiagonal())
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(sign, ref_sign, atol=1e-10)
    assert jnp.allclose(lad, ref_lad, rtol=1e-12)


def test_tridiagonal_slogdet_graded_float32(getkey):
    """As `test_tridiagonal_slogdet_graded`, in float32.

    The limit on tolerated grading is set by the exponent range, so float32 divides it
    by about eight: 8 decades rather than 63 at `_SLOGDET_BLOCK = 4`. 5 leaves margin.
    """
    n = 128
    span = 5.0
    grade = (10.0 ** (-span * jnp.arange(n, dtype=jnp.float32) / n)).astype(jnp.float32)
    diagonal = (jr.normal(getkey(), (n,), dtype=jnp.float32) + 4.0) * grade
    off = jr.normal(getkey(), (2, n - 1), dtype=jnp.float32) * 0.5 * grade[None, :-1]
    matrix = jnp.diag(diagonal) + jnp.diag(off[0], -1) + jnp.diag(off[1], 1)
    op = lx.MatrixLinearOperator(matrix, lx.tridiagonal_tag)
    _, lad = lx.slogdet(op, lx.Tridiagonal())
    _, ref_lad = jnp.linalg.slogdet(matrix.astype(jnp.float64))
    # Measured error over eight draws is 7e-10 to 2.4e-7, so this still leaves 10x.
    assert jnp.allclose(lad, ref_lad.astype(jnp.float32), rtol=1e-6)


@pytest.mark.parametrize("power", [0, 40, 77, 150])
def test_tridiagonal_slogdet_asymmetric_off_diagonals(power):
    """Off-diagonals of wildly different sizes but a coupling of exactly 1.

    The minor recurrence only ever sees `lower * upper`, so this operator is a
    permuted identity: its determinant is exactly 1 for every `power`. The prescale
    therefore has to measure the coupling as `sqrt(|l u|)` -- taking `max(|l|, |u|)`
    instead rescales by 10**power too much and underflows the whole recurrence.
    """
    n = 6
    diagonal = jnp.ones(n, dtype=jnp.float64)
    lower = jnp.full(n - 1, 10.0**power, dtype=jnp.float64)
    upper = jnp.full(n - 1, 10.0**-power, dtype=jnp.float64)
    op = lx.TridiagonalLinearOperator(diagonal, lower, upper)
    sign, lad = lx.slogdet(op, lx.Tridiagonal())
    assert sign == 1
    assert jnp.allclose(lad, 0.0, atol=1e-12)


@pytest.mark.parametrize("span", [0.0, 40.0, 60.0])
def test_tridiagonal_slogdet_graded_jvp(span, getkey):
    """The gradient has to tolerate the same grading as the primal.

    The renormalisation scale cancels out of `lad` exactly, so its tangent is zero --
    but only if we say so. Letting AD differentiate it costs a `1/scale**2`, which
    overflows once a block's scale falls below sqrt(smallest normal), i.e. at half
    the grading the primal handles. That put the cliff at `span = 40`, which is
    exactly what `test_tridiagonal_slogdet_graded` pins. The two implementations have
    different cliffs -- 40 for the scan, 55 for the tree -- so the largest span here
    has to clear both, or this is vacuous on one platform.
    """
    n = 64
    grade = 10.0 ** (-span * jnp.arange(n, dtype=jnp.float64) / n)
    diagonal = (jr.normal(getkey(), (n,), dtype=jnp.float64) + 4.0) * grade
    off = jr.normal(getkey(), (2, n - 1), dtype=jnp.float64) * 0.5 * grade[None, :-1]
    matrix = jnp.diag(diagonal) + jnp.diag(off[0], -1) + jnp.diag(off[1], 1)

    def lad_of(m):
        return lx.slogdet(lx.MatrixLinearOperator(m, lx.tridiagonal_tag))[1]

    grad = jax.grad(lad_of)(matrix)
    assert jnp.all(jnp.isfinite(grad))
    # d(log|det A|)/dA = (A^-1)^T, of which only the band is meaningful here.
    expected = jnp.linalg.inv(matrix).T
    for k in (-1, 0, 1):
        assert jnp.allclose(jnp.diag(grad, k), jnp.diag(expected, k), rtol=1e-8)

    # Again against each implementation directly. Going through `lx.slogdet` only
    # exercises whichever one `lax.platform_dependent` picks here, and the two carry
    # their own `stop_gradient`s with their own cliffs.
    want = (jnp.diag(expected), jnp.diag(expected, -1), jnp.diag(expected, 1))
    for impl in (_slogdet_scan, _slogdet_associative_reduce):
        direct = jax.grad(lambda a, b, c: impl(a, b, c)[1], argnums=(0, 1, 2))(
            diagonal, off[0], off[1]
        )
        for actual, target in zip(direct, want):
            assert jnp.all(jnp.isfinite(actual))
            assert jnp.allclose(actual, target, rtol=1e-8)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.complex128])
@pytest.mark.parametrize("n", [1, 2, 3, 5, 16, 17, 18, 25, 64, 129])
def test_tridiagonal_slogdet_implementations_agree(n, dtype, getkey):
    """`lax.platform_dependent` runs only one implementation per platform.

    So CI on CPU never exercises `_slogdet_associative_reduce` and CI on GPU never
    exercises `_slogdet_scan`, and nothing otherwise compares them. Call both
    directly on whatever device is to hand: they compute the same recurrence and must
    agree, with each other and with a dense reference.

    `n = 18` and `n = 25` are here because they are the smallest sizes that reach
    `_reduce_level`'s identity-padding branch (`n_chunks` not a multiple of the radix);
    every other size in this file skips it, and a wrong pad stays finite.
    """
    tol = 1e-4 if dtype == jnp.float32 else 1e-10
    diagonal = jr.normal(getkey(), (n,), dtype=dtype) + 3.0
    lower = jr.normal(getkey(), (n - 1,), dtype=dtype)
    upper = jr.normal(getkey(), (n - 1,), dtype=dtype)
    matrix = jnp.diag(diagonal) + jnp.diag(lower, -1) + jnp.diag(upper, 1)
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)

    scan_sign, scan_lad = _slogdet_scan(diagonal, lower, upper)
    assert jnp.allclose(scan_lad, ref_lad, rtol=tol)
    assert jnp.allclose(scan_sign, ref_sign, rtol=tol, atol=tol)
    if n > 1:
        tree_sign, tree_lad = _slogdet_associative_reduce(diagonal, lower, upper)
        assert jnp.allclose(tree_lad, ref_lad, rtol=tol)
        # Against the reference, not just against the scan: two identically wrong
        # signs would agree with each other.
        assert jnp.allclose(tree_sign, ref_sign, rtol=tol, atol=tol)
        assert jnp.allclose(scan_sign, tree_sign, rtol=tol, atol=tol)
    else:
        # `_slogdet_associative_reduce` cannot do n = 1: there are no recurrence
        # steps, so the chunked scan has nothing to reduce. This is why
        # `Tridiagonal.slogdet` special-cases it before dispatching, and CPU cannot
        # otherwise see that --
        # `lax.platform_dependent` only traces the branch it will run.
        with pytest.raises(IndexError):
            _slogdet_associative_reduce(diagonal, lower, upper)

    op = lx.TridiagonalLinearOperator(diagonal, lower, upper)
    assert jnp.allclose(lx.slogdet(op, lx.Tridiagonal())[1], ref_lad, rtol=tol)


@pytest.mark.parametrize("n", [16, 18, 64])
def test_tridiagonal_slogdet_implementation_gradients_agree(n, getkey):
    """As above, for the backward pass.

    The `stop_gradient` on the renormalisation scale is separately present in each
    implementation, so a test that only differentiates through the shipped dispatch
    pins whichever one this platform happens to run.
    """
    diagonal = jr.normal(getkey(), (n,), dtype=jnp.float64) + 3.0
    lower = jr.normal(getkey(), (n - 1,), dtype=jnp.float64)
    upper = jr.normal(getkey(), (n - 1,), dtype=jnp.float64)
    matrix = jnp.diag(diagonal) + jnp.diag(lower, -1) + jnp.diag(upper, 1)
    # d(log|det A|)/dA = (A^-1)^T, of which only the band is meaningful.
    inverse = jnp.linalg.inv(matrix).T
    expected = (jnp.diag(inverse), jnp.diag(inverse, -1), jnp.diag(inverse, 1))
    for impl in (_slogdet_scan, _slogdet_associative_reduce):
        grad = jax.grad(lambda a, b, c: impl(a, b, c)[1], argnums=(0, 1, 2))(
            diagonal, lower, upper
        )
        for actual, want in zip(grad, expected):
            assert jnp.allclose(actual, want, rtol=1e-10)


def test_tridiagonal_slogdet_dispatch_is_platform_correct(getkey):
    """The scan belongs on CPU and the associative reduce everywhere else.

    Both implementations are correct, so no comparison of values can see this
    inverted -- but doing so costs 202x on GPU at n = 8192, and about 2x on CPU. They
    do different amounts of arithmetic, though (the reduce multiplies 2x2 matrices
    where the scan takes two products per step), so the compiled FLOP count identifies
    which one was chosen without depending on how either happens to be written.
    """
    n = 1024
    diagonal = jr.normal(getkey(), (n,), dtype=jnp.float64) + 4.0
    lower = jr.normal(getkey(), (n - 1,), dtype=jnp.float64) * 0.3
    upper = jr.normal(getkey(), (n - 1,), dtype=jnp.float64) * 0.3
    solver = lx.Tridiagonal()
    state = solver.init(lx.TridiagonalLinearOperator(diagonal, lower, upper), {})

    dispatched = _flops(lambda s: solver.slogdet(s, {}), state)
    if dispatched is None:
        pytest.skip("backend does not report a FLOP count")
    arrays = (diagonal, lower, upper)
    scan = _flops(lambda *a: _slogdet_scan(*a)[1], *arrays)
    reduce_ = _flops(lambda *a: _slogdet_associative_reduce(*a)[1], *arrays)
    assert scan != reduce_, "the two implementations have become indistinguishable"
    expected = scan if jax.default_backend() == "cpu" else reduce_
    # Not exactly equal: `Tridiagonal.slogdet` adds the n = 1 guard around the call.
    assert abs(dispatched - expected) <= 8, (
        f"dispatched {dispatched}, scan {scan}, associative reduce {reduce_}"
    )


def test_tridiagonal_slogdet_singular():
    """A singular operator gives `(0, -inf)`, not `nan`.

    Only for this operator, not in general: the division-free minor recurrence is not
    backward stable when the answer is exactly zero, so a singular operator whose zero
    determinant arises from cancellation across many steps can come out as a small
    finite value instead, and the two implementations need not agree on which. Detecting
    exact singularity is not something this solver promises; use a rank-revealing
    factorisation (`lx.SVD`) if that is what you need.
    """
    matrix = jnp.array(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64
    )
    op = lx.MatrixLinearOperator(matrix, lx.tridiagonal_tag)
    sign, lad = lx.slogdet(op, lx.Tridiagonal())
    assert sign == 0
    assert lad == -jnp.inf


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


def _flops(fn, *args):
    """Compiled FLOP count, or `None` where the backend does not report one.

    Not a foolproof measure. It is XLA's own accounting, it can under-report work
    inside an FFI call, and nothing obliges it to stay stable across JAX versions. So
    the two tests using it are checked empirically rather than trusted: on jax 0.11.0,
    between them they catch every failure we actually expect -- inverting the platform
    dispatch, the fast JVP going quadratic, and dropping any one of the four solvers
    from `_jvp_through_state`. If a JAX upgrade makes them flaky, delete
    them rather than tune the numbers. They guard performance properties, which
    `benchmarks/determinant_speeds.py` also reports.
    """
    analysis = jax.jit(fn).lower(*args).compile().cost_analysis()
    if analysis is None:
        return None
    if isinstance(analysis, list | tuple):
        analysis = analysis[0]
    return analysis.get("flops")


def _structured_case(kind, n, key, complex_):
    """(operator, tangent_operator, solver) for each structure with a fast JVP."""
    dtype = jnp.complex128 if complex_ else jnp.float64
    keys = jr.split(key, 2)
    make: Callable
    args: Callable
    if kind == "diagonal":
        make = lx.DiagonalLinearOperator
        args = lambda k: (jr.normal(k, (n,), dtype=dtype) + 4.0,)  # noqa: E731
        solver = lx.Diagonal(well_posed=True)
    elif kind == "tridiagonal":
        make = lx.TridiagonalLinearOperator
        args = lambda k: (  # noqa: E731
            jr.normal(jr.split(k)[0], (n,), dtype=dtype) + 4.0,
            jr.normal(jr.split(k)[1], (n - 1,), dtype=dtype) * 0.3,
            jr.normal(k, (n - 1,), dtype=dtype) * 0.3,
        )
        solver = lx.Tridiagonal()
    elif kind == "circulant":
        make = lx.CirculantLinearOperator
        args = lambda k: (  # noqa: E731
            jr.normal(k, (n,), dtype=dtype).at[0].add(n),
        )
        solver = lx.Circulant(well_posed=True)
    elif kind == "triangular":
        make = lambda matrix: (  # noqa: E731
            lx.MatrixLinearOperator(matrix, lx.lower_triangular_tag)
        )
        args = lambda k: (  # noqa: E731
            jnp.tril(jr.normal(k, (n, n), dtype=dtype)) + 4.0 * jnp.eye(n, dtype=dtype),
        )
        solver = lx.Triangular()
    elif kind == "tridiagonal_tagged":
        # A dense matrix that merely *promises* to be tridiagonal: the fast path has to
        # reach it through `tridiagonal(...)`, including on the tangent operator.
        def _tagged_args(k):
            k0, k1, k2 = jr.split(k, 3)
            d = jr.normal(k0, (n,), dtype=dtype) + 4.0
            lo = jr.normal(k1, (n - 1,), dtype=dtype) * 0.3
            up = jr.normal(k2, (n - 1,), dtype=dtype) * 0.3
            return (jnp.diag(d) + jnp.diag(lo, -1) + jnp.diag(up, 1),)

        make = lambda matrix: (  # noqa: E731
            lx.MatrixLinearOperator(matrix, lx.tridiagonal_tag)
        )
        args = _tagged_args
        solver = lx.Tridiagonal()
    else:
        raise AssertionError(kind)
    return make(*args(keys[0])), make(*args(keys[1])), solver


@pytest.mark.parametrize(
    "kind",
    ["diagonal", "triangular", "tridiagonal", "tridiagonal_tagged", "circulant"],
)
@pytest.mark.parametrize("complex_", [False, True])
def test_slogdet_structured_jvp(kind, complex_, getkey):
    """Structured operators take a fast JVP path; it must match `trace(A^-1 dA)`.

    `lx.slogdet`'s generic rule solves once per column of the tangent. For operators
    whose determinant is a cheap pure-JAX function of their structure we differentiate
    that function instead, which is O(n) (O(n log n) for circulant). This checks the
    two agree, on the sign as well as the log-magnitude -- the sign tangent is only
    nonzero in the complex case, and is where `jnp.sign` would silently give zero.
    """
    n = 32
    op, t_op, solver = _structured_case(kind, n, getkey(), complex_)
    (sign, lad), (sign_dot, lad_dot) = jax.jvp(
        lambda o: lx.slogdet(o, solver), (op,), (t_op,)
    )
    matrix = op.as_matrix()
    t_matrix = t_op.as_matrix()
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    trace = jnp.trace(jnp.linalg.solve(matrix, t_matrix))
    ref_lad_dot = jnp.real(trace)
    if complex_:
        # The sign carries the imaginary part of the trace; cast explicitly, as lineax
        # runs under strict dtype promotion.
        ref_sign_dot = (trace - ref_lad_dot.astype(trace.dtype)) * ref_sign
    else:
        ref_sign_dot = jnp.zeros_like(sign)
    assert jnp.allclose(sign, ref_sign, atol=1e-10)
    assert jnp.allclose(lad, ref_lad, rtol=1e-10)
    assert jnp.allclose(lad_dot, ref_lad_dot, rtol=1e-8)
    assert jnp.allclose(sign_dot, ref_sign_dot, atol=1e-8)


@pytest.mark.parametrize("use_default_solver", [False, True])
@pytest.mark.parametrize("kind", ["diagonal", "tridiagonal", "circulant"])
def test_slogdet_structured_jvp_is_not_quadratic(kind, use_default_solver, getkey):
    """The fast JVP must not cost O(n**2).

    The generic rule solves once per column, in both work and memory -- at n = 65536
    it needs a 34GB tangent -- so a regression here is a memory blow-up rather than a
    slowdown. Rather than differentiate at a size whose dense tangent would not fit,
    which turns a regression into an OOM of the test runner, compare the compiled FLOP
    count at two sizes: the fast path grows by 1.9x to 2.3x per doubling (n, or
    n log n for `Circulant`), the generic rule by 4.0x to 4.4x. Both figures are the
    same on CPU and GPU, so the threshold is not platform-specific tuning.

    `diagonal` is pinned here too, although it takes the solve-based rule rather than
    the fast path: `trace(A^-1 dA)` dispatches structurally to one solve and an
    elementwise product, and this is what holds that at O(n).

    A FLOP count can under-report, since work inside an FFI custom call is opaque to
    it -- but not here, and not by luck: what makes the generic rule quadratic in this
    measurement is materialising the dense tangent and carrying n columns through the
    solve, which is ordinary XLA array traffic. Whatever the solve itself costs is on
    top of a signal that is already unambiguous.

    Only structures whose operator is O(n) to store are here. For one held as a dense
    matrix the operator itself is quadratic, so a growth rate cannot separate the two
    paths; `Triangular`'s membership of the fast set is pinned semantically by
    `test_slogdet_unit_diagonal_jvp` instead.

    This pins the cost of the fast path alone, and deliberately says nothing about
    whether the fast path was chosen: it compares against a fixed expectation of
    linear growth, not against the fallback, so it would go quiet if the fallback ever
    stopped being quadratic. `test_slogdet_fast_path_beats_the_fallback` is what
    guards the choice.

    `use_default_solver` covers the `AutoLinearSolver` look-through, which nothing else
    pins: without it the default solver silently falls back to the generic rule.
    """
    counts = []
    for n in (128, 256):
        op, _, solver = _structured_case(kind, n, getkey(), False)
        if use_default_solver:
            solver = None

        def lad(o, solver=solver):
            return lx.slogdet(o)[1] if solver is None else lx.slogdet(o, solver)[1]

        flops = _flops(jax.grad(lad), op)
        if flops is None:
            pytest.skip("backend does not report a FLOP count")
        counts.append(flops)
    growth = counts[1] / counts[0]
    assert growth < 3.0, f"gradient FLOPs grew {growth:.2f}x per doubling"


@pytest.mark.parametrize("use_default_solver", [False, True])
@pytest.mark.parametrize(
    "kind",
    ["tridiagonal", "tridiagonal_tagged", "triangular", "circulant"],
)
def test_slogdet_fast_path_beats_the_fallback(kind, use_default_solver, getkey):
    """Differentiating these solvers directly must be cheaper than not.

    This is the property that justifies the list in `_jvp_through_state` at all, so
    assert it against whatever the fallback currently is, rather than against a fixed
    idea of what the fallback costs. Today that is the structural
    `trace(A^-1 dA)` rule. If the fallback ever gets cheaper still, this keeps
    comparing like for like, and the two outcomes are both useful: it still passes if
    direct differentiation remains the better route, and it fails if the fallback
    overtakes it, which is exactly when someone should be reconsidering the list
    rather than trusting it.

    A FLOP count can under-report, since work inside an FFI custom call is opaque to
    it, and a structural trace would reach `A^-1` through one -- `Tridiagonal.compute`
    is `lax.linalg.tridiagonal_solve`. Measured, it is not opaque: materialising a
    tridiagonal inverse costs 12 FLOPs per matrix entry on CPU and 10 on GPU, growing
    4.00x per doubling, against 2 per entry for a `Diagonal` solve that involves no
    FFI at all. Extracting a diagonal of `A^-1` needs `A^-1`, so that quadratic cost
    is on the fallback's critical path and this comparison can see it.

    This already happened once: `diagonal(invert(A))` needs no materialisation and is
    linear, so the structural trace costs 0.75x what differentiating `Diagonal.slogdet`
    directly did (measured at every n from 128 to 1024) -- which is why `Diagonal` is
    no longer in the list and no longer in this parametrisation. `Tridiagonal` goes
    the other way -- because the band of its inverse cannot be had without the
    inverse.

    Contrast `test_slogdet_structured_jvp_is_not_quadratic`, which pins the cost of
    the fast path alone and would go quiet if the fallback stopped being quadratic.
    """
    n = 256
    op, _, solver = _structured_case(kind, n, getkey(), False)
    if use_default_solver:
        solver = None

    def lad(o):
        return lx.slogdet(o)[1] if solver is None else lx.slogdet(o, solver)[1]

    fast = _flops(jax.grad(lad), op)
    if fast is None:
        pytest.skip("backend does not report a FLOP count")
    # Force the fallback for the same operator and solver. Nothing else can produce
    # it: the choice is made on the solver's type, so there is no operator to pass
    # that would take the slow route while staying comparable.
    original = _determinant._jvp_through_state
    try:
        _determinant._jvp_through_state = lambda solver, operator: False
        fallback = _flops(jax.grad(lad), op)
    finally:
        _determinant._jvp_through_state = original
    assert fast < fallback, (
        f"differentiating {kind} directly costs {fast} FLOPs against {fallback} for "
        "the fallback, so it is no longer worth special-casing"
    )


@pytest.mark.parametrize(
    "kind",
    ["diagonal", "tridiagonal", "tridiagonal_tagged", "triangular", "circulant"],
)
@pytest.mark.parametrize("complex_", [False, True])
def test_slogdet_structured_grad(kind, complex_, getkey):
    """Reverse mode through the fast path.

    `jax.jvp` is what the rule defines, but `jax.grad` is what a loss function calls,
    and it is the transposition rather than the rule itself that a change to the
    implementation is most likely to break. Checking the transposition identity
    `<grad f, t> == jvp(f, t)` rather than a dense reference keeps this agnostic to
    how many matrix entries each stored parameter appears in -- a circulant column
    entry appears n times, a diagonal entry once.
    """
    n = 16
    op, t_op, solver = _structured_case(kind, n, getkey(), complex_)

    def lad(o):
        return lx.slogdet(o, solver)[1]

    _, tangent = jax.jvp(lad, (op,), (t_op,))
    grad = jax.grad(lad)(op)
    # `jax.grad` of a real-valued function of a complex input returns the conjugated
    # gradient, so the plain product's real part is the directional derivative.
    paired = sum(
        jnp.sum(g * t).real
        for g, t in zip(jtu.tree_leaves(grad), jtu.tree_leaves(t_op))
    )
    assert jnp.allclose(paired, tangent, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("kind", ["diagonal", "tridiagonal", "circulant"])
def test_slogdet_structured_jvp_jvp(kind, getkey):
    """Second order. `test_slogdet_jvp_jvp` covers only LU and QR."""
    n = 8
    op, t_op, solver = _structured_case(kind, n, getkey(), False)

    def lad(o):
        return lx.slogdet(o, solver)[1]

    def dense_lad(m):
        return jnp.linalg.slogdet(m)[1]

    def jvp_of(f, primal, tangent):
        return jax.jvp(f, (primal,), (tangent,))[1]

    second = jax.jvp(lambda o: jvp_of(lad, o, t_op), (op,), (t_op,))[1]
    matrix = op.as_matrix()
    t_matrix = lx.TangentLinearOperator(op, t_op).as_matrix()
    expected = jax.jvp(
        lambda m: jvp_of(dense_lad, m, t_matrix), (matrix,), (t_matrix,)
    )[1]
    assert jnp.allclose(second, expected, rtol=1e-8)


@pytest.mark.parametrize("kind", ["diagonal", "tridiagonal"])
def test_slogdet_structured_vmap(kind, getkey):
    """`lx.slogdet` under `vmap`, primal and gradient. Nothing else covers vmap."""
    n = 8
    batch = 3
    ops = []
    solver = None
    for _ in range(batch):
        op, _, solver = _structured_case(kind, n, getkey(), False)
        ops.append(op)
    assert solver is not None
    stacked = jtu.tree_map(lambda *xs: jnp.stack(xs), *ops)

    def lad(o):
        return lx.slogdet(o, solver)[1]

    batched = jax.vmap(lad)(stacked)
    expected = jnp.stack([lad(o) for o in ops])
    assert jnp.allclose(batched, expected, rtol=1e-10)
    batched_grad = jax.vmap(jax.grad(lad))(stacked)
    expected_grad = jtu.tree_map(
        lambda *xs: jnp.stack(xs), *[jax.grad(lad)(o) for o in ops]
    )
    for actual, want in zip(
        jtu.tree_leaves(batched_grad), jtu.tree_leaves(expected_grad)
    ):
        assert jnp.allclose(actual, want, rtol=1e-10)


def test_slogdet_structured_state_is_differentiable(getkey):
    """For the solvers we differentiate directly, `state` carries the derivative.

    The state is the thing being differentiated -- the fast rule differentiates the
    solver's own `slogdet`, whose only input is the state -- so `lx.slogdet` hands it
    over with its tangent intact rather than stopping it. That keeps a single `init` in
    the graph, where rebuilding it inside the rule would leave two and lean on the
    compiler to notice they are the same. Three consequences, each differing from the
    generic rule:

    1. Not passing a state works as before, the state being built from `operator`.
    2. Passing one built *inside* the differentiated function differentiates through
       it, which is the case a caller sharing a factorisation actually has.
    3. Passing one built outside gives a zero derivative -- correctly, since then the
       value does not depend on `operator` at all. `jax.jvp`'s primal agrees with the
       undifferentiated call, which is what makes that self-consistent.
    """
    solver = lx.Tridiagonal()
    off = jnp.zeros(2)
    diagonal = jnp.array([2.0, 3.0, 4.0])
    make = lambda d: lx.TridiagonalLinearOperator(d, off, off)
    operator = make(diagonal)
    t_operator = make(jnp.ones(3))
    expected = jnp.sum(1.0 / diagonal)

    primal, tangent = jax.jvp(
        lambda o: lx.slogdet(o, solver)[1], (operator,), (t_operator,)
    )
    assert jnp.allclose(primal, jnp.sum(jnp.log(diagonal)))
    assert jnp.allclose(tangent, expected)

    def shared(o):
        return lx.slogdet(o, solver, state=solver.init(o, {}))[1]

    primal, tangent = jax.jvp(shared, (operator,), (t_operator,))
    assert jnp.allclose(primal, jnp.sum(jnp.log(diagonal)))
    assert jnp.allclose(tangent, expected)

    foreign_diagonal = jnp.array([10.0, 20.0, 30.0])
    foreign = solver.init(make(foreign_diagonal), {})

    def with_foreign(o):
        return lx.slogdet(o, solver, state=foreign)[1]

    primal, tangent = jax.jvp(with_foreign, (operator,), (t_operator,))
    assert jnp.allclose(primal, with_foreign(operator))
    assert jnp.allclose(primal, jnp.sum(jnp.log(foreign_diagonal)))
    assert jnp.allclose(tangent, 0.0)


def test_slogdet_generic_state_stays_nondifferentiable():
    """The guard is lifted only for the solvers that need it.

    Every other solver gets a state that is a factorisation the generic rule merely
    solves against, where differentiating it would be a mistake rather than the point.
    """
    operator = lx.MatrixLinearOperator(jnp.eye(3) * 2.0)
    t_operator = lx.MatrixLinearOperator(jnp.eye(3))
    primal, tangent = jax.jvp(
        lambda o: lx.slogdet(o, lx.LU())[1], (operator,), (t_operator,)
    )
    assert jnp.allclose(primal, jnp.sum(jnp.log(jnp.full(3, 2.0))))
    assert jnp.allclose(tangent, 1.5)


def test_slogdet_singular_grad_through_state_vs_solve():
    """What a singular operator does to the gradient depends on the rule.

    A through-state solver differentiates its own `slogdet`, where `d log|det A|`
    genuinely does not exist, so a non-finite gradient is the honest answer. The
    solve-based rule instead surfaces the failed tangent solve loudly: it passes
    `throw=True`, and the non-finite solution against a singular operator raises.
    Both pinned here, since each is a deliberate choice documented beside the
    `throw=True` in `_slogdet_jvp`.
    """
    off = jnp.zeros(2)
    tri_op = lx.TridiagonalLinearOperator(jnp.array([1.0, 0.0, 3.0]), off, off)
    grad = jax.grad(lambda o: lx.slogdet(o, lx.Tridiagonal())[1])(tri_op)
    assert not jnp.all(jnp.isfinite(grad.diagonal))

    diag_op = lx.DiagonalLinearOperator(jnp.array([1.0, 0.0, 3.0]))
    with pytest.raises(Exception, match="non-finite"):
        # `block_until_ready`: the runtime error is raised when the value is
        # materialised, not when the (asynchronously dispatched) call returns.
        jax.block_until_ready(
            jax.grad(lambda o: lx.slogdet(o, lx.Diagonal(well_posed=True))[1])(diag_op)
        )


def test_slogdet_circulant_pseudodeterminant_jvp(getkey):
    """`Circulant(well_posed=False)`: the other pseudodeterminant path.

    A circulant with a zero eigenvalue -- here the all-ones operator, whose only
    non-zero eigenvalue is the DC term.
    """
    n = 4
    column = jnp.ones(n, dtype=jnp.float64)
    t_column = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    op = lx.CirculantLinearOperator(column)
    t_op = lx.CirculantLinearOperator(t_column)
    solver = lx.Circulant(well_posed=False)
    (sign, lad), (_, lad_dot) = jax.jvp(lambda o: lx.slogdet(o, solver), (op,), (t_op,))
    # The surviving eigenvalue is `sum(column) = n`, so `lad = log n` and its
    # derivative in the direction `t_column` is `sum(t_column) / n`.
    assert jnp.allclose(sign, 1.0)
    assert jnp.allclose(lad, jnp.log(jnp.asarray(float(n))))
    assert jnp.allclose(lad_dot, jnp.sum(t_column) / n)


def test_slogdet_pseudodeterminant_jvp(getkey):
    """A rank-deficient solver computes a pseudodeterminant, a different function.

    The solve-based rule needs no rank guard: `trace(A^+ dA)` solves with
    `Diagonal(well_posed=False)`, which masks the same (near-)zero entries the
    pseudodeterminant drops. This pins that, since getting it wrong would silently
    differentiate the full determinant of a singular operator.
    """
    diag = jnp.array([2.0, 3.0, 0.0, 5.0])
    t_diag = jnp.array([1.0, 1.0, 1.0, 1.0])
    op = lx.DiagonalLinearOperator(diag)
    t_op = lx.DiagonalLinearOperator(t_diag)
    solver = lx.Diagonal(well_posed=False)
    (_, lad), (_, lad_dot) = jax.jvp(lambda o: lx.slogdet(o, solver), (op,), (t_op,))
    # Pseudodeterminant over the nonzero entries: 2 * 3 * 5, and its tangent
    # sum(t / d) over those same entries.
    assert jnp.allclose(lad, jnp.log(30.0))
    assert jnp.allclose(lad_dot, 1 / 2 + 1 / 3 + 1 / 5)


@pytest.mark.parametrize("solver", ["lu", "auto"])
def test_slogdet_generic_jvp_nonflat_structure(solver):
    """The generic rule must handle a pytree-structured operator too.

    It materialises the tangent with `as_matrix`, which flattens, and then solves
    against each column -- but the operator may take a pytree, in which case a raw
    column is the wrong shape and `linear_solve` rightly refuses it. Nothing about
    this is specific to a tag or a solver: before the columns were unravelled, every
    pytree-structured operator raised here, while its primal was fine.
    """
    matrix = jnp.diag(jnp.array([3.0, 4.0, 5.0, 6.0]))
    matrix += jnp.diag(jnp.array([0.1, 0.2, 0.3]), -1)
    matrix += jnp.diag(jnp.array([0.4, 0.5, 0.6]), 1)
    t_matrix = jnp.arange(16.0).reshape(4, 4) * 0.01
    struct = {
        "a": jax.ShapeDtypeStruct((2,), jnp.float64),
        "b": jax.ShapeDtypeStruct((2,), jnp.float64),
    }

    def make(m):
        tree = {
            "a": {"a": m[:2, :2], "b": m[:2, 2:]},
            "b": {"a": m[2:, :2], "b": m[2:, 2:]},
        }
        return lx.PyTreeLinearOperator(tree, struct)

    op, t_op = make(matrix), make(t_matrix)

    def lad(o):
        return lx.slogdet(o)[1] if solver == "auto" else lx.slogdet(o, lx.LU())[1]

    primal, tangent = jax.jvp(lad, (op,), (t_op,))
    _, ref_lad = jnp.linalg.slogdet(matrix)
    ref_dot = jnp.trace(jnp.linalg.solve(matrix, t_matrix))
    assert jnp.allclose(primal, ref_lad, rtol=1e-10)
    assert jnp.allclose(tangent, ref_dot, rtol=1e-10)


def test_slogdet_structured_jvp_nonflat_structure(getkey):
    """`is_tridiagonal` does not imply a flat in/out structure, and need not.

    A pytree-structured operator can carry the tag; `tridiagonal(...)` ravels it, so
    the fast path applies there too. (`try_structured_materialise` guards on flatness
    for a different reason: `TridiagonalLinearOperator` cannot represent a pytree.)
    """
    matrix = jnp.diag(jnp.array([3.0, 4.0, 5.0, 6.0]))
    matrix += jnp.diag(jnp.array([0.1, 0.2, 0.3]), -1)
    matrix += jnp.diag(jnp.array([0.4, 0.5, 0.6]), 1)
    t_matrix = jnp.arange(16.0).reshape(4, 4) * jnp.where(
        jnp.abs(jnp.arange(4)[:, None] - jnp.arange(4)[None, :]) <= 1, 1.0, 0.0
    )

    struct = {
        "a": jax.ShapeDtypeStruct((2,), jnp.float64),
        "b": jax.ShapeDtypeStruct((2,), jnp.float64),
    }

    def to_pytree(m):
        return {
            "a": {"a": m[:2, :2], "b": m[:2, 2:]},
            "b": {"a": m[2:, :2], "b": m[2:, 2:]},
        }

    def make(m):
        return lx.PyTreeLinearOperator(to_pytree(m), struct, lx.tridiagonal_tag)

    op, t_op = make(matrix), make(t_matrix)
    assert not isinstance(op.in_structure(), jax.ShapeDtypeStruct)
    (_, lad), (_, lad_dot) = jax.jvp(
        lambda o: lx.slogdet(o, lx.Tridiagonal()), (op,), (t_op,)
    )
    _, ref_lad = jnp.linalg.slogdet(matrix)
    ref_dot = jnp.trace(jnp.linalg.solve(matrix, t_matrix))
    assert jnp.allclose(lad, ref_lad, rtol=1e-10)
    assert jnp.allclose(lad_dot, ref_dot, rtol=1e-8)


@pytest.mark.parametrize("lower", [True, False])
def test_slogdet_unit_diagonal_jvp(lower):
    """A unit diagonal pins the determinant to 1 *for a triangular operator*.

    The solver honours the promise rather than reading the stored diagonal, so the
    fast path must too -- and must not extend the reasoning to operators that merely
    happen to have ones on the diagonal.
    """
    matrix = jnp.array([[1.0, 0.0], [3.0, 1.0]])
    t_matrix = jnp.array([[5.0, 0.0], [7.0, 11.0]])
    if not lower:
        matrix, t_matrix = matrix.T, t_matrix.T
    tag = (lx.lower_triangular_tag, lx.unit_diagonal_tag)
    if not lower:
        tag = (lx.upper_triangular_tag, lx.unit_diagonal_tag)
    op = lx.MatrixLinearOperator(matrix, tag)
    t_op = lx.MatrixLinearOperator(t_matrix, tag)
    (sign, lad), (sign_dot, lad_dot) = jax.jvp(
        lambda o: lx.slogdet(o, lx.Triangular()), (op,), (t_op,)
    )
    assert jnp.allclose(sign, 1.0)
    assert jnp.allclose(lad, 0.0)
    assert jnp.allclose(sign_dot, 0.0)
    assert jnp.allclose(lad_dot, 0.0)


def test_slogdet_pseudodeterminant_grad_masked_entry_is_zero():
    """Reverse mode: the pseudodeterminant's gradient at a masked entry is zero.

    The pseudodeterminant is locally constant in the entries it drops, so their
    gradient is an exact zero. The solve-based rule gets this from the masked
    pseudoinverse solve; differentiating `Diagonal.slogdet`'s masked `log` directly
    instead produced `nan` here (`0/0` in the `where`'s transpose), which is the
    regression this pins against.
    """
    diag = jnp.array([2.0, 3.0, 0.0, 5.0])
    solver = lx.Diagonal(well_posed=False)
    grad = jax.grad(lambda d: lx.slogdet(lx.DiagonalLinearOperator(d), solver)[1])(diag)
    assert jnp.allclose(grad, jnp.array([1 / 2, 1 / 3, 0.0, 1 / 5]))


def test_slogdet_pseudodeterminant_complex_sign_jvp():
    """The masked pseudodeterminant's *sign* also has a tangent, for complex operators.

    The solve-based rule carries the phase in the imaginary part of `trace(A^+ dA)`,
    and the masked pseudoinverse solve confines the trace to the retained entries.
    """
    diag = jnp.array([2.0 + 1.0j, 3.0 - 2.0j, 0.0 + 0.0j, 5.0 + 4.0j])
    t_diag = jnp.array([1.0 + 1.0j, 1.0 - 1.0j, 1.0 + 0.0j, 1.0 + 2.0j])
    op = lx.DiagonalLinearOperator(diag)
    t_op = lx.DiagonalLinearOperator(t_diag)
    solver = lx.Diagonal(well_posed=False)
    (sign, _), (sign_dot, lad_dot) = jax.jvp(
        lambda o: lx.slogdet(o, solver), (op,), (t_op,)
    )
    # Over the three retained entries, d log(det) = sum(t / d); the real part is the
    # tangent of log|det| and the rest turns the phase.
    kept = jnp.array([0, 1, 3])
    trace = jnp.sum(t_diag[kept] / diag[kept])
    assert jnp.allclose(lad_dot, jnp.real(trace))
    assert jnp.allclose(sign_dot, (trace - jnp.real(trace).astype(trace.dtype)) * sign)


@pytest.mark.parametrize("kind", ["function", "jacobian"])
def test_slogdet_structured_jvp_opaque_operator(kind):
    """An operator carrying a structure tag need not be made of arrays alone.

    `FunctionLinearOperator` and `JacobianLinearOperator` hold a callable, so the fast
    path has to differentiate them with `eqx.filter_jvp`; plain `jax.jvp` rejects the
    tangent, whose callable leaf is `None`. These reach the fast path like any other
    tagged operator, and used to reach the generic one, so a regression here is a
    crash on operators that previously worked.
    """
    n = 8
    key = jr.PRNGKey(0)
    k0, k1, k2, k3 = jr.split(key, 4)
    matrix = (
        jnp.diag(jr.normal(k0, (n,)) + 4.0)
        + jnp.diag(jr.normal(k1, (n - 1,)) * 0.3, -1)
        + jnp.diag(jr.normal(k2, (n - 1,)) * 0.3, 1)
    )
    tangent = jnp.diag(jr.normal(k3, (n,)))
    structure = jax.ShapeDtypeStruct((n,), matrix.dtype)

    def make(m):
        if kind == "function":
            return lx.FunctionLinearOperator(
                lambda v: m @ v, structure, lx.tridiagonal_tag
            )
        return lx.JacobianLinearOperator(
            lambda x, args: m @ x,
            jnp.zeros(n, matrix.dtype),
            None,
            tags=lx.tridiagonal_tag,
        )

    (_, lad), (_, lad_dot) = eqx.filter_jvp(
        lambda m: lx.slogdet(make(m), lx.Tridiagonal()), (matrix,), (tangent,)
    )
    _, ref_lad = jnp.linalg.slogdet(matrix)
    ref_dot = jnp.trace(jnp.linalg.solve(matrix, tangent))
    assert jnp.allclose(lad, ref_lad, rtol=1e-10)
    assert jnp.allclose(lad_dot, ref_dot, rtol=1e-8)
