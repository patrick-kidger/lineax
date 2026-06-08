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
    finite_difference_jvp,
    make_function_operator,
    make_jac_operator,
    make_jacfwd_operator,
    make_jacrev_operator,
    make_matrix_operator,
    make_trivial_pytree_operator,
    tree_allclose,
)


# Projection is only interesting for rectangular / rank-deficient `A`, so we use
# tall operators throughout. Constructors that support non-square matrices:
nonsquare_operators = (
    make_matrix_operator,
    make_function_operator,
    make_jac_operator,
    make_jacfwd_operator,
    make_jacrev_operator,
    make_trivial_pytree_operator,
)

# Operators that support forward-mode autodiff (excludes `make_jacrev_operator`,
# which uses a `custom_vjp` with no JVP).
fwd_operators = (make_jac_operator, make_jacfwd_operator)

# Full-rank-assuming solvers: handle tall full-rank `A` (m > n).
full_rank_solvers = (
    lx.AutoLinearSolver(well_posed=None),
    lx.QR(),
    lx.Normal(lx.Cholesky()),
)
# Pseudoinverse solvers: handle (tagged) rank-deficient `A`.
pseudo_solvers = (lx.SVD(), lx.AutoLinearSolver(well_posed=False))

_M, _N = 5, 3  # tall: m > n
_RANK = 2  # exact rank for the rank-deficient cases


def _tall_full_rank(getkey, dtype):
    return jr.normal(getkey(), (_M, _N), dtype=dtype)


def _tall_rank_deficient(getkey, dtype):
    # Exactly rank `_RANK`: (m, r) @ (r, n). Use *exact* (tagged) rank deficiency
    # -- the SVD fast path applies `U U^H` over every cached singular direction
    # and does not rcond-mask, so a merely near-singular matrix would mismatch.
    b = jr.normal(getkey(), (_M, _RANK), dtype=dtype)
    c = jr.normal(getkey(), (_RANK, _N), dtype=dtype)
    return b @ c


# (kind, solver, tags): full-rank with full-rank solvers; rank-deficient (tagged)
# with pseudoinverse solvers.
structural_cases = [
    *[("full", s, ()) for s in full_rank_solvers],
    *[("deficient", s, lx.MaxRankTag(_RANK)) for s in pseudo_solvers],
]


def _make_matrix(getkey, kind, dtype):
    if kind == "full":
        return _tall_full_rank(getkey, dtype)
    else:
        return _tall_rank_deficient(getkey, dtype)


@pytest.mark.parametrize("make_operator", nonsquare_operators)
@pytest.mark.parametrize("kind, solver, tags", structural_cases)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_project(make_operator, kind, solver, tags, dtype, getkey):
    matrix = _make_matrix(getkey, kind, dtype)
    operator = make_operator(getkey, matrix, tags)
    A = operator.as_matrix()
    P = lx.project(operator, solver)
    v = jr.normal(getkey(), (_M,), dtype=dtype)
    Pv = P.mv(v)

    # Idempotent: P P v == P v.
    assert tree_allclose(P.mv(Pv), Pv, rtol=1e-5, atol=1e-6)

    # Matches the dense projector A A^†.
    dense = (A @ jnp.linalg.pinv(A)) @ v
    assert tree_allclose(Pv, dense, rtol=1e-5, atol=1e-6)

    # Hermitian: P == P^H, and conj(P).T == P.
    Pmat = P.as_matrix()
    assert tree_allclose(Pmat, Pmat.conj().T, rtol=1e-5, atol=1e-6)
    assert tree_allclose(lx.conj(P).T.as_matrix(), Pmat, rtol=1e-5, atol=1e-6)

    # Tags. (`project` returns a `FunctionLinearOperator` for these tall operators.)
    assert isinstance(P, lx.FunctionLinearOperator)
    assert lx.is_positive_semidefinite(P)
    assert lx.symmetric_tag not in P.tags
    # Symmetry is inferred for real dtypes, not for complex (where P is Hermitian).
    if jnp.iscomplexobj(jnp.zeros((), dtype=dtype)):
        assert not lx.is_symmetric(P)
    else:
        assert lx.is_symmetric(P)
    assert lx.max_rank(P) == lx.max_rank(operator)


def test_project_diagonal(getkey):
    # Projection of a diagonal operator is diagonal (0/1 on the diagonal). Use a
    # diagonal-with-zeros operator with an *exact* rank tag so the SVD fast path
    # truncates to the nonzero directions.
    for dtype in (jnp.float64, jnp.complex128):
        diag = jnp.array([1.0, 2.0, 0.0, 3.0, 0.0], dtype=dtype)
        nnz = 3
        operator = lx.MatrixLinearOperator(
            jnp.diag(diag), (lx.diagonal_tag, lx.MaxRankTag(nnz))
        )
        P = lx.project(operator, lx.SVD())
        assert lx.is_diagonal(P)
        assert lx.max_rank(P) == nnz
        v = jr.normal(getkey(), (5,), dtype=dtype)
        dense = (operator.as_matrix() @ jnp.linalg.pinv(operator.as_matrix())) @ v
        assert tree_allclose(P.mv(v), dense, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradients (Golub-Pereyra). The custom JVP/transpose rules should be the
# efficient equivalent of the naive `A (A^† v)` autodiff. We check this against
# two references:
#   * `lineax.invert` (`A @ invert(A) @ v`), which uses the *same* solver and
#     adjoint machinery -- it handles rank deficiency (via the pseudoinverse
#     solver) and shares `project`'s complex (Wirtinger/adjoint) convention, so
#     it is valid for full-rank/rank-deficient and real/complex alike; and
#   * finite differences, an entirely lineax-independent oracle, used for the
#     real full-rank case (complex FD is non-holomorphic, but `project` matches
#     it to FD tolerance anyway since the convention bug is fixed).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind, solver, tags", structural_cases)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_project_jvp_invert(kind, solver, tags, dtype, getkey):
    matrix = _make_matrix(getkey, kind, dtype)
    t_matrix = _make_matrix(getkey, kind, dtype)
    vec = jr.normal(getkey(), (_M,), dtype=dtype)
    t_vec = jr.normal(getkey(), (_M,), dtype=dtype)

    def run(M, v):
        return lx.project(lx.MatrixLinearOperator(M, tags), solver).mv(v)

    def ref(M, v):
        op = lx.MatrixLinearOperator(M, tags)
        return op.mv(lx.invert(op, solver).mv(v))

    out, t_out = eqx.filter_jvp(run, (matrix, vec), (t_matrix, t_vec))
    ref_out, t_ref = eqx.filter_jvp(ref, (matrix, vec), (t_matrix, t_vec))
    assert tree_allclose(out, ref_out, rtol=1e-4, atol=1e-6)
    assert tree_allclose(t_out, t_ref, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("solver", full_rank_solvers)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_project_jvp_fd(solver, dtype, getkey):
    # JVP against finite differences -- a lineax-independent ground truth. (The
    # complex case is formally non-holomorphic, but with the Hermitian B-term the
    # custom JVP still matches FD to FD tolerance.)
    matrix = _tall_full_rank(getkey, dtype)
    t_matrix = _tall_full_rank(getkey, dtype)
    vec = jr.normal(getkey(), (_M,), dtype=dtype)
    t_vec = jr.normal(getkey(), (_M,), dtype=dtype)

    def run(M, v):
        return lx.project(lx.MatrixLinearOperator(M), solver).mv(v)

    out, t_out = eqx.filter_jvp(run, (matrix, vec), (t_matrix, t_vec))
    expected, t_expected = finite_difference_jvp(run, (matrix, vec), (t_matrix, t_vec))
    assert tree_allclose(out, expected, rtol=1e-4, atol=1e-6)
    assert tree_allclose(t_out, t_expected, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("solver", full_rank_solvers)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_project_grad_invert(solver, dtype, getkey):
    # Reverse-mode gradient of a real-valued loss matches autodiff through
    # `lineax.invert` -- the quantity actually consumed downstream (separable NLLS).
    matrix = _tall_full_rank(getkey, dtype)
    vec = jr.normal(getkey(), (_M,), dtype=dtype)

    def loss_project(M):
        P = lx.project(lx.MatrixLinearOperator(M), solver)
        return jnp.sum(jnp.abs(P.mv(vec)) ** 2)

    def loss_invert(M):
        op = lx.MatrixLinearOperator(M)
        return jnp.sum(jnp.abs(op.mv(lx.invert(op, solver).mv(vec))) ** 2)

    assert tree_allclose(
        jax.grad(loss_project)(matrix),
        jax.grad(loss_invert)(matrix),
        rtol=1e-4,
        atol=1e-6,
    )


@pytest.mark.parametrize("make_operator", fwd_operators)
@pytest.mark.parametrize("solver", full_rank_solvers)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_project_jvp_operator(make_operator, solver, dtype, getkey):
    # The operator-tangent path (TangentLinearOperator / linearise) goes through
    # `JacobianLinearOperator`. Finite-differencing the operator pytree does not
    # perturb closure-captured arrays, so we instead check the analytic JVP
    # through the operator matches the analytic JVP through the equivalent
    # `MatrixLinearOperator` (grounded against finite differences / pinv above).
    matrix = _tall_full_rank(getkey, dtype)
    t_matrix = _tall_full_rank(getkey, dtype)
    vec = jr.normal(getkey(), (_M,), dtype=dtype)

    make_op = ft.partial(make_operator, getkey)
    operator, t_operator = eqx.filter_jvp(make_op, (matrix, ()), (t_matrix, ()))
    make_mat = ft.partial(make_matrix_operator, getkey)
    mat_op, t_mat_op = eqx.filter_jvp(make_mat, (matrix, ()), (t_matrix, ()))

    f = lambda op: lx.project(op, solver).mv(vec)
    out, t_out = eqx.filter_jvp(f, (operator,), (t_operator,))
    out_ref, t_out_ref = eqx.filter_jvp(f, (mat_op,), (t_mat_op,))
    assert tree_allclose(out, out_ref, rtol=1e-4, atol=1e-6)
    assert tree_allclose(t_out, t_out_ref, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("solver", full_rank_solvers)
def test_project_jacfwd_jacrev(solver, getkey):
    matrix = _tall_full_rank(getkey, jnp.float64)
    operator = lx.MatrixLinearOperator(matrix)
    vec = jr.normal(getkey(), (_M,), dtype=jnp.float64)
    f = lambda op: lx.project(op, solver).mv(vec)
    jfwd = eqx.filter_jacfwd(f)(operator)
    jrev = eqx.filter_jacrev(f)(operator)
    assert tree_allclose(jfwd, jrev, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("solver", full_rank_solvers)
def test_project_second_order(solver, getkey):
    # Second-order finite-difference check (real dtype), guarding the primitive
    # re-binding in the JVP/transpose rules (which is what makes higher-order
    # autodiff recurse).
    matrix = _tall_full_rank(getkey, jnp.float64)
    t_matrix = _tall_full_rank(getkey, jnp.float64)
    tt_matrix = _tall_full_rank(getkey, jnp.float64)
    vec = jr.normal(getkey(), (_M,), dtype=jnp.float64)

    def run(M):
        return lx.project(lx.MatrixLinearOperator(M), solver).mv(vec)

    # First-order directional derivative as a function of `M`.
    jvp1 = lambda M: eqx.filter_jvp(run, (M,), (t_matrix,))[1]

    out, t_out = eqx.filter_jvp(jvp1, (matrix,), (tt_matrix,))
    expected, t_expected = finite_difference_jvp(jvp1, (matrix,), (tt_matrix,))
    assert tree_allclose(out, expected, rtol=1e-3, atol=1e-4)
    assert tree_allclose(t_out, t_expected, rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# Targeted cases.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", full_rank_solvers)
def test_project_identity_shortcut(solver, getkey):
    # Full-rank solver + m <= n => P = I exactly, returned as IdentityLinearOperator.
    matrix = jr.normal(getkey(), (_N, _M), dtype=jnp.float64)  # wide: m < n
    operator = lx.MatrixLinearOperator(matrix)
    P = lx.project(operator, solver)
    assert isinstance(P, lx.IdentityLinearOperator)


@pytest.mark.parametrize("solver_cls", (lx.SVD, lx.QR))
def test_project_fast_path(solver_cls, getkey):
    # The SVD / QR registrations reproduce the dense projector, and report a fast
    # path (not NotImplemented) for a tall operator.
    for dtype in (jnp.float64, jnp.complex128):
        matrix = _tall_full_rank(getkey, dtype)
        operator = lx.MatrixLinearOperator(matrix)
        solver = solver_cls()
        P = lx.project(operator, solver)
        v = jr.normal(getkey(), (_M,), dtype=dtype)
        dense = (matrix @ jnp.linalg.pinv(matrix)) @ v
        assert tree_allclose(P.mv(v), dense, rtol=1e-5, atol=1e-6)
        state = solver.init(operator, {})
        out = lx.projection_mv(solver, state, v, {})
        assert out is not NotImplemented
        assert tree_allclose(out, dense, rtol=1e-5, atol=1e-6)


def test_project_options_threaded(getkey):
    # `options` must be threaded into every inner solve (as in `invert`), not just used
    # in `solver.init`. Exercise the options path with an iterative inner solver and a
    # `preconditioner` option: `project` must accept it and still equal the dense
    # projector. (A primitive-signature regression that dropped `options` would error.)
    matrix = _tall_full_rank(getkey, jnp.float64)
    operator = lx.MatrixLinearOperator(matrix)
    v = jr.normal(getkey(), (_M,), dtype=jnp.float64)
    preconditioner = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((_N,), jnp.float64))
    solver = lx.Normal(lx.CG(rtol=1e-10, atol=1e-10))
    P = lx.project(operator, solver, options={"preconditioner": preconditioner})
    dense = (matrix @ jnp.linalg.pinv(matrix)) @ v
    assert tree_allclose(P.mv(v), dense, rtol=1e-5, atol=1e-6)


def test_project_closure_operator(getkey):
    # `project` differentiates correctly through an operator that closes over a
    # parameter (with no explicit pytree leaf for it) -- matching an explicit
    # `MatrixLinearOperator`. So `project` needs no separate `check_closure` guard:
    # `filter_primitive_bind` converts the closure to operator leaves (seen by the JVP),
    # and the solves delegate to the already-guarded `linear_solve`.
    matrix = _tall_full_rank(getkey, jnp.float64)
    z = jr.normal(getkey(), (_M,), dtype=jnp.float64)

    def loss_closure(m):
        op = lx.FunctionLinearOperator(
            lambda c: m @ c, jax.ShapeDtypeStruct((_N,), m.dtype)
        )
        return jnp.sum(lx.project(op, lx.SVD()).mv(z) ** 2)

    def loss_matrix(m):
        return jnp.sum(lx.project(lx.MatrixLinearOperator(m), lx.SVD()).mv(z) ** 2)

    assert tree_allclose(
        jax.grad(loss_closure)(matrix),
        jax.grad(loss_matrix)(matrix),
        rtol=1e-5,
        atol=1e-6,
    )


def test_projection_mv_nondifferentiable(getkey):
    # `projection_mv` is a primal-only fast path: it applies the projector using the
    # *frozen* factorisation, so differentiating it directly would give an incorrect
    # projector gradient. Its output is `eqxi.nondifferentiable`, so accidental
    # autodiff raises rather than silently returning a wrong gradient. (Differentiable
    # projections go through `lineax.project`, which supplies the Golub-Pereyra rule.)
    matrix = _tall_full_rank(getkey, jnp.float64)
    operator = lx.MatrixLinearOperator(matrix)
    solver = lx.SVD()
    state = solver.init(operator, {})
    v = jr.normal(getkey(), (_M,), dtype=jnp.float64)
    # Primal evaluation works.
    assert lx.projection_mv(solver, state, v, {}) is not NotImplemented
    # Differentiating it directly raises (both forward and reverse mode).
    f = lambda vec: lx.projection_mv(solver, state, vec, {})
    with pytest.raises(Exception):
        eqx.filter_jvp(f, (v,), (v,))
    with pytest.raises(Exception):
        eqx.filter_grad(lambda vec: jnp.sum(f(vec)))(v)


def test_project_normal_tall_generic(getkey):
    # Tall Normal(Cholesky) has no projection fast path: P = A (A^H A)^† A^H lives
    # in m-space, not the n-space inner B B^†. The registration declines
    # (NotImplemented) and the generic path is used.
    matrix = _tall_full_rank(getkey, jnp.float64)
    operator = lx.MatrixLinearOperator(matrix)
    solver = lx.Normal(lx.Cholesky())
    state = solver.init(operator, {})
    v = jr.normal(getkey(), (_M,), dtype=jnp.float64)
    assert lx.projection_mv(solver, state, v, {}) is NotImplemented
    # The generic path still produces the correct projector.
    P = lx.project(operator, solver)
    dense = (matrix @ jnp.linalg.pinv(matrix)) @ v
    assert tree_allclose(P.mv(v), dense, rtol=1e-5, atol=1e-6)


def _multileaf_pytree_operator(getkey, dtype):
    # Multi-leaf pytree out-structure: in (3,) -> {"a": (2,), "b": (4,)}, tall full rank
    M1 = jr.normal(getkey(), (2, 3), dtype=dtype)
    M2 = jr.normal(getkey(), (4, 3), dtype=dtype)
    in_struct = jax.ShapeDtypeStruct((3,), dtype)
    return lx.FunctionLinearOperator(lambda x: {"a": M1 @ x, "b": M2 @ x}, in_struct)


@pytest.mark.parametrize("solver", full_rank_solvers)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_project_multileaf_pytree(solver, dtype, getkey):
    # `project` must support operators whose out-structure is a multi-leaf pytree
    # (its primitive's abstract-eval / JVP / transpose are pytree-aware).
    operator = _multileaf_pytree_operator(getkey, dtype)
    P = lx.project(operator, solver)
    vec = {
        "a": jr.normal(getkey(), (2,), dtype=dtype),
        "b": jr.normal(getkey(), (4,), dtype=dtype),
    }
    Pv = P.mv(vec)
    # Idempotent.
    assert tree_allclose(P.mv(Pv), Pv, rtol=1e-5, atol=1e-6)
    # Matches the dense projector (leaves flatten in pytree order: "a" then "b").
    A = operator.as_matrix()
    flat = jnp.concatenate([vec["a"], vec["b"]])
    dense = (A @ jnp.linalg.pinv(A)) @ flat
    assert tree_allclose(
        jnp.concatenate([Pv["a"], Pv["b"]]), dense, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("solver", full_rank_solvers)
def test_project_multileaf_pytree_grad(solver, getkey):
    # JVP (vs finite differences) and jacfwd == jacrev through a multi-leaf pytree.
    M2 = jr.normal(getkey(), (4, 3), dtype=jnp.float64)
    vec = {
        "a": jr.normal(getkey(), (2,), dtype=jnp.float64),
        "b": jr.normal(getkey(), (4,), dtype=jnp.float64),
    }

    def run(M1):
        op = lx.FunctionLinearOperator(
            lambda x: {"a": M1 @ x, "b": M2 @ x},
            jax.ShapeDtypeStruct((3,), jnp.float64),
        )
        Pv = lx.project(op, solver).mv(vec)
        return jnp.concatenate([Pv["a"], Pv["b"]])

    M1 = jr.normal(getkey(), (2, 3), dtype=jnp.float64)
    t_M1 = jr.normal(getkey(), (2, 3), dtype=jnp.float64)
    _, t_out = eqx.filter_jvp(run, (M1,), (t_M1,))
    _, t_fd = finite_difference_jvp(run, (M1,), (t_M1,))
    assert tree_allclose(t_out, t_fd, rtol=1e-3, atol=1e-4)
    assert tree_allclose(jax.jacfwd(run)(M1), jax.jacrev(run)(M1), rtol=1e-4, atol=1e-6)


def test_project_errors(getkey):
    # Raw array rejected. (Under the jaxtyping/beartype import hook used in the
    # test suite this surfaces as a `TypeError`; the explicit `ValueError` in
    # `project` fires when the hook is absent.)
    with pytest.raises((ValueError, TypeError)):
        lx.project(jnp.eye(3))  # pyright: ignore

    # Zero-dimension operator rejected.
    with pytest.raises(ValueError):
        lx.project(lx.MatrixLinearOperator(jnp.zeros((3, 0))))

    # Full-rank solver on a declared rank-deficient operator raises (rank check).
    matrix = _tall_rank_deficient(getkey, jnp.float64)
    operator = lx.MatrixLinearOperator(matrix, lx.MaxRankTag(_RANK))
    with pytest.raises(ValueError):
        lx.project(operator, lx.AutoLinearSolver(well_posed=None))
