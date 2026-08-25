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

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from .helpers import construct_poisson_matrix, tree_allclose


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

    assert tree_allclose(lx_soln, true_x, atol=tol, rtol=tol)


def test_nontrivial_pytree_operator():
    x = [[1, 5.0], [jnp.array(-2), jnp.array(-2.0)]]
    y = [3, 4]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct)
    out = lx.linear_solve(operator, y).value
    true_out = [jnp.array(-3.25), jnp.array(1.25)]
    assert tree_allclose(out, true_out)


def test_nontrivial_diagonal_operator():
    x = (8.0, jnp.array([1, 2, 3]), {"a": jnp.array([4, 5]), "b": 6})
    y = (4.0, jnp.array([7, 8, 9]), {"a": jnp.array([2, 10]), "b": 12})
    operator = lx.DiagonalLinearOperator(x)
    out = lx.linear_solve(operator, y).value
    true_out = (
        jnp.array(0.5),
        jnp.array([7.0, 4.0, 3.0]),
        {"a": jnp.array([0.5, 2.0]), "b": jnp.array(2.0)},
    )
    assert tree_allclose(out, true_out)


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
    assert tree_allclose(out, true_out)


@pytest.mark.parametrize("solver", (lx.LU(), lx.QR(), lx.SVD()))
def test_mixed_dtypes_complex(solver):
    c64 = lambda x: jnp.array(x, dtype=jnp.complex64)
    c128 = lambda x: jnp.array(x, dtype=jnp.complex128)
    x = [[c64(1), c128(5.0j)], [c64(2.0j), c128(-2)]]
    y = [c128(3), c128(4)]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct)
    out = lx.linear_solve(operator, y, solver=solver).value
    true_out = [c64(-0.75 - 2.5j), c128(0.5 - 0.75j)]
    assert tree_allclose(out, true_out)


@pytest.mark.parametrize("solver", (lx.LU(), lx.QR(), lx.SVD()))
def test_mixed_dtypes_complex_real(solver):
    f64 = lambda x: jnp.array(x, dtype=jnp.float64)
    c128 = lambda x: jnp.array(x, dtype=jnp.complex128)
    x = [[f64(1), c128(-5.0j)], [f64(2.0), c128(-2j)]]
    y = [c128(3), c128(4)]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct)
    out = lx.linear_solve(operator, y, solver=solver).value
    true_out = [f64(1.75), c128(0.25j)]
    assert tree_allclose(out, true_out)


def test_mixed_dtypes_triangular():
    f32 = lambda x: jnp.array(x, dtype=jnp.float32)
    f64 = lambda x: jnp.array(x, dtype=jnp.float64)
    x = [[f32(1), f64(0)], [f32(-2), f64(-2)]]
    y = [f64(3), f64(4)]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct, lx.lower_triangular_tag)
    out = lx.linear_solve(operator, y, solver=lx.Triangular()).value
    true_out = [f32(3), f64(-5)]
    assert tree_allclose(out, true_out)


def test_mixed_dtypes_complex_triangular():
    c64 = lambda x: jnp.array(x, dtype=jnp.complex64)
    c128 = lambda x: jnp.array(x, dtype=jnp.complex128)
    x = [[c64(1), c128(0)], [c64(2.0j), c128(-2)]]
    y = [c128(3), c128(4)]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct, lx.lower_triangular_tag)
    out = lx.linear_solve(operator, y, solver=lx.Triangular()).value
    true_out = [c64(3), c128(-2 + 3.0j)]
    assert tree_allclose(out, true_out)


def test_mixed_dtypes_complex_real_triangular():
    f64 = lambda x: jnp.array(x, dtype=jnp.float64)
    c128 = lambda x: jnp.array(x, dtype=jnp.complex128)
    x = [[f64(1), c128(0)], [f64(2.0), c128(2j)]]
    y = [c128(3), c128(4)]
    struct = jax.eval_shape(lambda: y)
    operator = lx.PyTreeLinearOperator(x, struct, lx.lower_triangular_tag)
    out = lx.linear_solve(operator, y, solver=lx.Triangular()).value
    true_out = [f64(3), c128(1j)]
    assert tree_allclose(out, true_out)


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
    assert tree_allclose(grad, -z / (x**2))
    assert tree_allclose(sol, z / x)


def test_grad_vmap_symbolic_cotangent():
    def f(x):
        return x[0], x[1]

    @jax.vmap
    def to_vmap(x):
        op = lx.FunctionLinearOperator(f, jax.eval_shape(lambda: x))
        sol = lx.linear_solve(op, x)
        return sol.value[0]

    @jax.grad
    def to_grad(x):
        return jnp.sum(to_vmap(x))

    x = (jnp.arange(3.0), jnp.arange(3.0))
    to_grad(x)


@pytest.mark.parametrize(
    "solver",
    (
        lx.CG(0.0, 0.0, max_steps=2),
        lx.Normal(lx.CG(0.0, 0.0, max_steps=2)),
        lx.BiCGStab(0.0, 0.0, max_steps=2),
        lx.GMRES(0.0, 0.0, max_steps=2),
        lx.LSMR(0.0, 0.0, max_steps=2),
    ),
)
def test_iterative_solver_max_steps_only(solver):
    """Iterative solvers should work with max_steps only (no Equinox errors)."""
    SIZE = 100

    poisson_matrix = construct_poisson_matrix(SIZE)
    poisson_operator = lx.MatrixLinearOperator(
        poisson_matrix, tags=(lx.negative_semidefinite_tag, lx.symmetric_tag)
    )
    rhs = jax.random.normal(jax.random.key(0), (SIZE,))

    lx.linear_solve(poisson_operator, rhs, solver)


def test_solver_init_not_differentiated(getkey):
    """stop_gradient should be applied before solver.init, not after.

    Also checks that dynamic arrays in options don't cause issues.
    """

    class DisallowGradWrapper(lx._solve.AbstractLinearSolver):
        solver: lx._solve.AbstractLinearSolver

        def init(self, operator, options):
            @jax.custom_jvp
            def f(operator, dummy):
                del dummy
                return self.solver.init(operator, options)

            @f.defjvp
            def _(*args):
                raise NotImplementedError("solver.init should not be differentiated")

            return f(operator, options.get("dummy"))

        def compute(self, state, vector, options):
            return self.solver.compute(state, vector, options)

        def transpose(self, state, options):
            return self.solver.transpose(state, options)

        def conj(self, state, options):
            return self.solver.conj(state, options)

        def assume_full_rank(self):
            return self.solver.assume_full_rank()

    m = jax.random.normal(getkey(), (3, 3))
    mt = jax.random.normal(getkey(), (3, 3))
    v = jax.random.normal(getkey(), (3,))
    dummy = jnp.array(1.0)

    def f(m):
        op = lx.MatrixLinearOperator(m)
        return lx.linear_solve(
            op, v, solver=DisallowGradWrapper(lx.QR()), options={"dummy": dummy}
        ).value

    # Differentiating through operator only, but options has a dynamic array.
    # solver.init should not be differentiated through.
    jax.jvp(f, (m,), (mt,))

    _, f_vjp = jax.vjp(f, m)
    f_vjp(v)


def test_nonfinite_input():
    operator = lx.DiagonalLinearOperator((1.0, 1.0))
    vector = (1.0, jnp.inf)
    sol = lx.linear_solve(operator, vector, throw=False)
    assert sol.result == lx.RESULTS.nonfinite_input

    vector = (1.0, jnp.nan)
    sol = lx.linear_solve(operator, vector, throw=False)
    assert sol.result == lx.RESULTS.nonfinite_input

    vector = (jnp.nan, jnp.inf)
    sol = lx.linear_solve(operator, vector, throw=False)
    assert sol.result == lx.RESULTS.nonfinite_input


@pytest.mark.parametrize("sign", (1.0, -1.0))
def test_cholesky_semidefinite_tag_dynamic_sign(getkey, sign):
    """`lx.semidefinite_tag` records that an operator is definite, without pinning
    down its sign -- e.g. because it results from scaling by a traced value of
    unknown sign. `Cholesky` should determine the sign cheaply at trace time (via
    the diagonal entry of largest magnitude) rather than requiring it statically.
    """
    m = jr.normal(getkey(), (5, 5))
    psd = m @ m.T + 5 * jnp.eye(5)  # well-conditioned, strictly positive definite
    operator = lx.MatrixLinearOperator(psd, lx.positive_semidefinite_tag)
    b = jr.normal(getkey(), (5,))

    @jax.jit
    def solve(scalar):
        scaled = operator * scalar
        # The sign is genuinely unknown at trace time...
        assert not lx.is_positive_semidefinite(scaled)
        assert not lx.is_negative_semidefinite(scaled)
        # ...yet `Cholesky` should still accept it, via `is_semidefinite`.
        return lx.linear_solve(scaled, b, solver=lx.Cholesky()).value

    x = solve(jnp.asarray(sign))
    expected = jnp.linalg.solve(sign * psd, b)
    assert tree_allclose(x, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("sign", (1.0, -1.0))
def test_cg_semidefinite_tag_dynamic_sign(getkey, sign):
    """As above, but for `lx.CG`, which accepts the tag without resolving the sign at
    all: its recurrence is invariant under negating the operator and the right hand
    side together, so the same iterates come out either way.
    """
    tol = 1e-10 if jax.config.jax_enable_x64 else 1e-4  # pyright: ignore
    m = jr.normal(getkey(), (5, 5))
    psd = m @ m.T + 5 * jnp.eye(5)  # well-conditioned, strictly positive definite
    operator = lx.MatrixLinearOperator(psd, lx.positive_semidefinite_tag)
    b = jr.normal(getkey(), (5,))
    solver = lx.CG(rtol=tol, atol=tol)

    @jax.jit
    def solve(scalar):
        scaled = operator * scalar
        assert not lx.is_positive_semidefinite(scaled)
        assert not lx.is_negative_semidefinite(scaled)
        return lx.linear_solve(scaled, b, solver=solver).value

    x = solve(jnp.asarray(sign))
    expected = jnp.linalg.solve(sign * psd, b)
    assert tree_allclose(x, expected, atol=1e-4, rtol=1e-4)


def test_cg_is_indifferent_to_the_operator_sign(getkey):
    """`CG` deliberately never resolves which sign a semidefinite operator has, because
    its recurrence is invariant under `(A, b) -> (-A, -b)`: substituting `r -> -r`,
    `p -> -p`, `z -> -z` leaves `gamma`, `beta`, `diff` and every norm in the
    termination check alone, and flips `alpha` and `inner_prod` together.

    Pin that down by *lying* about the sign. A negative semidefinite operator
    mistagged as positive semidefinite must still produce the right answer, in the same
    number of steps -- including with a non-identity preconditioner and a nonzero `y0`,
    which are the parts of the recurrence where a sign could plausibly leak in. If this
    test fails, `CG.init` has grown a dependency on the sign and needs to resolve it.
    """
    tol = 1e-12 if jax.config.jax_enable_x64 else 1e-6  # pyright: ignore
    m = jr.normal(getkey(), (6, 6))
    nsd = -(m @ m.T + 3 * jnp.eye(6))  # strictly negative definite
    b = jr.normal(getkey(), (6,))
    expected = jnp.linalg.solve(nsd, b)
    solver = lx.CG(rtol=tol, atol=tol)
    diag = jnp.abs(jr.normal(getkey(), (6,))) + 1.0
    options = (
        {},
        {
            "preconditioner": lx.MatrixLinearOperator(
                jnp.diag(diag), lx.positive_semidefinite_tag
            ),
            "y0": jr.normal(getkey(), (6,)),
        },
    )
    for opts in options:
        truthful = lx.linear_solve(
            lx.MatrixLinearOperator(nsd, lx.negative_semidefinite_tag),
            b,
            solver,
            options=opts,
        )
        # A deliberate lie: `nsd` is *not* positive semidefinite.
        mistagged = lx.linear_solve(
            lx.MatrixLinearOperator(nsd, lx.positive_semidefinite_tag),
            b,
            solver,
            options=opts,
        )
        assert tree_allclose(truthful.value, expected, atol=1e-6, rtol=1e-6)
        assert tree_allclose(mistagged.value, truthful.value)
        assert mistagged.stats["num_steps"] == truthful.stats["num_steps"]


def test_cg_singular_semidefinite_consistent_rhs(getkey):
    """Unlike `Cholesky`, `CG` documents support for a *singular* semidefinite operator
    provided `b` lies in its range: the Krylov space is generated by `r0 = b - A y0`,
    which stays in `range(A)` (as `A` is Hermitian, so `range(A)` is the orthogonal
    complement of `null(A)`). So the iterates never enter the null space and the
    minimum-norm solution comes out.
    """
    tol = 1e-12 if jax.config.jax_enable_x64 else 1e-6  # pyright: ignore
    q, _ = jnp.linalg.qr(jr.normal(getkey(), (6, 6)))
    # Rank 4 of 6: two exactly-zero eigenvalues, so `null(A)` is `q[:, :2]`.
    eigvals = jnp.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    matrix = (q * eigvals[None, :]) @ q.T
    matrix = (matrix + matrix.T) / 2
    b = q[:, 2:] @ jr.normal(getkey(), (4,))  # consistent: `b` in `range(A)`

    for tag in (lx.positive_semidefinite_tag, lx.semidefinite_tag):
        operator = lx.MatrixLinearOperator(matrix, tag)
        sol = lx.linear_solve(operator, b, lx.CG(rtol=tol, atol=tol), throw=False)
        assert sol.result == lx.RESULTS.successful
        assert tree_allclose(matrix @ sol.value, b, atol=1e-6, rtol=1e-6)
        # Minimum-norm: no component in the null space.
        assert tree_allclose(q[:, :2].T @ sol.value, jnp.zeros(2), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("solver", (lx.Cholesky(), lx.CG(rtol=1e-10, atol=1e-10)))
def test_semidefinite_tag_dynamic_sign_batched(getkey, solver):
    """`Cholesky`'s sign-selecting `jnp.where` must batch correctly under `vmap` -- in
    particular when different batch elements resolve to *different* signs, not just a
    sign-uniform batch. `CG` is included as a control: it does not resolve the sign at
    all, so it must be indifferent to the mix.
    """
    m = jr.normal(getkey(), (5, 5))
    psd = m @ m.T + 5 * jnp.eye(5)
    operator = lx.MatrixLinearOperator(psd, lx.positive_semidefinite_tag)
    b = jr.normal(getkey(), (5,))
    signs = jnp.array([2.0, -3.0, -1.5, 0.5])  # mixed signs within one batch

    def solve(scalar):
        return lx.linear_solve(operator * scalar, b, solver=solver).value

    batched = jax.jit(jax.vmap(solve))(signs)
    expected = jax.vmap(lambda s: jnp.linalg.solve(s * psd, b))(signs)
    assert tree_allclose(batched, expected, atol=1e-4, rtol=1e-4)


def test_gmres_wide_dynamic_range_rhs():
    # Regression test for the iterative breakdown caused by elementwise tolerances
    # (#230). Well-conditioned, consistent system with a wide-dynamic-range RHS: with
    # the scalar stopping rule `norm(r) <= atol + rtol * norm(b)` this converges even
    # at a tolerance near the round-off floor, whereas the old per-component rule
    # `|r_i| <= atol + rtol * |b_i|` spuriously reported breakdown.
    A = jnp.array([[1.0, 0.5, 0.3], [0.2, 2.0, 0.7], [0.6, 0.1, 1.5]])
    b = jnp.array([1e8, 1.0, 1.0])
    operator = lx.MatrixLinearOperator(A)
    sol = lx.linear_solve(operator, b, lx.GMRES(rtol=1e-12, atol=1e-12), throw=False)
    assert sol.result == lx.RESULTS.successful
    assert tree_allclose(sol.value, jnp.linalg.solve(A, b))
