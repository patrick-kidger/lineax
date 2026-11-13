import equinox as ex
import jax
import jax.numpy as jnp
import lineax as lx
import pytest


solver = lx.LSMR(1e-10, 1e-10)
Aill = lx.DiagonalLinearOperator(jnp.array([1e8, 1e6, 1e4, 1e2, 1]))
Awell = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0, 5.0, 8.0, 10.0]))
Asing = lx.DiagonalLinearOperator(jnp.array([0.0, 4.0, 5.0, 8.0, 10.0]))


def test_ill_conditioned():
    try:
        lx.linear_solve(Aill, jnp.ones(5), solver=solver)
    except ex.EquinoxRuntimeError as e:
        assert "Condition number" in str(e)


def test_zero_rhs():
    # b=0, so x=0 is solution
    sol = lx.linear_solve(Aill, jnp.zeros(5), solver=solver)
    assert (sol.value == 0).all()
    sol = lx.linear_solve(Awell, jnp.zeros(5), solver=solver)
    assert (sol.value == 0).all()
    sol = lx.linear_solve(Asing, jnp.zeros(5), solver=solver)
    assert (sol.value == 0).all()
    # b lies in null space of A, so x=0 is minimum norm solution
    sol = lx.linear_solve(Asing, jnp.zeros(5).at[0].set(1), solver=solver)
    assert (sol.value == 0).all()


@pytest.mark.skip("Damp support is disabled.")
def test_damp_regularizes():
    solution_ill = lx.linear_solve(Aill, jnp.ones(5), solver=solver, options={})
    assert solution_ill.stats["istop"] == 1

    solution_damped = lx.linear_solve(
        Aill, jnp.ones(5), solver=solver, options={"damp": 100.0}
    )
    assert solution_damped.stats["istop"] == 2

    assert solution_damped.stats["num_steps"] < solution_ill.stats["num_steps"]


@pytest.mark.skip("Damp support is disabled.")
def test_damp():
    solution_damped = lx.linear_solve(
        Awell, jnp.ones(5), solver=solver, options={"damp": 1.0}
    )
    assert jnp.allclose(
        solution_damped.value,
        jnp.array([0.4, 0.23529412, 0.19230769, 0.12307692, 0.0990099]),
    )
    solution_damped = lx.linear_solve(
        Awell, jnp.ones(5), solver=solver, options={"damp": 1000.0}
    )
    assert jnp.allclose(
        solution_damped.value, jnp.array([2e-6, 4e-6, 5e-6, 8e-6, 10.0e-6])
    )


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("shape", ("square", "tall", "wide"))
@pytest.mark.parametrize("zero_rhs", (False, True))
@pytest.mark.parametrize("jit", (False, True))
def test_lsmr_preserves_precision(dtype, shape, zero_rhs, jit):
    matrix = jnp.array([[2.0, 1.0, -0.5], [0.5, 3.0, 1.0]], dtype=dtype)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        matrix = matrix + 1j * jnp.array(
            [[0.5, 1.0, 0.0], [-0.5, 0.0, 1.0]], dtype=dtype
        )
    if shape == "square":
        matrix = matrix[:, :2]
    elif shape == "tall":
        matrix = matrix.T
    rhs = jnp.arange(1, matrix.shape[0] + 1).astype(dtype)
    if zero_rhs:
        rhs = jnp.zeros_like(rhs)

    def solve(rhs):
        return lx.linear_solve(
            lx.MatrixLinearOperator(matrix), rhs, lx.LSMR(1e-6, 1e-6)
        ).value

    if jit:
        solve = jax.jit(solve)
    actual = solve(rhs)
    expected = jnp.linalg.lstsq(matrix, rhs, rcond=None)[0]
    assert actual.dtype == expected.dtype == jnp.dtype(dtype)
    assert jnp.allclose(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
@pytest.mark.parametrize("values", ((0.0, 2.0), (2.0, 0.0), (1.0, 2.0), (2.0, 1.0)))
@pytest.mark.parametrize("jit", (False, True))
def test_lsmr_givens_preserves_precision(dtype, values, jit):
    a, b = (jnp.asarray(value, dtype=dtype) for value in values)
    givens = solver._givens
    if jit:
        givens = jax.jit(givens)
    c, s, r = givens(a, b)
    assert c.dtype == s.dtype == r.dtype == jnp.dtype(dtype)
    assert jnp.allclose(c * a + s * b, r)
    assert jnp.allclose(-s * a + c * b, jnp.zeros_like(a), atol=1e-6)
    assert jnp.allclose(c**2 + s**2, jnp.ones_like(a))


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("tall", (False, True))
@pytest.mark.parametrize("jit", (False, True))
def test_lsmr_precision_gradients(dtype, tall, jit):
    matrix = jnp.array([[2.0, 1.0, -0.5], [0.5, 3.0, 1.0]], dtype=dtype)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        matrix = matrix + 1j * jnp.array(
            [[0.5, 1.0, 0.0], [-0.5, 0.0, 1.0]], dtype=dtype
        )
    if tall:
        matrix = matrix.T
    rhs = jnp.arange(1, matrix.shape[0] + 1).astype(dtype)

    def loss(matrix, rhs):
        value = lx.linear_solve(
            lx.MatrixLinearOperator(matrix), rhs, lx.LSMR(1e-6, 1e-6)
        ).value
        return jnp.real(jnp.vdot(value, value))

    def reference(matrix, rhs):
        value = jnp.linalg.lstsq(matrix, rhs, rcond=None)[0]
        return jnp.real(jnp.vdot(value, value))

    actual = jax.value_and_grad(loss, argnums=(0, 1))
    expected = jax.value_and_grad(reference, argnums=(0, 1))
    if jit:
        actual = jax.jit(actual)
        expected = jax.jit(expected)
    actual_value, actual_grads = actual(matrix, rhs)
    expected_value, expected_grads = expected(matrix, rhs)
    assert jnp.allclose(actual_value, expected_value, rtol=1e-4, atol=1e-5)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert actual_grad.dtype == expected_grad.dtype == jnp.dtype(dtype)
        assert jnp.allclose(actual_grad, expected_grad, rtol=1e-4, atol=1e-5)
