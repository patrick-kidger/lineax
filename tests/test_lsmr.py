import equinox as ex
import jax.numpy as jnp
import lineax as lx
import pytest


solver = lx.LSMR(1e-10, 1e-10)
Aill = lx.DiagonalLinearOperator(jnp.array([1e8, 1e6, 1e4, 1e2, 1]))
Awell = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0, 5.0, 8.0, 10.0]))


def test_ill_conditioned():
    try:
        lx.linear_solve(Aill, jnp.ones(5), solver=solver)
    except ex.EquinoxRuntimeError as e:
        assert "Condition number" in str(e)


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
