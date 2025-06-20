import equinox as ex
import jax.numpy as jnp
import lineax as lx


def test_ill_conditioned():
    solver = lx.LSMR(1e-10, 1e-10, conlim=1e3)
    Aill = lx.DiagonalLinearOperator(jnp.array([1e8, 1e6, 1e4, 1e2, 1]))
    try:
        lx.linear_solve(Aill, jnp.ones(5), solver=solver)
    except ex.EquinoxRuntimeError as e:
        assert "Condition number" in str(e)


def test_damp_regularizes():
    solver = lx.LSMR(1e-10, 1e-10)
    Aill = lx.DiagonalLinearOperator(jnp.array([1e8, 1e6, 1e4, 1e2, 1]))
    solution_ill = lx.linear_solve(Aill, jnp.ones(5), solver=solver, options={})
    assert solution_ill.stats["istop"] == 1

    solution_damped = lx.linear_solve(
        Aill, jnp.ones(5), solver=solver, options={"damp": 100.0}
    )
    assert solution_damped.stats["istop"] == 2

    assert solution_damped.stats["num_steps"] < solution_ill.stats["num_steps"]
