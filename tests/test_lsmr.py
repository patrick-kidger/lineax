import jax.numpy as jnp
import lineax as lx


solver = lx.LSMR(0, 0)


def test_ill_conditioned():
    Aill = lx.DiagonalLinearOperator(jnp.array([1e5, 1]))
    solution_ill = lx.linear_solve(
        Aill, jnp.ones(2), solver=solver, options={"condlim": 1e4}
    )
    assert solution_ill.stats["istop"] == 3


def test_damp_regularizes():
    Aill = lx.DiagonalLinearOperator(jnp.array([1e5, 1]))
    solution_ill = lx.linear_solve(Aill, jnp.ones(2), solver=solver, options={})
    assert solution_ill.stats["istop"] == 1

    solution_damped = lx.linear_solve(
        Aill, jnp.ones(2), solver=solver, options={"damp": 1}
    )
    assert solution_damped.stats["istop"] == 1

    assert solution_damped.stats["num_steps"] < solution_ill.stats["num_steps"]
