import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest
from lineax import FunctionLinearOperator

from .helpers import (
    make_identity_operator,
    make_jacrev_operator,
    make_operators,
    make_tridiagonal_operator,
    make_trivial_diagonal_operator,
    tree_allclose,
)


@pytest.mark.parametrize("make_operator", make_operators)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_adjoint(make_operator, dtype, getkey):
    if (
        make_operator is make_trivial_diagonal_operator
        or make_operator is make_identity_operator
    ):
        matrix = jnp.eye(4, dtype=dtype)
        tags = lx.diagonal_tag
        in_size = out_size = 4
    elif make_operator is make_tridiagonal_operator:
        matrix = jnp.eye(4, dtype=dtype)
        tags = lx.tridiagonal_tag
        in_size = out_size = 4
    else:
        matrix = jr.normal(getkey(), (3, 5), dtype=dtype)
        tags = ()
        in_size = 5
        out_size = 3
    if make_operator is make_jacrev_operator and dtype is jnp.complex128:
        # JacobianLinearOperator does not support complex dtypes when jac="bwd"
        return
    operator = make_operator(getkey, matrix, tags)
    v1, v2 = (
        jr.normal(getkey(), (in_size,), dtype=dtype),
        jr.normal(getkey(), (out_size,), dtype=dtype),
    )

    inner1 = operator.mv(v1) @ v2.conj()
    adjoint_op1 = lx.conj(operator).transpose()
    ov2 = adjoint_op1.mv(v2)
    inner2 = v1 @ ov2.conj()
    assert tree_allclose(inner1, inner2)

    adjoint_op2 = lx.conj(operator.transpose())
    ov2 = adjoint_op2.mv(v2)
    inner2 = v1 @ ov2.conj()
    assert tree_allclose(inner1, inner2)


def test_functional_pytree_adjoint():
    def fn(y):
        return {"b": y["a"]}

    y_struct = jax.eval_shape(lambda: {"a": 0.0})
    operator = FunctionLinearOperator(fn, y_struct)
    conj_operator = lx.conj(operator)
    assert tree_allclose(lx.materialise(conj_operator), lx.materialise(operator))


def test_functional_pytree_adjoint_complex():
    def fn(y):
        return {"b": y["a"]}

    y_struct = jax.eval_shape(lambda: {"a": 0.0j})
    operator = FunctionLinearOperator(fn, y_struct)
    conj_operator = lx.conj(operator)
    assert tree_allclose(lx.materialise(conj_operator), lx.materialise(operator))


if jax.config.jax_enable_x64:  # pyright: ignore
    tol = 1e-12
else:
    tol = 1e-6


@pytest.mark.parametrize(
    "solver",
    [
        # in theory only 1 iteration is needed, but stopping criteria are
        # complicated, see gh #160
        lx.GMRES(tol, tol, max_steps=4, restart=1),
        lx.BiCGStab(tol, tol, max_steps=3),
        lx.Normal(lx.CG(tol, tol, max_steps=4)),
        lx.CG(tol, tol, max_steps=3),
    ],
)
def test_preconditioner_adjoint(solver):
    """Test for fix to gh #160"""
    # Nonsymmetric poorly conditioned matrix. Without preconditioning,
    # this would take 20+ iterations (100s for GMRES)
    key = jax.random.key(123)
    key, subkey = jax.random.split(key)
    A = jax.random.uniform(key, (10, 10))
    A += jnp.diag(jnp.arange(A.shape[0]) ** 6).astype(A.dtype)
    b = jax.random.uniform(subkey, (A.shape[0],))
    if isinstance(solver, lx.CG):
        A = A.T @ A
        tags = (lx.positive_semidefinite_tag,)
    else:
        tags = ()

    A = lx.MatrixLinearOperator(A, tags=tags)
    # exact inverse, should only take ~1 iteration
    M = lx.MatrixLinearOperator(
        jnp.linalg.inv(A.matrix),
        tags=tags,
    )

    def solve(b):
        out = lx.linear_solve(
            A, b, solver=solver, options={"preconditioner": M}, throw=True
        )
        return out.value

    # if they don't converge then this will throw an error
    _ = solve(b)
    A1 = jax.jacfwd(solve)(b)
    A2 = jax.jacrev(solve)(b)

    # we also do a sanity check, dx/db should give A^{-1}
    assert tree_allclose(A1, jnp.linalg.inv(A.matrix), atol=tol, rtol=tol)
    assert tree_allclose(A2, jnp.linalg.inv(A.matrix), atol=tol, rtol=tol)
