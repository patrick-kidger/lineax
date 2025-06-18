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
        pytest.skip(
            'JacobianLinearOperator does not support complex dtypes when jac="bwd"'
        )
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
