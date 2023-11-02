import jax.numpy as jnp
import jax.random as jr
import pytest

import lineax as lx

from .helpers import (
    make_diagonal_operator,
    make_operators,
    make_tridiagonal_operator,
    shaped_allclose,
)


@pytest.mark.parametrize("make_operator", make_operators)
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.complex64))
def test_adjoint(make_operator, dtype, getkey):
    if make_operator is make_diagonal_operator:
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
    operator = make_operator(matrix, tags)
    v1, v2 = jr.normal(getkey(), (in_size,), dtype=dtype), jr.normal(
        getkey(), (out_size,), dtype=dtype
    )

    inner1 = operator.mv(v1) @ v2.conj()
    adjoint_op1 = lx.conj(operator).transpose()
    ov2 = adjoint_op1.mv(v2)
    inner2 = v1 @ ov2.conj()
    assert shaped_allclose(inner1, inner2)

    adjoint_op2 = lx.conj(operator.transpose())
    ov2 = adjoint_op2.mv(v2)
    inner2 = v1 @ ov2.conj()
    assert shaped_allclose(inner1, inner2)
