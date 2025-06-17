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

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from .helpers import (
    make_identity_operator,
    make_operators,
    make_tridiagonal_operator,
    make_trivial_diagonal_operator,
    tree_allclose,
)


@pytest.mark.parametrize("make_operator", make_operators)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_ops(make_operator, getkey, dtype):
    if (
        make_operator is make_trivial_diagonal_operator
        or make_operator is make_identity_operator
    ):
        matrix = jnp.eye(3, dtype=dtype)
        tags = lx.diagonal_tag
    elif make_operator is make_tridiagonal_operator:
        matrix = jnp.eye(3, dtype=dtype)
        tags = lx.tridiagonal_tag
    else:
        matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
        tags = ()
    matrix1 = make_operator(getkey, matrix, tags)
    matrix2 = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3), dtype=dtype))
    scalar = jr.normal(getkey(), (), dtype=dtype)
    add = matrix1 + matrix2
    composed = matrix1 @ matrix2
    mul = matrix1 * scalar
    rmul = cast(lx.AbstractLinearOperator, scalar * matrix1)
    div = matrix1 / scalar
    vec = jr.normal(getkey(), (3,), dtype=dtype)

    assert tree_allclose(matrix1.mv(vec) + matrix2.mv(vec), add.mv(vec))
    assert tree_allclose(matrix1.mv(matrix2.mv(vec)), composed.mv(vec))
    scalar_matvec = scalar * matrix1.mv(vec)
    assert tree_allclose(scalar_matvec, mul.mv(vec))
    assert tree_allclose(scalar_matvec, rmul.mv(vec))
    assert tree_allclose(matrix1.mv(vec) / scalar, div.mv(vec))

    add_matrix = matrix1.as_matrix() + matrix2.as_matrix()
    composed_matrix = matrix1.as_matrix() @ matrix2.as_matrix()
    mul_matrix = scalar * matrix1.as_matrix()
    div_matrix = matrix1.as_matrix() / scalar
    assert tree_allclose(add_matrix, add.as_matrix())
    assert tree_allclose(composed_matrix, composed.as_matrix())
    assert tree_allclose(mul_matrix, mul.as_matrix())
    assert tree_allclose(mul_matrix, rmul.as_matrix())
    assert tree_allclose(div_matrix, div.as_matrix())

    assert tree_allclose(add_matrix.T, add.T.as_matrix())
    assert tree_allclose(composed_matrix.T, composed.T.as_matrix())
    assert tree_allclose(mul_matrix.T, mul.T.as_matrix())
    assert tree_allclose(mul_matrix.T, rmul.T.as_matrix())
    assert tree_allclose(div_matrix.T, div.T.as_matrix())


@pytest.mark.parametrize("make_operator", make_operators)
def test_structures_vector(make_operator, getkey):
    if (
        make_operator is make_trivial_diagonal_operator
        or make_operator is make_identity_operator
    ):
        matrix = jnp.eye(4)
        tags = lx.diagonal_tag
        in_size = out_size = 4
    elif make_operator is make_tridiagonal_operator:
        matrix = jnp.eye(4)
        tags = lx.tridiagonal_tag
        in_size = out_size = 4
    else:
        matrix = jr.normal(getkey(), (3, 5))
        tags = ()
        in_size = 5
        out_size = 3
    operator = make_operator(getkey, matrix, tags)
    in_structure = jax.ShapeDtypeStruct((in_size,), jnp.float64)
    out_structure = jax.ShapeDtypeStruct((out_size,), jnp.float64)
    assert tree_allclose(in_structure, operator.in_structure())
    assert tree_allclose(out_structure, operator.out_structure())


def _setup(getkey, matrix, tag: object | frozenset[object] = frozenset()):
    for make_operator in make_operators:
        if make_operator is make_trivial_diagonal_operator and tag != lx.diagonal_tag:
            continue
        if make_operator is make_tridiagonal_operator and tag not in (
            lx.tridiagonal_tag,
            lx.diagonal_tag,
            lx.symmetric_tag,
        ):
            continue
        if make_operator is make_identity_operator and tag not in (
            lx.tridiagonal_tag,
            lx.diagonal_tag,
            lx.symmetric_tag,
        ):
            continue
        operator = make_operator(getkey, matrix, tag)
        yield operator


def _assert_except_diag(cond_fun, operators, flip_cond):
    if flip_cond:
        _cond_fun = cond_fun
        cond_fun = lambda x: not _cond_fun(x)
    for operator in operators:
        if isinstance(operator, lx.DiagonalLinearOperator):
            assert not cond_fun(operator)
        else:
            assert cond_fun(operator)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_linearise(dtype, getkey):
    operators = _setup(getkey, jr.normal(getkey(), (3, 3), dtype=dtype))
    for operator in operators:
        lx.linearise(operator)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_materialise(dtype, getkey):
    operators = _setup(getkey, jr.normal(getkey(), (3, 3), dtype=dtype))
    for operator in operators:
        lx.materialise(operator)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_materialise_large(dtype, getkey):
    operators = _setup(getkey, jr.normal(getkey(), (200, 500), dtype=dtype))
    for operator in operators:
        lx.materialise(operator)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_diagonal(dtype, getkey):
    matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
    matrix_diag = jnp.diag(matrix)
    operators = _setup(getkey, matrix)
    for operator in operators:
        assert jnp.allclose(lx.diagonal(operator), matrix_diag)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_is_symmetric(dtype, getkey):
    matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
    symmetric_operators = _setup(getkey, matrix.T @ matrix, lx.symmetric_tag)
    for operator in symmetric_operators:
        assert lx.is_symmetric(operator)

    not_symmetric_operators = _setup(getkey, matrix)
    _assert_except_diag(lx.is_symmetric, not_symmetric_operators, flip_cond=True)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_is_diagonal(dtype, getkey):
    matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
    diagonal_operators = _setup(getkey, jnp.diag(jnp.diag(matrix)), lx.diagonal_tag)
    for operator in diagonal_operators:
        assert lx.is_diagonal(operator)

    not_diagonal_operators = _setup(getkey, matrix)
    _assert_except_diag(lx.is_diagonal, not_diagonal_operators, flip_cond=True)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_is_diagonal_scalar(dtype, getkey):
    matrix = jr.normal(getkey(), (1, 1), dtype=dtype)
    diagonal_operators = _setup(getkey, matrix)
    for operator in diagonal_operators:
        assert lx.is_diagonal(operator)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_is_diagonal_tridiagonal(dtype, getkey):
    diag1 = jr.normal(getkey(), (1,), dtype=dtype)
    diag2 = jnp.zeros((0,), dtype=dtype)
    op1 = lx.TridiagonalLinearOperator(diag1, diag2, diag2)
    assert lx.is_diagonal(op1)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_has_unit_diagonal(dtype, getkey):
    matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
    not_unit_diagonal = _setup(getkey, matrix)
    for operator in not_unit_diagonal:
        assert not lx.has_unit_diagonal(operator)

    matrix_unit_diag = matrix.at[jnp.arange(3), jnp.arange(3)].set(1)
    unit_diagonal = _setup(getkey, matrix_unit_diag, lx.unit_diagonal_tag)
    _assert_except_diag(lx.has_unit_diagonal, unit_diagonal, flip_cond=False)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_is_lower_triangular(dtype, getkey):
    matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
    lower_triangular = _setup(getkey, jnp.tril(matrix), lx.lower_triangular_tag)
    for operator in lower_triangular:
        assert lx.is_lower_triangular(operator)

    not_lower_triangular = _setup(getkey, matrix)
    _assert_except_diag(lx.is_lower_triangular, not_lower_triangular, flip_cond=True)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_is_upper_triangular(dtype, getkey):
    matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
    upper_triangular = _setup(getkey, jnp.triu(matrix), lx.upper_triangular_tag)
    for operator in upper_triangular:
        assert lx.is_upper_triangular(operator)

    not_upper_triangular = _setup(getkey, matrix)
    _assert_except_diag(lx.is_upper_triangular, not_upper_triangular, flip_cond=True)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_is_positive_semidefinite(dtype, getkey):
    matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
    not_positive_semidefinite = _setup(getkey, matrix)
    for operator in not_positive_semidefinite:
        assert not lx.is_positive_semidefinite(operator)

    positive_semidefinite = _setup(
        getkey, matrix.T.conj() @ matrix, lx.positive_semidefinite_tag
    )
    _assert_except_diag(
        lx.is_positive_semidefinite, positive_semidefinite, flip_cond=False
    )


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_is_negative_semidefinite(dtype, getkey):
    matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
    not_negative_semidefinite = _setup(getkey, matrix)
    for operator in not_negative_semidefinite:
        assert not lx.is_negative_semidefinite(operator)

    negative_semidefinite = _setup(
        getkey, -matrix.T.conj() @ matrix, lx.negative_semidefinite_tag
    )
    _assert_except_diag(
        lx.is_negative_semidefinite, negative_semidefinite, flip_cond=False
    )


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_tridiagonal(dtype, getkey):
    matrix = jr.normal(getkey(), (5, 5), dtype=dtype)
    matrix_diag = jnp.diag(matrix)
    matrix_lower_diag = jnp.diag(matrix, k=-1)
    matrix_upper_diag = jnp.diag(matrix, k=1)
    tridiag_matrix = (
        jnp.diag(matrix_diag)
        + jnp.diag(matrix_lower_diag, k=-1)
        + jnp.diag(matrix_upper_diag, k=1)
    )
    print(tridiag_matrix)
    operators = _setup(getkey, tridiag_matrix)
    for operator in operators:
        diag, lower_diag, upper_diag = lx.tridiagonal(operator)
        assert jnp.allclose(diag, matrix_diag)
        assert jnp.allclose(lower_diag, matrix_lower_diag)
        assert jnp.allclose(upper_diag, matrix_upper_diag)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_is_tridiagonal(dtype, getkey):
    diag1 = jr.normal(getkey(), (5,), dtype=dtype)
    diag2 = jr.normal(getkey(), (4,), dtype=dtype)
    diag3 = jr.normal(getkey(), (4,), dtype=dtype)
    op1 = lx.TridiagonalLinearOperator(diag1, diag2, diag3)
    op2 = lx.IdentityLinearOperator(jax.eval_shape(lambda: diag1))
    op3 = lx.MatrixLinearOperator(jnp.diag(diag1))
    assert lx.is_tridiagonal(op1)
    assert lx.is_tridiagonal(op2)
    assert not lx.is_tridiagonal(op3)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_tangent_as_matrix(dtype, getkey):
    def _list_setup(matrix):
        return list(_setup(getkey, matrix))

    matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
    t_matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
    operators, t_operators = eqx.filter_jvp(_list_setup, (matrix,), (t_matrix,))
    for operator, t_operator in zip(operators, t_operators):
        t_operator = lx.TangentLinearOperator(operator, t_operator)
        if isinstance(operator, lx.DiagonalLinearOperator):
            assert jnp.allclose(operator.as_matrix(), jnp.diag(jnp.diag(matrix)))
            assert jnp.allclose(t_operator.as_matrix(), jnp.diag(jnp.diag(t_matrix)))
        else:
            assert jnp.allclose(operator.as_matrix(), matrix)
            assert jnp.allclose(t_operator.as_matrix(), t_matrix)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_materialise_function_linear_operator(dtype, getkey):
    x = (
        jr.normal(getkey(), (5, 9), dtype=dtype),
        jr.normal(getkey(), (3,), dtype=dtype),
    )
    input_structure = jax.eval_shape(lambda: x)
    fn = lambda x: {"a": jnp.broadcast_to(jnp.sum(x[0]), (1, 2))}
    output_structure = jax.eval_shape(fn, input_structure)
    operator = lx.FunctionLinearOperator(fn, input_structure)
    materialised_operator = lx.materialise(operator)
    assert materialised_operator.in_structure() == input_structure
    assert materialised_operator.out_structure() == output_structure
    assert isinstance(materialised_operator, lx.PyTreeLinearOperator)
    expected_struct = {
        "a": (
            jax.ShapeDtypeStruct((1, 2, 5, 9), dtype),
            jax.ShapeDtypeStruct((1, 2, 3), dtype),
        )
    }
    assert jax.eval_shape(lambda: materialised_operator.pytree) == expected_struct


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_pytree_transpose(dtype, getkey):
    out_struct = jax.eval_shape(
        lambda: ({"a": jnp.zeros((2, 3, 3), dtype=dtype)}, jnp.zeros((2,), dtype=dtype))
    )
    in_struct = jax.eval_shape(lambda: {"b": jnp.zeros((4,), dtype=dtype)})
    leaf1 = jr.normal(getkey(), (2, 3, 3, 4), dtype=dtype)
    leaf2 = jr.normal(getkey(), (2, 4), dtype=dtype)
    pytree = ({"a": {"b": leaf1}}, {"b": leaf2})
    operator = lx.PyTreeLinearOperator(pytree, out_struct)
    assert operator.in_structure() == in_struct
    assert operator.out_structure() == out_struct
    leaf1_T = jnp.moveaxis(leaf1, -1, 0)
    leaf2_T = jnp.moveaxis(leaf2, -1, 0)
    pytree_T = {"b": ({"a": leaf1_T}, leaf2_T)}
    operator_T = operator.T
    assert operator_T.in_structure() == out_struct
    assert operator_T.out_structure() == in_struct
    assert eqx.tree_equal(operator_T.pytree, pytree_T)  # pyright: ignore


def test_diagonal_tangent():
    diag = jnp.array([1.0, 2.0, 3.0])
    t_diag = jnp.array([4.0, 5.0, 6.0])

    def run(diag):
        op = lx.DiagonalLinearOperator(diag)
        out = lx.linear_solve(op, jnp.array([1.0, 1.0, 1.0]), solver=lx.Diagonal())
        return out.value

    jax.jvp(run, (diag,), (t_diag,))


def test_identity_with_different_structures():
    structure1 = (
        jax.ShapeDtypeStruct((), jnp.float32),
        jax.ShapeDtypeStruct((2, 3), jnp.float16),
    )
    structure2 = {"a": jax.ShapeDtypeStruct((5,), jnp.float32)}
    # structure3 = (None, jax.ShapeDtypeStruct((2, 3), jnp.float16))
    op1 = lx.IdentityLinearOperator(structure1, structure2)
    op2 = lx.IdentityLinearOperator(structure2, structure1)
    # op3 = lx.IdentityLinearOperator(structure3, structure2)

    assert op1.T == op2
    # assert op2.transpose((True, False)) == op3
    assert jnp.array_equal(op1.as_matrix(), jnp.eye(5, 7, dtype=jnp.float32))
    assert op1.in_size() == 7
    assert op1.out_size() == 5
    vec1 = (
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([[2, 3, 4], [5, 6, 7]], dtype=jnp.float16),
    )
    vec2 = {"a": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)}
    vec1b = (
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([[2, 3, 4], [5, 0, 0]], dtype=jnp.float16),
    )
    assert tree_allclose(op1.mv(vec1), vec2)
    assert tree_allclose(op2.mv(vec2), vec1b)


def test_identity_with_different_structures_complex():
    structure1 = (
        jax.ShapeDtypeStruct((), jnp.complex128),
        jax.ShapeDtypeStruct((2, 3), jnp.float16),
    )
    structure2 = {"a": jax.ShapeDtypeStruct((5,), jnp.complex128)}
    # structure3 = (None, jax.ShapeDtypeStruct((2, 3), jnp.float16))
    op1 = lx.IdentityLinearOperator(structure1, structure2)
    op2 = lx.IdentityLinearOperator(structure2, structure1)
    # op3 = lx.IdentityLinearOperator(structure3, structure2)

    assert op1.T == op2
    # assert op2.transpose((True, False)) == op3
    assert jnp.array_equal(op1.as_matrix(), jnp.eye(5, 7, dtype=jnp.complex128))
    assert op1.in_size() == 7
    assert op1.out_size() == 5
    vec1 = (
        jnp.array(1.0, dtype=jnp.complex128),
        jnp.array([[2, 3, 4], [5, 6, 7]], dtype=jnp.float16),
    )
    vec2 = {"a": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.complex128)}
    vec1b = (
        jnp.array(1.0, dtype=jnp.complex128),
        jnp.array([[2, 3, 4], [5, 0, 0]], dtype=jnp.float16),
    )
    assert tree_allclose(op1.mv(vec1), vec2)
    assert tree_allclose(op2.mv(vec2), vec1b)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_zero_pytree_as_matrix(dtype):
    a = jnp.array([], dtype=dtype).reshape(2, 1, 0, 2, 1, 0)
    struct = jax.ShapeDtypeStruct((2, 1, 0), a.dtype)
    op = lx.PyTreeLinearOperator(a, struct)
    assert op.as_matrix().shape == (0, 0)


def test_jacrev_operator():
    @jax.custom_vjp
    def f(x, _):
        return dict(foo=x["bar"] + 2)

    def f_fwd(x, _):
        return f(x, None), None

    def f_bwd(_, g):
        return dict(bar=g["foo"] + 5), None

    f.defvjp(f_fwd, f_bwd)

    x = dict(bar=jnp.arange(2.0))
    rev_op = lx.JacobianLinearOperator(f, x, jac="bwd")
    as_matrix = jnp.array([[6.0, 5.0], [5.0, 6.0]])
    assert tree_allclose(rev_op.as_matrix(), as_matrix)

    y = dict(bar=jnp.arange(2.0) + 1)
    true_out = dict(foo=jnp.array([16.0, 17.0]))
    for op in (rev_op, lx.materialise(rev_op)):
        out = op.mv(y)
        assert tree_allclose(out, true_out)

    fwd_op = lx.JacobianLinearOperator(f, x, jac="fwd")
    with pytest.raises(TypeError, match="can't apply forward-mode autodiff"):
        fwd_op.mv(y)
    with pytest.raises(TypeError, match="can't apply forward-mode autodiff"):
        lx.materialise(fwd_op)
