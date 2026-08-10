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
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from .helpers import (
    make_circulant_operator,
    make_identity_operator,
    make_jacrev_operator,
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
    elif make_operator is make_circulant_operator:
        column = jr.normal(getkey(), (3,), dtype=dtype)
        i, j = jnp.ogrid[:3, :3]
        matrix = column[(i - j) % 3]
        tags = lx.circulant_tag
    else:
        matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
        tags = ()
    if make_operator is make_jacrev_operator and dtype is jnp.complex128:
        # JacobianLinearOperator does not support complex dtypes when jac="bwd"
        return
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
    elif make_operator is make_circulant_operator:
        column = jr.normal(getkey(), (4,))
        i, j = jnp.ogrid[:4, :4]
        matrix = column[(i - j) % 4]
        tags = lx.circulant_tag
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
        if make_operator is make_circulant_operator and tag is not lx.circulant_tag:
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
        jitted_identity = eqx.filter_jit(lambda x: x)
        if isinstance(operator, lx.DiagonalLinearOperator):
            assert not cond_fun(operator)
            assert not cond_fun(jitted_identity(operator))
        else:
            assert cond_fun(operator)
            assert cond_fun(jitted_identity(operator))


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_linearise(dtype, getkey):
    matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
    operators = list(_setup(getkey, matrix))
    vec = jr.normal(getkey(), (3,), dtype=dtype)
    for operator in operators:
        # Skip jacrev operators with complex dtype (jacrev doesn't support complex)
        if (
            isinstance(operator, lx.JacobianLinearOperator)
            and operator.jac == "bwd"
            and dtype is jnp.complex128
        ):
            continue
        linearised = lx.linearise(operator)
        # Actually evaluate the linearised operator to ensure it works
        result = linearised.mv(vec)
        expected = operator.mv(vec)
        assert tree_allclose(result, expected)


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
    # test we properly extract diagonal from a dense matrix when not tagged
    operators = _setup(getkey, matrix)
    for operator in operators:
        assert jnp.allclose(lx.diagonal(operator), matrix_diag)
    # test we properly extract diagonal from diagonal matrix when tagged
    operators = _setup(getkey, jnp.diag(matrix_diag), lx.diagonal_tag)
    for operator in operators:
        if isinstance(operator, lx.IdentityLinearOperator):
            assert jnp.allclose(lx.diagonal(operator), jnp.ones(3, dtype))
        else:
            assert jnp.allclose(lx.diagonal(operator), matrix_diag)


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
    operators = _setup(getkey, tridiag_matrix, lx.tridiagonal_tag)
    for operator in operators:
        diag, lower_diag, upper_diag = lx.tridiagonal(operator)
        if isinstance(operator, lx.IdentityLinearOperator):
            assert jnp.allclose(diag, jnp.ones(5, dtype))
            assert jnp.allclose(lower_diag, jnp.zeros(4, dtype))
            assert jnp.allclose(upper_diag, jnp.zeros(4, dtype))
        else:
            assert jnp.allclose(diag, matrix_diag)
            assert jnp.allclose(lower_diag, matrix_lower_diag)
            assert jnp.allclose(upper_diag, matrix_upper_diag)

    # Test ComposedLinearOperator: diagonal @ tridiagonal and tridiagonal @ diagonal
    random_diag = jr.normal(getkey(), (5,), dtype=dtype)
    tridiag_op = lx.TridiagonalLinearOperator(
        matrix_diag, matrix_lower_diag, matrix_upper_diag
    )
    diag_op = lx.DiagonalLinearOperator(random_diag)

    # diagonal @ tridiagonal (row scaling)
    dt_matrix = jnp.matmul(jnp.diag(random_diag), tridiag_matrix)
    diag, lower_diag, upper_diag = lx.tridiagonal(diag_op @ tridiag_op)
    assert jnp.allclose(diag, jnp.diagonal(dt_matrix, 0))
    assert jnp.allclose(lower_diag, jnp.diagonal(dt_matrix, -1))
    assert jnp.allclose(upper_diag, jnp.diagonal(dt_matrix, 1))

    # tridiagonal @ diagonal (column scaling)
    td_matrix = jnp.matmul(tridiag_matrix, jnp.diag(random_diag))
    diag, lower_diag, upper_diag = lx.tridiagonal(tridiag_op @ diag_op)
    assert jnp.allclose(diag, jnp.diagonal(td_matrix, 0))
    assert jnp.allclose(lower_diag, jnp.diagonal(td_matrix, -1))
    assert jnp.allclose(upper_diag, jnp.diagonal(td_matrix, 1))


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_first_column(dtype, getkey):
    column = jr.normal(getkey(), (5,), dtype=dtype)
    i, j = jnp.ogrid[:5, :5]
    circulant_matrix = column[(i - j) % 5]
    operators = _setup(getkey, circulant_matrix, lx.circulant_tag)
    for operator in operators:
        col = lx.first_column(operator)
        assert jnp.allclose(col, column)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
@pytest.mark.parametrize(
    "tree_sizes",
    # (size1, size2), (size2, size3)  ..., (size_nm1, size_n)
    [
        ([4, {"a": 2, "b": 2}], [{"a": 2, "b": 2}, 3]),
        ([{"a": 2, "b": 2}, 4], [4, {"a": 2, "b": 1}]),
        ([[2, 1], [2, 3]], [[2, 3], 3]),
        ([4, 5], [5, 2]),
        (
            [4, {"a": 2, "b": 2}],
            [{"a": 2, "b": 2}, {"a": 2, "b": 1}],
            [{"a": 2, "b": 1}, {"a": 1, "b": 1}],
        ),
    ],
)
def test_first_column_composite(dtype, tree_sizes, getkey):
    operators = []
    for out_size, inp_size in tree_sizes:
        out_struct = jax.tree_util.tree_map(
            lambda size: jax.ShapeDtypeStruct((size,), dtype), out_size
        )
        pytree = jax.tree_util.tree_map(
            lambda out: jax.tree_util.tree_map(
                lambda inp: jr.normal(getkey(), (out, inp), dtype=dtype), inp_size
            ),
            out_size,
        )
        operators.append(lx.PyTreeLinearOperator(pytree, out_struct))

    composite = ft.reduce(lambda a, b: a @ b, operators)
    column = lx.first_column(composite)
    column_matrix = composite.as_matrix()[:, 0]
    assert jnp.allclose(column, column_matrix)
    assert column.dtype == dtype


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
def test_is_diagonal_circulant(dtype, getkey):
    column = jr.normal(getkey(), (1,), dtype=dtype)
    op1 = lx.CirculantLinearOperator(column)
    assert lx.is_diagonal(op1)

    column = jnp.zeros(3, dtype=dtype).at[0].set(2.0)
    op2 = lx.TaggedLinearOperator(lx.CirculantLinearOperator(column), lx.diagonal_tag)
    assert lx.is_diagonal(op2)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_has_unit_diagonal(dtype, getkey):
    matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
    not_unit_diagonal = _setup(getkey, matrix)
    for operator in not_unit_diagonal:
        assert not lx.has_unit_diagonal(operator)

    matrix_unit_diag = matrix.at[jnp.arange(3), jnp.arange(3)].set(1)
    unit_diagonal = _setup(getkey, matrix_unit_diag, lx.unit_diagonal_tag)
    _assert_except_diag(lx.has_unit_diagonal, unit_diagonal, flip_cond=False)
    assert not lx.has_unit_diagonal(
        2 * lx.MatrixLinearOperator(matrix, tags=lx.unit_diagonal_tag)
    )


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
def test_is_circulant(dtype, getkey):
    column1 = jr.normal(getkey(), (5,), dtype=dtype)
    op1 = lx.CirculantLinearOperator(column1)
    assert lx.is_circulant(op1)

    # C1 + C2 is circulant
    column2 = jr.normal(getkey(), (5,), dtype=dtype)
    op2 = lx.CirculantLinearOperator(column2)
    assert lx.is_circulant(op1 + op2)
    assert jnp.allclose(lx.first_column(op1 + op2), column1 + column2)

    # C1 @ C2 is Circulant
    assert lx.is_circulant(op1 @ op2)
    assert jnp.allclose(
        lx.first_column(op1 @ op2), (op1.as_matrix() @ op2.as_matrix())[:, 0]
    )

    # C1 @ Diag is not circulant
    op3 = lx.DiagonalLinearOperator(column2)
    assert not lx.is_circulant(op1 @ op3)

    # Untagged
    op4 = lx.MatrixLinearOperator(op1.as_matrix())
    assert not lx.is_circulant(op4)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_tangent_as_matrix(dtype, getkey):
    def _list_setup(matrix):
        # Exclude jacrev operator: jac="bwd" uses custom_vjp which doesn't support JVP
        return [
            op
            for op in _setup(getkey, matrix)
            if not (isinstance(op, lx.JacobianLinearOperator) and op.jac == "bwd")
        ]

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


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.complex128))
def test_identity_with_different_structures(dtype):
    # Same number of elements, laid out differently across the PyTree.
    structure1 = (
        jax.ShapeDtypeStruct((), dtype),
        jax.ShapeDtypeStruct((2, 3), jnp.float16),
    )
    structure2 = {"a": jax.ShapeDtypeStruct((7,), dtype)}
    op1 = lx.IdentityLinearOperator(structure1, structure2)
    op2 = lx.IdentityLinearOperator(structure2, structure1)

    assert op1.T == op2
    assert jnp.array_equal(op1.as_matrix(), jnp.eye(7, dtype=dtype))
    assert op1.in_size() == 7
    assert op1.out_size() == 7
    vec1 = (
        jnp.array(1.0, dtype=dtype),
        jnp.array([[2, 3, 4], [5, 6, 7]], dtype=jnp.float16),
    )
    vec2 = {"a": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=dtype)}
    assert tree_allclose(op1.mv(vec1), vec2)
    # Unlike the truncating behaviour this replaced, the round trip is exact.
    assert tree_allclose(op2.mv(vec2), vec1)


def test_identity_must_be_square():
    structure1 = (
        jax.ShapeDtypeStruct((), jnp.float32),
        jax.ShapeDtypeStruct((2, 3), jnp.float16),
    )
    structure2 = {"a": jax.ShapeDtypeStruct((5,), jnp.float32)}
    with pytest.raises(ValueError, match="same number of elements"):
        lx.IdentityLinearOperator(structure1, structure2)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64, jnp.complex128))
def test_identity_diagonal_dtype(dtype):
    # These used to fall back to the default floating dtype, which then blew up under
    # strict dtype promotion when combined with a non-default-dtype operator.
    operator = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), dtype))
    assert lx.diagonal(operator).dtype == dtype
    assert all(x.dtype == dtype for x in lx.tridiagonal(operator))
    assert operator.as_matrix().dtype == dtype


def test_compose_identity_with_different_structures():
    structure1 = (
        jax.ShapeDtypeStruct((), jnp.float32),
        jax.ShapeDtypeStruct((2,), jnp.float32),
    )
    structure2 = {"a": jax.ShapeDtypeStruct((3,), jnp.float32)}
    op1 = lx.IdentityLinearOperator(structure1, structure2)
    diagonal = lx.DiagonalLinearOperator(
        (
            jnp.array(2.0, dtype=jnp.float32),
            jnp.array([3.0, 4.0], dtype=jnp.float32),
        )
    )

    # Diagonal, but the composition does not land back in `structure1`, so it is not
    # symmetric and must not be rejected for having mismatched structures.
    composed = op1 @ diagonal
    assert lx.is_diagonal(composed)
    assert not lx.is_symmetric(composed)
    assert jnp.allclose(
        composed.as_matrix(), jnp.diag(jnp.array([2.0, 3.0, 4.0], dtype=jnp.float32))
    )
    vector = {"a": jnp.array([2.0, 6.0, 12.0], dtype=jnp.float32)}
    solution = lx.linear_solve(composed, vector).value
    assert tree_allclose(
        solution,
        (
            jnp.array(1.0, dtype=jnp.float32),
            jnp.array([2.0, 3.0], dtype=jnp.float32),
        ),
    )

    # But composing back to `structure1` is genuinely symmetric, even though neither
    # operand has matching input and output structures.
    op2 = lx.IdentityLinearOperator(structure2, structure1)
    round_trip = op2 @ op1
    assert lx.is_symmetric(round_trip)
    assert jnp.array_equal(round_trip.as_matrix(), jnp.eye(3, dtype=jnp.float32))


def test_identity_solve_with_different_structures():
    structure1 = (
        jax.ShapeDtypeStruct((), jnp.float32),
        jax.ShapeDtypeStruct((2, 3), jnp.float32),
    )
    structure2 = {"a": jax.ShapeDtypeStruct((7,), jnp.float32)}
    operator = lx.IdentityLinearOperator(structure1, structure2)
    vector = {"a": jnp.arange(1.0, 8.0, dtype=jnp.float32)}
    expected = (
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([[2, 3, 4], [5, 6, 7]], dtype=jnp.float32),
    )
    # The solution lives in the operator's in-structure, not its out-structure.
    solution = lx.linear_solve(operator, vector).value
    assert tree_allclose(solution, expected)
    assert tree_allclose(operator.mv(solution), vector)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_zero_pytree_as_matrix(dtype):
    a = jnp.array([], dtype=dtype).reshape(2, 1, 0, 2, 1, 0)
    struct = jax.ShapeDtypeStruct((2, 1, 0), a.dtype)
    op = lx.PyTreeLinearOperator(a, struct)
    assert op.as_matrix().shape == (0, 0)


def test_jacrev_operator():
    # Test that custom_vjp is respected. The custom backward multiplies by 3
    # instead of the true derivative (which would be 2).
    # This tests that lineax uses the custom_vjp, not the true derivative.
    @jax.custom_vjp
    def f(x, _):
        return dict(foo=x["bar"] * 2)  # forward: multiply by 2

    def f_fwd(x, _):
        return f(x, None), None

    def f_bwd(_, g):
        # Custom backward: multiply by 3 (not the true derivative 2)
        # This must be linear in g for linear_transpose to work correctly.
        return dict(bar=g["foo"] * 3), None

    f.defvjp(f_fwd, f_bwd)

    x = dict(bar=jnp.arange(2.0))
    rev_op = lx.JacobianLinearOperator(f, x, jac="bwd")
    # Jacobian is 3*I (from custom backward, not 2*I from true derivative)
    as_matrix = jnp.array([[3.0, 0.0], [0.0, 3.0]])
    assert tree_allclose(rev_op.as_matrix(), as_matrix)

    y = dict(bar=jnp.arange(2.0) + 1)  # y = [1, 2]
    true_out = dict(foo=jnp.array([3.0, 6.0]))  # 3*I @ [1, 2] = [3, 6]
    for op in (rev_op, lx.materialise(rev_op)):
        out = op.mv(y)
        assert tree_allclose(out, true_out)

    fwd_op = lx.JacobianLinearOperator(f, x, jac="fwd")
    with pytest.raises(TypeError, match="can't apply forward-mode autodiff"):
        fwd_op.mv(y)
    with pytest.raises(TypeError, match="can't apply forward-mode autodiff"):
        lx.materialise(fwd_op)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_circulant_tags_preserved(dtype, getkey):
    # Palindromic column -> symmetric, and eigenvalues [6.5, 3.5, 2.5, 3.5] > 0,
    # so the tags below are truthful rather than merely asserted.
    column = jnp.array([4.0, 1.0, 0.5, 1.0], dtype=dtype)

    # `CirculantLinearOperator` takes no tags of its own, so extra properties are
    # declared by wrapping. `TaggedLinearOperator` unions its tags with the inner
    # operator's checks, so circulance survives alongside the declared tag.
    op = lx.TaggedLinearOperator(
        lx.CirculantLinearOperator(column), lx.positive_semidefinite_tag
    )
    assert lx.is_positive_semidefinite(op.T)
    assert lx.is_positive_semidefinite(lx.conj(op))
    assert lx.is_circulant(op.T)
    assert lx.is_circulant(lx.conj(op))
    # The wrapper must not cost us the cheap first-column extraction.
    assert jnp.allclose(lx.first_column(op), column)

    op_sym = lx.TaggedLinearOperator(
        lx.CirculantLinearOperator(column), lx.symmetric_tag
    )
    assert lx.is_symmetric(op_sym.T)
