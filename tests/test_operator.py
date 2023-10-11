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

from typing import cast, Union

import equinox as eqx
import jax
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


def test_ops(getkey):
    matrix1 = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    matrix2 = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    scalar = jr.normal(getkey(), ())
    add = matrix1 + matrix2
    composed = matrix1 @ matrix2
    mul = matrix1 * scalar
    rmul = cast(lx.AbstractLinearOperator, scalar * matrix1)
    div = matrix1 / scalar
    vec = jr.normal(getkey(), (3,))

    assert shaped_allclose(matrix1.mv(vec) + matrix2.mv(vec), add.mv(vec))
    assert shaped_allclose(matrix1.mv(matrix2.mv(vec)), composed.mv(vec))
    scalar_matvec = scalar * matrix1.mv(vec)
    assert shaped_allclose(scalar_matvec, mul.mv(vec))
    assert shaped_allclose(scalar_matvec, rmul.mv(vec))
    assert shaped_allclose(matrix1.mv(vec) / scalar, div.mv(vec))

    add_matrix = matrix1.as_matrix() + matrix2.as_matrix()
    composed_matrix = matrix1.as_matrix() @ matrix2.as_matrix()
    mul_matrix = scalar * matrix1.as_matrix()
    div_matrix = matrix1.as_matrix() / scalar
    assert shaped_allclose(add_matrix, add.as_matrix())
    assert shaped_allclose(composed_matrix, composed.as_matrix())
    assert shaped_allclose(mul_matrix, mul.as_matrix())
    assert shaped_allclose(mul_matrix, rmul.as_matrix())
    assert shaped_allclose(div_matrix, div.as_matrix())

    assert shaped_allclose(add_matrix.T, add.T.as_matrix())
    assert shaped_allclose(composed_matrix.T, composed.T.as_matrix())
    assert shaped_allclose(mul_matrix.T, mul.T.as_matrix())
    assert shaped_allclose(mul_matrix.T, rmul.T.as_matrix())
    assert shaped_allclose(div_matrix.T, div.T.as_matrix())


@pytest.mark.parametrize("make_operator", make_operators)
def test_structures_vector(make_operator, getkey):
    if make_operator is make_diagonal_operator:
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
    operator = make_operator(matrix, tags)
    in_structure = jax.ShapeDtypeStruct((in_size,), jnp.float64)
    out_structure = jax.ShapeDtypeStruct((out_size,), jnp.float64)
    assert shaped_allclose(in_structure, operator.in_structure())
    assert shaped_allclose(out_structure, operator.out_structure())


def _setup(matrix, tag: Union[object, frozenset[object]] = frozenset()):
    for make_operator in make_operators:
        if make_operator is make_diagonal_operator and tag != lx.diagonal_tag:
            continue
        if make_operator is make_tridiagonal_operator and tag not in (
            lx.tridiagonal_tag,
            lx.diagonal_tag,
            lx.symmetric_tag,
        ):
            continue
        operator = make_operator(matrix, tag)
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


def test_linearise(getkey):
    operators = _setup(jr.normal(getkey(), (3, 3)))
    for operator in operators:
        lx.linearise(operator)


def test_materialise(getkey):
    operators = _setup(jr.normal(getkey(), (3, 3)))
    for operator in operators:
        lx.materialise(operator)


def test_diagonal(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    matrix_diag = jnp.diag(matrix)
    operators = _setup(matrix)
    for operator in operators:
        assert jnp.allclose(lx.diagonal(operator), matrix_diag)


def test_is_symmetric(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    symmetric_operators = _setup(matrix.T @ matrix, lx.symmetric_tag)
    for operator in symmetric_operators:
        assert lx.is_symmetric(operator)

    not_symmetric_operators = _setup(matrix)
    _assert_except_diag(lx.is_symmetric, not_symmetric_operators, flip_cond=True)


def test_is_diagonal(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    diagonal_operators = _setup(jnp.diag(jnp.diag(matrix)), lx.diagonal_tag)
    for operator in diagonal_operators:
        assert lx.is_diagonal(operator)

    not_diagonal_operators = _setup(matrix)
    _assert_except_diag(lx.is_diagonal, not_diagonal_operators, flip_cond=True)


def test_has_unit_diagonal(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    not_unit_diagonal = _setup(matrix)
    for operator in not_unit_diagonal:
        assert not lx.has_unit_diagonal(operator)

    matrix_unit_diag = matrix.at[jnp.arange(3), jnp.arange(3)].set(1)
    unit_diagonal = _setup(matrix_unit_diag, lx.unit_diagonal_tag)
    _assert_except_diag(lx.has_unit_diagonal, unit_diagonal, flip_cond=False)


def test_is_lower_triangular(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    lower_triangular = _setup(jnp.tril(matrix), lx.lower_triangular_tag)
    for operator in lower_triangular:
        assert lx.is_lower_triangular(operator)

    not_lower_triangular = _setup(matrix)
    _assert_except_diag(lx.is_lower_triangular, not_lower_triangular, flip_cond=True)


def test_is_upper_triangular(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    upper_triangular = _setup(jnp.triu(matrix), lx.upper_triangular_tag)
    for operator in upper_triangular:
        assert lx.is_upper_triangular(operator)

    not_upper_triangular = _setup(matrix)
    _assert_except_diag(lx.is_upper_triangular, not_upper_triangular, flip_cond=True)


def test_is_positive_semidefinite(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    not_positive_semidefinite = _setup(matrix)
    for operator in not_positive_semidefinite:
        assert not lx.is_positive_semidefinite(operator)

    positive_semidefinite = _setup(matrix.T @ matrix, lx.positive_semidefinite_tag)
    _assert_except_diag(
        lx.is_positive_semidefinite, positive_semidefinite, flip_cond=False
    )


def test_is_negative_semidefinite(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    not_negative_semidefinite = _setup(matrix)
    for operator in not_negative_semidefinite:
        assert not lx.is_negative_semidefinite(operator)

    negative_semidefinite = _setup(-matrix.T @ matrix, lx.negative_semidefinite_tag)
    _assert_except_diag(
        lx.is_negative_semidefinite, negative_semidefinite, flip_cond=False
    )


def test_is_tridiagonal(getkey):
    diag1 = jr.normal(getkey(), (5,))
    diag2 = jr.normal(getkey(), (4,))
    diag3 = jr.normal(getkey(), (4,))
    op1 = lx.TridiagonalLinearOperator(diag1, diag2, diag3)
    op2 = lx.IdentityLinearOperator(jax.eval_shape(lambda: diag1))
    op3 = lx.MatrixLinearOperator(jnp.diag(diag1))
    assert lx.is_tridiagonal(op1)
    assert lx.is_tridiagonal(op2)
    assert not lx.is_tridiagonal(op3)


def test_tangent_as_matrix(getkey):
    def _list_setup(matrix):
        return list(_setup(matrix))

    matrix = jr.normal(getkey(), (3, 3))
    t_matrix = jr.normal(getkey(), (3, 3))
    operators, t_operators = eqx.filter_jvp(_list_setup, (matrix,), (t_matrix,))
    for operator, t_operator in zip(operators, t_operators):
        t_operator = lx.TangentLinearOperator(operator, t_operator)
        if isinstance(operator, lx.DiagonalLinearOperator):
            assert jnp.allclose(operator.as_matrix(), jnp.diag(jnp.diag(matrix)))
            assert jnp.allclose(t_operator.as_matrix(), jnp.diag(jnp.diag(t_matrix)))
        else:
            assert jnp.allclose(operator.as_matrix(), matrix)
            assert jnp.allclose(t_operator.as_matrix(), t_matrix)


def test_materialise_function_linear_operator(getkey):
    x = (jr.normal(getkey(), (5, 9)), jr.normal(getkey(), (3,)))
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
            jax.ShapeDtypeStruct((1, 2, 5, 9), jnp.float64),
            jax.ShapeDtypeStruct((1, 2, 3), jnp.float64),
        )
    }
    assert jax.eval_shape(lambda: materialised_operator.pytree) == expected_struct


def test_pytree_transpose(getkey):
    out_struct = jax.eval_shape(lambda: ({"a": jnp.zeros((2, 3, 3))}, jnp.zeros((2,))))
    in_struct = jax.eval_shape(lambda: {"b": jnp.zeros((4,))})
    leaf1 = jr.normal(getkey(), (2, 3, 3, 4))
    leaf2 = jr.normal(getkey(), (2, 4))
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
    assert jnp.array_equal(op1.as_matrix(), jnp.eye(5, 7))
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
    assert shaped_allclose(op1.mv(vec1), vec2)
    assert shaped_allclose(op2.mv(vec2), vec1b)
