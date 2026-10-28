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
from lineax._operator.base import (
    diagonal_via_mv,
    is_materialised,
    tridiagonal_via_coloring,
)
from lineax._operator.wrapper import TangentLinearOperator

from .helpers import (
    make_circulant_operator,
    make_identity_operator,
    make_jacrev_operator,
    make_operators,
    make_tridiagonal_operator,
    make_trivial_diagonal_operator,
    tree_allclose,
)


def _square_matrix_and_tags(make_operator, getkey, dtype, size=3):
    if (
        make_operator is make_trivial_diagonal_operator
        or make_operator is make_identity_operator
    ):
        matrix = jnp.eye(size, dtype=dtype)
        tags = lx.diagonal_tag
    elif make_operator is make_tridiagonal_operator:
        matrix = jnp.eye(size, dtype=dtype)
        tags = lx.tridiagonal_tag
    elif make_operator is make_circulant_operator:
        column = jr.normal(getkey(), (size,), dtype=dtype)
        i, j = jnp.ogrid[:size, :size]
        matrix = column[(i - j) % size]
        tags = lx.circulant_tag
    else:
        matrix = jr.normal(getkey(), (size, size), dtype=dtype)
        tags = ()
    return matrix, tags


@pytest.mark.parametrize("make_operator", make_operators)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_ops(make_operator, getkey, dtype):
    matrix, tags = _square_matrix_and_tags(make_operator, getkey, dtype)
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
def test_diagonal_tridiagonal_tagged_wraps_untagged_operator(dtype, getkey):
    # `TaggedLinearOperator` retroactively asserting is_diagonal/is_tridiagonal on an
    # opaque (`FunctionLinearOperator`) operator that doesn't know about it itself
    # should take the coloring-based fast path, not fall through to materialising it.
    # Using a dense (rather than genuinely diagonal/tridiagonal) matrix here means
    # `diagonal`/`tridiagonal` would give a different answer had they instead fallen
    # through to materialising: matching diagonal_via_mv/tridiagonal_via_coloring
    # directly is what proves it's the fast path that actually ran.
    size = 4
    matrix = jr.normal(getkey(), (size, size), dtype=dtype)
    in_struct = jax.ShapeDtypeStruct((size,), dtype)
    fn_op = lx.FunctionLinearOperator(lambda x: matrix @ x, in_struct)

    diag_wrapped = lx.TaggedLinearOperator(fn_op, lx.diagonal_tag)
    assert jnp.allclose(lx.diagonal(diag_wrapped), diagonal_via_mv(diag_wrapped))

    tridiag_wrapped = lx.TaggedLinearOperator(fn_op, lx.tridiagonal_tag)
    assert tree_allclose(
        lx.tridiagonal(tridiag_wrapped), tridiagonal_via_coloring(tridiag_wrapped)
    )


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_has_unit_diagonal_tagged_wrapper_trusts_tag(dtype, getkey):
    # The has_unit_diagonal fast path must trust `TaggedLinearOperator`'s own tag
    # rather than the wrapped operator's. Tag a genuinely non-unit-diagonal function
    # (`2x`, real diagonal `[2, 2, ...]`) as unit-diagonal anyway: getting back all
    # ones proves the fast path fired (trusting the tag), rather than falling through
    # to the wrapped operator and materialising its real diagonal.
    size = 4
    in_struct = jax.ShapeDtypeStruct((size,), dtype)
    fn_op = lx.FunctionLinearOperator(lambda x: 2.0 * x, in_struct)
    wrapped = lx.TaggedLinearOperator(fn_op, lx.unit_diagonal_tag)
    assert jnp.allclose(lx.diagonal(wrapped), jnp.ones(size, dtype))
    assert jnp.allclose(lx.trace(wrapped), size)


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


@pytest.mark.parametrize("make_operator2", make_operators)
@pytest.mark.parametrize("make_operator1", make_operators)
def test_diagonal_composed(make_operator1, make_operator2, getkey):
    # Sweeping every pair of operator flavours naturally exercises every fast path in
    # `diagonal(ComposedLinearOperator)`: diagonal-or-anything (make_trivial_diagonal/
    # make_identity crossed with anything), tridiagonal-or-anything, and circulant @
    # circulant, in addition to the generic materialising fallback.
    dtype = jnp.float64
    matrix1, tags1 = _square_matrix_and_tags(make_operator1, getkey, dtype)
    matrix2, tags2 = _square_matrix_and_tags(make_operator2, getkey, dtype)
    op1 = make_operator1(getkey, matrix1, tags1)
    op2 = make_operator2(getkey, matrix2, tags2)
    composed_matrix = op1.as_matrix() @ op2.as_matrix()
    composed = op1 @ op2
    assert tree_allclose(lx.diagonal(composed), jnp.diag(composed_matrix))
    assert tree_allclose(lx.trace(composed), jnp.trace(composed_matrix))


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_diagonal_composed_triangular(dtype, getkey):
    # `make_operators` has no triangular-tagged flavour, so this isn't reachable via
    # the sweep in `test_diagonal_composed` above and needs its own construction.
    size = 5

    def make_triangular(triangularise, tag):
        matrix = triangularise(jr.normal(getkey(), (size, size), dtype=dtype))
        return lx.MatrixLinearOperator(matrix, tags=tag), matrix

    def check(composed, composed_matrix):
        assert tree_allclose(lx.diagonal(composed), jnp.diag(composed_matrix))
        assert tree_allclose(lx.trace(composed), jnp.trace(composed_matrix))

    # same-orientation triangular @ triangular
    lower_op, lower_matrix = make_triangular(jnp.tril, lx.lower_triangular_tag)
    lower_op2, lower_matrix2 = make_triangular(jnp.tril, lx.lower_triangular_tag)
    check(lower_op @ lower_op2, lower_matrix @ lower_matrix2)

    upper_op, upper_matrix = make_triangular(jnp.triu, lx.upper_triangular_tag)
    upper_op2, upper_matrix2 = make_triangular(jnp.triu, lx.upper_triangular_tag)
    check(upper_op @ upper_op2, upper_matrix @ upper_matrix2)

    # mixed-orientation triangular @ triangular: falls back to materialising, but
    # should still be correct
    check(lower_op @ upper_op, lower_matrix @ upper_matrix)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_tangent_operator_structure(dtype, getkey):
    # A tangent operator inherits its primal's structure -- extraction returns the
    # *tangent's* entries (consistent with `TangentLinearOperator.as_matrix`), by
    # differentiating whatever fast path the primal has.
    size = 5

    primal_diag = jr.normal(getkey(), (size,), dtype=dtype)
    tangent_diag = jr.normal(getkey(), (size,), dtype=dtype)
    tangent_op = TangentLinearOperator(
        lx.DiagonalLinearOperator(primal_diag), lx.DiagonalLinearOperator(tangent_diag)
    )
    assert lx.is_diagonal(tangent_op)
    assert jnp.allclose(lx.diagonal(tangent_op), tangent_diag)
    assert jnp.allclose(lx.trace(tangent_op), jnp.sum(tangent_diag))

    primal_col = jr.normal(getkey(), (size,), dtype=dtype)
    tangent_col = jr.normal(getkey(), (size,), dtype=dtype)
    tangent_op = TangentLinearOperator(
        lx.CirculantLinearOperator(primal_col), lx.CirculantLinearOperator(tangent_col)
    )
    assert lx.is_circulant(tangent_op)
    assert jnp.allclose(lx.first_column(tangent_op), tangent_col)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_tangent_operator_tridiagonal_probes(dtype, getkey):
    # An opaque tridiagonal-tagged primal extracts by colouring; the tangent's bands
    # come from differentiating that probe, with no materialisation. The tangent
    # operator is built the way lineax itself builds them: as the jvp of the operator's
    # construction, so the primal and tangent share their static (closure) structure.
    size = 5
    band = lambda m: jnp.tril(jnp.triu(m, -1), 1)
    primal_matrix = band(jr.normal(getkey(), (size, size), dtype=dtype))
    tangent_matrix = band(jr.normal(getkey(), (size, size), dtype=dtype))
    in_struct = jax.ShapeDtypeStruct((size,), dtype)

    def make(m):
        return lx.TaggedLinearOperator(
            lx.FunctionLinearOperator(lambda x: m @ x, in_struct), lx.tridiagonal_tag
        )

    primal_op, t_op = eqx.filter_jvp(make, (primal_matrix,), (tangent_matrix,))
    tangent_op = TangentLinearOperator(primal_op, t_op)
    assert lx.is_tridiagonal(tangent_op)
    main, lower, upper = lx.tridiagonal(tangent_op)
    assert jnp.allclose(main, jnp.diag(tangent_matrix))
    assert jnp.allclose(lower, jnp.diag(tangent_matrix, -1))
    assert jnp.allclose(upper, jnp.diag(tangent_matrix, 1))


def test_tangent_operator_unit_diagonal_is_zero(getkey):
    # The tangent of a unit-diagonal family has a *zero* diagonal, so the tag must not
    # be inherited; nor is semidefiniteness of either sign.
    size = 4
    in_struct = jax.ShapeDtypeStruct((size,), jnp.float64)

    def make(scale):
        return lx.TaggedLinearOperator(
            lx.FunctionLinearOperator(lambda x: scale * x, in_struct),
            (lx.diagonal_tag, lx.unit_diagonal_tag, lx.positive_semidefinite_tag),
        )

    primal_op, t_op = eqx.filter_jvp(make, (jnp.array(1.0),), (jnp.array(0.5),))
    tangent_op = TangentLinearOperator(primal_op, t_op)
    assert lx.is_diagonal(tangent_op)
    assert not lx.has_unit_diagonal(tangent_op)
    assert not lx.is_positive_semidefinite(tangent_op)
    assert not lx.is_negative_semidefinite(tangent_op)
    # The unit-diagonal tag means the primal's diagonal is *constant* ones whatever
    # `scale` is, so the honest tangent diagonal here is zero, not 0.5: the tag wins
    # over the arithmetic, exactly as it does for the primal.
    assert jnp.allclose(lx.diagonal(tangent_op), 0.0)


def test_tangent_operator_max_rank(getkey):
    # A rank bound doubles rather than transfers: writing the family as
    # `A(t) = U(t) V(t)^T` with rank <= k, the tangent `dU V^T + U dV^T` has rank up
    # to `2k` -- still capped by the dimension bound.
    def make(m, r):
        return lx.MatrixLinearOperator(m, lx.MaxRankTag(r))

    matrix = jr.normal(getkey(), (6, 6))
    t_matrix = jr.normal(getkey(), (6, 6))
    tangent_op = TangentLinearOperator(make(matrix, 1), make(t_matrix, 1))
    assert lx.max_rank(tangent_op) == 2
    tangent_op = TangentLinearOperator(make(matrix, 4), make(t_matrix, 4))
    assert lx.max_rank(tangent_op) == 6


def test_is_materialised_recurses_through_wrappers(getkey):
    # A flat isinstance check is not enough here: `fn_op + fn_op` is an
    # `AddLinearOperator`, whose `as_matrix` still costs one `mv` per column of each
    # operand. The predicate has to recurse.
    matrix = jr.normal(getkey(), (3, 3))
    mat_op = lx.MatrixLinearOperator(matrix)
    in_struct = jax.ShapeDtypeStruct((3,), matrix.dtype)
    fn_op = lx.FunctionLinearOperator(lambda x: matrix @ x, in_struct)
    assert is_materialised(mat_op)
    assert not is_materialised(fn_op)
    assert is_materialised(lx.TaggedLinearOperator(mat_op, ()))
    assert not is_materialised(lx.TaggedLinearOperator(fn_op, ()))
    assert is_materialised(mat_op + mat_op)
    assert not is_materialised(fn_op + fn_op)
    assert not is_materialised(mat_op + fn_op)
    assert is_materialised(-mat_op)
    assert is_materialised(2.0 * mat_op)
    assert not is_materialised(fn_op / 2.0)
    # A composition's matrix is computed on demand, never stored.
    assert not is_materialised(mat_op @ mat_op)


@pytest.mark.parametrize("make_operator", make_operators)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_trace(make_operator, dtype, getkey):
    matrix, tags = _square_matrix_and_tags(make_operator, getkey, dtype)
    if make_operator is make_jacrev_operator and dtype is jnp.complex128:
        # JacobianLinearOperator does not support complex dtypes when jac="bwd"
        return
    operator = make_operator(getkey, matrix, tags)
    assert jnp.allclose(lx.trace(operator), jnp.trace(matrix))
    assert jnp.allclose(lx.trace(operator), jnp.sum(lx.diagonal(operator)))


def test_trace_is_not_singledispatch():
    # `trace` is documented as `jnp.sum(diagonal(operator))` and nothing more, with
    # all fast paths belonging in `diagonal` -- so it should stay a plain function,
    # not a `functools.singledispatch` one (which would expose a `.register` method).
    assert not hasattr(lx.trace, "register")


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
def test_is_hermitian(dtype, getkey):
    matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
    hermitian_operators = _setup(getkey, matrix + matrix.conj().T, lx.hermitian_tag)
    for operator in hermitian_operators:
        assert lx.is_hermitian(operator)

    not_hermitian_operators = _setup(getkey, matrix)
    _assert_except_diag(lx.is_hermitian, not_hermitian_operators, flip_cond=True)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_is_hermitian_implications(dtype, getkey):
    matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
    real = jnp.issubdtype(dtype, jnp.floating)

    # PSD/NSD are Hermitian for both real and complex dtypes.
    psd = matrix @ matrix.conj().T
    assert lx.is_hermitian(lx.MatrixLinearOperator(psd, lx.positive_semidefinite_tag))
    assert lx.is_hermitian(lx.MatrixLinearOperator(-psd, lx.negative_semidefinite_tag))

    # Symmetric (A = Aᵀ) and diagonal operators are Hermitian iff real-valued.
    sym = lx.MatrixLinearOperator(matrix + matrix.T, lx.symmetric_tag)
    assert lx.is_hermitian(sym) == real
    assert lx.is_hermitian(lx.DiagonalLinearOperator(jnp.diag(matrix))) == real

    # Conversely a Hermitian operator is symmetric iff real-valued.
    herm = lx.MatrixLinearOperator(matrix + matrix.conj().T, lx.hermitian_tag)
    assert lx.is_hermitian(herm)
    assert lx.is_symmetric(herm) == real


def test_hermitian_tag_propagation(getkey):
    # Hermitian-ness is preserved through transpose and inversion, addition, and real
    # (but not complex) scaling.
    assert lx.hermitian_tag in lx.transpose_tags(frozenset({lx.hermitian_tag}))
    assert lx.hermitian_tag in lx.invert_tags(frozenset({lx.hermitian_tag}))

    def herm_op():
        m = jr.normal(getkey(), (3, 3), dtype=jnp.complex128)
        return lx.MatrixLinearOperator(m + m.conj().T, lx.hermitian_tag)

    op = herm_op()
    assert lx.is_hermitian(op + herm_op())  # sum of Hermitian is Hermitian
    assert lx.is_hermitian(-op)  # negation preserves
    assert lx.is_hermitian(op * 2.0)  # real scaling preserves
    assert not lx.is_hermitian(op * (1.0 + 1j))  # complex scaling does not


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
def test_is_semidefinite(dtype, getkey):
    matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
    not_semidefinite = _setup(getkey, matrix)
    for operator in not_semidefinite:
        assert not lx.is_semidefinite(operator)

    semidefinite = _setup(getkey, matrix.T.conj() @ matrix, lx.semidefinite_tag)
    _assert_except_diag(lx.is_semidefinite, semidefinite, flip_cond=False)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_is_semidefinite_implications(dtype, getkey):
    matrix = jr.normal(getkey(), (3, 3), dtype=dtype)
    psd = matrix @ matrix.conj().T

    # A known sign implies the weaker, sign-agnostic `is_semidefinite`.
    assert lx.is_semidefinite(
        lx.MatrixLinearOperator(psd, lx.positive_semidefinite_tag)
    )
    assert lx.is_semidefinite(
        lx.MatrixLinearOperator(-psd, lx.negative_semidefinite_tag)
    )

    # But not the other way around: the weaker tag doesn't imply a known sign.
    ambiguous = lx.MatrixLinearOperator(psd, lx.semidefinite_tag)
    assert lx.is_semidefinite(ambiguous)
    assert not lx.is_positive_semidefinite(ambiguous)
    assert not lx.is_negative_semidefinite(ambiguous)

    # Semidefinite (of either sign) implies Hermitian, same as PSD/NSD.
    assert lx.is_hermitian(ambiguous)


def test_semidefinite_tag_propagation(getkey):
    # Semidefiniteness is preserved through transpose and inversion, negation, and
    # real (but not complex) scaling -- including scaling by a value whose sign isn't
    # known statically.
    assert lx.semidefinite_tag in lx.transpose_tags(frozenset({lx.semidefinite_tag}))
    assert lx.semidefinite_tag in lx.invert_tags(frozenset({lx.semidefinite_tag}))

    def semidefinite_op():
        m = jr.normal(getkey(), (3, 3), dtype=jnp.complex128)
        psd = m @ m.conj().T
        return lx.MatrixLinearOperator(psd, lx.semidefinite_tag)

    op = semidefinite_op()
    assert lx.is_semidefinite(-op)  # negation preserves
    assert lx.is_semidefinite(op * 2.0)  # real scaling preserves
    assert lx.is_semidefinite(op * -2.0)  # ...regardless of the scalar's sign
    assert not lx.is_semidefinite(op * (1.0 + 1j))  # complex scaling does not

    @jax.jit
    def scale_by_traced(operator, scalar):
        scaled = operator * scalar
        # The specific sign is genuinely unknown at trace time...
        assert not lx.is_positive_semidefinite(scaled)
        assert not lx.is_negative_semidefinite(scaled)
        # ...but it's still recognised as semidefinite.
        return lx.is_semidefinite(scaled)

    assert scale_by_traced(op, jnp.asarray(-3.0))


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


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64, jnp.complex128))
def test_unit_diagonal_dtype_matches_untagged(dtype, getkey):
    # The has_unit_diagonal fast path (`in_dtype`/`in_size`) must agree, dtype
    # included, with the ordinary is_diagonal/materialise route taken when the
    # operator isn't tagged -- not just agree on values.
    size = 4
    in_struct = jax.ShapeDtypeStruct((size,), dtype)
    tagged = lx.FunctionLinearOperator(
        lambda x: x, in_struct, tags=lx.unit_diagonal_tag
    )
    untagged = lx.FunctionLinearOperator(lambda x: x, in_struct)
    fast_path = lx.diagonal(tagged)
    slow_path = lx.diagonal(untagged)
    assert fast_path.dtype == slow_path.dtype
    assert jnp.allclose(fast_path, slow_path)


def test_unit_diagonal_mixed_dtype_structure():
    # The has_unit_diagonal fast path (`in_dtype`/`in_size`) must promote across
    # leaves the same way `IdentityLinearOperator` does, or this blows up under
    # strict dtype promotion. Unlike `test_unit_diagonal_dtype_matches_untagged`, this
    # can't be cross-checked against the untagged form: `materialise` itself builds
    # its basis via a plain (non-"standard"-promoting) `ravel_pytree`, so it has the
    # same failure mode for a genuinely mixed-dtype structure -- a separate,
    # pre-existing limitation, not something introduced by the fast path here.
    in_struct = {
        "a": jax.ShapeDtypeStruct((2,), jnp.float32),
        "b": jax.ShapeDtypeStruct((3,), jnp.float64),
    }
    operator = lx.FunctionLinearOperator(
        lambda x: x, in_struct, tags=lx.unit_diagonal_tag
    )
    assert jnp.allclose(lx.diagonal(operator), jnp.ones(5))
    assert lx.trace(operator) == 5


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
