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


import equinox as eqx
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
from equinox.internal import ω

from .base import (
    AbstractLinearOperator,
    conj,
    diagonal,
    first_column,
    has_unit_diagonal,
    is_circulant,
    is_diagonal,
    is_lower_triangular,
    is_negative_semidefinite,
    is_positive_semidefinite,
    is_symmetric,
    is_tridiagonal,
    is_upper_triangular,
    linearise,
    materialise,
    max_rank,
    tridiagonal,
)
from .core import try_structured_materialise
from .structured import IdentityLinearOperator


class AddLinearOperator(AbstractLinearOperator):
    """A linear operator formed by adding two other linear operators together.

    !!! Example

        ```python
        x = MatrixLinearOperator(...)
        y = MatrixLinearOperator(...)
        assert isinstance(x + y, AddLinearOperator)
        ```
    """

    operator1: AbstractLinearOperator
    operator2: AbstractLinearOperator

    def __check_init__(self):
        if self.operator1.in_structure() != self.operator2.in_structure():
            raise ValueError("Incompatible linear operator structures")
        if self.operator1.out_structure() != self.operator2.out_structure():
            raise ValueError("Incompatible linear operator structures")

    def mv(self, vector):
        maybe_structured_op = try_structured_materialise(self)
        if maybe_structured_op is not self:
            return maybe_structured_op.mv(vector)
        mv1 = self.operator1.mv(vector)
        mv2 = self.operator2.mv(vector)
        return (mv1**ω + mv2**ω).ω

    def as_matrix(self):
        return self.operator1.as_matrix() + self.operator2.as_matrix()

    def transpose(self):
        return self.operator1.transpose() + self.operator2.transpose()

    def in_structure(self):
        return self.operator1.in_structure()

    def out_structure(self):
        return self.operator1.out_structure()


class ComposedLinearOperator(AbstractLinearOperator):
    """A linear operator formed by composing (matrix-multiplying) two other linear
    operators together.

    !!! Example

        ```python
        x = MatrixLinearOperator(matrix1)
        y = MatrixLinearOperator(matrix2)
        composed = x @ y
        assert isinstance(composed, ComposedLinearOperator)
        assert jnp.allclose(composed.as_matrix(), matrix1 @ matrix2)
        ```
    """

    operator1: AbstractLinearOperator
    operator2: AbstractLinearOperator

    def __check_init__(self):
        if self.operator1.in_structure() != self.operator2.out_structure():
            raise ValueError("Incompatible linear operator structures")

    def mv(self, vector):
        maybe_structured_op = try_structured_materialise(self)
        if maybe_structured_op is not self:
            return maybe_structured_op.mv(vector)
        return self.operator1.mv(self.operator2.mv(vector))

    def as_matrix(self):
        if isinstance(self.operator1, IdentityLinearOperator):
            return self.operator2.as_matrix()
        if isinstance(self.operator2, IdentityLinearOperator):
            return self.operator1.as_matrix()
        _, unravel = eqx.filter_eval_shape(
            jfu.ravel_pytree, self.operator1.in_structure()
        )

        def mv_flat(v):
            out = self.operator1.mv(unravel(v))
            return jfu.ravel_pytree(out)[0]

        return jax.vmap(mv_flat, in_axes=1, out_axes=1)(self.operator2.as_matrix())

    def transpose(self):
        return self.operator2.transpose() @ self.operator1.transpose()

    def in_structure(self):
        return self.operator2.in_structure()

    def out_structure(self):
        return self.operator1.out_structure()


for transform in (linearise, diagonal):

    @transform.register(AddLinearOperator)  # pyright: ignore
    def _(operator, transform=transform):
        return transform(operator.operator1) + transform(operator.operator2)  # pyright: ignore


@materialise.register(AddLinearOperator)
def _(operator):
    maybe_structured_op = try_structured_materialise(operator)
    if maybe_structured_op is not operator:
        return maybe_structured_op
    return materialise(operator.operator1) + materialise(operator.operator2)


@tridiagonal.register(AddLinearOperator)
def _(operator):
    (diag1, lower1, upper1) = tridiagonal(operator.operator1)
    (diag2, lower2, upper2) = tridiagonal(operator.operator2)
    return (diag1 + diag2, lower1 + lower2, upper1 + upper2)


@first_column.register(AddLinearOperator)
def _(operator):
    return first_column(operator.operator1) + first_column(operator.operator2)


@linearise.register(ComposedLinearOperator)
def _(operator):
    return linearise(operator.operator1) @ linearise(operator.operator2)


@materialise.register(ComposedLinearOperator)
def _(operator):
    if isinstance(operator.operator1, IdentityLinearOperator):
        return materialise(operator.operator2)
    if isinstance(operator.operator2, IdentityLinearOperator):
        return materialise(operator.operator1)
    maybe_structured_op = try_structured_materialise(operator)
    if maybe_structured_op is not operator:
        return maybe_structured_op
    return materialise(operator.operator1) @ materialise(operator.operator2)


@diagonal.register(ComposedLinearOperator)
def _(operator):
    if is_diagonal(operator.operator1) and is_diagonal(operator.operator2):
        return diagonal(operator.operator1) * diagonal(operator.operator2)
    return jnp.diag(operator.as_matrix())


@tridiagonal.register(ComposedLinearOperator)
def _(operator):
    if is_diagonal(operator.operator1) and is_tridiagonal(operator.operator2):
        d = diagonal(operator.operator1)
        main, lower, upper = tridiagonal(operator.operator2)
        # D @ T scales rows: row i multiplied by d[i]
        return d * main, d[1:] * lower, d[:-1] * upper
    if is_diagonal(operator.operator2) and is_tridiagonal(operator.operator1):
        d = diagonal(operator.operator2)
        main, lower, upper = tridiagonal(operator.operator1)
        # T @ D scales columns: column j multiplied by d[j]
        return d * main, d[:-1] * lower, d[1:] * upper
    matrix = operator.as_matrix()
    assert matrix.ndim == 2
    main_diagonal = jnp.diagonal(matrix, offset=0)
    upper_diagonal = jnp.diagonal(matrix, offset=1)
    lower_diagonal = jnp.diagonal(matrix, offset=-1)
    return main_diagonal, lower_diagonal, upper_diagonal


@first_column.register(ComposedLinearOperator)
def _(operator):
    # The first column of `A @ B` is `A @ (B e_0)`.
    _, unravel = eqx.filter_eval_shape(
        jfu.ravel_pytree, operator.operator1.in_structure()
    )
    column = first_column(operator.operator2)
    out, _ = jfu.ravel_pytree(operator.operator1.mv(unravel(column)))
    return out


for check in (
    is_symmetric,
    is_diagonal,
    is_lower_triangular,
    is_upper_triangular,
    is_positive_semidefinite,
    is_negative_semidefinite,
    is_tridiagonal,
    is_circulant,
):

    @check.register(AddLinearOperator)
    def _(operator, check=check):
        return check(operator.operator1) and check(operator.operator2)


@has_unit_diagonal.register(AddLinearOperator)
def _(operator):
    return False


@max_rank.register(AddLinearOperator)
def _(operator):
    return min(
        max_rank(operator.operator1) + max_rank(operator.operator2),
        min(operator.out_size(), operator.in_size()),
    )


# These properties ARE preserved under composition.
for check in (
    is_diagonal,
    is_lower_triangular,
    is_upper_triangular,
    is_circulant,
):

    @check.register(ComposedLinearOperator)
    def _(operator, check=check):
        return check(operator.operator1) and check(operator.operator2)


# is_symmetric: A@B is symmetric only if A and B commute. Diagonal matrices commute.
# The structure check is on the composition itself, not on its operands: composing two
# operators that each map between differently-laid-out (but equal-sized) structures can
# still land back where it started, and the result is then genuinely symmetric.
@is_symmetric.register(ComposedLinearOperator)
def _(operator):
    if eqx.tree_equal(operator.in_structure(), operator.out_structure()) is not True:
        return False
    return is_diagonal(operator.operator1) and is_diagonal(operator.operator2)


# is_tridiagonal: tridiagonal @ tridiagonal = pentadiagonal, but
# tridiagonal @ diagonal = tridiagonal and diagonal @ tridiagonal = tridiagonal
@is_tridiagonal.register(ComposedLinearOperator)
def _(operator):
    if is_diagonal(operator.operator1):
        return is_tridiagonal(operator.operator2)
    if is_diagonal(operator.operator2):
        return is_tridiagonal(operator.operator1)
    return False


# PSD/NSD: not preserved under composition in general.
@is_positive_semidefinite.register(ComposedLinearOperator)
@is_negative_semidefinite.register(ComposedLinearOperator)
def _(operator):
    return False


@has_unit_diagonal.register(ComposedLinearOperator)
def _(operator):
    a = is_diagonal(operator)
    b = is_lower_triangular(operator)
    c = is_upper_triangular(operator)
    d = has_unit_diagonal(operator.operator1)
    e = has_unit_diagonal(operator.operator2)
    return (a or b or c) and d and e


@max_rank.register(ComposedLinearOperator)
def _(operator):
    return min(max_rank(operator.operator1), max_rank(operator.operator2))


@conj.register(AddLinearOperator)
def _(operator):
    return conj(operator.operator1) + conj(operator.operator2)


@conj.register(ComposedLinearOperator)
def _(operator):
    return conj(operator.operator1) @ conj(operator.operator2)
