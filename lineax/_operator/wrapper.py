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

import enum
from collections.abc import Iterable

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω
from jaxtyping import (
    ScalarLike,
)

from .._tags import (
    diagonal_tag,
    hermitian_tag,
    lower_triangular_tag,
    MaxRankTag,
    negative_semidefinite_tag,
    positive_semidefinite_tag,
    symmetric_tag,
    transpose_tags,
    tridiagonal_tag,
    unit_diagonal_tag,
    upper_triangular_tag,
)
from .base import (
    AbstractLinearOperator,
    as_frozenset,
    conj,
    diagonal,
    has_real_dtype,
    has_unit_diagonal,
    is_diagonal,
    is_hermitian,
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


class TaggedLinearOperator(AbstractLinearOperator):
    """Wraps another linear operator and specifies that it has certain tags, e.g.
    representing symmetry.

    !!! Example

        ```python
        # Some other operator.
        operator = lx.MatrixLinearOperator(some_jax_array)

        # Now symmetric! But the type system doesn't know this.
        sym_operator = operator + operator.T
        assert lx.is_symmetric(sym_operator) == False

        # We can declare that our operator has a particular property.
        sym_operator = lx.TaggedLinearOperator(sym_operator, lx.symmetric_tag)
        assert lx.is_symmetric(sym_operator) == True
        ```
    """

    operator: AbstractLinearOperator
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self, operator: AbstractLinearOperator, tags: object | Iterable[object]
    ):
        """**Arguments:**

        - `operator`: some other linear operator to wrap.
        - `tags`: any tags indicating whether this operator has any particular
            properties, like symmetry or positive-definite-ness. Note that these
            properties are unchecked and you may get incorrect values elsewhere if these
            tags are wrong.
        """
        self.operator = operator
        self.tags = as_frozenset(tags)

    def mv(self, vector):
        return self.operator.mv(vector)

    def as_matrix(self):
        return self.operator.as_matrix()

    def transpose(self):
        return TaggedLinearOperator(
            self.operator.transpose(), transpose_tags(self.tags)
        )

    def in_structure(self):
        return self.operator.in_structure()

    def out_structure(self):
        return self.operator.out_structure()


#
# All operators below here are private to lineax.
#


def _is_none(x):
    return x is None


class TangentLinearOperator(AbstractLinearOperator):
    """Internal to lineax. Used to represent the tangent (jvp) computation with
    respect to the linear operator in a linear solve.
    """

    primal: AbstractLinearOperator
    tangent: AbstractLinearOperator

    def __check_init__(self):
        assert type(self.primal) is type(self.tangent)  # noqa: E721

    def mv(self, vector):
        mv = lambda operator: operator.mv(vector)
        out, t_out = eqx.filter_jvp(mv, (self.primal,), (self.tangent,))
        return jtu.tree_map(eqxi.materialise_zeros, out, t_out, is_leaf=_is_none)

    def as_matrix(self):
        as_matrix = lambda operator: operator.as_matrix()
        out, t_out = eqx.filter_jvp(as_matrix, (self.primal,), (self.tangent,))
        return jtu.tree_map(eqxi.materialise_zeros, out, t_out, is_leaf=_is_none)

    def transpose(self):
        transpose = lambda operator: operator.transpose()
        primal_out, tangent_out = eqx.filter_jvp(
            transpose, (self.primal,), (self.tangent,)
        )
        return TangentLinearOperator(primal_out, tangent_out)

    def in_structure(self):
        return self.primal.in_structure()

    def out_structure(self):
        return self.primal.out_structure()


class MulLinearOperator(AbstractLinearOperator):
    """A linear operator formed by multiplying a linear operator by a scalar.

    !!! Example

        ```python
        x = MatrixLinearOperator(...)
        y = 0.5
        assert isinstance(x * y, MulLinearOperator)
        ```
    """

    operator: AbstractLinearOperator
    scalar: ScalarLike

    def mv(self, vector):
        return (self.operator.mv(vector) ** ω * self.scalar).ω

    def as_matrix(self):
        return self.operator.as_matrix() * self.scalar

    def transpose(self):
        return self.operator.transpose() * self.scalar

    def in_structure(self):
        return self.operator.in_structure()

    def out_structure(self):
        return self.operator.out_structure()


# Not just `MulLinearOperator(..., -1)` for compatibility with
# `jax_numpy_dtype_promotion=strict`.
class NegLinearOperator(AbstractLinearOperator):
    """A linear operator formed by computing the negative of a linear operator.

    !!! Example

        ```python
        x = MatrixLinearOperator(...)
        assert isinstance(-x, NegLinearOperator)
        ```
    """

    operator: AbstractLinearOperator

    def mv(self, vector):
        return (-(self.operator.mv(vector) ** ω)).ω

    def as_matrix(self):
        return -self.operator.as_matrix()

    def transpose(self):
        return -self.operator.transpose()

    def in_structure(self):
        return self.operator.in_structure()

    def out_structure(self):
        return self.operator.out_structure()


class DivLinearOperator(AbstractLinearOperator):
    """A linear operator formed by dividing a linear operator by a scalar.

    !!! Example

        ```python
        x = MatrixLinearOperator(...)
        y = 0.5
        assert isinstance(x / y, DivLinearOperator)
        ```
    """

    operator: AbstractLinearOperator
    scalar: ScalarLike

    def mv(self, vector):
        with jax.numpy_dtype_promotion("standard"):
            return (self.operator.mv(vector) ** ω / self.scalar).ω

    def as_matrix(self):
        return self.operator.as_matrix() / self.scalar

    def transpose(self):
        return self.operator.transpose() / self.scalar

    def in_structure(self):
        return self.operator.in_structure()

    def out_structure(self):
        return self.operator.out_structure()


@linearise.register(TaggedLinearOperator)
def _(operator):
    return TaggedLinearOperator(linearise(operator.operator), operator.tags)


@materialise.register(TaggedLinearOperator)
def _(operator):
    return TaggedLinearOperator(materialise(operator.operator), operator.tags)


@diagonal.register(TaggedLinearOperator)
def _(operator):
    return diagonal(operator.operator)


@tridiagonal.register(TaggedLinearOperator)
def _(operator):
    return tridiagonal(operator.operator)


for transform in (linearise, materialise, diagonal):

    @transform.register(MulLinearOperator)
    def _(operator, transform=transform):
        return transform(operator.operator) * operator.scalar

    @transform.register(NegLinearOperator)  # pyright: ignore
    def _(operator, transform=transform):
        return -transform(operator.operator)

    @transform.register(DivLinearOperator)
    def _(operator, transform=transform):
        return transform(operator.operator) / operator.scalar


@linearise.register(TangentLinearOperator)
def _(operator):
    primal_out, tangent_out = eqx.filter_jvp(
        linearise, (operator.primal,), (operator.tangent,)
    )
    return TangentLinearOperator(primal_out, tangent_out)


@materialise.register(TangentLinearOperator)
def _(operator):
    primal_out, tangent_out = eqx.filter_jvp(
        materialise, (operator.primal,), (operator.tangent,)
    )
    return TangentLinearOperator(primal_out, tangent_out)


@diagonal.register(TangentLinearOperator)
def _(operator):
    # Should be unreachable: TangentLinearOperator is used for a narrow set of
    # operations only (mv; transpose) inside the JVP rule linear_solve_p.
    raise NotImplementedError(
        "Please open a GitHub issue: https://github.com/google/lineax"
    )


@tridiagonal.register(TangentLinearOperator)
def _(operator):
    # Should be unreachable: TangentLinearOperator is used for a narrow set of
    # operations only (mv; transpose) inside the JVP rule linear_solve_p.
    raise NotImplementedError(
        "Please open a GitHub issue: https://github.com/google/lineax"
    )


@tridiagonal.register(MulLinearOperator)
def _(operator):
    (diag, lower, upper) = tridiagonal(operator.operator)
    return (diag * operator.scalar, lower * operator.scalar, upper * operator.scalar)


@tridiagonal.register(NegLinearOperator)
def _(operator):
    (diag, lower, upper) = tridiagonal(operator.operator)
    return (-diag, -lower, -upper)


@tridiagonal.register(DivLinearOperator)
def _(operator):
    (diag, lower, upper) = tridiagonal(operator.operator)
    return (diag / operator.scalar, lower / operator.scalar, upper / operator.scalar)


for check in (
    is_symmetric,
    is_hermitian,
    is_diagonal,
    has_unit_diagonal,
    is_lower_triangular,
    is_upper_triangular,
    is_tridiagonal,
    is_positive_semidefinite,
    is_negative_semidefinite,
    max_rank,
):

    @check.register(TangentLinearOperator)  # pyright: ignore
    def _(operator, check=check):
        return check(operator.primal)


# Scaling/negating preserves these structural properties
for check in (
    is_symmetric,
    is_diagonal,
    is_lower_triangular,
    is_upper_triangular,
    is_tridiagonal,
):

    @check.register(MulLinearOperator)
    @check.register(NegLinearOperator)
    @check.register(DivLinearOperator)
    def _(operator, check=check):
        return check(operator.operator)


def _scalar_is_real(scalar) -> bool:
    """Whether a scalar is statically known to be real-valued.

    A real dtype guarantees a real value, so this is known even for JAX tracers (whose
    runtime value is unknown at trace time): only the dtype matters, not the value.
    Returns `False` only for genuinely complex-typed scalars.
    """
    return not jnp.issubdtype(jnp.result_type(scalar), jnp.complexfloating)


# Hermitian-ness preserved by negation and scaling by any real scalar
@is_hermitian.register(NegLinearOperator)
def _(operator):
    return is_hermitian(operator.operator)


@is_hermitian.register(MulLinearOperator)
@is_hermitian.register(DivLinearOperator)
def _(operator):
    return _scalar_is_real(operator.scalar) and is_hermitian(operator.operator)


# has_unit_diagonal is NOT preserved by negation
@has_unit_diagonal.register(NegLinearOperator)
def _(operator):
    return False


# has_unit_diagonal is preserved by scaling/dividing only when scalar == 1
@has_unit_diagonal.register(MulLinearOperator)
@has_unit_diagonal.register(DivLinearOperator)
def _(operator):
    scalar = operator.scalar
    if not isinstance(scalar, (int, float, np.ndarray, np.generic)):
        return False
    return float(scalar) == 1.0 and has_unit_diagonal(operator.operator)


class _ScalarSign(enum.Enum):
    positive = enum.auto()
    negative = enum.auto()
    zero = enum.auto()
    unknown = enum.auto()


def _scalar_sign(scalar) -> _ScalarSign:
    """Returns the sign of a scalar, or unknown for JAX tracers."""
    if isinstance(scalar, (int, float, np.ndarray, np.generic)):
        scalar = float(scalar)
        if scalar > 0:
            return _ScalarSign.positive
        elif scalar < 0:
            return _ScalarSign.negative
        else:
            return _ScalarSign.zero
    else:
        return _ScalarSign.unknown


# PSD/NSD for MulLinearOperator: depends on sign of scalar
# Zero scalar gives zero matrix which is both PSD and NSD
@is_positive_semidefinite.register(MulLinearOperator)
def _(operator):
    sign = _scalar_sign(operator.scalar)
    if sign is _ScalarSign.positive:
        return is_positive_semidefinite(operator.operator)
    elif sign is _ScalarSign.negative:
        return is_negative_semidefinite(operator.operator)
    elif sign is _ScalarSign.zero:
        return True  # zero matrix is PSD
    return False


@is_negative_semidefinite.register(MulLinearOperator)
def _(operator):
    sign = _scalar_sign(operator.scalar)
    if sign is _ScalarSign.positive:
        return is_negative_semidefinite(operator.operator)
    elif sign is _ScalarSign.negative:
        return is_positive_semidefinite(operator.operator)
    elif sign is _ScalarSign.zero:
        return True  # zero matrix is NSD
    return False


# PSD/NSD for DivLinearOperator: depends on sign of scalar
# Zero scalar is division by zero - return False (conservative)
@is_positive_semidefinite.register(DivLinearOperator)
def _(operator):
    sign = _scalar_sign(operator.scalar)
    if sign is _ScalarSign.positive:
        return is_positive_semidefinite(operator.operator)
    elif sign is _ScalarSign.negative:
        return is_negative_semidefinite(operator.operator)
    return False


@is_negative_semidefinite.register(DivLinearOperator)
def _(operator):
    sign = _scalar_sign(operator.scalar)
    if sign is _ScalarSign.positive:
        return is_negative_semidefinite(operator.operator)
    elif sign is _ScalarSign.negative:
        return is_positive_semidefinite(operator.operator)
    return False


# PSD/NSD for NegLinearOperator: negation swaps PSD <-> NSD
@is_positive_semidefinite.register(NegLinearOperator)
def _(operator):
    return is_negative_semidefinite(operator.operator)


@is_negative_semidefinite.register(NegLinearOperator)
def _(operator):
    return is_positive_semidefinite(operator.operator)


# Multiplying an operator by a scalar  preserves its rank
# unless the scalar is statically known to be zero
@max_rank.register(MulLinearOperator)
def _(operator):
    if _scalar_sign(operator.scalar) is _ScalarSign.zero:
        return 0
    return max_rank(operator.operator)


@max_rank.register(DivLinearOperator)
@max_rank.register(NegLinearOperator)
def _(operator):
    return max_rank(operator.operator)


for check, tag in (
    (is_symmetric, symmetric_tag),
    (is_diagonal, diagonal_tag),
    (has_unit_diagonal, unit_diagonal_tag),
    (is_lower_triangular, lower_triangular_tag),
    (is_upper_triangular, upper_triangular_tag),
    (is_positive_semidefinite, positive_semidefinite_tag),
    (is_negative_semidefinite, negative_semidefinite_tag),
    (is_tridiagonal, tridiagonal_tag),
):

    @check.register(TaggedLinearOperator)
    def _(operator, check=check, tag=tag):
        return (tag in operator.tags) or check(operator.operator)


# `is_hermitian` is special-cased rather than handled by the loop above: a tag other
# than `hermitian_tag` can still imply Hermitian-ness. PSD/NSD operators are Hermitian
# (real or complex), and real symmetric/diagonal operators are Hermitian too. This
# mirrors the cross-implications encoded for the core operators.
@is_hermitian.register(TaggedLinearOperator)
def _(operator):
    tags = operator.tags
    if is_hermitian(operator.operator):
        return True
    if (
        hermitian_tag in tags
        or positive_semidefinite_tag in tags
        or negative_semidefinite_tag in tags
    ):
        return True
    if symmetric_tag in tags or diagonal_tag in tags:
        return has_real_dtype(operator)
    return False


@max_rank.register(TaggedLinearOperator)
def _(operator):
    inner = max_rank(operator.operator)
    bounds = [t.r for t in operator.tags if isinstance(t, MaxRankTag)]
    return min(min(bounds), inner) if bounds else inner


# conj


@conj.register(TaggedLinearOperator)
def _(operator):
    return TaggedLinearOperator(conj(operator.operator), operator.tags)


@conj.register(TangentLinearOperator)
def _(operator):
    c = lambda operator: conj(operator)
    primal_out, tangent_out = eqx.filter_jvp(c, (operator.primal,), (operator.tangent,))
    return TangentLinearOperator(primal_out, tangent_out)


def _scalar_conj(scalar):
    # Preserve Python scalar types so that weak-typed Python ints/floats
    # don't get promoted to numpy generics (which equinox treats as arrays
    # and thus become strong-typed JAX tracers under strict dtype promotion).
    if isinstance(scalar, (int, float)):
        return scalar
    if isinstance(scalar, complex):
        return scalar.conjugate()
    if isinstance(scalar, (np.ndarray, np.generic)):
        return np.conj(scalar)
    return jnp.conj(scalar)


@conj.register(MulLinearOperator)
def _(operator):
    return conj(operator.operator) * _scalar_conj(operator.scalar)


@conj.register(NegLinearOperator)
def _(operator):
    return -conj(operator.operator)


@conj.register(DivLinearOperator)
def _(operator):
    return conj(operator.operator) / _scalar_conj(operator.scalar)
