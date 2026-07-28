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

import math
import warnings

import equinox as eqx
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω
from jaxtyping import (
    Array,
    ArrayLike,
    Inexact,
    PyTree,  # pyright: ignore
)

from .._custom_types import sentinel
from .._misc import (
    default_floating_dtype,
    inexact_asarray,
    strip_weak_dtype,
)
from .._tags import (
    diagonal_tag,
    lower_triangular_tag,
    negative_semidefinite_tag,
    positive_semidefinite_tag,
    symmetric_tag,
    tridiagonal_tag,
    unit_diagonal_tag,
    upper_triangular_tag,
    transpose_tags,
)
from .base import (
    AbstractLinearOperator,
    as_frozenset,
    circulant_column,
    conj,
    diagonal,
    FlatPyTree,
    has_unit_diagonal,
    inexact_structure,
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
    tridiagonal,
)


def _has_real_dtype(operator) -> bool:
    """Check if all dtypes in an operator's structure are real (not complex)."""
    leaves = jtu.tree_leaves((operator.in_structure(), operator.out_structure()))
    dtype = jnp.result_type(*leaves)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return False
    elif jnp.issubdtype(dtype, jnp.floating):
        return True
    else:
        assert False, (
            "Only `jnp.floating` and `jnp.complexfloating` dtypes are understood."
        )


# `structure` must be static as with `JacobianLinearOperator`
class IdentityLinearOperator(AbstractLinearOperator):
    """Represents the identity transformation `X -> X`, where each `x in X` is some
    PyTree of floating-point JAX arrays.
    """

    input_structure: FlatPyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    output_structure: FlatPyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)

    def __init__(
        self,
        input_structure: PyTree[jax.ShapeDtypeStruct],
        output_structure: PyTree[jax.ShapeDtypeStruct] = sentinel,
    ):
        """**Arguments:**

        - `input_structure`: A PyTree of `jax.ShapeDtypeStruct`s specifying the
            structure of the the input space. (When later calling `self.mv(x)`
            then this should match the structure of `x`, i.e.
            `jax.eval_shape(lambda: x)`.)
        - `output_structure`: A PyTree of `jax.ShapeDtypeStruct`s specifying the
            structure of the the output space. If not passed then this defaults to the
            same as `input_structure`. If passed then it must have the same number of
            elements as `input_structure`, so that the operator is square.
        """
        if output_structure is sentinel:
            output_structure = input_structure
        input_structure = inexact_structure(input_structure)
        output_structure = inexact_structure(output_structure)
        self.input_structure = jtu.tree_flatten(input_structure)
        self.output_structure = jtu.tree_flatten(output_structure)

    def mv(self, vector):
        if not eqx.tree_equal(
            strip_weak_dtype(jax.eval_shape(lambda: vector)),
            strip_weak_dtype(self.in_structure()),
        ):
            raise ValueError("Vector and operator structures do not match")
        elif self.input_structure == self.output_structure:
            return vector  # fast-path for common special case
        else:
            # TODO(kidger): this could be done slightly more efficiently, by iterating
            #     leaf-by-leaf.
            leaves = jtu.tree_leaves(vector)
            with jax.numpy_dtype_promotion("standard"):
                dtype = jnp.result_type(*leaves)
            vector = jnp.concatenate([x.astype(dtype).reshape(-1) for x in leaves])
            out_size = self.out_size()
            if vector.size < out_size:
                vector = jnp.concatenate(
                    [vector, jnp.zeros(out_size - vector.size, vector.dtype)]
                )
            else:
                vector = vector[:out_size]
            leaves, treedef = jtu.tree_flatten(self.out_structure())
            sizes = np.cumsum([math.prod(x.shape) for x in leaves[:-1]])
            split = jnp.split(vector, sizes)
            assert len(split) == len(leaves)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ignore complex-to-real cast warning
                shaped = [
                    x.reshape(y.shape).astype(y.dtype) for x, y in zip(split, leaves)
                ]
            return jtu.tree_unflatten(treedef, shaped)

    def as_matrix(self):
        leaves = jtu.tree_leaves(self.in_structure())
        with jax.numpy_dtype_promotion("standard"):
            dtype = (
                default_floating_dtype()
                if len(leaves) == 0
                else jnp.result_type(*leaves)
            )
        return jnp.eye(self.out_size(), self.in_size(), dtype=dtype)

    def transpose(self):
        return IdentityLinearOperator(self.out_structure(), self.in_structure())

    def in_structure(self):
        leaves, treedef = self.input_structure
        return jtu.tree_unflatten(treedef, leaves)

    def out_structure(self):
        leaves, treedef = self.output_structure
        return jtu.tree_unflatten(treedef, leaves)

    @property
    def tags(self):
        return frozenset()


class DiagonalLinearOperator(AbstractLinearOperator):
    """A diagonal linear operator, e.g. for a diagonal matrix. Only the diagonal is
    stored (for memory efficiency). Matrix-vector products are computed by doing a
    pointwise diagonal * vector, rather than a full matrix @ vector (for speed).

    The diagonal may also be a PyTree, rather than a 1D array. When materialising the
    matrix, the diagonal is taken to be defined by the flattened PyTree (i.e. values
    show up in the same order.)
    """

    diagonal: PyTree[Inexact[Array, "..."]]

    def __init__(self, diagonal: PyTree[ArrayLike]):
        """**Arguments:**

        - `diagonal`: an array or PyTree defining the diagonal of the matrix.
        """
        self.diagonal = jtu.tree_map(inexact_asarray, diagonal)

    def mv(self, vector):
        return (ω(self.diagonal) * ω(vector)).ω

    def as_matrix(self):
        return jnp.diag(diagonal(self))

    def transpose(self):
        return self

    def in_structure(self):
        return jax.eval_shape(lambda: self.diagonal)

    def out_structure(self):
        return jax.eval_shape(lambda: self.diagonal)


class TridiagonalLinearOperator(AbstractLinearOperator):
    """As [`lineax.MatrixLinearOperator`][], but for specifically a tridiagonal
    matrix.
    """

    diagonal: Inexact[Array, " size"]
    lower_diagonal: Inexact[Array, " size-1"]
    upper_diagonal: Inexact[Array, " size-1"]

    def __init__(
        self,
        diagonal: Inexact[Array, " size"],
        lower_diagonal: Inexact[Array, " size-1"],
        upper_diagonal: Inexact[Array, " size-1"],
    ):
        """**Arguments:**

        - `diagonal`: A rank-one JAX array. This is the diagonal of the matrix.
        - `lower_diagonal`: A rank-one JAX array. This is the lower diagonal of the
            matrix.
        - `upper_diagonal`: A rank-one JAX array. This is the upper diagonal of the
            matrix.

        If `diagonal` has shape `(a,)` then `lower_diagonal` and `upper_diagonal` should
        both have shape `(a - 1,)`.
        """
        self.diagonal = inexact_asarray(diagonal)
        self.lower_diagonal = inexact_asarray(lower_diagonal)
        self.upper_diagonal = inexact_asarray(upper_diagonal)
        (size,) = self.diagonal.shape
        if self.lower_diagonal.shape != (size - 1,):
            raise ValueError("lower_diagonal and diagonal do not have consistent size")
        if self.upper_diagonal.shape != (size - 1,):
            raise ValueError("upper_diagonal and diagonal do not have consistent size")

    def mv(self, vector):
        a = self.upper_diagonal * vector[1:]
        b = self.diagonal * vector
        c = self.lower_diagonal * vector[:-1]
        return b.at[:-1].add(a).at[1:].add(c)

    def as_matrix(self):
        (size,) = jnp.shape(self.diagonal)
        matrix = jnp.zeros((size, size), self.diagonal.dtype)
        arange = np.arange(size)
        matrix = matrix.at[arange, arange].set(self.diagonal)
        matrix = matrix.at[arange[1:], arange[:-1]].set(self.lower_diagonal)
        matrix = matrix.at[arange[:-1], arange[1:]].set(self.upper_diagonal)
        return matrix

    def transpose(self):
        return TridiagonalLinearOperator(
            self.diagonal, self.upper_diagonal, self.lower_diagonal
        )

    def in_structure(self):
        (size,) = jnp.shape(self.diagonal)
        return jax.ShapeDtypeStruct(shape=(size,), dtype=self.diagonal.dtype)

    def out_structure(self):
        (size,) = jnp.shape(self.diagonal)
        return jax.ShapeDtypeStruct(shape=(size,), dtype=self.diagonal.dtype)


class CirculantLinearOperator(AbstractLinearOperator):
    column: Inexact[Array, " size"]
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        column: Inexact[Array, " size"],
        tags: object | frozenset[object] = (),
    ):
        self.column = inexact_asarray(column)
        if self.column.ndim != 1:
            raise ValueError("Circulant must have exactly 1 dimension.")
        self.tags = as_frozenset(tags)

    def mv(self, vector):
        if jnp.issubdtype(self.column.dtype, jnp.complexfloating):
            freq_circulant = jnp.fft.fft(self.column)
            freq_vector = jnp.fft.fft(vector)
            result = jnp.fft.ifft(freq_circulant * freq_vector)
        else:
            freq_circulant = jnp.fft.rfft(self.column)
            freq_vector = jnp.fft.rfft(vector)
            (size,) = self.column.shape
            result = jnp.fft.irfft(freq_circulant * freq_vector, n=size)
        return result

    def transpose(self):
        return CirculantLinearOperator(
            jnp.concatenate([self.column[:1], self.column[1:][::-1]]),
            transpose_tags(self.tags),
        )

    def as_matrix(self):
        (size,) = jnp.shape(self.column)
        # static indices, use numpy
        i, j = np.ogrid[:size, :size]
        return self.column[(i - j) % size]

    def in_structure(self):
        (size,) = jnp.shape(self.column)
        return jax.ShapeDtypeStruct(shape=(size,), dtype=self.column.dtype)

    def out_structure(self):
        return self.in_structure()


for transform in (linearise, materialise):

    @transform.register(IdentityLinearOperator)
    @transform.register(DiagonalLinearOperator)
    @transform.register(TridiagonalLinearOperator)
    @transform.register(CirculantLinearOperator)
    def _(operator):
        return operator


@diagonal.register(IdentityLinearOperator)
def _(operator):
    return jnp.ones(operator.in_size())


@diagonal.register(DiagonalLinearOperator)
def _(operator):
    diagonal, _ = jfu.ravel_pytree(operator.diagonal)
    return diagonal


@diagonal.register(TridiagonalLinearOperator)
def _(operator):
    return operator.diagonal


@diagonal.register(CirculantLinearOperator)
def _(operator):
    return jnp.full_like(operator.column, operator.column[0])


@tridiagonal.register(IdentityLinearOperator)
def _(operator):
    size = operator.in_size()
    main_diagonal = jnp.ones(size)
    off_diagonal = jnp.zeros(size - 1)
    return main_diagonal, off_diagonal, off_diagonal


@tridiagonal.register(DiagonalLinearOperator)
def _(operator):
    diag = diagonal(operator)
    upper_diag = jnp.zeros(diag.size - 1)
    lower_diag = jnp.zeros(diag.size - 1)
    return diag, lower_diag, upper_diag


@tridiagonal.register(TridiagonalLinearOperator)
def _(operator):
    return operator.diagonal, operator.lower_diagonal, operator.upper_diagonal


@tridiagonal.register(CirculantLinearOperator)
def _(operator):
    diag = diagonal(operator)
    if diag.size == 1:
        upper_diag = jnp.zeros(0, dtype=diag.dtype)
        lower_diag = jnp.zeros(0, dtype=diag.dtype)
    else:
        upper_diag = jnp.full(diag.size - 1, operator.column[-1], dtype=diag.dtype)
        lower_diag = jnp.full(diag.size - 1, operator.column[1], dtype=diag.dtype)
    return diag, lower_diag, upper_diag


@circulant_column.register(IdentityLinearOperator)
def _(operator):
    size = operator.in_size()
    dtype = jtu.tree_leaves(operator.in_structure())[0].dtype
    return jnp.zeros(size, dtype).at[0].set(1)


@circulant_column.register(CirculantLinearOperator)
def _(operator):
    return operator.column


@is_symmetric.register(IdentityLinearOperator)
def _(operator):
    return eqx.tree_equal(operator.in_structure(), operator.out_structure()) is True


@is_symmetric.register(DiagonalLinearOperator)
def _(operator):
    return True


@is_symmetric.register(TridiagonalLinearOperator)
def _(operator):
    return False


@is_symmetric.register(CirculantLinearOperator)
def _(operator):
    # Symmetric (A = A^T) if explicitly tagged symmetric or diagonal
    if symmetric_tag in operator.tags or diagonal_tag in operator.tags:
        return True
    # PSD/NSD implies symmetric only for real dtypes; for complex, it's Hermitian
    if (
        positive_semidefinite_tag in operator.tags
        or negative_semidefinite_tag in operator.tags
    ):
        return _has_real_dtype(operator)
    return False


@is_diagonal.register(IdentityLinearOperator)
@is_diagonal.register(DiagonalLinearOperator)
def _(operator):
    return True


@is_diagonal.register(TridiagonalLinearOperator)
def _(operator):
    return operator.in_size() == 1


@is_diagonal.register(CirculantLinearOperator)
def _(operator):
    return diagonal_tag in operator.tags or (operator.in_size() == 1)


for check in (is_lower_triangular, is_upper_triangular):

    @check.register(IdentityLinearOperator)
    @check.register(DiagonalLinearOperator)
    def _(operator):
        return True

    @check.register(TridiagonalLinearOperator)  # pyright: ignore
    def _(operator):
        return False


for check, tag in (
    (has_unit_diagonal, unit_diagonal_tag),
    (is_lower_triangular, lower_triangular_tag),
    (is_upper_triangular, upper_triangular_tag),
    (is_positive_semidefinite, positive_semidefinite_tag),
    (is_negative_semidefinite, negative_semidefinite_tag),
):

    @check.register(CirculantLinearOperator)  # pyright: ignore
    def _(operator, tag=tag):
        return tag in operator.tags


@is_tridiagonal.register(IdentityLinearOperator)
@is_tridiagonal.register(DiagonalLinearOperator)
@is_tridiagonal.register(TridiagonalLinearOperator)
def _(operator):
    return True


@is_tridiagonal.register(CirculantLinearOperator)
def _(operator):
    return (
        operator.in_size() < 3
        or tridiagonal_tag in operator.tags
        or diagonal_tag in operator.tags
    )


@is_circulant.register(IdentityLinearOperator)
def _(operator):
    # A non-square `IdentityLinearOperator` is not circulant.
    return eqx.tree_equal(operator.in_structure(), operator.out_structure()) is True


@is_circulant.register(CirculantLinearOperator)
def _(operator):
    return True


# A diagonal matrix is circulant iff every diagonal entry is equal, and a tridiagonal
# matrix is circulant only for sizes below three (larger ones need the wrap-around
# corners). Neither can be checked at trace time, so we conservatively report circulance
# only in the size-one case, where it holds unconditionally.
@is_circulant.register(DiagonalLinearOperator)
@is_circulant.register(TridiagonalLinearOperator)
def _(operator):
    return operator.in_size() == 1


@has_unit_diagonal.register(IdentityLinearOperator)
def _(operator):
    return True


@is_positive_semidefinite.register(IdentityLinearOperator)
def _(operator):
    return eqx.tree_equal(operator.in_structure(), operator.out_structure()) is True


@is_negative_semidefinite.register(IdentityLinearOperator)
def _(operator):
    return False


# TODO: refine these. For now we conservatively report Diagonal and Tridiagonal
# operators as not having unit diagonal and as not being (semi)definite.
for check in (has_unit_diagonal, is_positive_semidefinite, is_negative_semidefinite):

    @check.register(DiagonalLinearOperator)
    @check.register(TridiagonalLinearOperator)
    def _(operator):
        return False


@conj.register(IdentityLinearOperator)
def _(operator):
    return operator


@conj.register(DiagonalLinearOperator)
def _(operator):
    diagonal_conj = jtu.tree_map(lambda x: x.conj(), operator.diagonal)
    return DiagonalLinearOperator(diagonal_conj)


@conj.register(TridiagonalLinearOperator)
def _(operator):
    return TridiagonalLinearOperator(
        operator.diagonal.conj(),
        operator.lower_diagonal.conj(),
        operator.upper_diagonal.conj(),
    )


@conj.register(CirculantLinearOperator)
def _(operator):
    return CirculantLinearOperator(
        operator.column.conj(),
        operator.tags,
    )
