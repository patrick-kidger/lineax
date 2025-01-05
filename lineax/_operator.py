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

import abc
import functools as ft
import math
import warnings
from collections.abc import Callable
from typing import Any, Iterable, Literal, NoReturn, Optional, TypeVar, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.flatten_util as jfu
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω
from jaxtyping import (
    Array,
    ArrayLike,
    Inexact,
    PyTree,  # pyright: ignore
    Scalar,
    Shaped,
)

from ._custom_types import sentinel
from ._misc import (
    default_floating_dtype,
    inexact_asarray,
    jacobian,
    NoneAux,
    strip_weak_dtype,
)
from ._tags import (
    diagonal_tag,
    lower_triangular_tag,
    negative_semidefinite_tag,
    positive_semidefinite_tag,
    symmetric_tag,
    transpose_tags,
    tridiagonal_tag,
    unit_diagonal_tag,
    upper_triangular_tag,
)


def _frozenset(x: Union[object, Iterable[object]]) -> frozenset[object]:
    try:
        iter_x = iter(x)  # pyright: ignore
    except TypeError:
        return frozenset([x])
    else:
        return frozenset(iter_x)


class AbstractLinearOperator(eqx.Module, strict=True):
    """Abstract base class for all linear operators.

    Linear operators can act between PyTrees. Each `AbstractLinearOperator` is thought
    of as a linear function `X -> Y`, where each element of `X` is as PyTree of
    floating-point JAX arrays, and each element of `Y` is a PyTree of floating-point
    JAX arrays.

    Abstract linear operators support some operations:
    ```python
    op1 + op2  # addition of two operators
    op1 @ op2  # composition of two operators.
    op1 * 3.2  # multiplication by a scalar
    op1 / 3.2  # division by a scalar
    ```
    """

    def __check_init__(self):
        if is_symmetric(self):
            # In particular, we check that dtypes match.
            in_structure = self.in_structure()
            out_structure = self.out_structure()
            # `is` check to handle the possibility of a tracer.
            if eqx.tree_equal(in_structure, out_structure) is not True:
                raise ValueError(
                    "Symmetric matrices must have matching input and output "
                    f"structures. Got input structure {in_structure} and output "
                    f"structure {out_structure}."
                )

    @abc.abstractmethod
    def mv(
        self, vector: PyTree[Inexact[Array, " _b"]]
    ) -> PyTree[Inexact[Array, " _a"]]:
        """Computes a matrix-vector product between this operator and a `vector`.

        **Arguments:**

        - `vector`: Should be some PyTree of floating-point arrays, whose structure
            should match `self.in_structure()`.

        **Returns:**

        A PyTree of floating-point arrays, with structure that matches
        `self.out_structure()`.
        """

    @abc.abstractmethod
    def as_matrix(self) -> Inexact[Array, "a b"]:
        """Materialises this linear operator as a matrix.

        Note that this can be a computationally (time and/or memory) expensive
        operation, as many linear operators are defined implicitly, e.g. in terms of
        their action on a vector.

        **Arguments:** None.

        **Returns:**

        A 2-dimensional floating-point JAX array.
        """

    @abc.abstractmethod
    def transpose(self) -> "AbstractLinearOperator":
        """Transposes this linear operator.

        This can be called as either `operator.T` or `operator.transpose()`.

        **Arguments:** None.

        **Returns:**

        Another [`lineax.AbstractLinearOperator`][].
        """

    @abc.abstractmethod
    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        """Returns the expected input structure of this linear operator.

        **Arguments:** None.

        **Returns:**

        A PyTree of `jax.ShapeDtypeStruct`.
        """

    @abc.abstractmethod
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        """Returns the expected output structure of this linear operator.

        **Arguments:** None.

        **Returns:**

        A PyTree of `jax.ShapeDtypeStruct`.
        """

    def in_size(self) -> int:
        """Returns the total number of scalars in the input of this linear operator.

        That is, the dimensionality of its input space.

        **Arguments:** None.

        **Returns:** An integer.
        """
        leaves = jtu.tree_leaves(self.in_structure())
        return sum(math.prod(leaf.shape) for leaf in leaves)  # pyright: ignore

    def out_size(self) -> int:
        """Returns the total number of scalars in the output of this linear operator.

        That is, the dimensionality of its output space.

        **Arguments:** None.

        **Returns:** An integer.
        """
        leaves = jtu.tree_leaves(self.out_structure())
        return sum(math.prod(leaf.shape) for leaf in leaves)  # pyright: ignore

    @property
    def T(self) -> "AbstractLinearOperator":
        """Equivalent to [`lineax.AbstractLinearOperator.transpose`][]"""
        return self.transpose()

    def __add__(self, other) -> "AbstractLinearOperator":
        if not isinstance(other, AbstractLinearOperator):
            raise ValueError("Can only add AbstractLinearOperators together.")
        return AddLinearOperator(self, other)

    def __sub__(self, other) -> "AbstractLinearOperator":
        if not isinstance(other, AbstractLinearOperator):
            raise ValueError("Can only add AbstractLinearOperators together.")
        return AddLinearOperator(self, -other)

    def __mul__(self, other) -> "AbstractLinearOperator":
        other = jnp.asarray(other)
        if other.shape != ():
            raise ValueError("Can only multiply AbstractLinearOperators by scalars.")
        return MulLinearOperator(self, other)

    def __rmul__(self, other) -> "AbstractLinearOperator":
        return self * other

    def __matmul__(self, other) -> "AbstractLinearOperator":
        if not isinstance(other, AbstractLinearOperator):
            raise ValueError("Can only compose AbstractLinearOperators together.")
        return ComposedLinearOperator(self, other)

    def __truediv__(self, other) -> "AbstractLinearOperator":
        other = jnp.asarray(other)
        if other.shape != ():
            raise ValueError("Can only divide AbstractLinearOperators by scalars.")
        return DivLinearOperator(self, other)

    def __neg__(self) -> "AbstractLinearOperator":
        return NegLinearOperator(self)


class MatrixLinearOperator(AbstractLinearOperator, strict=True):
    """Wraps a 2-dimensional JAX array into a linear operator.

    If the matrix has shape `(a, b)` then matrix-vector multiplication (`self.mv`) is
    defined in the usual way: as performing a matrix-vector that accepts a vector of
    shape `(a,)` and returns a vector of shape `(b,)`.
    """

    matrix: Inexact[Array, "a b"]
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self, matrix: Shaped[Array, "a b"], tags: Union[object, frozenset[object]] = ()
    ):
        """**Arguments:**

        - `matrix`: a two-dimensional JAX array. For an array with shape `(a, b)` then
            this operator can perform matrix-vector products on a vector of shape
            `(b,)` to return a vector of shape `(a,)`.
        - `tags`: any tags indicating whether this matrix has any particular properties,
            like symmetry or positive-definite-ness. Note that these properties are
            unchecked and you may get incorrect values elsewhere if these tags are
            wrong.
        """
        if jnp.ndim(matrix) != 2:
            raise ValueError(
                "`MatrixLinearOperator(matrix=...)` should be 2-dimensional."
            )
        if not jnp.issubdtype(matrix, jnp.inexact):
            matrix = matrix.astype(jnp.float32)
        self.matrix = matrix
        self.tags = _frozenset(tags)

    def mv(self, vector):
        return jnp.matmul(self.matrix, vector, precision=lax.Precision.HIGHEST)

    def as_matrix(self):
        return self.matrix

    def transpose(self):
        if symmetric_tag in self.tags:
            return self
        return MatrixLinearOperator(self.matrix.T, transpose_tags(self.tags))

    def in_structure(self):
        _, in_size = jnp.shape(self.matrix)
        return jax.ShapeDtypeStruct(shape=(in_size,), dtype=self.matrix.dtype)

    def out_structure(self):
        out_size, _ = jnp.shape(self.matrix)
        return jax.ShapeDtypeStruct(shape=(out_size,), dtype=self.matrix.dtype)


def _matmul(matrix: ArrayLike, vector: ArrayLike) -> Array:
    # matrix has structure [leaf(out), leaf(in)]
    # vector has structure [leaf(in)]
    # return has structure [leaf(out)]
    return jnp.tensordot(
        matrix, vector, axes=jnp.ndim(vector), precision=lax.Precision.HIGHEST
    )


def _tree_matmul(matrix: PyTree[ArrayLike], vector: PyTree[ArrayLike]) -> PyTree[Array]:
    # matrix has structure [tree(in), leaf(out), leaf(in)]
    # vector has structure [tree(in), leaf(in)]
    # return has structure [leaf(out)]
    matrix = jtu.tree_leaves(matrix)
    vector = jtu.tree_leaves(vector)
    assert len(matrix) == len(vector)
    return sum([_matmul(m, v) for m, v in zip(matrix, vector)])


# Needed as static fields must be hashable and eq-able, and custom pytrees might have
# e.g. define custom __eq__ methods.
_T = TypeVar("_T")
_FlatPyTree = tuple[list[_T], jtu.PyTreeDef]


def _inexact_structure_impl2(x):
    if jnp.issubdtype(x.dtype, jnp.inexact):
        return x
    else:
        return x.astype(default_floating_dtype())


def _inexact_structure_impl(x):
    return jtu.tree_map(_inexact_structure_impl2, x)


def _inexact_structure(x: PyTree[jax.ShapeDtypeStruct]) -> PyTree[jax.ShapeDtypeStruct]:
    return strip_weak_dtype(jax.eval_shape(_inexact_structure_impl, x))


class _Leaf:  # not a pytree
    def __init__(self, value):
        self.value = value


# The `{input,output}_structure`s have to be static because otherwise abstract
# evaluation rules will promote them to ShapedArrays.
class PyTreeLinearOperator(AbstractLinearOperator, strict=True):
    """Represents a PyTree of floating-point JAX arrays as a linear operator.

    This is basically a generalisation of [`lineax.MatrixLinearOperator`][], from
    taking just a single array to take a PyTree-of-arrays. (And likewise from returning
    a single array to returning a PyTree-of-arrays.)

    Specifically, suppose we want this to be a linear operator `X -> Y`, for which
    elements of `X` are PyTrees with structure `T` whose `i`th leaf is a floating-point
    JAX array of shape `x_shape_i`, and elements of `Y` are PyTrees with structure `S`
    whose `j`th leaf is a floating-point JAX array of has shape `y_shape_j`. Then the
    input PyTree should have structure `T`-compose-`S`, and its `(i, j)`-th  leaf should
    be a floating-point JAX array of shape `(*x_shape_i, *y_shape_j)`.

    !!! Example

        ```python
        # Suppose `x` is a member of our input space, with the following pytree
        # structure:
        eqx.tree_pprint(x)  # [f32[5, 9], f32[3]]

        # Suppose `y` is a member of our output space, with the following pytree
        # structure:
        eqx.tree_pprint(y)
        # {"a": f32[1, 2]}

        # then `pytree` should be a pytree with the following structure:
        eqx.tree_pprint(pytree)  # {"a": [f32[1, 2, 5, 9], f32[1, 2, 3]]}
        ```
    """

    pytree: PyTree[Inexact[Array, "..."]]
    output_structure: _FlatPyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)
    input_structure: _FlatPyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)

    def __init__(
        self,
        pytree: PyTree[ArrayLike],
        output_structure: PyTree[jax.ShapeDtypeStruct],
        tags: Union[object, frozenset[object]] = (),
    ):
        """**Arguments:**

        - `pytree`: this should be a PyTree, with structure as specified in
            [`lineax.PyTreeLinearOperator`][].
        - `output_structure`: the structure of the output space. This should be a PyTree
            of `jax.ShapeDtypeStruct`s. (The structure of the input space is then
            automatically derived from the structure of `pytree`.)
        - `tags`: any tags indicating whether this operator has any particular
            properties, like symmetry or positive-definite-ness. Note that these
            properties are unchecked and you may get incorrect values elsewhere if these
            tags are wrong.
        """
        output_structure = _inexact_structure(output_structure)
        self.pytree = jtu.tree_map(inexact_asarray, pytree)
        self.output_structure = jtu.tree_flatten(output_structure)
        self.tags = _frozenset(tags)

        # self.out_structure() has structure [tree(out)]
        # self.pytree has structure [tree(out), tree(in), leaf(out), leaf(in)]
        def get_structure(struct, subpytree):
            # subpytree has structure [tree(in), leaf(out), leaf(in)]
            def sub_get_structure(leaf):
                shape = jnp.shape(leaf)  # [leaf(out), leaf(in)]
                ndim = len(struct.shape)
                if shape[:ndim] != struct.shape:
                    raise ValueError(
                        "`pytree` and `output_structure` are not consistent"
                    )
                return jax.ShapeDtypeStruct(shape=shape[ndim:], dtype=jnp.dtype(leaf))

            return _Leaf(jtu.tree_map(sub_get_structure, subpytree))

        if output_structure is None:
            # Implies that len(input_structures) > 0
            raise ValueError("Cannot have trivial output_structure")
        input_structures = jtu.tree_map(get_structure, output_structure, self.pytree)
        input_structures = jtu.tree_leaves(input_structures)
        input_structure = input_structures[0].value
        for val in input_structures[1:]:
            if eqx.tree_equal(input_structure, val.value) is not True:
                raise ValueError(
                    "`pytree` does not have a consistent `input_structure`"
                )
        self.input_structure = jtu.tree_flatten(input_structure)

    def mv(self, vector):
        # vector has structure [tree(in), leaf(in)]
        # self.out_structure() has structure [tree(out)]
        # self.pytree has structure [tree(out), tree(in), leaf(out), leaf(in)]
        # return has struture [tree(out), leaf(out)]
        def matmul(_, matrix):
            return _tree_matmul(matrix, vector)

        return jtu.tree_map(matmul, self.out_structure(), self.pytree)

    def as_matrix(self):
        with jax.numpy_dtype_promotion("standard"):
            dtype = jnp.result_type(*jtu.tree_leaves(self.pytree))

        def concat_in(struct, subpytree):
            leaves = jtu.tree_leaves(subpytree)
            assert all(leaf.shape[: struct.ndim] == struct.shape for leaf in leaves)
            leaves = [
                leaf.astype(dtype).reshape(
                    struct.size, math.prod(leaf.shape[struct.ndim :])
                )
                for leaf in leaves
            ]
            return jnp.concatenate(leaves, axis=1)

        matrix = jtu.tree_map(concat_in, self.out_structure(), self.pytree)
        matrix = jtu.tree_leaves(matrix)
        return jnp.concatenate(matrix, axis=0)

    def transpose(self):
        if symmetric_tag in self.tags:
            return self

        def _transpose(struct, subtree):
            def _transpose_impl(leaf):
                return jnp.moveaxis(leaf, source, dest)

            source = list(range(struct.ndim))
            dest = list(range(-struct.ndim, 0))
            return jtu.tree_map(_transpose_impl, subtree)

        pytree_transpose = jtu.tree_map(_transpose, self.out_structure(), self.pytree)
        pytree_transpose = jtu.tree_transpose(
            jtu.tree_structure(self.out_structure()),
            jtu.tree_structure(self.in_structure()),
            pytree_transpose,
        )
        return PyTreeLinearOperator(
            pytree_transpose, self.in_structure(), transpose_tags(self.tags)
        )

    def in_structure(self):
        leaves, treedef = self.input_structure
        return jtu.tree_unflatten(treedef, leaves)

    def out_structure(self):
        leaves, treedef = self.output_structure
        return jtu.tree_unflatten(treedef, leaves)


class DiagonalLinearOperator(AbstractLinearOperator, strict=True):
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


class _NoAuxIn(eqx.Module):
    fn: Callable
    args: Any

    def __call__(self, x):
        return self.fn(x, self.args)


class _NoAuxOut(eqx.Module):
    fn: Callable

    def __call__(self, x):
        f, _ = self.fn(x)
        return f


class _Unwrap(eqx.Module):
    fn: Callable

    def __call__(self, x):
        (f,) = self.fn(x)
        return f


class JacobianLinearOperator(AbstractLinearOperator, strict=True):
    """Given a function `fn: X -> Y`, and a point `x in X`, then this defines the
    linear operator (also a function `X -> Y`) given by the Jacobian `(d(fn)/dx)(x)`.

    For example if the inputs and outputs are just arrays, then this is equivalent to
    `MatrixLinearOperator(jax.jacfwd(fn)(x))`.

    The Jacobian is not materialised; matrix-vector products, which are in fact
    Jacobian-vector products, are computed using autodifferentiation, specifically
    `jax.jvp`. Thus, `JacobianLinearOperator(fn, x).mv(v)` is equivalent to
    `jax.jvp(fn, (x,), (v,))`.

    See also [`lineax.linearise`][], which caches the primal computation, i.e.
    it returns `_, lin = jax.linearize(fn, x); FunctionLinearOperator(lin, ...)`

    See also [`lineax.materialise`][], which materialises the whole Jacobian in
    memory.
    """

    fn: Callable[
        [PyTree[Inexact[Array, "..."]], PyTree[Any]], PyTree[Inexact[Array, "..."]]
    ]
    x: PyTree[Inexact[Array, "..."]]
    args: PyTree[Any]
    tags: frozenset[object] = eqx.field(static=True)
    jac: Optional[Literal["fwd", "bwd"]]

    @eqxi.doc_remove_args("closure_convert", "_has_aux")
    def __init__(
        self,
        fn: Callable,
        x: PyTree[ArrayLike],
        args: PyTree[Any] = None,
        tags: Union[object, Iterable[object]] = (),
        closure_convert: bool = True,
        _has_aux: bool = False,  # TODO(kidger): remove, no longer used
        jac: Optional[Literal["fwd", "bwd"]] = None,
    ):
        """**Arguments:**

        - `fn`: A function `(x, args) -> y`. The Jacobian `d(fn)/dx` is used as the
            linear operator, and `args` are just any other arguments that should not be
            differentiated.
        - `x`: The point to evaluate `d(fn)/dx` at: `(d(fn)/dx)(x, args)`.
        - `args`: As `x`; this is the point to evaluate `d(fn)/dx` at:
            `(d(fn)/dx)(x, args)`.
        - `tags`: any tags indicating whether this operator has any particular
            properties, like symmetry or positive-definite-ness. Note that these
            properties are unchecked and you may get incorrect values elsewhere if these
            tags are wrong.
        - `jac`: allows to use specific jacobian computation method. If `jac=fwd`
           forces `jax.jacfwd` to be used, similarly `jac=bwd` mandates the use of
           `jax.jacrev`. Otherwise, if not specified it will be chosen
           by default according to input and output shape.
        """
        if not _has_aux:
            fn = NoneAux(fn)
        # Flush out any closed-over values, so that we can safely pass `self`
        # across API boundaries. (In particular, across `linear_solve_p`.)
        # We don't use `jax.closure_convert` as that only flushes autodiffable
        # (=floating-point) constants. It probably doesn't matter, but if `fn` is a
        # PyTree capturing non-floating-point constants, we should probably continue
        # to respect that, and keep any non-floating-point constants as part of the
        # PyTree structure.
        x = jtu.tree_map(inexact_asarray, x)
        if closure_convert:
            fn = eqx.filter_closure_convert(fn, x, args)
        self.fn = fn
        self.x = x
        self.args = args
        self.tags = _frozenset(tags)
        self.jac = jac

    def mv(self, vector):
        fn = _NoAuxOut(_NoAuxIn(self.fn, self.args))
        if self.jac == "fwd" or self.jac is None:
            _, out = jax.jvp(fn, (self.x,), (vector,))
        elif self.jac == "bwd":
            jac = jax.jacrev(fn)(self.x)
            out = PyTreeLinearOperator(jac, output_structure=self.out_structure()).mv(
                vector
            )
        else:
            raise ValueError("`jac` should be either `'fwd'`, `'bwd'`, or `None`.")
        return out

    def as_matrix(self):
        return materialise(self).as_matrix()

    def transpose(self):
        if symmetric_tag in self.tags:
            return self
        fn = _NoAuxOut(_NoAuxIn(self.fn, self.args))
        # Works because vjpfn is a PyTree
        _, vjpfn = jax.vjp(fn, self.x)
        vjpfn = _Unwrap(vjpfn)
        return FunctionLinearOperator(
            vjpfn, self.out_structure(), transpose_tags(self.tags)
        )

    def in_structure(self):
        return strip_weak_dtype(jax.eval_shape(lambda: self.x))

    def out_structure(self):
        fn = _NoAuxOut(_NoAuxIn(self.fn, self.args))
        return strip_weak_dtype(eqxi.cached_filter_eval_shape(fn, self.x))


# `input_structure` must be static as with `JacobianLinearOperator`
class FunctionLinearOperator(AbstractLinearOperator, strict=True):
    """Wraps a *linear* function `fn: X -> Y` into a linear operator. (So that
    `self.mv(x)` is defined by `self.mv(x) == fn(x)`.)

    See also [`lineax.materialise`][], which materialises the whole linear operator
    in memory. (Similar to `.as_matrix()`.)
    """

    fn: Callable[[PyTree[Inexact[Array, "..."]]], PyTree[Inexact[Array, "..."]]]
    input_structure: _FlatPyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    @eqxi.doc_remove_args("closure_convert")
    def __init__(
        self,
        fn: Callable[[PyTree[Inexact[Array, "..."]]], PyTree[Inexact[Array, "..."]]],
        input_structure: PyTree[jax.ShapeDtypeStruct],
        tags: Union[object, Iterable[object]] = (),
        closure_convert: bool = True,
    ):
        """**Arguments:**

        - `fn`: a linear function. Should accept a PyTree of floating-point JAX arrays,
            and return a PyTree of floating-point JAX arrays.
        - `input_structure`: A PyTree of `jax.ShapeDtypeStruct`s specifying the
            structure of the input to the function. (When later calling `self.mv(x)`
            then this should match the structure of `x`, i.e.
            `jax.eval_shape(lambda: x)`.)
        - `tags`: any tags indicating whether this operator has any particular
            properties, like symmetry or positive-definite-ness. Note that these
            properties are unchecked and you may get incorrect values elsewhere if these
            tags are wrong.
        """
        # See matching comment in JacobianLinearOperator.
        input_structure = _inexact_structure(input_structure)
        if closure_convert:
            fn = eqx.filter_closure_convert(fn, input_structure)
        self.fn = fn
        self.input_structure = jtu.tree_flatten(input_structure)
        self.tags = _frozenset(tags)

    def mv(self, vector):
        return self.fn(vector)

    def as_matrix(self):
        return materialise(self).as_matrix()

    def transpose(self):
        if symmetric_tag in self.tags:
            return self
        transpose_fn = jax.linear_transpose(self.fn, self.in_structure())

        def _transpose_fn(vector):
            (out,) = transpose_fn(vector)
            return out

        # Works because transpose_fn is a PyTree
        return FunctionLinearOperator(
            _transpose_fn, self.out_structure(), transpose_tags(self.tags)
        )

    def in_structure(self):
        leaves, treedef = self.input_structure
        return jtu.tree_unflatten(treedef, leaves)

    def out_structure(self):
        return strip_weak_dtype(
            eqxi.cached_filter_eval_shape(self.fn, self.in_structure())
        )


# `structure` must be static as with `JacobianLinearOperator`
class IdentityLinearOperator(AbstractLinearOperator, strict=True):
    """Represents the identity transformation `X -> X`, where each `x in X` is some
    PyTree of floating-point JAX arrays.
    """

    input_structure: _FlatPyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    output_structure: _FlatPyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)

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
        input_structure = _inexact_structure(input_structure)
        output_structure = _inexact_structure(output_structure)
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


class TridiagonalLinearOperator(AbstractLinearOperator, strict=True):
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


class TaggedLinearOperator(AbstractLinearOperator, strict=True):
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
        self, operator: AbstractLinearOperator, tags: Union[object, Iterable[object]]
    ):
        """**Arguments:**

        - `operator`: some other linear operator to wrap.
        - `tags`: any tags indicating whether this operator has any particular
            properties, like symmetry or positive-definite-ness. Note that these
            properties are unchecked and you may get incorrect values elsewhere if these
            tags are wrong.
        """
        self.operator = operator
        self.tags = _frozenset(tags)

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


class TangentLinearOperator(AbstractLinearOperator, strict=True):
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


class AddLinearOperator(AbstractLinearOperator, strict=True):
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


class MulLinearOperator(AbstractLinearOperator, strict=True):
    """A linear operator formed by multiplying a linear operator by a scalar.

    !!! Example

        ```python
        x = MatrixLinearOperator(...)
        y = 0.5
        assert isinstance(x * y, MulLinearOperator)
        ```
    """

    operator: AbstractLinearOperator
    scalar: Scalar

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
class NegLinearOperator(AbstractLinearOperator, strict=True):
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


class DivLinearOperator(AbstractLinearOperator, strict=True):
    """A linear operator formed by dividing a linear operator by a scalar.

    !!! Example

        ```python
        x = MatrixLinearOperator(...)
        y = 0.5
        assert isinstance(x / y, DivLinearOperator)
        ```
    """

    operator: AbstractLinearOperator
    scalar: Scalar

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


class ComposedLinearOperator(AbstractLinearOperator, strict=True):
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
        return self.operator1.mv(self.operator2.mv(vector))

    def as_matrix(self):
        return jnp.matmul(
            self.operator1.as_matrix(),
            self.operator2.as_matrix(),
            precision=lax.Precision.HIGHEST,  # pyright: ignore
        )

    def transpose(self):
        return self.operator2.transpose() @ self.operator1.transpose()

    def in_structure(self):
        return self.operator2.in_structure()

    def out_structure(self):
        return self.operator1.out_structure()


class AuxLinearOperator(AbstractLinearOperator, strict=True):
    """Internal to lineax. Used to represent a linear operator with additional
    metadata attached.
    """

    operator: AbstractLinearOperator
    aux: PyTree[Array]

    def mv(self, vector):
        return self.operator.mv(vector)

    def as_matrix(self):
        return self.operator.as_matrix()

    def transpose(self):
        return self.operator.transpose()

    def in_structure(self):
        return self.operator.in_structure()

    def out_structure(self):
        return self.operator.out_structure()


#
# Operations on `AbstractLinearOperator`s.
# These are done through `singledispatch` rather than as methods.
#
# If an end user ever wanted to add something analogous to
# `diagonal: AbstractLinearOperator -> Array`
# then of course they don't get to edit our base class and add overloads to all
# subclasses.
# They'd have to use `singledispatch` to get the desired behaviour. (Or maybe just
# hardcode compatibility with only some `AbstractLinearOperator` subclasses, eurgh.)
# So for consistency we do the same thing here, rather than adding privileged behaviour
# for just the operations we happen to support.
#
# (Something something Julia something something orphan problem etc.)
#


def _default_not_implemented(name: str, operator: AbstractLinearOperator) -> NoReturn:
    msg = f"`lineax.{name}` has not been implemented for {type(operator)}"
    if type(operator).__module__.startswith("lineax"):
        assert False, msg + ". Please file a bug against Lineax."
    else:
        raise NotImplementedError(msg)


# linearise


@ft.singledispatch
def linearise(operator: AbstractLinearOperator) -> AbstractLinearOperator:
    """Linearises a linear operator. This returns another linear operator.

    Mathematically speaking this is just the identity function. And indeed most linear
    operators will be returned unchanged.

    For specifically [`lineax.JacobianLinearOperator`][], then this will cache the
    primal pass, so that it does not need to be recomputed each time. That is, it uses
    some memory to improve speed. (This is the precisely same distinction as `jax.jvp`
    versus `jax.linearize`.)

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    Another linear operator. Mathematically it performs matrix-vector products
    (`operator.mv`) that produce the same results as the input `operator`.
    """
    _default_not_implemented("linearise", operator)


@linearise.register(MatrixLinearOperator)
@linearise.register(PyTreeLinearOperator)
@linearise.register(FunctionLinearOperator)
@linearise.register(IdentityLinearOperator)
@linearise.register(DiagonalLinearOperator)
@linearise.register(TridiagonalLinearOperator)
def _(operator):
    return operator


@linearise.register(JacobianLinearOperator)
def _(operator):
    fn = _NoAuxIn(operator.fn, operator.args)
    (_, aux), lin = jax.linearize(fn, operator.x)
    lin = _NoAuxOut(lin)
    out = FunctionLinearOperator(lin, operator.in_structure(), operator.tags)
    return AuxLinearOperator(out, aux)


# materialise


@ft.singledispatch
def materialise(operator: AbstractLinearOperator) -> AbstractLinearOperator:
    """Materialises a linear operator. This returns another linear operator.

    Mathematically speaking this is just the identity function. And indeed most linear
    operators will be returned unchanged.

    For specifically [`lineax.JacobianLinearOperator`][] and
    [`lineax.FunctionLinearOperator`][] then the linear operator is materialised in
    memory. That is, it becomes defined as a matrix (or pytree of arrays), rather
    than being defined only through its matrix-vector product
    ([`lineax.AbstractLinearOperator.mv`][]).

    Materialisation sometimes improves compile time or run time. It usually increases
    memory usage.

    For example:
    ```python
    large_function = ...
    operator = lx.FunctionLinearOperator(large_function, ...)

    # Option 1
    out1 = operator.mv(vector1)  # Traces and compiles `large_function`
    out2 = operator.mv(vector2)  # Traces and compiles `large_function` again!
    out3 = operator.mv(vector3)  # Traces and compiles `large_function` a third time!
    # All that compilation might lead to long compile times.
    # If `large_function` takes a long time to run, then this might also lead to long
    # run times.

    # Option 2
    operator = lx.materialise(operator)  # Traces and compiles `large_function` and
                                           # stores the result as a matrix.
    out1 = operator.mv(vector1)  # Each of these just computes a matrix-vector product
    out2 = operator.mv(vector2)  # against the stored matrix.
    out3 = operator.mv(vector3)  #
    # Now, `large_function` is only compiled once, and only ran once.
    # However, storing the matrix might take a lot of memory, and the initial
    # computation may-or-may-not take a long time to run.
    ```
    Generally speaking it is worth first setting up your problem without
    `lx.materialise`, and using it as an optional optimisation if you find that it
    helps your particular problem.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    Another linear operator. Mathematically it performs matrix-vector products
    (`operator.mv`) that produce the same results as the input `operator`.
    """
    _default_not_implemented("materialise", operator)


@materialise.register(MatrixLinearOperator)
@materialise.register(PyTreeLinearOperator)
@materialise.register(IdentityLinearOperator)
@materialise.register(DiagonalLinearOperator)
@materialise.register(TridiagonalLinearOperator)
def _(operator):
    return operator


@materialise.register(JacobianLinearOperator)
def _(operator):
    fn = _NoAuxIn(operator.fn, operator.args)
    jac, aux = jacobian(
        fn,
        operator.in_size(),
        operator.out_size(),
        holomorphic=any(jnp.iscomplexobj(xi) for xi in jtu.tree_leaves(operator.x)),
        has_aux=True,
        jac=operator.jac,
    )(operator.x)
    out = PyTreeLinearOperator(jac, operator.out_structure(), operator.tags)
    return AuxLinearOperator(out, aux)


@materialise.register(FunctionLinearOperator)
def _(operator):
    flat, unravel = strip_weak_dtype(
        eqx.filter_eval_shape(jfu.ravel_pytree, operator.in_structure())
    )
    eye = jnp.eye(flat.size, dtype=flat.dtype)
    jac = jax.vmap(lambda x: operator.fn(unravel(x)), out_axes=-1)(eye)

    def batch_unravel(x):
        assert x.ndim > 0
        unravel_ = unravel
        for _ in range(x.ndim - 1):
            unravel_ = jax.vmap(unravel_)
        return unravel_(x)

    jac = jtu.tree_map(batch_unravel, jac)
    return PyTreeLinearOperator(jac, operator.out_structure(), operator.tags)


# diagonal


@ft.singledispatch
def diagonal(operator: AbstractLinearOperator) -> Shaped[Array, " size"]:
    """Extracts the diagonal from a linear operator, and returns a vector.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    A rank-1 JAX array. (That is, it has shape `(a,)` for some integer `a`.)

    For most operators this is just `jnp.diag(operator.as_matrix())`. Some operators
    (e.g. [`lineax.DiagonalLinearOperator`][]) can have more efficient
    implementations. If you don't know what kind of operator you might have, then this
    function ensures that you always get the most efficient implementation.
    """
    _default_not_implemented("diagonal", operator)


@diagonal.register(MatrixLinearOperator)
@diagonal.register(PyTreeLinearOperator)
@diagonal.register(JacobianLinearOperator)
@diagonal.register(FunctionLinearOperator)
def _(operator):
    return jnp.diag(operator.as_matrix())


@diagonal.register(DiagonalLinearOperator)
def _(operator):
    diagonal, _ = jfu.ravel_pytree(operator.diagonal)
    return diagonal


@diagonal.register(IdentityLinearOperator)
def _(operator):
    return jnp.ones(operator.in_size())


@diagonal.register(TridiagonalLinearOperator)
def _(operator):
    return operator.diagonal


# tridiagonal


@ft.singledispatch
def tridiagonal(
    operator: AbstractLinearOperator,
) -> tuple[Shaped[Array, " size"], Shaped[Array, " size-1"], Shaped[Array, " size-1"]]:
    """Extracts the diagonal, lower diagonal, and upper diagonal, from a linear
    operator. Returns three vectors.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    A 3-tuple, consisting of:

    - The diagonal of the matrix, represented as a vector.
    - The lower diagonal of the matrix, represented as a vector.
    - The upper diagonal of the matrix, represented as a vector.

    If the diagonal has shape `(a,)` then the lower and upper diagonals will have shape
    `(a - 1,)`.

    For most operators these are computed by materialising the array and then extracting
    the relevant elements, e.g. getting the main diagonal via
    `jnp.diag(operator.as_matrix())`. Some operators (e.g.
    [`lineax.TridiagonalLinearOperator`][]) can have more efficient implementations.
    If you don't know what kind of operator you might have, then this function ensures
    that you always get the most efficient implementation.
    """
    _default_not_implemented("tridiagonal", operator)


@tridiagonal.register(MatrixLinearOperator)
@tridiagonal.register(PyTreeLinearOperator)
@tridiagonal.register(JacobianLinearOperator)
@tridiagonal.register(FunctionLinearOperator)
def _(operator):
    matrix = operator.as_matrix()
    assert matrix.ndim == 2
    diagonal = jnp.diagonal(matrix, offset=0)
    upper_diagonal = jnp.diagonal(matrix, offset=1)
    lower_diagonal = jnp.diagonal(matrix, offset=-1)
    return diagonal, lower_diagonal, upper_diagonal


@tridiagonal.register(DiagonalLinearOperator)
def _(operator):
    diag = diagonal(operator)
    upper_diag = jnp.zeros(diag.size - 1)
    lower_diag = jnp.zeros(diag.size - 1)
    return diag, lower_diag, upper_diag


@tridiagonal.register(IdentityLinearOperator)
def _(operator):
    size = operator.in_size()
    diagonal = jnp.ones(size)
    off_diagonal = jnp.zeros(size - 1)
    return diagonal, off_diagonal, off_diagonal


@tridiagonal.register(TridiagonalLinearOperator)
def _(operator):
    return operator.diagonal, operator.lower_diagonal, operator.upper_diagonal


# is_symmetric


@ft.singledispatch
def is_symmetric(operator: AbstractLinearOperator) -> bool:
    """Returns whether an operator is marked as symmetric.

    See [the documentation on linear operator tags](../api/tags.md) for more
    information.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    Either `True` or `False.`
    """
    _default_not_implemented("is_symmetric", operator)


@is_symmetric.register(MatrixLinearOperator)
@is_symmetric.register(PyTreeLinearOperator)
@is_symmetric.register(JacobianLinearOperator)
@is_symmetric.register(FunctionLinearOperator)
def _(operator):
    return any(
        tag in operator.tags
        for tag in (
            symmetric_tag,
            positive_semidefinite_tag,
            negative_semidefinite_tag,
            diagonal_tag,
        )
    )


@is_symmetric.register(IdentityLinearOperator)
def _(operator):
    return eqx.tree_equal(operator.in_structure(), operator.out_structure()) is True


@is_symmetric.register(DiagonalLinearOperator)
def _(operator):
    return True


@is_symmetric.register(TridiagonalLinearOperator)
def _(operator):
    return False


# is_diagonal


@ft.singledispatch
def is_diagonal(operator: AbstractLinearOperator) -> bool:
    """Returns whether an operator is marked as diagonal.

    See [the documentation on linear operator tags](../api/tags.md) for more
    information.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    Either `True` or `False.`
    """
    _default_not_implemented("is_diagonal", operator)


@is_diagonal.register(MatrixLinearOperator)
@is_diagonal.register(PyTreeLinearOperator)
@is_diagonal.register(JacobianLinearOperator)
@is_diagonal.register(FunctionLinearOperator)
def _(operator):
    return diagonal_tag in operator.tags or (
        operator.in_size() == 1 and operator.out_size() == 1
    )


@is_diagonal.register(IdentityLinearOperator)
@is_diagonal.register(DiagonalLinearOperator)
def _(operator):
    return True


@is_diagonal.register(TridiagonalLinearOperator)
def _(operator):
    return operator.in_size() == 1


# is_tridiagonal


@ft.singledispatch
def is_tridiagonal(operator: AbstractLinearOperator) -> bool:
    """Returns whether an operator is marked as tridiagonal.

    See [the documentation on linear operator tags](../api/tags.md) for more
    information.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    Either `True` or `False.`
    """
    _default_not_implemented("is_tridiagonal", operator)


@is_tridiagonal.register(MatrixLinearOperator)
@is_tridiagonal.register(PyTreeLinearOperator)
@is_tridiagonal.register(JacobianLinearOperator)
@is_tridiagonal.register(FunctionLinearOperator)
def _(operator):
    return tridiagonal_tag in operator.tags or diagonal_tag in operator.tags


@is_tridiagonal.register(IdentityLinearOperator)
@is_tridiagonal.register(DiagonalLinearOperator)
@is_tridiagonal.register(TridiagonalLinearOperator)
def _(operator):
    return True


# has_unit_diagonal


@ft.singledispatch
def has_unit_diagonal(operator: AbstractLinearOperator) -> bool:
    """Returns whether an operator is marked as having unit diagonal.

    See [the documentation on linear operator tags](../api/tags.md) for more
    information.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    Either `True` or `False.`
    """
    _default_not_implemented("has_unit_diagonal", operator)


@has_unit_diagonal.register(MatrixLinearOperator)
@has_unit_diagonal.register(PyTreeLinearOperator)
@has_unit_diagonal.register(JacobianLinearOperator)
@has_unit_diagonal.register(FunctionLinearOperator)
def _(operator):
    return unit_diagonal_tag in operator.tags


@has_unit_diagonal.register(IdentityLinearOperator)
def _(operator):
    return True


@has_unit_diagonal.register(DiagonalLinearOperator)
@has_unit_diagonal.register(TridiagonalLinearOperator)
def _(operator):
    # TODO: refine this
    return False


# is_lower_triangular


@ft.singledispatch
def is_lower_triangular(operator: AbstractLinearOperator) -> bool:
    """Returns whether an operator is marked as lower triangular.

    See [the documentation on linear operator tags](../api/tags.md) for more
    information.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    Either `True` or `False.`
    """
    _default_not_implemented("is_lower_triangular", operator)


@is_lower_triangular.register(MatrixLinearOperator)
@is_lower_triangular.register(PyTreeLinearOperator)
@is_lower_triangular.register(JacobianLinearOperator)
@is_lower_triangular.register(FunctionLinearOperator)
def _(operator):
    return lower_triangular_tag in operator.tags


@is_lower_triangular.register(IdentityLinearOperator)
@is_lower_triangular.register(DiagonalLinearOperator)
def _(operator):
    return True


@is_lower_triangular.register(TridiagonalLinearOperator)
def _(operator):
    return False


# is_upper_triangular


@ft.singledispatch
def is_upper_triangular(operator: AbstractLinearOperator) -> bool:
    """Returns whether an operator is marked as upper triangular.

    See [the documentation on linear operator tags](../api/tags.md) for more
    information.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    Either `True` or `False.`
    """
    _default_not_implemented("is_upper_triangular", operator)


@is_upper_triangular.register(MatrixLinearOperator)
@is_upper_triangular.register(PyTreeLinearOperator)
@is_upper_triangular.register(JacobianLinearOperator)
@is_upper_triangular.register(FunctionLinearOperator)
def _(operator):
    return upper_triangular_tag in operator.tags


@is_upper_triangular.register(IdentityLinearOperator)
@is_upper_triangular.register(DiagonalLinearOperator)
def _(operator):
    return True


@is_upper_triangular.register(TridiagonalLinearOperator)
def _(operator):
    return False


# is_positive_semidefinite


@ft.singledispatch
def is_positive_semidefinite(operator: AbstractLinearOperator) -> bool:
    """Returns whether an operator is marked as positive semidefinite.

    See [the documentation on linear operator tags](../api/tags.md) for more
    information.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    Either `True` or `False.`
    """
    _default_not_implemented("is_positive_semidefinite", operator)


@is_positive_semidefinite.register(MatrixLinearOperator)
@is_positive_semidefinite.register(PyTreeLinearOperator)
@is_positive_semidefinite.register(JacobianLinearOperator)
@is_positive_semidefinite.register(FunctionLinearOperator)
def _(operator):
    return positive_semidefinite_tag in operator.tags


@is_positive_semidefinite.register(IdentityLinearOperator)
def _(operator):
    return True


@is_positive_semidefinite.register(DiagonalLinearOperator)
@is_positive_semidefinite.register(TridiagonalLinearOperator)
def _(operator):
    # TODO: refine this
    return False


# is_negative_semidefinite


@ft.singledispatch
def is_negative_semidefinite(operator: AbstractLinearOperator) -> bool:
    """Returns whether an operator is marked as negative semidefinite.

    See [the documentation on linear operator tags](../api/tags.md) for more
    information.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    Either `True` or `False.`
    """
    _default_not_implemented("is_negative_semidefinite", operator)


@is_negative_semidefinite.register(MatrixLinearOperator)
@is_negative_semidefinite.register(PyTreeLinearOperator)
@is_negative_semidefinite.register(JacobianLinearOperator)
@is_negative_semidefinite.register(FunctionLinearOperator)
def _(operator):
    return negative_semidefinite_tag in operator.tags


@is_negative_semidefinite.register(IdentityLinearOperator)
def _(operator):
    return False


@is_negative_semidefinite.register(DiagonalLinearOperator)
@is_negative_semidefinite.register(TridiagonalLinearOperator)
def _(operator):
    # TODO: refine this
    return False


# ops for wrapper operators


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

    @transform.register(AddLinearOperator)  # pyright: ignore
    def _(operator, transform=transform):
        return transform(operator.operator1) + transform(operator.operator2)

    @transform.register(MulLinearOperator)
    def _(operator, transform=transform):
        return transform(operator.operator) * operator.scalar

    @transform.register(NegLinearOperator)  # pyright: ignore
    def _(operator, transform=transform):
        return -transform(operator.operator)

    @transform.register(DivLinearOperator)
    def _(operator, transform=transform):
        return transform(operator.operator) / operator.scalar

    @transform.register(AuxLinearOperator)  # pyright: ignore
    def _(operator, transform=transform):
        return transform(operator.operator)


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


@tridiagonal.register(AddLinearOperator)
def _(operator):
    (diag1, lower1, upper1) = tridiagonal(operator.operator1)
    (diag2, lower2, upper2) = tridiagonal(operator.operator2)
    return (diag1 + diag2, lower1 + lower2, upper1 + upper2)


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


@tridiagonal.register(AuxLinearOperator)
def _(operator):
    return tridiagonal(operator.operator)


@linearise.register(ComposedLinearOperator)
def _(operator):
    return linearise(operator.operator1) @ linearise(operator.operator2)


@materialise.register(ComposedLinearOperator)
def _(operator):
    return materialise(operator.operator1) @ materialise(operator.operator2)


@diagonal.register(ComposedLinearOperator)
def _(operator):
    return jnp.diag(operator.as_matrix())


@tridiagonal.register(ComposedLinearOperator)
def _(operator):
    matrix = operator.as_matrix()
    assert matrix.ndim == 2
    diagonal = jnp.diagonal(matrix, offset=0)
    upper_diagonal = jnp.diagonal(matrix, offset=1)
    lower_diagonal = jnp.diagonal(matrix, offset=-1)
    return diagonal, lower_diagonal, upper_diagonal


for check in (
    is_symmetric,
    is_diagonal,
    has_unit_diagonal,
    is_lower_triangular,
    is_upper_triangular,
    is_tridiagonal,
):

    @check.register(TangentLinearOperator)
    def _(operator, check=check):
        return check(operator.primal)

    @check.register(MulLinearOperator)
    @check.register(NegLinearOperator)
    @check.register(DivLinearOperator)
    @check.register(AuxLinearOperator)
    def _(operator, check=check):
        return check(operator.operator)


for check in (is_positive_semidefinite, is_negative_semidefinite):

    @check.register(TangentLinearOperator)
    def _(operator):
        # Should be unreachable: TangentLinearOperator is used for a narrow set of
        # operations only (mv; transpose) inside the JVP rule linear_solve_p.
        raise NotImplementedError(
            "Please open a GitHub issue: https://github.com/google/lineax"
        )

    @check.register(MulLinearOperator)
    @check.register(DivLinearOperator)
    def _(operator):
        return False  # play it safe, no way to tell.

    @check.register(NegLinearOperator)
    def _(operator, check=check):
        return not check(operator.operator)

    @check.register(AuxLinearOperator)
    def _(operator, check=check):
        return check(operator.operator)


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


for check in (
    is_symmetric,
    is_diagonal,
    is_lower_triangular,
    is_upper_triangular,
    is_positive_semidefinite,
    is_negative_semidefinite,
    is_tridiagonal,
):

    @check.register(AddLinearOperator)
    def _(operator, check=check):
        return check(operator.operator1) and check(operator.operator2)


@has_unit_diagonal.register(AddLinearOperator)
def _(operator):
    return False


for check in (
    is_symmetric,
    is_diagonal,
    is_lower_triangular,
    is_upper_triangular,
    is_positive_semidefinite,
    is_negative_semidefinite,
    is_tridiagonal,
):

    @check.register(ComposedLinearOperator)
    def _(operator, check=check):
        return check(operator.operator1) and check(operator.operator2)


@has_unit_diagonal.register(ComposedLinearOperator)
def _(operator):
    a = is_diagonal(operator)
    b = is_lower_triangular(operator)
    c = is_upper_triangular(operator)
    d = has_unit_diagonal(operator.operator1)
    e = has_unit_diagonal(operator.operator2)
    return (a or b or c) and d and e


# conj


@ft.singledispatch
def conj(operator: AbstractLinearOperator) -> AbstractLinearOperator:
    """Elementwise conjugate of a linear operator. This returns another linear operator.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    Another linear operator.
    """
    _default_not_implemented("conj", operator)


@conj.register(MatrixLinearOperator)
def _(operator):
    return MatrixLinearOperator(operator.matrix.conj(), operator.tags)


@conj.register(PyTreeLinearOperator)
def _(operator):
    pytree_conj = jtu.tree_map(lambda x: x.conj(), operator.pytree)
    return PyTreeLinearOperator(pytree_conj, operator.out_structure(), operator.tags)


@conj.register(DiagonalLinearOperator)
def _(operator):
    diagonal_conj = jtu.tree_map(lambda x: x.conj(), operator.diagonal)
    return DiagonalLinearOperator(diagonal_conj)


@conj.register(JacobianLinearOperator)
def _(operator):
    return conj(linearise(operator))


@conj.register(FunctionLinearOperator)
def _(operator):
    return FunctionLinearOperator(
        lambda vec: jtu.tree_map(jnp.conj, operator.mv(jtu.tree_map(jnp.conj, vec))),
        operator.in_structure(),
        operator.tags,
    )


@conj.register(IdentityLinearOperator)
def _(operator):
    return operator


@conj.register(TridiagonalLinearOperator)
def _(operator):
    return TridiagonalLinearOperator(
        operator.diagonal.conj(),
        operator.lower_diagonal.conj(),
        operator.upper_diagonal.conj(),
    )


@conj.register(TaggedLinearOperator)
def _(operator):
    return TaggedLinearOperator(conj(operator.operator), operator.tags)


@conj.register(TangentLinearOperator)
def _(operator):
    c = lambda operator: conj(operator)
    primal_out, tangent_out = eqx.filter_jvp(c, (operator.primal,), (operator.tangent,))
    return TangentLinearOperator(primal_out, tangent_out)


@conj.register(AddLinearOperator)
def _(operator):
    return conj(operator.operator1) + conj(operator.operator2)


@conj.register(MulLinearOperator)
def _(operator):
    return conj(operator.operator) * operator.scalar.conj()


@conj.register(NegLinearOperator)
def _(operator):
    return -conj(operator.operator)


@conj.register(DivLinearOperator)
def _(operator):
    return conj(operator.operator) / operator.scalar.conj()


@conj.register(ComposedLinearOperator)
def _(operator):
    return conj(operator.operator1) @ conj(operator.operator2)


@conj.register(AuxLinearOperator)
def _(operator):
    return conj(operator.operator)
