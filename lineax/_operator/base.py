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
from collections.abc import Iterable
from typing import NoReturn, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import (
    Array,
    Inexact,
    PyTree,  # pyright: ignore
    Shaped,
)

from .._misc import (
    default_floating_dtype,
    strip_weak_dtype,
)
from .._tags import MaxRankTag


def as_frozenset(x: object | Iterable[object]) -> frozenset[object]:
    try:
        iter_x = iter(x)  # pyright: ignore
    except TypeError:
        return frozenset([x])
    else:
        return frozenset(iter_x)


# Needed as static fields must be hashable and eq-able, and custom pytrees might have
# e.g. define custom __eq__ methods.
_T = TypeVar("_T")
FlatPyTree = tuple[list[_T], jtu.PyTreeDef]


def _inexact_structure_impl2(x):
    if jnp.issubdtype(x.dtype, jnp.inexact):
        return x
    else:
        return x.astype(default_floating_dtype())


def _inexact_structure_impl(x):
    return jtu.tree_map(_inexact_structure_impl2, x)


def inexact_structure(x: PyTree[jax.ShapeDtypeStruct]) -> PyTree[jax.ShapeDtypeStruct]:
    return strip_weak_dtype(jax.eval_shape(_inexact_structure_impl, x))


class AbstractLinearOperator(eqx.Module):
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
        if (
            is_symmetric(self)
            or is_positive_semidefinite(self)
            or is_negative_semidefinite(self)
        ):
            # In particular, we check that dtypes match.
            in_structure = self.in_structure()
            out_structure = self.out_structure()
            # `is` check to handle the possibility of a tracer.
            if eqx.tree_equal(in_structure, out_structure) is not True:
                raise ValueError(
                    "Symmetric/Hermitian matrices must have matching input and output "
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
        # Local imports to avoid a circular dependency: `binary`/`wrapper` import
        # `base`, so the operators built here are imported lazily at call time.
        from .binary import AddLinearOperator

        if not isinstance(other, AbstractLinearOperator):
            raise ValueError("Can only add AbstractLinearOperators together.")
        return AddLinearOperator(self, other)

    def __sub__(self, other) -> "AbstractLinearOperator":
        from .binary import AddLinearOperator

        if not isinstance(other, AbstractLinearOperator):
            raise ValueError("Can only add AbstractLinearOperators together.")
        return AddLinearOperator(self, -other)

    def __mul__(self, other) -> "AbstractLinearOperator":
        from .wrapper import MulLinearOperator

        if np.ndim(other) != 0:
            raise ValueError("Can only multiply AbstractLinearOperators by scalars.")
        return MulLinearOperator(self, other)

    def __rmul__(self, other) -> "AbstractLinearOperator":
        return self * other

    def __matmul__(self, other) -> "AbstractLinearOperator":
        from .binary import ComposedLinearOperator

        if not isinstance(other, AbstractLinearOperator):
            raise ValueError("Can only compose AbstractLinearOperators together.")
        return ComposedLinearOperator(self, other)

    def __truediv__(self, other) -> "AbstractLinearOperator":
        from .wrapper import DivLinearOperator

        if np.ndim(other) != 0:
            raise ValueError("Can only divide AbstractLinearOperators by scalars.")
        return DivLinearOperator(self, other)

    def __neg__(self) -> "AbstractLinearOperator":
        from .wrapper import NegLinearOperator

        return NegLinearOperator(self)


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


@ft.singledispatch
def is_circulant(operator: AbstractLinearOperator) -> bool:
    """Returns whether an operator is marked as circulant.

    See [the documentation on linear operator tags](../api/tags.md) for more
    information.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    Either `True` or `False.`
    """
    _default_not_implemented("is_circulant", operator)


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


@ft.singledispatch
def conj(operator: AbstractLinearOperator) -> AbstractLinearOperator:
    """Elementwise conjugate of a linear operator. This returns another linear operator.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    Another linear operator.
    """
    _default_not_implemented("conj", operator)


# max_rank


@ft.singledispatch
def max_rank(operator: AbstractLinearOperator) -> int:
    """Returns the maximum possible rank of the linear operator.

    Maximum possible rank is inferred from:

    - Shape: `min(out_size, in_size)`
    - User-provided tags: i.e. `MaxRankTag(r)`
    - Composition rules: e.g. `max_rank(A @ B) = min(max_rank(A), max_rank(B))`

    The tightest bound is always returned.

    The return value is a plain Python `int` (never a JAX tracer), suitable
    for use in solver-dispatch control flow.

    **Arguments:**

    - `operator`: a linear operator.

    **Returns:**

    A non-negative integer. This is `0` for known zero operators
    (e.g. multiplication by a static zero scalar).
    """
    # Unlike the boolean `is_*` tag-checking functions, this deliberately
    # does **not** raise `NotImplementedError` for unregistered types.
    # `min(out_size, in_size)` is a correct (conservative) answer for any
    # linear operator, so third-party subclasses that do not register a
    # dispatch still produce a valid result.
    dim_bound = min(operator.out_size(), operator.in_size())
    tags = getattr(operator, "tags", ())
    bounds = [t.r for t in tags if isinstance(t, MaxRankTag)]
    if bounds:
        return min(min(bounds), dim_bound)
    return dim_bound


# All public API operators use default max_rank except for TaggedLinearOperator
