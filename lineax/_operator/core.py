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
from collections.abc import Callable, Iterable
from typing import Any, Literal

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.flatten_util as jfu
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import (
    Array,
    ArrayLike,
    Inexact,
    PyTree,  # pyright: ignore
    Shaped,
)

from .._misc import (
    inexact_asarray,
    jacobian,
    strip_weak_dtype,
)
from .._tags import (
    circulant_tag,
    diagonal_tag,
    hermitian_tag,
    lower_triangular_tag,
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
    first_column,
    FlatPyTree,
    has_real_dtype,
    has_unit_diagonal,
    inexact_structure,
    is_circulant,
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
    tridiagonal,
)
from .structured import (
    CirculantLinearOperator,
    DiagonalLinearOperator,
    TridiagonalLinearOperator,
)


class MatrixLinearOperator(AbstractLinearOperator):
    """Wraps a 2-dimensional JAX array into a linear operator.

    If the matrix has shape `(a, b)` then matrix-vector multiplication (`self.mv`) is
    defined in the usual way: as performing a matrix-vector that accepts a vector of
    shape `(a,)` and returns a vector of shape `(b,)`.
    """

    matrix: Inexact[Array, "a b"]
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self, matrix: Shaped[Array, "a b"], tags: object | frozenset[object] = ()
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
        if not jnp.issubdtype(matrix.dtype, jnp.inexact):
            matrix = matrix.astype(jnp.float32)
        self.matrix = matrix
        self.tags = as_frozenset(tags)

    def mv(self, vector):
        maybe_structured_op = try_structured_materialise(self)
        if maybe_structured_op is not self:
            return maybe_structured_op.mv(vector)
        return jnp.matmul(self.matrix, vector, precision=lax.Precision.HIGHEST)

    def as_matrix(self):
        return self.matrix

    def transpose(self):
        if is_symmetric(self):
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


class _Leaf:  # not a pytree
    def __init__(self, value):
        self.value = value


# The `{input,output}_structure`s have to be static because otherwise abstract
# evaluation rules will promote them to ShapedArrays.
class PyTreeLinearOperator(AbstractLinearOperator):
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
    output_structure: FlatPyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)
    input_structure: FlatPyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)

    def __init__(
        self,
        pytree: PyTree[ArrayLike],
        output_structure: PyTree[jax.ShapeDtypeStruct],
        tags: object | frozenset[object] = (),
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
        output_structure = inexact_structure(output_structure)
        self.pytree = jtu.tree_map(inexact_asarray, pytree)
        self.output_structure = jtu.tree_flatten(output_structure)
        self.tags = as_frozenset(tags)

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
                return jax.ShapeDtypeStruct(
                    shape=shape[ndim:], dtype=jnp.result_type(leaf)
                )

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
        # return has structure [tree(out), leaf(out)]
        maybe_structured_op = try_structured_materialise(self)
        if maybe_structured_op is not self:
            return maybe_structured_op.mv(vector)

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
        if is_symmetric(self):
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


class _NoAuxIn(eqx.Module):
    fn: Callable
    args: Any

    def __call__(self, x):
        return self.fn(x, self.args)


class _Unwrap(eqx.Module):
    fn: Callable

    def __call__(self, x):
        (f,) = self.fn(x)
        return f


class JacobianLinearOperator(AbstractLinearOperator):
    """Given a function `fn: X -> Y`, and a point `x in X`, then this defines the
    linear operator (also a function `X -> Y`) given by the Jacobian `(d(fn)/dx)(x)`.

    For example if the inputs and outputs are just arrays, then this is equivalent to
    `MatrixLinearOperator(jax.jacfwd(fn)(x))`.

    The Jacobian is not materialised; matrix-vector products, which are in fact
    Jacobian-vector products, are computed using autodifferentiation. By default
    (or with `jac="fwd"`), `JacobianLinearOperator(fn, x).mv(v)` is equivalent to
    `jax.jvp(fn, (x,), (v,))`. For `jac="bwd"`, `jax.vjp` is combined with
    `jax.linear_transpose`, which works even with functions
    that only define a custom VJP (via `jax.custom_vjp`) and don't support
    forward-mode differentiation.

    See also [`lineax.materialise`][], which materialises the whole Jacobian in
    memory.

    !!! tip

        For repeated `mv()` calls, consider using [`lineax.linearise`][] to cache
        the primal computation,  e.g. for `jac="fwd"/None` it returns
        `_, lin = jax.linearize(fn, x); FunctionLinearOperator(lin, ...)`
    """

    fn: Callable[
        [PyTree[Inexact[Array, "..."]], PyTree[Any]], PyTree[Inexact[Array, "..."]]
    ]
    x: PyTree[Inexact[Array, "..."]]
    args: PyTree[Any]
    tags: frozenset[object] = eqx.field(static=True)
    jac: Literal["fwd", "bwd"] | None

    @eqxi.doc_remove_args("closure_convert")
    def __init__(
        self,
        fn: Callable,
        x: PyTree[ArrayLike],
        args: PyTree[Any] = None,
        tags: object | Iterable[object] = (),
        jac: Literal["fwd", "bwd"] | None = None,
        closure_convert: bool = True,
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
        if jac not in [None, "fwd", "bwd"]:
            raise ValueError(
                "`jac` argument of `JacobianLinearOperator` should be either "
                "`'fwd'`, `'bwd'`, or `None`."
            )
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
        self.tags = as_frozenset(tags)
        self.jac = jac

    def mv(self, vector):
        fn = _NoAuxIn(self.fn, self.args)
        if self.jac == "fwd" or self.jac is None:
            _, out = jax.jvp(fn, (self.x,), (vector,))
        elif self.jac == "bwd":
            # Use VJP + linear_transpose instead of materializing full Jacobian.
            # This works even for custom_vjp functions that don't have JVP rules.
            _, vjp_fn = jax.vjp(fn, self.x)
            if is_symmetric(self):
                # For symmetric operators, J = J.T, so vjp directly gives J @ v
                (out,) = vjp_fn(vector)
            else:
                # For non-symmetric, transpose the VJP to get J @ v from J.T @ v
                transpose_vjp = jax.linear_transpose(
                    lambda g: vjp_fn(g)[0], self.out_structure()
                )
                (out,) = transpose_vjp(vector)
        else:
            raise ValueError("`jac` should be either `'fwd'`, `'bwd'`, or `None`.")
        return out

    def as_matrix(self):
        return materialise(self).as_matrix()

    def transpose(self):
        if is_symmetric(self):
            return self
        fn = _NoAuxIn(self.fn, self.args)
        # Works because vjpfn is a PyTree
        _, vjpfn = jax.vjp(fn, self.x)
        vjpfn = _Unwrap(vjpfn)
        return FunctionLinearOperator(
            vjpfn, self.out_structure(), transpose_tags(self.tags)
        )

    def in_structure(self):
        return strip_weak_dtype(jax.eval_shape(lambda: self.x))

    def out_structure(self):
        fn = _NoAuxIn(self.fn, self.args)
        return strip_weak_dtype(eqxi.cached_filter_eval_shape(fn, self.x))


# `input_structure` must be static as with `JacobianLinearOperator`
class FunctionLinearOperator(AbstractLinearOperator):
    """Wraps a *linear* function `fn: X -> Y` into a linear operator. (So that
    `self.mv(x)` is defined by `self.mv(x) == fn(x)`.)

    See also [`lineax.materialise`][], which materialises the whole linear operator
    in memory. (Similar to `.as_matrix()`.)
    """

    fn: Callable[[PyTree[Inexact[Array, "..."]]], PyTree[Inexact[Array, "..."]]]
    input_structure: FlatPyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    @eqxi.doc_remove_args("closure_convert")
    def __init__(
        self,
        fn: Callable[[PyTree[Inexact[Array, "..."]]], PyTree[Inexact[Array, "..."]]],
        input_structure: PyTree[jax.ShapeDtypeStruct],
        tags: object | Iterable[object] = (),
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
        input_structure = inexact_structure(input_structure)
        if closure_convert:
            fn = eqx.filter_closure_convert(fn, input_structure)
        self.fn = fn
        self.input_structure = jtu.tree_flatten(input_structure)
        self.tags = as_frozenset(tags)

    def mv(self, vector):
        return self.fn(vector)

    def as_matrix(self):
        return materialise(self).as_matrix()

    def transpose(self):
        if is_symmetric(self):
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


def try_structured_materialise(
    operator: AbstractLinearOperator,
) -> AbstractLinearOperator:
    """Try to materialise to a structured operator.

    Returns a structured operator
    (`DiagonalLinearOperator`/`TridiagonalLinearOperator`/`CirculantLinearOperator`)
    if the operator is known to have the required structure (e.g through tags),
    otherwise returns the original operator unchanged. The resulting operator
    preserves the input/output structure of the original operator.
    """
    if is_diagonal(operator):
        diag_flat = diagonal(operator)
        _, unravel = eqx.filter_eval_shape(jfu.ravel_pytree, operator.in_structure())
        diag_pytree = unravel(diag_flat)
        return DiagonalLinearOperator(diag_pytree)
    # TridiagonalLinearOperator only supports flat in and out structures
    if (
        is_tridiagonal(operator)
        and isinstance(operator.in_structure(), jax.ShapeDtypeStruct)
        and isinstance(operator.out_structure(), jax.ShapeDtypeStruct)
    ):
        return TridiagonalLinearOperator(*tridiagonal(operator))
    if (
        is_circulant(operator)
        and isinstance(operator.in_structure(), jax.ShapeDtypeStruct)
        and isinstance(operator.out_structure(), jax.ShapeDtypeStruct)
    ):
        return CirculantLinearOperator(first_column(operator))
    return operator


# linearise


@linearise.register(MatrixLinearOperator)
@linearise.register(PyTreeLinearOperator)
@linearise.register(FunctionLinearOperator)
def _(operator):
    return operator


@linearise.register(JacobianLinearOperator)
def _(operator):
    fn = _NoAuxIn(operator.fn, operator.args)
    if operator.jac == "bwd":
        # For backward mode, use VJP + linear_transpose.
        # This works even with custom_vjp functions that don't support forward-mode AD.
        _, vjp_fn = jax.vjp(fn, operator.x)
        if is_symmetric(operator):
            # For symmetric: J = J.T, so vjp directly gives J @ v
            lin = _Unwrap(vjp_fn)
        else:
            # Transpose the VJP to get J @ v from J.T @ v
            lin = _Unwrap(
                jax.linear_transpose(lambda g: vjp_fn(g)[0], operator.out_structure())
            )
    else:  # "fwd" or None
        _, lin = jax.linearize(fn, operator.x)
    return FunctionLinearOperator(lin, operator.in_structure(), operator.tags)


# materialise


@materialise.register(MatrixLinearOperator)
@materialise.register(PyTreeLinearOperator)
def _(operator):
    return try_structured_materialise(operator)


@materialise.register(JacobianLinearOperator)
def _(operator):
    maybe_structured_op = try_structured_materialise(operator)
    if maybe_structured_op is not operator:
        return maybe_structured_op
    fn = _NoAuxIn(operator.fn, operator.args)
    jac = jacobian(
        fn,
        operator.in_size(),
        operator.out_size(),
        holomorphic=any(jnp.iscomplexobj(xi) for xi in jtu.tree_leaves(operator.x)),
        jac=operator.jac,
    )(operator.x)
    return PyTreeLinearOperator(jac, operator.out_structure(), operator.tags)


@materialise.register(FunctionLinearOperator)
def _(operator):
    maybe_structured_op = try_structured_materialise(operator)
    if maybe_structured_op is not operator:
        return maybe_structured_op
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


def _leaf_from_keypath(pytree: PyTree, keypath: jtu.KeyPath) -> Array:
    """Extract the leaf from a pytree at the given keypath."""
    for path, leaf in jtu.tree_leaves_with_path(pytree):
        if path == keypath:
            return leaf
    raise ValueError(f"Leaf not found at keypath {keypath}")


@diagonal.register(MatrixLinearOperator)
def _(operator):
    return jnp.diag(operator.as_matrix())


@diagonal.register(PyTreeLinearOperator)
def _(operator):
    if is_diagonal(operator):

        def extract_diag(keypath, struct, subpytree):
            block = _leaf_from_keypath(subpytree, keypath)
            return jnp.diag(block.reshape(struct.size, struct.size))

        diags = jtu.tree_map_with_path(
            extract_diag, operator.out_structure(), operator.pytree
        )
        return jnp.concatenate(jtu.tree_leaves(diags))
    else:
        return jnp.diag(operator.as_matrix())


@diagonal.register(JacobianLinearOperator)
@diagonal.register(FunctionLinearOperator)
def _(operator):
    if is_diagonal(operator):
        with jax.ensure_compile_time_eval():
            basis = jtu.tree_map(
                lambda s: jnp.ones(s.shape, s.dtype), operator.in_structure()
            )
        diag_as_pytree = operator.mv(basis)
        diag, _ = jfu.ravel_pytree(diag_as_pytree)
        return diag
    return diagonal(materialise(operator))


# tridiagonal


@tridiagonal.register(MatrixLinearOperator)
@tridiagonal.register(PyTreeLinearOperator)
def _(operator):
    matrix = operator.as_matrix()
    assert matrix.ndim == 2
    main_diagonal = jnp.diagonal(matrix, offset=0)
    upper_diagonal = jnp.diagonal(matrix, offset=1)
    lower_diagonal = jnp.diagonal(matrix, offset=-1)
    return main_diagonal, lower_diagonal, upper_diagonal


@tridiagonal.register(JacobianLinearOperator)
@tridiagonal.register(FunctionLinearOperator)
def _(operator):
    if is_tridiagonal(operator):
        with jax.ensure_compile_time_eval():
            flat, unravel = strip_weak_dtype(
                eqx.filter_eval_shape(jfu.ravel_pytree, operator.in_structure())
            )

            basis = jnp.zeros((3, flat.size), dtype=flat.dtype)
            for i in range(3):
                basis = basis.at[i, i::3].set(1.0)

            basis = jax.vmap(unravel)(basis)

            coloring = jnp.arange(flat.size) % 3

        compressed_as_pytree = jax.vmap(operator.mv)(basis)
        compressed_flat = jax.vmap(lambda x: jfu.ravel_pytree(x)[0])(
            compressed_as_pytree
        )

        # unique_indices propagates through linear_transpose to set unique_indices=True
        # on the scatter, allowing assignment rather than accumulation.
        rows = jnp.arange(flat.size)
        diag = compressed_flat.at[coloring, rows].get(
            wrap_negative_indices=False, unique_indices=True
        )
        lower_diag = compressed_flat.at[coloring[:-1], rows[1:]].get(
            wrap_negative_indices=False, unique_indices=True
        )
        upper_diag = compressed_flat.at[coloring[1:], rows[:-1]].get(
            wrap_negative_indices=False, unique_indices=True
        )

        return diag, lower_diag, upper_diag
    matrix = operator.as_matrix()
    assert matrix.ndim == 2
    main_diagonal = jnp.diagonal(matrix, offset=0)
    upper_diagonal = jnp.diagonal(matrix, offset=1)
    lower_diagonal = jnp.diagonal(matrix, offset=-1)
    return main_diagonal, lower_diagonal, upper_diagonal


# first_column


@first_column.register(MatrixLinearOperator)
@first_column.register(PyTreeLinearOperator)
def _(operator):
    return operator.as_matrix()[:, 0]


# checks


@is_symmetric.register(MatrixLinearOperator)
@is_symmetric.register(PyTreeLinearOperator)
@is_symmetric.register(JacobianLinearOperator)
@is_symmetric.register(FunctionLinearOperator)
def _(operator):
    # Symmetric (A = A^T) if explicitly tagged symmetric or diagonal
    if symmetric_tag in operator.tags or diagonal_tag in operator.tags:
        return True
    # PSD/NSD/Hermitian imply A = A^T only for real dtypes
    if (
        positive_semidefinite_tag in operator.tags
        or negative_semidefinite_tag in operator.tags
        or hermitian_tag in operator.tags
    ):
        return has_real_dtype(operator)
    return False


@is_hermitian.register(MatrixLinearOperator)
@is_hermitian.register(PyTreeLinearOperator)
@is_hermitian.register(JacobianLinearOperator)
@is_hermitian.register(FunctionLinearOperator)
def _(operator):
    if (
        hermitian_tag in operator.tags
        or positive_semidefinite_tag in operator.tags
        or negative_semidefinite_tag in operator.tags
    ):
        return True
    if symmetric_tag in operator.tags or diagonal_tag in operator.tags:
        return has_real_dtype(operator)
    return False


@is_diagonal.register(MatrixLinearOperator)
@is_diagonal.register(PyTreeLinearOperator)
@is_diagonal.register(JacobianLinearOperator)
@is_diagonal.register(FunctionLinearOperator)
def _(operator):
    return diagonal_tag in operator.tags or (
        operator.in_size() == 1 and operator.out_size() == 1
    )


@is_tridiagonal.register(MatrixLinearOperator)
@is_tridiagonal.register(PyTreeLinearOperator)
@is_tridiagonal.register(JacobianLinearOperator)
@is_tridiagonal.register(FunctionLinearOperator)
def _(operator):
    return tridiagonal_tag in operator.tags or diagonal_tag in operator.tags


# The remaining checks are true iff the operator carries the corresponding tag.
for check, tag in (
    (has_unit_diagonal, unit_diagonal_tag),
    (is_lower_triangular, lower_triangular_tag),
    (is_upper_triangular, upper_triangular_tag),
    (is_positive_semidefinite, positive_semidefinite_tag),
    (is_negative_semidefinite, negative_semidefinite_tag),
    (is_circulant, circulant_tag),
):

    @check.register(MatrixLinearOperator)
    @check.register(PyTreeLinearOperator)
    @check.register(JacobianLinearOperator)
    @check.register(FunctionLinearOperator)
    def _(operator, tag=tag):
        return tag in operator.tags


# conj


@conj.register(MatrixLinearOperator)
def _(operator):
    return MatrixLinearOperator(operator.matrix.conj(), operator.tags)


@conj.register(PyTreeLinearOperator)
def _(operator):
    pytree_conj = jtu.tree_map(lambda x: x.conj(), operator.pytree)
    return PyTreeLinearOperator(pytree_conj, operator.out_structure(), operator.tags)


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
