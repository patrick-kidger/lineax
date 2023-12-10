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
import math

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Inexact, PyTree, Scalar


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def tree_dot(tree1: PyTree[ArrayLike], tree2: PyTree[ArrayLike]) -> Inexact[Array, ""]:
    """Compute the dot product of two pytrees of arrays with the same pytree
    structure."""
    leaves1, treedef1 = jtu.tree_flatten(tree1)
    leaves2, treedef2 = jtu.tree_flatten(tree2)
    if treedef1 != treedef2:
        raise ValueError("trees must have the same structure")
    assert len(leaves1) == len(leaves2)
    dots = []
    for leaf1, leaf2 in zip(leaves1, leaves2):
        dots.append(
            jnp.dot(
                jnp.reshape(leaf1, -1),
                jnp.conj(leaf2).reshape(-1),
                precision=jax.lax.Precision.HIGHEST,  # pyright: ignore
            )
        )
    if len(dots) == 0:
        return jnp.array(0, default_floating_dtype())
    else:
        return ft.reduce(jnp.add, dots)


def sum_squares(x: PyTree[ArrayLike]) -> Scalar:
    """Computes the square of the L2 norm of a PyTree of arrays.

    Considering the input `x` as a flat vector `(x_1, ..., x_n)`, then this computes
    `Σ_i x_i^2`
    """
    return tree_dot(x, x).real


@jax.custom_jvp
def two_norm(x: PyTree[ArrayLike]) -> Scalar:
    """Computes the L2 norm of a PyTree of arrays.

    Considering the input `x` as a flat vector `(x_1, ..., x_n)`, then this computes
    `sqrt(Σ_i x_i^2)`
    """
    leaves = jtu.tree_leaves(x)
    size = sum([jnp.size(xi) for xi in leaves])
    if size == 1:
        # Avoid needless squaring-and-then-rooting.
        for leaf in leaves:
            if jnp.size(leaf) == 1:
                return jnp.abs(jnp.reshape(leaf, ()))
        else:
            assert False
    else:
        return jnp.sqrt(sum_squares(x))


@two_norm.defjvp
def _two_norm_jvp(x, tx):
    (x,) = x
    (tx,) = tx
    out = two_norm(x)
    # Get zero gradient, rather than NaN gradient, in these cases.
    pred = (out == 0) | jnp.isinf(out)
    denominator = jnp.where(pred, 1, out)
    # We could also switch the dot and the division.
    # This approach is a bit more expensive (more divisions), but should be more
    # numerically stable (`x` and `denominator` should be of the same scale; `tx` is of
    # unknown scale).
    t_out = tree_dot((x**ω / denominator).ω, tx).real
    t_out = jnp.where(pred, 0, t_out)
    return out, t_out


def rms_norm(x: PyTree[ArrayLike]) -> Scalar:
    """Compute the RMS (root-mean-squared) norm of a PyTree of arrays.

    This is the same as the L2 norm, averaged by the size of the input `x`. Considering
    the input `x` as a flat vector `(x_1, ..., x_n)`, then this computes
    `sqrt((Σ_i x_i^2)/n)`
    """
    leaves = jtu.tree_leaves(x)
    size = sum([jnp.size(xi) for xi in leaves])
    if size == 0:
        if len(leaves) == 0:
            dtype = default_floating_dtype()
        else:
            dtype = jnp.result_type(*leaves)
        return jnp.array(0.0, dtype)
    else:
        return two_norm(x) / math.sqrt(size)


def max_norm(x: PyTree[ArrayLike]) -> Scalar:
    """Compute the L-infinity norm of a PyTree of arrays.

    This is the largest absolute elementwise value. Considering the input `x` as a flat
    vector `(x_1, ..., x_n)`, then this computes `max_i |x_i|`.
    """
    leaves = jtu.tree_leaves(x)
    leaf_maxes = [jnp.max(jnp.abs(xi)) for xi in leaves if jnp.size(xi) > 0]
    if len(leaf_maxes) == 0:
        if len(leaves) == 0:
            dtype = default_floating_dtype()
        else:
            dtype = jnp.result_type(*leaves)
        return jnp.array(0.0, dtype)
    else:
        out = ft.reduce(jnp.maximum, leaf_maxes)
        return _zero_grad_at_zero(out)


@jax.custom_jvp
def _zero_grad_at_zero(x):
    return x


@_zero_grad_at_zero.defjvp
def _zero_grad_at_zero_jvp(primals, tangents):
    (out,) = primals
    (t_out,) = tangents
    return out, jnp.where(out == 0, 0, t_out)
