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
from typing import Any, NewType

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, PyTree, Shaped

from .._operator import (
    AbstractLinearOperator,
    IdentityLinearOperator,
    is_positive_semidefinite,
)


def preconditioner_and_y0(
    operator: AbstractLinearOperator, vector: PyTree[Array], options: dict[str, Any]
):
    structure = operator.in_structure()
    try:
        preconditioner = options["preconditioner"]
    except KeyError:
        preconditioner = IdentityLinearOperator(structure)
    else:
        if not isinstance(preconditioner, AbstractLinearOperator):
            raise ValueError("The preconditioner must be a linear operator.")
        if preconditioner.in_structure() != structure:
            raise ValueError(
                "The preconditioner must have `in_structure` that matches the "
                "operator's `in_strucure`."
            )
        if preconditioner.out_structure() != structure:
            raise ValueError(
                "The preconditioner must have `out_structure` that matches the "
                "operator's `in_structure`."
            )
        if not is_positive_semidefinite(preconditioner):
            raise ValueError("The preconditioner must be positive definite.")
    try:
        y0 = options["y0"]
    except KeyError:
        y0 = jtu.tree_map(jnp.zeros_like, vector)
    else:
        if jax.eval_shape(lambda: y0) != jax.eval_shape(lambda: vector):
            raise ValueError(
                "`y0` must have the same structure, shape, and dtype as `vector`"
            )
    return preconditioner, y0


PackedStructures = NewType("PackedStructures", eqxi.Static)


def pack_structures(operator: AbstractLinearOperator) -> PackedStructures:
    structures = operator.out_structure(), operator.in_structure()
    leaves, treedef = jtu.tree_flatten(structures)  # handle nonhashable pytrees
    return PackedStructures(eqxi.Static((leaves, treedef)))


def ravel_vector(
    pytree: PyTree[Array], packed_structures: PackedStructures
) -> Shaped[Array, " size"]:
    leaves, treedef = packed_structures.value
    out_structure, _ = jtu.tree_unflatten(treedef, leaves)
    # `is` in case `tree_equal` returns a Tracer.
    if eqx.tree_equal(jax.eval_shape(lambda: pytree), out_structure) is not True:
        raise ValueError("pytree does not match out_structure")
    # not using `ravel_pytree` as that doesn't come with guarantees about order
    leaves = jtu.tree_leaves(pytree)
    dtype = jnp.result_type(*leaves)
    return jnp.concatenate([x.astype(dtype).reshape(-1) for x in leaves])


def unravel_solution(
    solution: Shaped[Array, " size"], packed_structures: PackedStructures
) -> PyTree[Array]:
    leaves, treedef = packed_structures.value
    _, in_structure = jtu.tree_unflatten(treedef, leaves)
    leaves, treedef = jtu.tree_flatten(in_structure)
    sizes = np.cumsum([math.prod(x.shape) for x in leaves[:-1]])
    split = jnp.split(solution, sizes)
    assert len(split) == len(leaves)
    shaped = [x.reshape(y.shape).astype(y.dtype) for x, y in zip(split, leaves)]
    return jtu.tree_unflatten(treedef, shaped)


def transpose_packed_structures(
    packed_structures: PackedStructures,
) -> PackedStructures:
    leaves, treedef = packed_structures.value
    out_structure, in_structure = jtu.tree_unflatten(treedef, leaves)
    leaves, treedef = jtu.tree_flatten((in_structure, out_structure))
    return PackedStructures(eqxi.Static((leaves, treedef)))
