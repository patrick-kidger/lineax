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

from collections.abc import Callable

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, Bool, PyTree  # pyright:ignore


def tree_where(
    pred: Bool[ArrayLike, ""], true: PyTree[ArrayLike], false: PyTree[ArrayLike]
) -> PyTree[Array]:
    keep = lambda a, b: jnp.where(pred, a, b)
    return jtu.tree_map(keep, true, false)


def resolve_rcond(rcond, n, m, dtype):
    if rcond is None:
        return jnp.finfo(dtype).eps * max(n, m)
    else:
        return jnp.where(rcond < 0, jnp.finfo(dtype).eps, rcond)


class NoneAux(eqx.Module, strict=True):
    fn: Callable

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs), None


def jacobian(fn, in_size, out_size, holomorphic=False, has_aux=False, jac=None):
    if jac is None:
        # Heuristic for which is better in each case
        # These could probably be tuned a lot more.
        jac_fwd = (in_size < 100) or (in_size <= 1.5 * out_size)
    elif jac == "fwd":
        jac_fwd = True
    elif jac == "bwd":
        jac_fwd = False
    else:
        raise ValueError("`jac` should either be None, 'fwd', or 'bwd'.")
    if jac_fwd:
        return jax.jacfwd(fn, holomorphic=holomorphic, has_aux=has_aux)
    else:
        return jax.jacrev(fn, holomorphic=holomorphic, has_aux=has_aux)


def _asarray(dtype, x):
    return jnp.asarray(x, dtype=dtype)


# Work around JAX issue #15676
_asarray = jax.custom_jvp(_asarray, nondiff_argnums=(0,))


@_asarray.defjvp
def _asarray_jvp(dtype, x, tx):
    (x,) = x
    (tx,) = tx
    return _asarray(dtype, x), _asarray(dtype, tx)


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def inexact_asarray(x):
    dtype = jnp.result_type(x)
    if not jnp.issubdtype(jnp.result_type(x), jnp.inexact):
        dtype = default_floating_dtype()
    return _asarray(dtype, x)


def complex_to_real_dtype(dtype):
    return jnp.finfo(dtype).dtype
