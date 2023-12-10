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

# Core JAX has some numerical issues with their lstsq gradients.
# See https://github.com/google/jax/issues/14868
# This demonstrates that we don't have the same issue!

import sys

import jax
import jax.numpy as jnp
import lineax as lx


sys.path.append("../tests")
from helpers import finite_difference_jvp  # pyright: ignore


a_primal = (jnp.eye(3),)
a_tangent = (jnp.zeros((3, 3)),)


def jax_solve(a):
    sol, _, _, _ = jnp.linalg.lstsq(a, jnp.arange(3))  # pyright: ignore
    return sol


def lx_solve(a):
    op = lx.MatrixLinearOperator(a)
    return lx.linear_solve(op, jnp.arange(3)).value


_, true_jvp = finite_difference_jvp(jax_solve, a_primal, a_tangent)
_, jax_jvp = jax.jvp(jax_solve, a_primal, a_tangent)
_, lx_jvp = jax.jvp(lx_solve, a_primal, a_tangent)
assert jnp.isnan(jax_jvp).all()
assert jnp.allclose(true_jvp, lx_jvp)
