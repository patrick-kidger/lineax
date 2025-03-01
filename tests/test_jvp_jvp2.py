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

import jax.numpy as jnp
import pytest

from .helpers import (
    construct_matrix,
    construct_singular_matrix,
    jvp_jvp_impl,
    make_jac_operator,
    make_matrix_operator,
    solvers_tags_pseudoinverse,
)


@pytest.mark.parametrize("solver, tags, pseudoinverse", solvers_tags_pseudoinverse)
@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("use_state", (True, False))
@pytest.mark.parametrize("make_matrix", (construct_matrix, construct_singular_matrix))
@pytest.mark.parametrize("dtype", (jnp.complex128,))
def test_jvp_jvp(
    getkey, solver, tags, pseudoinverse, make_operator, use_state, make_matrix, dtype
):
    jvp_jvp_impl(
        getkey,
        solver,
        tags,
        pseudoinverse,
        make_operator,
        use_state,
        make_matrix,
        dtype,
    )
