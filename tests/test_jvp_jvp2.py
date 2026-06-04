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

import equinox as eqx
import jax.numpy as jnp
import pytest

from .helpers import (
    all_test_cases,
    jvp_jvp_impl,
    make_jac_operator,
    make_matrix_operator,
)


# Workaround for https://github.com/jax-ml/jax/issues/27201
@pytest.fixture(autouse=True)
def _clear_cache():
    eqx.clear_caches()


@pytest.mark.parametrize("solver, tags, rows, cols, full_rank", all_test_cases())
@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("use_state", (True, False))
@pytest.mark.parametrize("dtype", (jnp.complex128,))
def test_jvp_jvp(
    getkey,
    solver,
    tags,
    rows,
    cols,
    full_rank,
    make_operator,
    use_state,
    dtype,
):
    jvp_jvp_impl(
        getkey,
        solver,
        tags,
        make_operator,
        use_state,
        rows,
        cols,
        full_rank,
        dtype,
    )
