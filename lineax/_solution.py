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

from typing import Any

import equinox as eqx
import equinox.internal as eqxi
from jaxtyping import Array, ArrayLike, PyTree


_singular_msg = """
A linear solver returned non-finite (NaN or inf) output. This usually means that an
operator was not well-posed, and that its solver does not support this.

If you are trying solve a linear least-squares problem then you should pass
`solver=AutoLinearSolver(well_posed=False)`. By default `lineax.linear_solve`
assumes that the operator is square and nonsingular.

If you *were* expecting this solver to work with this operator, then it may be because:

(a) the operator is singular, and your code has a bug; or

(b) the operator was nearly singular (i.e. it had a high condition number:
    `jnp.linalg.cond(operator.as_matrix())` is large), and the solver suffered from
    numerical instability issues; or

(c) the operator is declared to exhibit a certain property (e.g. positive definiteness)
    that is does not actually satisfy.
""".strip()


_nonfinite_msg = """
A linear solver received non-finite (NaN or inf) input and cannot determine a 
solution. 

This means that you have a bug upstream of Lineax and should check the inputs to 
`lineax.linear_solve` for non-finite values.
""".strip()


class RESULTS(eqxi.Enumeration):
    successful = ""
    max_steps_reached = (
        "The maximum number of solver steps was reached. Try increasing `max_steps`."
    )
    singular = _singular_msg
    breakdown = (
        "A form of iterative breakdown has occured in a linear solve. "
        "Try using a different solver for this problem or increase `restart` "
        "if using GMRES."
    )
    stagnation = (
        "A stagnation in an iterative linear solve has occurred. Try increasing "
        "`stagnation_iters` or `restart`."
    )
    nonfinite_input = _nonfinite_msg


class Solution(eqx.Module, strict=True):
    """The solution to a linear solve.

    **Attributes:**

    - `value`: The solution to the solve.
    - `result`: An integer representing whether the solve was successful or not. This
        can be converted into a human-readable error message via
        `lineax.RESULTS[result]`.
    - `stats`: Statistics about the solver, e.g. the number of steps that were required.
    - `state`: The internal state of the solver. The meaning of this is specific to each
        solver.
    """

    value: PyTree[Array]
    result: RESULTS
    stats: dict[str, PyTree[ArrayLike]]
    state: PyTree[Any]
