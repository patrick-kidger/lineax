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
from typing import Any, ClassVar, Optional
from typing_extensions import TYPE_CHECKING, TypeAlias

import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, PyTree, Scalar


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox.internal import AbstractClassVar

from .._misc import resolve_rcond, structure_equal, tree_where
from .._norm import max_norm, tree_dot
from .._operator import (
    AbstractLinearOperator,
    conj,
    is_negative_semidefinite,
    is_positive_semidefinite,
    linearise,
)
from .._solution import RESULTS
from .._solve import AbstractLinearSolver
from .misc import preconditioner_and_y0


_CGState: TypeAlias = tuple[AbstractLinearOperator, bool]


# TODO(kidger): this is pretty slow to compile.
# - CG evaluates `operator.mv` three times.
# - Normal CG evaluates `operator.mv` seven (!) times.
# Possibly this can be cheapened a bit somehow?
class _AbstractCG(AbstractLinearSolver[_CGState], strict=True):
    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm
    stabilise_every: Optional[int] = 10
    max_steps: Optional[int] = None

    _normal: AbstractClassVar[bool]

    def __check_init__(self):
        if isinstance(self.rtol, (int, float)) and self.rtol < 0:
            raise ValueError("Tolerances must be non-negative.")
        if isinstance(self.atol, (int, float)) and self.atol < 0:
            raise ValueError("Tolerances must be non-negative.")

        if isinstance(self.atol, (int, float)) and isinstance(self.rtol, (int, float)):
            if self.atol == 0 and self.rtol == 0 and self.max_steps is None:
                raise ValueError(
                    "Must specify `rtol`, `atol`, or `max_steps` (or some combination "
                    "of all three)."
                )

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        del options
        is_nsd = is_negative_semidefinite(operator)
        if not self._normal:
            if not structure_equal(operator.in_structure(), operator.out_structure()):
                raise ValueError(
                    "`CG()` may only be used for linear solves with " "square matrices."
                )
            if not (is_positive_semidefinite(operator) | is_nsd):
                raise ValueError(
                    "`CG()` may only be used for positive "
                    "or negative definite linear operators"
                )
            if is_nsd:
                operator = -operator
        return operator, is_nsd

    # This differs from jax.scipy.sparse.linalg.cg in:
    # 1. Every few steps we calculate the residual directly, rather than by cheaply
    #    using the existing quantities. This improves numerical stability.
    # 2. We use a more sophisticated termination condition. To begin with we have an
    #    rtol and atol in the conventional way, inducing a vector-valued scale. This is
    #    then checked in both the `y` and `b` domains (for `Ay = b`).
    # 3. We return the number of steps, and whether or not the solve succeeded, as
    #    additional information.
    # 4. We don't try to support complex numbers. (Yet.)
    def compute(
        self, state: _CGState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        operator, is_nsd = state
        if self._normal:
            # Linearise if JacobianLinearOperator, to avoid computing the forward
            # pass separately for mv and transpose_mv.
            # This choice is "fast by default", even at the expense of memory.
            # If a downstream user wants to avoid this then they can call
            # ```
            # linear_solve(
            #     conj(operator.T) @ operator, operator.mv(b), solver=CG()
            # )
            # ```
            # directly.
            operator = linearise(operator)

            _mv = operator.mv
            _transpose_mv = conj(operator.transpose()).mv

            def mv(vector: PyTree) -> PyTree:
                return _transpose_mv(_mv(vector))

            vector = _transpose_mv(vector)
        else:
            mv = operator.mv
        preconditioner, y0 = preconditioner_and_y0(operator, vector, options)
        leaves, _ = jtu.tree_flatten(vector)
        size = sum(leaf.size for leaf in leaves)
        if self.max_steps is None:
            max_steps = 10 * size  # Copied from SciPy!
        else:
            max_steps = self.max_steps
        r0 = (vector**ω - mv(y0) ** ω).ω
        p0 = preconditioner.mv(r0)
        gamma0 = tree_dot(p0, r0)
        rcond = resolve_rcond(None, size, size, jnp.result_type(*leaves))
        initial_value = (
            ω(y0).call(lambda x: jnp.full_like(x, jnp.inf)).ω,
            y0,
            r0,
            p0,
            gamma0,
            0,
        )
        has_scale = not (
            isinstance(self.atol, (int, float))
            and isinstance(self.rtol, (int, float))
            and self.atol == 0
            and self.rtol == 0
        )
        if has_scale:
            b_scale = (self.atol + self.rtol * ω(vector).call(jnp.abs)).ω

        def not_converged(r, diff, y):
            # The primary tolerance check.
            # Given Ay=b, then we have to be doing better than `scale` in both
            # the `y` and the `b` spaces.
            if has_scale:
                with jax.numpy_dtype_promotion("standard"):
                    y_scale = (self.atol + self.rtol * ω(y).call(jnp.abs)).ω
                    norm1 = self.norm((r**ω / b_scale**ω).ω)  # pyright: ignore
                    norm2 = self.norm((diff**ω / y_scale**ω).ω)
                return (norm1 > 1) | (norm2 > 1)
            else:
                return True

        def cond_fun(value):
            diff, y, r, _, gamma, step = value
            out = gamma > 0
            out = out & (step < max_steps)
            out = out & not_converged(r, diff, y)
            return out

        def body_fun(value):
            _, y, r, p, gamma, step = value
            mat_p = mv(p)
            inner_prod = tree_dot(mat_p, p)
            alpha = gamma / inner_prod
            alpha = tree_where(
                jnp.abs(inner_prod) > 100 * rcond * jnp.abs(gamma), alpha, jnp.nan
            )
            diff = (alpha * p**ω).ω
            y = (y**ω + diff**ω).ω
            step = step + 1

            # E.g. see B.2 of
            # https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
            # We compute the residual the "expensive" way every now and again, so as to
            # correct numerical rounding errors.
            def stable_r():
                return (vector**ω - mv(y) ** ω).ω

            def cheap_r():
                return (r**ω - alpha * mat_p**ω).ω

            if self.stabilise_every == 1:
                r = stable_r()
            elif self.stabilise_every is None:
                r = cheap_r()
            else:
                stable_step = (eqxi.unvmap_max(step) % self.stabilise_every) == 0
                stable_step = eqxi.nonbatchable(stable_step)
                r = lax.cond(stable_step, stable_r, cheap_r)

            z = preconditioner.mv(r)
            gamma_prev = gamma
            gamma = tree_dot(z, r)
            beta = gamma / gamma_prev
            p = (z**ω + beta * p**ω).ω
            return diff, y, r, p, gamma, step

        _, solution, _, _, _, num_steps = lax.while_loop(
            cond_fun, body_fun, initial_value
        )

        if (self.max_steps is None) or (max_steps < self.max_steps):
            result = RESULTS.where(
                num_steps == max_steps,
                RESULTS.singular,
                RESULTS.successful,
            )
        else:
            result = RESULTS.where(
                num_steps == max_steps,
                RESULTS.max_steps_reached if has_scale else RESULTS.successful,
                RESULTS.successful,
            )

        if is_nsd and not self._normal:
            solution = -(solution**ω).ω
        stats = {"num_steps": num_steps, "max_steps": self.max_steps}
        return solution, result, stats

    def transpose(self, state: _CGState, options: dict[str, Any]):
        del options
        psd_op, is_nsd = state
        transpose_state = psd_op.transpose(), is_nsd
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _CGState, options: dict[str, Any]):
        del options
        psd_op, is_nsd = state
        conj_state = conj(psd_op), is_nsd
        conj_options = {}
        return conj_state, conj_options


class CG(_AbstractCG, strict=True):
    """Conjugate gradient solver for linear systems.

    The operator should be positive or negative definite.

    Equivalent to `scipy.sparse.linalg.cg`.

    This supports the following `options` (as passed to
    `lx.linear_solve(..., options=...)`).

    - `preconditioner`: A positive definite [`lineax.AbstractLinearOperator`][]
        to be used as preconditioner. Defaults to
        [`lineax.IdentityLinearOperator`][].
    - `y0`: The initial estimate of the solution to the linear system. Defaults to all
        zeros.

    !!! info


    """

    _normal: ClassVar[bool] = False

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False


class NormalCG(_AbstractCG, strict=True):
    """Conjugate gradient applied to the normal equations:

    `A^T A = A^T b`

    of a system of linear equations. Note that this squares the condition
    number, so it is not recommended. This is a fast but potentially inaccurate
    method, especially in 32 bit floating point precision.

    This can handle nonsquare operators provided they are full-rank.

    This supports the following `options` (as passed to
    `lx.linear_solve(..., options=...)`).

    - `preconditioner`: A positive definite [`lineax.AbstractLinearOperator`][]
        to be used as preconditioner. Defaults to
        [`lineax.IdentityLinearOperator`][].
    - `y0`: The initial estimate of the solution to the linear system. Defaults to all
        zeros.

    !!! info


    """

    _normal: ClassVar[bool] = True

    def allow_dependent_columns(self, operator):
        rows = operator.out_size()
        columns = operator.in_size()
        return columns > rows

    def allow_dependent_rows(self, operator):
        rows = operator.out_size()
        columns = operator.in_size()
        return rows > columns


CG.__init__.__doc__ = r"""**Arguments:**

- `rtol`: Relative tolerance for terminating solve.
- `atol`: Absolute tolerance for terminating solve.
- `norm`: The norm to use when computing whether the error falls within the tolerance.
    Defaults to the max norm.
- `stabilise_every`: The conjugate gradient is an iterative method that produces
    candidate solutions $x_1, x_2, \ldots$, and terminates once $r_i = \| Ax_i - b \|$
    is small enough. For computational efficiency, the values $r_i$ are computed using
    other internal quantities, and not by directly evaluating the formula above.
    However, this computation of $r_i$ is susceptible to drift due to limited
    floating-point precision. Every `stabilise_every` steps, then $r_i$ is computed
    directly using the formula above, in order to stabilise the computation.
- `max_steps`: The maximum number of iterations to run the solver for. If more steps
    than this are required, then the solve is halted with a failure.
"""

NormalCG.__init__.__doc__ = r"""**Arguments:**

- `rtol`: Relative tolerance for terminating solve.
- `atol`: Absolute tolerance for terminating solve.
- `norm`: The norm to use when computing whether the error falls within the tolerance.
    Defaults to the max norm.
- `stabilise_every`: The conjugate gradient is an iterative method that produces
    candidate solutions $x_1, x_2, \ldots$, and terminates once $r_i = \| Ax_i - b \|$
    is small enough. For computational efficiency, the values $r_i$ are computed using
    other internal quantities, and not by directly evaluating the formula above.
    However, this computation of $r_i$ is susceptible to drift due to limited
    floating-point precision. Every `stabilise_every` steps, then $r_i$ is computed
    directly using the formula above, in order to stabilise the computation.
- `max_steps`: The maximum number of iterations to run the solver for. If more steps
    than this are required, then the solve is halted with a failure.
"""
