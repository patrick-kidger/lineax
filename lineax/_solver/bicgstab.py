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
from typing import Any, Optional
from typing_extensions import TypeAlias

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, PyTree

from .._norm import max_norm, tree_dot
from .._operator import AbstractLinearOperator, conj
from .._solution import RESULTS
from .._solve import AbstractLinearSolver
from .misc import preconditioner_and_y0


_BiCGStabState: TypeAlias = AbstractLinearOperator


class BiCGStab(AbstractLinearSolver[_BiCGStabState], strict=True):
    """Biconjugate gradient stabilised method for linear systems.

    The operator should be square.

    Equivalent to `jax.scipy.sparse.linalg.bicgstab`.

    This supports the following `options` (as passed to
    `lx.linear_solve(..., options=...)`).

    - `preconditioner`: A positive definite [`lineax.AbstractLinearOperator`][]
        to be used as a preconditioner. Defaults to
        [`lineax.IdentityLinearOperator`][].
    - `y0`: The initial estimate of the solution to the linear system. Defaults to all
        zeros.
    """

    rtol: float
    atol: float
    norm: Callable = max_norm
    max_steps: Optional[int] = None

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
        if operator.in_structure() != operator.out_structure():
            raise ValueError(
                "`BiCGstab(..., normal=False)` may only be used for linear solves with "
                "square matrices."
            )
        return operator

    def compute(
        self, state: _BiCGStabState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        operator = state
        preconditioner, y0 = preconditioner_and_y0(operator, vector, options)
        leaves, _ = jtu.tree_flatten(vector)
        if self.max_steps is None:
            size = sum(leaf.size for leaf in leaves)
            max_steps = 10 * size
        else:
            max_steps = self.max_steps
        has_scale = not (
            isinstance(self.atol, (int, float))
            and isinstance(self.rtol, (int, float))
            and self.atol == 0
            and self.rtol == 0
        )
        if has_scale:
            b_scale = (self.atol + self.rtol * ω(vector).call(jnp.abs)).ω

        # This implementation is the same a jax.scipy.sparse.linalg.bicgstab
        # but with AbstractLinearOperator.
        # We use the notation found on the wikipedia except with y instead of x:
        # https://en.wikipedia.org/wiki/
        # Biconjugate_gradient_stabilized_method#Preconditioned_BiCGSTAB
        # preconditioner in this case is K2^(-1) (i.e., right preconditioning)

        r0 = (vector**ω - operator.mv(y0) ** ω).ω

        def breakdown_occurred(omega, alpha, rho):
            # Empirically, the tolerance checks for breakdown are very tight.
            # These specific tolerances are heuristic.
            if jax.config.jax_enable_x64:  # pyright: ignore
                return (omega == 0.0) | (alpha == 0.0) | (rho == 0.0)
            else:
                return (omega < 1e-16) | (alpha < 1e-16) | (rho < 1e-16)

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

        def cond_fun(carry):
            y, r, alpha, omega, rho, _, _, diff, step = carry
            out = jnp.invert(breakdown_occurred(omega, alpha, rho))
            out = out & not_converged(r, diff, y)
            out = out & (step < max_steps)
            return out

        def body_fun(carry):
            y, r, alpha, omega, rho, p, v, diff, step = carry

            rho_new = tree_dot(r0, r)
            beta = (rho_new / rho) * (alpha / omega)
            p_new = (r**ω + beta * (p**ω - omega * v**ω)).ω

            # TODO(raderj): reduce this to a single operator.mv call
            # by using the scan trick.
            x = preconditioner.mv(p_new)
            v_new = operator.mv(x)

            alpha_new = rho_new / tree_dot(r0, v_new)
            s = (r**ω - alpha_new * v_new**ω).ω

            z = preconditioner.mv(s)
            t = operator.mv(z)

            omega_new = tree_dot(s, t) / tree_dot(t, t)

            diff = (alpha_new * x**ω + omega_new * z**ω).ω
            y_new = (y**ω + diff**ω).ω
            r_new = (s**ω - omega_new * t**ω).ω
            return (
                y_new,
                r_new,
                alpha_new,
                omega_new,
                rho_new,
                p_new,
                v_new,
                diff,
                step + 1,
            )

        p0 = v0 = jtu.tree_map(jnp.zeros_like, vector)
        alpha = omega = rho = jnp.array(1.0)

        init_carry = (
            y0,
            r0,
            alpha,
            omega,
            rho,
            p0,
            v0,
            ω(y0).call(lambda x: jnp.full_like(x, jnp.inf)).ω,
            0,
        )
        solution, residual, alpha, omega, rho, _, _, diff, num_steps = lax.while_loop(
            cond_fun, body_fun, init_carry
        )

        if self.max_steps is None:
            result = RESULTS.where(
                (num_steps == max_steps), RESULTS.singular, RESULTS.successful
            )
        else:
            result = RESULTS.where(
                (num_steps == self.max_steps),
                RESULTS.max_steps_reached if has_scale else RESULTS.successful,
                RESULTS.successful,
            )
        # breakdown is only an issue if we did not converge
        breakdown = breakdown_occurred(omega, alpha, rho) & not_converged(
            residual, diff, solution
        )
        result = RESULTS.where(breakdown, RESULTS.breakdown, result)

        stats = {"num_steps": num_steps, "max_steps": self.max_steps}
        return solution, result, stats

    def transpose(self, state: _BiCGStabState, options: dict[str, Any]):
        del options
        operator = state
        transpose_options = {}
        return operator.transpose(), transpose_options

    def conj(self, state: _BiCGStabState, options: dict[str, Any]):
        del options
        operator = state
        conj_options = {}
        return conj(operator), conj_options

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False


BiCGStab.__init__.__doc__ = r"""**Arguments:**

- `rtol`: Relative tolerance for terminating solve.
- `atol`: Absolute tolerance for terminating solve.
- `norm`: The norm to use when computing whether the error falls within the tolerance.
    Defaults to the max norm.
- `max_steps`: The maximum number of iterations to run the solver for. If more steps
    than this are required, then the solve is halted with a failure.
"""
