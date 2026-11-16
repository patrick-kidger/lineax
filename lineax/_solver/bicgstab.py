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
from typing import Any, TypeAlias

import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, PyTree

from .._misc import tree_where
from .._norm import max_norm, tree_dot, two_norm
from .._operator import AbstractLinearOperator, conj, linearise
from .._solution import RESULTS
from .base import AbstractLinearSolver
from .misc import preconditioner_and_y0


_BiCGStabState: TypeAlias = AbstractLinearOperator


class BiCGStab(AbstractLinearSolver[_BiCGStabState]):
    """Biconjugate gradient stabilised method for linear systems.

    The operator should be square.

    Equivalent to `jax.scipy.sparse.linalg.bicgstab`.

    This supports the following `options` (as passed to
    `lx.linear_solve(..., options=...)`).

    - `preconditioner`: A [`lineax.AbstractLinearOperator`][]
        to be used as a preconditioner. Defaults to
        [`lineax.IdentityLinearOperator`][]. This method uses right preconditioning.
    - `y0`: The initial estimate of the solution to the linear system. Defaults to all
        zeros.
    """

    rtol: float
    atol: float
    norm: Callable = max_norm
    max_steps: int | None = None

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
        return linearise(operator)

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
            b_scale = self.atol + self.rtol * self.norm(vector)

        # This implementation is the same a jax.scipy.sparse.linalg.bicgstab
        # but with AbstractLinearOperator.
        # We use the notation found on the wikipedia except with y instead of x:
        # https://en.wikipedia.org/wiki/
        # Biconjugate_gradient_stabilized_method#Preconditioned_BiCGSTAB
        # preconditioner in this case is K2^(-1) (i.e., right preconditioning)

        r0 = (vector**ω - operator.mv(y0) ** ω).ω
        r0_norm = two_norm(r0)
        eps = jnp.finfo(jnp.result_type(*leaves)).eps

        def not_converged(r, diff, y):
            # The primary tolerance check.
            # Given Ay=b, then we have to be doing better than `scale` in both
            # the `y` and the `b` spaces.
            if has_scale:
                # Standard relative-residual stopping rule: ‖r‖ ≤ atol + rtol·‖b‖
                # (and likewise for the increment in the `y` space). Note this uses
                # scalar norms, *not* an elementwise `atol + rtol·|b|` scale: the
                # latter is unsatisfiable for wide-dynamic-range `b`, where the
                # round-off floor of large components exceeds the absolute tolerance
                # demanded of small ones, yielding spurious non-convergence.
                y_scale = self.atol + self.rtol * self.norm(y)
                b_unconverged = self.norm(r) > b_scale  # pyright: ignore
                y_unconverged = self.norm(diff) > y_scale
                return b_unconverged | y_unconverged
            else:
                return True

        def is_negligible(r):
            # Whether the residual `r` is zero to working precision, so that `y` solves
            # the system exactly (e.g. because `r0` lies in an invariant subspace of
            # the preconditioned operator) and BiCGStab can make no further progress.
            # Deliberately not the (much looser) solver tolerance: `r` routinely meets
            # that a few steps before `diff` does, and those steps must still be taken.
            return two_norm(r) <= eps * r0_norm

        def numerically_orthogonal(dot, x_norm, y_norm):
            # Whether `dot = <x, y>` is indistinguishable from zero: below the rounding
            # error of computing it, which is at least `eps * ‖x‖ ‖y‖`. Comparing
            # against the norms, rather than against an absolute threshold, makes this
            # invariant to the scale of `b` (and of the operator), and taking `abs`
            # handles negative and complex dot products.
            return jnp.abs(dot) <= eps * x_norm * y_norm

        def cond_fun(carry):
            y, r, _, _, _, _, _, diff, step, breakdown = carry
            out = jnp.invert(breakdown) & jnp.invert(is_negligible(r))
            out = out & not_converged(r, diff, y)
            out = out & (step < max_steps)
            return out

        def body_fun(carry):
            y, r, alpha, omega, rho, p, v, diff, step, _ = carry

            # `r` is not negligible here (`cond_fun` stops if it is), so `rho_new` is
            # zero only on a genuine breakdown, as are the `<r0, v_new>` and `<s, t>`
            # below. Each is checked, and on a breakdown we stop and return the last
            # iterate, rather than divide by a value that is zero up to rounding.
            rho_new = tree_dot(r0, r)
            rho_breakdown = numerically_orthogonal(rho_new, r0_norm, two_norm(r))
            beta = (rho_new / rho) * (alpha / omega)
            p_new = (r**ω + beta * (p**ω - omega * v**ω)).ω

            # TODO(raderj): reduce this to a single operator.mv call
            # by using the scan trick.
            x = preconditioner.mv(p_new)
            v_new = operator.mv(x)

            r0_v_new = tree_dot(r0, v_new)
            alpha_breakdown = numerically_orthogonal(r0_v_new, r0_norm, two_norm(v_new))
            alpha_new = rho_new / jnp.where(alpha_breakdown, 1, r0_v_new)
            s = (r**ω - alpha_new * v_new**ω).ω

            # If the alpha-step alone zeroes the residual (so `t = A(M(s))` is also
            # ~0), skip the stabilising omega-step, as `scipy`/`jax.scipy`'s `bicgstab`
            # do: `omega_new = <s, t> / <t, t>` would be `0 / 0`. `cond_fun` then
            # stops, as the new residual is `s`.
            s_negligible = is_negligible(s)
            z = preconditioner.mv(s)
            t = operator.mv(z)
            s_t = tree_dot(s, t)
            omega_breakdown = jnp.invert(s_negligible) & numerically_orthogonal(
                s_t, two_norm(s), two_norm(t)
            )
            omega_new = s_t / jnp.where(
                s_negligible | omega_breakdown, 1, tree_dot(t, t)
            )
            omega_new = jnp.where(s_negligible, 0, omega_new)

            breakdown = rho_breakdown | alpha_breakdown | omega_breakdown
            diff_new = (alpha_new * x**ω + omega_new * z**ω).ω
            y_new = (y**ω + diff_new**ω).ω
            r_new = (s**ω - omega_new * t**ω).ω
            y_new, r_new, diff_new = tree_where(
                breakdown, (y, r, diff), (y_new, r_new, diff_new)
            )
            return (
                y_new,
                r_new,
                alpha_new,
                omega_new,
                rho_new,
                p_new,
                v_new,
                diff_new,
                step + 1,
                breakdown,
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
            jnp.array(False),
        )
        solution, residual, _, _, _, _, _, diff, num_steps, breakdown = lax.while_loop(
            cond_fun, body_fun, init_carry
        )

        if self.max_steps is None:
            result = RESULTS.where(
                num_steps == max_steps, RESULTS.singular, RESULTS.successful
            )
        elif has_scale:
            result = RESULTS.where(
                num_steps == max_steps, RESULTS.max_steps_reached, RESULTS.successful
            )
        else:
            result = RESULTS.successful
        # Breakdown is only an issue if we did not converge. As no further steps can
        # be taken, judge this by the residual alone: the `y`-space check would
        # otherwise fail on a "lucky" breakdown, in which the Krylov space is exhausted
        # and the last step -- however large -- solved the system to rounding error.
        if has_scale:
            breakdown = breakdown & (self.norm(residual) > b_scale)  # pyright: ignore
        result = RESULTS.where(breakdown, RESULTS.breakdown, result)

        stats = {"num_steps": num_steps, "max_steps": self.max_steps}
        return solution, result, stats

    def transpose(self, state: _BiCGStabState, options: dict[str, Any]):
        transpose_options = {}
        if "preconditioner" in options:
            transpose_options["preconditioner"] = options["preconditioner"].transpose()
        operator = state
        return operator.transpose(), transpose_options

    def conj(self, state: _BiCGStabState, options: dict[str, Any]):
        conj_options = {}
        if "preconditioner" in options:
            conj_options["preconditioner"] = conj(options["preconditioner"])
        operator = state
        return conj(operator), conj_options

    def assume_full_rank(self):
        return True


BiCGStab.__init__.__doc__ = r"""**Arguments:**

- `rtol`: Relative tolerance for terminating solve.
- `atol`: Absolute tolerance for terminating solve.
- `norm`: The norm to use when computing whether the error falls within the tolerance.
    Defaults to the max norm.
- `max_steps`: The maximum number of iterations to run the solver for. If more steps
    than this are required, then the solve is halted with a failure.
"""
