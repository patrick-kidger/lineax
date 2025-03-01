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
from collections.abc import Callable
from typing import Any, cast, Optional
from typing_extensions import TypeAlias

import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Bool, Float, Inexact, PyTree

from .._misc import structure_equal
from .._norm import max_norm, two_norm
from .._operator import (
    AbstractLinearOperator,
    conj,
    MatrixLinearOperator,
)
from .._solution import RESULTS
from .._solve import AbstractLinearSolver, linear_solve
from .misc import preconditioner_and_y0
from .qr import QR


_GMRESState: TypeAlias = AbstractLinearOperator


class GMRES(AbstractLinearSolver[_GMRESState], strict=True):
    """GMRES solver for linear systems.

    The operator should be square.

    Similar to `jax.scipy.sparse.linalg.gmres`.

    This supports the following `options` (as passed to
    `lx.linear_solve(..., options=...)`).

    - `preconditioner`: A positive definite [`lineax.AbstractLinearOperator`][]
        to be used as preconditioner. Defaults to
        [`lineax.IdentityLinearOperator`][].
    - `y0`: The initial estimate of the solution to the linear system. Defaults to all
        zeros.
    """

    rtol: float
    atol: float
    norm: Callable = max_norm
    max_steps: Optional[int] = None
    restart: int = 20
    stagnation_iters: int = 20

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
        if not structure_equal(operator.in_structure(), operator.out_structure()):
            raise ValueError(
                "`GMRES(..., normal=False)` may only be used for linear solves with "
                "square matrices."
            )
        return operator

    #
    # This differs from `jax.scipy.sparse.linalg.gmres` in a few ways:
    # 1. We use a more sophisticated termination condition. To begin with we have an
    #    rtol and atol in the conventional way, inducing a vector-valued scale. This is
    #    then checked in both the `y` and `b` domains (for `Ay = b`).
    # 2. We handle in-place updates with buffers to avoid generating unnecessary
    #    copies of arrays during the Gram-Schmidt procedure.
    # 3. We use a QR solve at the end of the batched Gram-Schmidt instead
    #    of a Cholesky solve of the normal equations. This is both faster and more
    #    numerically stable.
    # 4. We use tricks to compile `A y` fewer times throughout the code, including
    #    passing a dummy initial residual.
    # 5. We return the number of steps, and whether or not the solve succeeded, as
    #    additional information.
    # 6. We do not use the unnecessary loop within Gram-Schmidt, and simply compute
    #    this in a single pass.
    # 7. We add better safety checks for breakdown, and a safety check for stagnation
    #    of the iterates even when we don't explicitly get breakdown.
    #
    def compute(
        self,
        state: _GMRESState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        has_scale = not (
            isinstance(self.atol, (int, float))
            and isinstance(self.rtol, (int, float))
            and self.atol == 0
            and self.rtol == 0
        )
        if has_scale:
            b_scale = (self.atol + self.rtol * ω(vector).call(jnp.abs)).ω
        operator = state
        preconditioner, y0 = preconditioner_and_y0(operator, vector, options)
        leaves, _ = jtu.tree_flatten(vector)
        size = sum(leaf.size for leaf in leaves)
        if self.max_steps is None:
            max_steps = 10 * size  # Copied from SciPy!
        else:
            max_steps = self.max_steps
        restart = min(self.restart, size)

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
            y, r, _, deferred_breakdown, diff, _, step, stagnation_counter = carry
            # NOTE: we defer ending due to breakdown by one loop! This is nonstandard,
            # but lets us use a cauchy-like condition in the convergence criteria.
            # If we do not defer breakdown, breakdown may detect convergence when
            # the diff between two iterations is still quite large, and we only
            # consider convergence when the diff is small.
            out = jnp.invert(deferred_breakdown) & (
                stagnation_counter < self.stagnation_iters
            )
            out = out & not_converged(r, diff, y)
            out = out & (step < max_steps)
            # The first pass uses a dummy value for r0 in order to save on compiling
            # an extra matvec. The dummy step may raise a breakdown, and `step == 0`
            # avoids us from returning prematurely.
            return out | (step == 0)

        def body_fun(carry):
            # `breakdown` -> `deferred_breakdown` and `deferred_breakdown` -> `_`
            y, r, deferred_breakdown, _, diff, r_min, step, stagnation_counter = carry
            y_new, r_new, breakdown, diff_new = self._gmres_compute(
                operator, vector, y, r, restart, preconditioner, step == 0
            )

            #
            # If the minimum residual does not decrease for many iterations
            # ("many" is determined by self.stagnation_iters) then the iterative
            # solve has stagnated and we stop the loop. This bit keeps track of how
            # long it has been since the minimum has decreased, and updates the minimum
            # when a new minimum is encountered. As far as I (raderj) am
            # aware, this is custom to our implementation and not standard practice.
            #
            r_new_norm = self.norm(r_new)
            r_decreased = (r_new_norm - r_min) < 0
            stagnation_counter = jnp.where(r_decreased, 0, stagnation_counter + 1)
            stagnation_counter = cast(Array, stagnation_counter)
            r_min = jnp.minimum(r_new_norm, r_min)

            return (
                y_new,
                r_new,
                breakdown,
                deferred_breakdown,
                diff_new,
                r_min,
                step + 1,
                stagnation_counter,
            )

        # Initialise the residual r0 to the dummy value of all 0s. This means
        # the first iteration of Gram-Schmidt will do nothing, but it saves
        # us from compiling an extra matvec here.
        r0 = ω(vector).call(jnp.zeros_like).ω
        init_carry = (
            y0,  # y
            r0,  # residual
            False,  # breakdown
            False,  # deferred_breakdown
            ω(y0).call(lambda x: jnp.full_like(x, jnp.inf)).ω,  # diff
            jnp.inf,  # r_min
            0,  # steps
            jnp.array(0),  # stagnation counter
        )
        (
            solution,
            residual,
            _,  # breakdown
            breakdown,  # deferred_breakdown
            diff,
            _,
            num_steps,
            stagnation_counter,
        ) = lax.while_loop(cond_fun, body_fun, init_carry)

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
        result = RESULTS.where(
            stagnation_counter >= self.stagnation_iters, RESULTS.stagnation, result
        )

        # breakdown is only an issue if we broke down outside the tolerance
        # of the solution. If we get breakdown and are within the tolerance,
        # this is called convergence :)
        breakdown = breakdown & not_converged(residual, diff, solution)
        # breakdown is the most serious potential issue
        result = RESULTS.where(breakdown, RESULTS.breakdown, result)

        stats = {"num_steps": num_steps, "max_steps": self.max_steps}
        return solution, result, stats

    def _gmres_compute(
        self, operator, vector, y, r, restart, preconditioner, first_pass
    ):
        #
        # internal function for computing the bulk of the gmres. We seperate this out
        # for two reasons:
        # 1. avoid nested body and cond functions in the body and cond function of
        # `self.compute`. `self.compute` is primarily responsible for the restart
        # behavior of gmres.
        # 2. Like the jax.scipy implementation we may want to add an incremental
        # version at a later date.
        #

        def main_gmres(y):
            # see the comment at the end of `_arnoldi_gram_schmidt` for a discussion
            # of `initial_breakdown`
            r_normalised, r_norm, initial_breakdown = self._normalise(r, eps=None)
            basis_init = jtu.tree_map(
                lambda x: jnp.pad(x[..., None], ((0, 0),) * x.ndim + ((0, restart),)),
                r_normalised,
            )
            coeff_mat_init = jnp.eye(
                restart,
                restart + 1,
                dtype=jnp.result_type(*jtu.tree_leaves(r_normalised)),
            )

            def cond_fun(carry):
                _, _, breakdown, step = carry
                return (step < restart) & jnp.invert(breakdown)

            def body_fun(carry):
                basis, coeff_mat, breakdown, step = carry
                basis_new, coeff_mat_new, breakdown = self._arnoldi_gram_schmidt(
                    operator,
                    preconditioner,
                    basis,
                    coeff_mat,
                    step,
                    restart,
                    vector,
                    breakdown,
                )
                return basis_new, coeff_mat_new, breakdown, step + 1

            def buffers(carry):
                basis, coeff_mat, _, _ = carry
                return basis, coeff_mat

            init_carry = (basis_init, coeff_mat_init, initial_breakdown, 0)
            basis, coeff_mat, breakdown, steps = eqxi.while_loop(
                cond_fun, body_fun, init_carry, kind="lax", buffers=buffers
            )
            beta_vec = jnp.concatenate(
                (
                    r_norm[None].astype(jnp.result_type(coeff_mat)),
                    jnp.zeros_like(coeff_mat, shape=(restart,)),
                )
            )
            coeff_op_transpose = MatrixLinearOperator(coeff_mat.T)
            # TODO(raderj): move to a Hessenberg-specific solver
            z = linear_solve(coeff_op_transpose, beta_vec, QR(), throw=False).value
            diff = jtu.tree_map(
                lambda mat: jnp.tensordot(
                    mat[..., :-1], z, axes=1, precision=lax.Precision.HIGHEST
                ),
                basis,
            )
            y_new = (y**ω + diff**ω).ω
            return y_new, diff, breakdown

        def first_gmres(y):
            return y, ω(y).call(lambda x: jnp.full_like(x, jnp.inf)).ω, False

        first_pass = eqxi.unvmap_any(first_pass)
        y_new, diff, breakdown = lax.cond(first_pass, first_gmres, main_gmres, y)
        r_new = preconditioner.mv((vector**ω - operator.mv(y_new) ** ω).ω)

        return y_new, r_new, breakdown, diff

        # NOTE: in the jax implementation:
        # https://github.com/google/jax/blob/
        # c662fd216dec10cdb2cff4138b4318bb98853134/jax/_src/scipy/sparse/linalg.py#L327
        # _classical_iterative_gram_schmidt uses a while loop to call this.
        # However, max_iterations is set to 2 in all calls they make to the function,
        # and the condition function requires steps < (max_iterations - 1).
        # This means that in fact they only apply Gram-Schmidt once, and using a
        # while_loop is unnecessary.

    def _arnoldi_gram_schmidt(
        self,
        operator,
        preconditioner,
        basis,
        coeff_mat,
        step,
        restart,
        vector,
        initial_breakdown,
    ):
        #
        # compute `basis.T @ basis_step` for each leaf of pytree
        # and then compute the projected vector onto the basis
        #
        # `basis` is a pytree with buffers, meaning it can only be
        # indexed into. Through this section, there are terms like `lambda _, x: ...`
        # because`jtu.tree_map` only uses the first argument to determine the shape
        # of the pytree. Since _Buffer is considered part of the pytree
        # structure, we get leaves which are not buffers if we directly pass `basis`.
        # Instead, we make sure that the first argument of the tree map is something
        # with the correct pytree structure, such as `vector` in the dummy case and
        # basis_step when not, so that we correctly index into `basis`.
        #
        basis_step = preconditioner.mv(
            operator.mv(jtu.tree_map(lambda _, x: x[..., step], vector, basis))
        )
        step_norm = two_norm(basis_step)
        contract_matrix = lambda x, y: ft.partial(
            jnp.tensordot, axes=x.ndim, precision=lax.Precision.HIGHEST
        )(x, y[...].conj())
        _proj = jtu.tree_map(contract_matrix, basis_step, basis)
        proj = jtu.tree_reduce(lambda x, y: x + y, _proj)
        proj_on_cols = jtu.tree_map(lambda _, x: x[...] @ proj, vector, basis)
        # now remove the component of the vector in that subspace
        basis_step_new = (basis_step**ω - proj_on_cols**ω).ω
        eps = step_norm * jnp.finfo(proj.dtype).eps
        basis_step_normalised, step_norm_new, breakdown = self._normalise(
            basis_step_new, eps=eps
        )
        basis_new = jtu.tree_map(
            lambda y, mat: mat.at[..., step + 1].set(y),
            basis_step_normalised,
            basis,
        )
        proj_new = proj.at[step + 1].set(step_norm_new.astype(jnp.result_type(proj)))
        #
        # NOTE: two somewhat complicated things are going on here:
        #
        # The `coeff_mat` in_place update has a batch tracer, so we need to be
        # careful and wrap it in a buffer, hence the use of eqxi.while_loop
        # instead of lax.while_loop throughout.
        #
        # `initial_breakdown` occurs when the previous loop returns a
        # residual which is small enough to be interpreted as 0 by self._normalise,
        # but which was passed through the solver anyway. This occurs when
        # the residual is small but the diff is not, or if the
        # correct solution was given to GMRES from the start. Both of these tend to
        # happen at the start of `gmres_compute`.
        # The latter may happen when using a sequence of iterative methods.
        # If `initial_breakdown` occurs, then we leave the `coeff_mat` as it was
        # at initialisation. Replacing it with the projection (which will be all 0s)
        # will mean `coeff_mat` is not full-rank, and `QR` can only handle nonsquare
        # matrices of full-rank.
        #
        coeff_mat_new = coeff_mat.at[step, :].set(
            proj_new, pred=jnp.invert(initial_breakdown)
        )
        return basis_new, coeff_mat_new, breakdown

    def _normalise(
        self, x: PyTree[Array], eps: Optional[Float[ArrayLike, ""]]
    ) -> tuple[PyTree[Array], Inexact[Array, ""], Bool[ArrayLike, ""]]:
        norm = two_norm(x)
        if eps is None:
            eps = jnp.finfo(norm.dtype).eps
        else:
            eps = jnp.astype(eps, norm.dtype)
        breakdown = norm < eps
        safe_norm = jnp.where(breakdown, jnp.inf, norm)
        with jax.numpy_dtype_promotion("standard"):
            x_normalised = (x**ω / safe_norm).ω
        return x_normalised, norm, breakdown

    def transpose(self, state: _GMRESState, options: dict[str, Any]):
        del options
        operator = state
        transpose_options = {}
        return operator.transpose(), transpose_options

    def conj(self, state: _GMRESState, options: dict[str, Any]):
        del options
        operator = state
        conj_options = {}
        return conj(operator), conj_options

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False


GMRES.__init__.__doc__ = r"""**Arguments:**

- `rtol`: Relative tolerance for terminating solve.
- `atol`: Absolute tolerance for terminating solve.
- `norm`: The norm to use when computing whether the error falls within the tolerance.
    Defaults to the max norm.
- `max_steps`: The maximum number of iterations to run the solver for. If more steps
    than this are required, then the solve is halted with a failure.
- `restart`: Size of the Krylov subspace built between restarts. The returned solution
    is the projection of the true solution onto this subpsace, so this direclty
    bounds the accuracy of the algorithm. Default is 20.
- `stagnation_iters`: The maximum number of iterations for which the solver may not
    decrease. If more than `stagnation_iters` restarts are performed without
    sufficient decrease in the residual, the algorithm is halted.
"""
