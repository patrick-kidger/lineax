from typing import Callable, Optional

import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω

from .._misc import max_norm, two_norm
from .._operator import (
    AbstractLinearOperator,
    IdentityLinearOperator,
    MatrixLinearOperator,
)
from .._solution import RESULTS
from .._solve import AbstractLinearSolver, linear_solve
from .qr import QR


class GMRES(AbstractLinearSolver):
    """GMRES solver for linear systems.

    The operator should be square.

    Similar to `jax.scipy.sparse.linalg.gmres`.

    This supports the following `options` (as passed to
    `optx.linear_solve(..., options=...)`).

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

    def __post_init__(self):
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

    def init(self, operator, options):
        if operator.in_structure() != operator.out_structure():
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
    def compute(self, state, vector, options):
        has_scale = not (
            isinstance(self.atol, (int, float))
            and isinstance(self.rtol, (int, float))
            and self.atol == 0
            and self.rtol == 0
        )
        if has_scale:
            b_scale = (self.atol + self.rtol * ω(vector).call(jnp.abs)).ω
        operator = state
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
        try:
            y0 = options["y0"]
        except KeyError:
            y0 = ω(vector).call(jnp.zeros_like).ω
        else:
            if jax.eval_shape(lambda: y0) != jax.eval_shape(lambda: vector):
                raise ValueError(
                    "`y0` must have the same structure, shape, and dtype as `vector`"
                )
        leaves, _ = jtu.tree_flatten(vector)
        size = sum(leaf.size for leaf in leaves)
        restart = min(self.restart, size)

        def not_converged(r, diff, y):
            # The primary tolerance check.
            # Given Ay=b, then we have to be doing better than `scale` in both
            # the `y` and the `b` spaces.
            if has_scale:
                y_scale = (self.atol + self.rtol * ω(y).call(jnp.abs)).ω
                norm1 = self.norm((r**ω / b_scale**ω).ω)
                norm2 = self.norm((diff**ω / y_scale**ω).ω)
                return (norm1 > 1) | (norm2 > 1)
            else:
                return True

        def cond_fun(carry):
            y, r, breakdown, diff, _, step, stagnation_counter = carry
            out = jnp.invert(breakdown) & (stagnation_counter < self.stagnation_iters)
            out = out & not_converged(r, diff, y)
            if self.max_steps is not None:
                out = out & (step < self.max_steps)
            # The first pass uses a dummy value for r0 in order to save on compiling
            # an extra matvec. The dummy step may raise a breakdown, and `step == 0`
            # avoids us from returning prematurely.
            return out | (step == 0)

        def body_fun(carry):
            y, r, breakdown, diff, r_min, step, stagnation_counter = carry
            y_new, r_new, breakdown, diff_new = self._gmres_compute(
                operator, vector, y, r, restart, preconditioner, b_scale, step == 0
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
            r_min = jnp.minimum(r_new_norm, r_min)

            return (
                y_new,
                r_new,
                breakdown,
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
            ω(y0).call(lambda x: jnp.full_like(x, jnp.inf)).ω,  # diff
            jnp.inf,  # r_min
            0,  # steps
            0,  # stagnation counter
        )
        (
            solution,
            residual,
            breakdown,
            diff,
            _,
            num_steps,
            stagnation_counter,
        ) = lax.while_loop(cond_fun, body_fun, init_carry)

        if self.max_steps is None:
            result = RESULTS.successful
        else:
            result = jnp.where(
                (num_steps == self.max_steps),
                RESULTS.max_steps_reached,
                RESULTS.successful,
            )
        result = jnp.where(
            stagnation_counter >= self.stagnation_iters,
            RESULTS.stagnation,
            result,
        )

        # breakdown is only an issue if we broke down outside the tolerance
        # of the solution. If we get breakdown and are within the tolerance,
        # this is called convergence :)
        breakdown = breakdown & not_converged(residual, diff, solution)

        # breakdown is the most serious potential issue
        result = jnp.where(breakdown, RESULTS.breakdown, result)
        return (
            solution,
            result,
            {"num_steps": num_steps, "max_steps": self.max_steps},
        )

    def _gmres_compute(
        self, operator, vector, y, r, restart, preconditioner, b_scale, first_pass
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
            r_normalised, r_norm = self._normalise(r, eps=None)
            basis_init = jtu.tree_map(
                lambda x: jnp.pad(x[..., None], ((0, 0),) * x.ndim + ((0, restart),)),
                r_normalised,
            )
            coeff_mat_init = jnp.eye(restart, restart + 1, dtype=r_norm.dtype)

            def cond_fun(carry):
                _, _, breakdown, step = carry
                return (step < restart) & jnp.invert(breakdown)

            def body_fun(carry):
                basis, coeff_mat, _, step = carry
                basis_new, coeff_mat_new, breakdown = self._arnoldi_gram_schmidt(
                    operator, preconditioner, basis, coeff_mat, step, restart, b_scale
                )
                return basis_new, coeff_mat_new, breakdown, step + 1

            def buffers(carry):
                basis, coeff_mat, _, _ = carry
                return (basis, coeff_mat)

            init_carry = (basis_init, coeff_mat_init, False, 0)
            basis, coeff_mat, breakdown, steps = eqxi.while_loop(
                cond_fun, body_fun, init_carry, kind="lax", buffers=buffers
            )
            beta_vec = jnp.concatenate(
                (r_norm[None], jnp.zeros_like(coeff_mat, shape=(restart,)))
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

    def _arnoldi_gram_schmidt(
        self, operator, preconditioner, basis, coeff_mat, step, restart, b_scale
    ):
        # Unwrap the buffer so we can do the matmuls in Gram-Schmidt easier.
        # Note that we've already written to `basis` with 0s, so we are not reading
        # from something which we have not yet written to. Further, we'll never
        # autodiff through this. In total, we are avoiding the common buffer pitfalls.
        basis_unwrapped = basis[...]
        basis_step = preconditioner.mv(operator.mv(ω(basis_unwrapped)[..., step].ω))
        step_norm = two_norm(basis_step)

        # Note that the jax implementation:
        # https://github.com/google/jax/blob/
        # c662fd216dec10cdb2cff4138b4318bb98853134/jax/_src/scipy/sparse/linalg.py#L327
        # of _classical_iterative_gram_schmidt uses a while loop to call this.
        # However, max_iterations is set to 2 in all calls they make to the function,
        # and the condition function requires steps < (max_iterations - 1).
        # This means that in fact they only apply Gram-Schmidt once, and using a
        # while_loop is unnecessary.
        contract_matrix = jax.vmap(
            lambda x, y: jnp.tensordot(
                x, y, axes=y.ndim, precision=lax.Precision.HIGHEST
            ),
            in_axes=(-1, None),
        )
        # `basis.T @ basis_step` for each leaf of pytree
        _proj = jtu.tree_map(contract_matrix, basis_unwrapped, basis_step)

        # projection coeffs of `basis_step` onto existing columns.
        # accumulated over all leaves of pytree.
        proj = jtu.tree_reduce(lambda x, y: x + y, _proj)

        proj_on_cols = jtu.tree_map(lambda x: x @ proj, basis_unwrapped)

        basis_step_new = (basis_step**ω - proj_on_cols**ω).ω

        eps = step_norm * jnp.finfo(proj.dtype).eps
        basis_step_normalised, step_norm_new = self._normalise(basis_step_new, eps=eps)
        # the dummy initial val is to get the right structure for indexing into the
        # buffer with a tree_map
        basis_new = jtu.tree_map(
            lambda y, mat: mat.at[..., step + 1].set(y),
            basis_step_normalised,
            basis,
        )

        proj_new = proj.at[step + 1].set(step_norm_new)
        # NOTE: this in_place update has a batch tracer, so we need to be
        # careful and wrap it in a buffer, hence the use of eqxi.while_loop
        # instead of lax.while_loop throughout.
        coeff_mat_new = coeff_mat.at[step, :].set(proj_new)
        breakdown = step_norm_new < jnp.finfo(step_norm_new.dtype).eps
        return basis_new, coeff_mat_new, breakdown

    def _normalise(self, x, eps):
        norm = two_norm(x)
        if eps is None:
            eps = jnp.finfo(norm.dtype).eps
        pred = norm > eps
        safe_norm = jnp.where(pred, norm, jnp.inf)
        x_normalised = (x**ω / safe_norm).ω
        return x_normalised, norm

    def transpose(self, state, options):
        del options
        operator = state
        transpose_options = {}
        return operator.transpose(), transpose_options

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
