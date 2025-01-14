from typing import Any, Callable, Optional
from typing_extensions import TypeAlias

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, PyTree

from .._norm import max_norm
from .._operator import AbstractLinearOperator
from .._solution import RESULTS
from .._solve import AbstractLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_GaussSeidelState: TypeAlias = tuple[tuple[Array, Array], PackedStructures]


class GaussSeidel(AbstractLinearSolver[_GaussSeidelState], strict=False):
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

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _GaussSeidelState:
        del options

        packed_structures = pack_structures(operator)
        # Needs to materialize the system matrix to decompose it
        system_matrix = operator.as_matrix()
        strictly_upper = jnp.triu(system_matrix, k=1)
        lower = system_matrix - strictly_upper
        return (lower, strictly_upper), packed_structures

    def compute(
        self,
        state: _GaussSeidelState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        (lower, strictly_upper), packed_structures = state

        leaves, _ = jtu.tree_flatten(vector)
        if self.max_steps is None:
            size = sum(leaf.size for leaf in leaves)
            max_steps = 10 * size
        else:
            max_steps = self.max_steps

        # Ravel the PyTree right hand side into a single vector to operate with
        # the materialized matrices
        vector = ravel_vector(vector, packed_structures)

        has_scale = not (
            isinstance(self.atol, (int, float))
            and isinstance(self.rtol, (int, float))
            and self.atol == 0
            and self.rtol == 0
        )
        if has_scale:
            b_scale = (self.atol + self.rtol * ω(vector).call(jnp.abs)).ω

        def compute_residual(y):
            return vector - (lower + strictly_upper) @ y

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
            diff, y, r, step = carry
            out = step < max_steps
            out = out & not_converged(r, diff, y)
            return out

        def body_fun(carry):
            _, y, _, step = carry
            y_prev = y
            y = jax.scipy.linalg.solve_triangular(
                lower,
                vector - strictly_upper @ y_prev,
                lower=True,
            )
            diff = y - y_prev
            r = compute_residual(y)
            step = step + 1
            return diff, y, r, step

        try:
            y0 = options["y0"]
        except KeyError:
            y0 = jtu.tree_map(jnp.zeros_like, vector)

        r0 = compute_residual(y0)

        initial_carry = (
            ω(y0).call(lambda x: jnp.full_like(x, jnp.inf)).ω,
            y0,
            r0,
            0,
        )

        _, solution, _, num_steps = lax.while_loop(cond_fun, body_fun, initial_carry)

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

        solution = unravel_solution(solution, packed_structures)
        stats = {"num_steps": num_steps, "max_steps": self.max_steps}
        return solution, result, stats

    def transpose(self, state: _GaussSeidelState, options: dict[str, Any]):
        del options
        (diagonal, strictly_lower_upper), packed_structures = state
        transposed_strictly_lower_upper = jnp.transpose(strictly_lower_upper)
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_state = (
            (diagonal, transposed_strictly_lower_upper),
            transposed_packed_structures,
        )
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _GaussSeidelState, options: dict[str, Any]):
        del options
        (diagonal, strictly_lower_upper), packed_structures = state
        conjugated_strictly_lower_upper = jnp.conj(strictly_lower_upper)
        conjugated_diagonal = jnp.conj(diagonal)
        conjugated_state = (
            (conjugated_diagonal, conjugated_strictly_lower_upper),
            packed_structures,
        )
        conjugated_options = {}
        return conjugated_state, conjugated_options

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False


GaussSeidel.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating solve.
- `atol`: Absolute tolerance for terminating solve.
- `norm`: The norm to use when computing whether the error falls within the tolerance.
    Defaults to the max norm.
- `max_steps`: The maximum number of iterations to run the solver for. If more steps
    than this are required, then the solve is halted with a failure.
"""
