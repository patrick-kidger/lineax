from typing import Any
from typing_extensions import TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._operator import (
    AbstractLinearOperator,
    MatrixLinearOperator,
    WoodburyLinearOperator,
)
from .._solution import RESULTS
from .._solve import AbstractLinearSolver, AutoLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_WoodburyState: TypeAlias = tuple[
    tuple[Array, Array, Array],
    tuple[AbstractLinearSolver, Any, AbstractLinearSolver, Any],
    PackedStructures,
]


def _compute_pushthrough(
    A_solver: AbstractLinearSolver, A_state: Any, C: Array, U: Array, V: Array
) -> tuple[AbstractLinearSolver, Any]:
    # Push through ( C^-1 + V A^-1 U) y = x
    vmapped_solve = jax.vmap(
        lambda x_vec: A_solver.compute(A_state, x_vec, {})[0], in_axes=1, out_axes=1
    )
    pushthrough_mat = jnp.linalg.inv(C) + V @ vmapped_solve(U)
    pushthrough_op = MatrixLinearOperator(pushthrough_mat)
    solver = AutoLinearSolver(well_posed=True).select_solver(pushthrough_op)
    state = solver.init(pushthrough_op, {})
    return solver, state


class Woodbury(AbstractLinearSolver[_WoodburyState]):
    """Solving system using Woodbury matrix identity"""

    def init(
        self,
        operator: AbstractLinearOperator,
        options: dict[str, Any],
        A_solver: AbstractLinearSolver = AutoLinearSolver(well_posed=True),
    ):
        del options
        if not isinstance(operator, WoodburyLinearOperator):
            raise ValueError(
                "`Woodbury` may only be used for linear solves with A + U C V structure"
            )
        else:
            A, C, U, V = operator.A, operator.C, operator.U, operator.V  # pyright: ignore
            if A.in_size() != A.out_size():
                raise ValueError("""A must be square""")
            # Find correct solvers and init for A
            A_state = A_solver.init(A, {})
            # Compute pushthrough operator
            pt_solver, pt_state = _compute_pushthrough(A_solver, A_state, C, U, V)
            return (
                (C, U, V),
                (A_solver, A_state, pt_solver, pt_state),
                pack_structures(A),
            )

    def compute(
        self,
        state: _WoodburyState,
        vector,
        options,
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        (
            (C, U, V),
            (A_solver, A_state, pt_solver, pt_state),
            A_packed_structures,
        ) = state
        del state, options
        vector = ravel_vector(vector, A_packed_structures)

        # Solution to A x = b
        # [0] selects the solution vector
        x_1 = A_solver.compute(A_state, vector, {})[0]
        # Push through U ( C^-1 + V A^-1 U)^-1 V (A^-1 b)
        # [0] selects the solution vector
        x_pushthrough = U @ pt_solver.compute(pt_state, V @ x_1, {})[0]
        # A^-1 on result of push through
        # [0] selects the solution vector
        x_2 = A_solver.compute(A_state, x_pushthrough, {})[0]
        # See https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        solution = x_1 - x_2

        solution = unravel_solution(solution, A_packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _WoodburyState, options: dict[str, Any]):
        (
            (C, U, V),
            (A_solver, A_state, pt_solver, pt_state),
            A_packed_structures,
        ) = state
        transposed_packed_structures = transpose_packed_structures(A_packed_structures)
        C = jnp.transpose(C)
        U = jnp.transpose(V)
        V = jnp.transpose(U)
        A_state, _ = A_solver.transpose(A_state, {})
        pt_solver, pt_state = _compute_pushthrough(A_solver, A_state, C, U, V)
        transpose_state = (
            (C, U, V),
            (A_solver, A_state, pt_solver, pt_state),
            transposed_packed_structures,
        )
        return transpose_state, options

    def conj(self, state: _WoodburyState, options: dict[str, Any]):
        (
            (C, U, V),
            (A_solver, A_state, pt_solver, pt_state),
            packed_structures,
        ) = state
        C = jnp.conj(C)
        U = jnp.conj(U)
        V = jnp.conj(V)
        A_state, _ = A_solver.conj(A_state, {})
        pt_solver, pt_state = _compute_pushthrough(A_solver, A_state, C, U, V)
        conj_state = (
            (C, U, V),
            (A_solver, A_state, pt_solver, pt_state),
            packed_structures,
        )
        return conj_state, options

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False
