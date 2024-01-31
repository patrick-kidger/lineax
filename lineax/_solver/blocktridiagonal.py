from typing import Any
from typing_extensions import TypeAlias

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._operator import (
    AbstractLinearOperator,
    blocktridiagonal,
    is_blocktridiagonal,
    MatrixLinearOperator,
)
from .._solution import RESULTS
from .._solve import AbstractLinearSolver, linear_solve
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_BlockTridiagonalState: TypeAlias = tuple[tuple[Array, Array, Array], PackedStructures]


class BlockTridiagonal(AbstractLinearSolver[_BlockTridiagonalState]):
    """Block tridiagonal solver for linear systems, using the Thomas algorithm."""

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                """`BlockTridiagonal` may only be used for linear solves with 
                square matrices"""
            )
        if not is_blocktridiagonal(operator):
            raise ValueError(
                """`BlockTridiagonal` may only be used for linear solves with 
                block tridiagonal `matrices`"""
            )
        return blocktridiagonal(operator), pack_structures(operator)

    def compute(
        self,
        state: _BlockTridiagonalState,
        vector,
        options,
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        (diagonal, lower_diagonal, upper_diagonal), packed_structures = state
        del state, options
        vector = ravel_vector(vector, packed_structures)

        #
        # Modifications to basic Thomas algorithm to work on block matrices
        # notation from: https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        # _p indicates prime, ie. `d_p` is the variable name for d' on wikipedia
        #

        block_size = diagonal.shape[1]
        size = diagonal.shape[0]

        vector = vector.reshape(size, block_size)

        def matrix_linear_solve(A, X):
            # Solving A X = B, where X and B are matrices
            def matrix_linear_solve_vec(step, x_vec):
                y = linear_solve(A, x_vec).value
                step += 1
                return step, y

            carry = 0
            _, B = lax.scan(matrix_linear_solve_vec, carry, X.T)
            return B.T

        def blockthomas_scan(prev_cd_carry, bd):
            c_p, d_p, step = prev_cd_carry
            # the index of `a` doesn't matter at step 0 as
            # we won't use it at all. Same for `c` at final step
            a_index = jnp.where(step > 0, step - 1, 0)
            c_index = jnp.where(step < size, step, 0)

            b, d = bd
            a, c = lower_diagonal[a_index, :, :], upper_diagonal[c_index, :, :]

            denom = MatrixLinearOperator(b - jnp.matmul(a, c_p))
            new_d_p = linear_solve(denom, d - jnp.matmul(a, d_p)).value
            new_c_p = matrix_linear_solve(denom, c)
            return (new_c_p, new_d_p, step + 1), (new_c_p, new_d_p)

        def backsub(prev_x_carry, cd_p):
            x_prev, step = prev_x_carry
            c_p, d_p = cd_p
            x_new = d_p - jnp.dot(c_p, x_prev)
            return (x_new, step + 1), x_new

        init_thomas = (jnp.zeros((block_size, block_size)), jnp.zeros(block_size), 0)
        init_backsub = (jnp.zeros(block_size), 0)
        diag_vec = (diagonal, vector)
        _, cd_p = lax.scan(blockthomas_scan, init_thomas, diag_vec, unroll=32)
        _, solution = lax.scan(backsub, init_backsub, cd_p, reverse=True, unroll=32)
        solution = solution.flatten()

        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _BlockTridiagonalState, options: dict[str, Any]):
        (diagonal, lower_diagonal, upper_diagonal), packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_diagonals = (
            jnp.transpose(diagonal, axes=[0, 2, 1]),
            jnp.transpose(upper_diagonal, axes=[0, 2, 1]),
            jnp.transpose(lower_diagonal, axes=[0, 2, 1]),
        )
        transpose_state = (transpose_diagonals, transposed_packed_structures)
        return transpose_state, options

    def conj(self, state: _BlockTridiagonalState, options: dict[str, Any]):
        (diagonal, lower_diagonal, upper_diagonal), packed_structures = state
        conj_diagonals = (diagonal.conj(), lower_diagonal.conj(), upper_diagonal.conj())
        conj_state = (conj_diagonals, packed_structures)
        return conj_state, options

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False
