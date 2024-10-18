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

from typing import Any, Optional
from typing_extensions import TypeAlias

import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._misc import resolve_rcond
from .._operator import AbstractLinearOperator, diagonal, has_unit_diagonal, is_diagonal
from .._solution import RESULTS
from .._solve import AbstractLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_DiagonalState: TypeAlias = tuple[Optional[Array], PackedStructures]


class Diagonal(AbstractLinearSolver[_DiagonalState], strict=True):
    """Diagonal solver for linear systems.

    Requires that the operator be diagonal. Then $Ax = b$, with $A = diag[a]$, is
    solved simply by doing an elementwise division $x = b / a$.

    This solver can handle singular operators (i.e. diagonal entries with value 0).
    """

    well_posed: bool = False
    rcond: Optional[float] = None

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _DiagonalState:
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`Diagonal` may only be used for linear solves with square matrices"
            )
        if not is_diagonal(operator):
            raise ValueError(
                "`Diagonal` may only be used for linear solves with diagonal matrices"
            )
        packed_structures = pack_structures(operator)
        if has_unit_diagonal(operator):
            return None, packed_structures
        else:
            return diagonal(operator), packed_structures

    def compute(
        self, state: _DiagonalState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        diag, packed_structures = state
        del state, options
        unit_diagonal = diag is None
        vector = ravel_vector(vector, packed_structures)
        if unit_diagonal:
            solution = vector
        else:
            if not self.well_posed:
                (size,) = diag.shape
                rcond = resolve_rcond(self.rcond, size, size, diag.dtype)
                abs_diag = jnp.abs(diag)
                diag = jnp.where(abs_diag > rcond * jnp.max(abs_diag), diag, jnp.inf)
            solution = vector / diag
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _DiagonalState, options: dict[str, Any]):
        del options
        diag, packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_state = diag, transposed_packed_structures
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _DiagonalState, options: dict[str, Any]):
        del options
        diag, packed_structures = state
        if diag is None:
            conj_diag = None
        else:
            conj_diag = diag.conj()
        conj_options = {}
        conj_state = conj_diag, packed_structures
        return conj_state, conj_options

    def allow_dependent_columns(self, operator):
        return not self.well_posed

    def allow_dependent_rows(self, operator):
        return not self.well_posed


Diagonal.__init__.__doc__ = """**Arguments**:

- `well_posed`: if `False`, then singular operators are accepted, and the pseudoinverse
    solution is returned. If `True` then passing a singular operator will cause an error
    to be raised instead.
- `rcond`: the cutoff for handling zero entries on the diagonal. Defaults to machine
    precision times `N`, where `N` is the input (or output) size of the operator.
    Only used if `well_posed=False`
"""
