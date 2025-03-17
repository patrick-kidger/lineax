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
from typing_extensions import TypeAlias

import jax.scipy as jsp
from jaxtyping import Array, PyTree

from .._operator import (
    AbstractLinearOperator,
    has_unit_diagonal,
    is_lower_triangular,
    is_upper_triangular,
)
from .._solution import RESULTS
from .._solve import AbstractLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_TriangularState: TypeAlias = tuple[Array, bool, bool, PackedStructures, bool]


class Triangular(AbstractLinearSolver[_TriangularState], strict=True):
    """Triangular solver for linear systems.

    The operator should either be lower triangular or upper triangular.
    """

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`Triangular` may only be used for linear solves with square matrices"
            )
        if not (is_lower_triangular(operator) or is_upper_triangular(operator)):
            raise ValueError(
                "`Triangular` may only be used for linear solves with triangular "
                "matrices"
            )
        return (
            operator.as_matrix(),
            is_lower_triangular(operator),
            has_unit_diagonal(operator),
            pack_structures(operator),
            False,  # transposed
        )

    def compute(
        self, state: _TriangularState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        matrix, lower, unit_diagonal, packed_structures, transpose = state
        del state, options
        vector = ravel_vector(vector, packed_structures)
        if transpose:
            trans = "T"
        else:
            trans = "N"
        solution = jsp.linalg.solve_triangular(
            matrix, vector, trans=trans, lower=lower, unit_diagonal=unit_diagonal
        )
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _TriangularState, options: dict[str, Any]):
        matrix, lower, unit_diagonal, packed_structures, transpose = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_state = (
            matrix,
            lower,
            unit_diagonal,
            transposed_packed_structures,
            not transpose,
        )
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _TriangularState, options: dict[str, Any]):
        matrix, lower, unit_diagonal, packed_structures, transpose = state
        conj_state = (
            matrix.conj(),
            lower,
            unit_diagonal,
            packed_structures,
            transpose,
        )
        conj_options = {}
        return conj_state, conj_options

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False


Triangular.__init__.__doc__ = """**Arguments:**

Nothing.
"""
