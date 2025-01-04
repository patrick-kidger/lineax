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

import importlib.metadata

from . import internal as internal
from ._operator import (
    AbstractLinearOperator as AbstractLinearOperator,
    AddLinearOperator as AddLinearOperator,
    AuxLinearOperator as AuxLinearOperator,
    ComposedLinearOperator as ComposedLinearOperator,
    conj as conj,
    diagonal as diagonal,
    DiagonalLinearOperator as DiagonalLinearOperator,
    DivLinearOperator as DivLinearOperator,
    FunctionLinearOperator as FunctionLinearOperator,
    has_unit_diagonal as has_unit_diagonal,
    IdentityLinearOperator as IdentityLinearOperator,
    is_diagonal as is_diagonal,
    is_lower_triangular as is_lower_triangular,
    is_negative_semidefinite as is_negative_semidefinite,
    is_positive_semidefinite as is_positive_semidefinite,
    is_symmetric as is_symmetric,
    is_tridiagonal as is_tridiagonal,
    is_upper_triangular as is_upper_triangular,
    JacobianLinearOperator as JacobianLinearOperator,
    linearise as linearise,
    materialise as materialise,
    MatrixLinearOperator as MatrixLinearOperator,
    MulLinearOperator as MulLinearOperator,
    NegLinearOperator as NegLinearOperator,
    PyTreeLinearOperator as PyTreeLinearOperator,
    TaggedLinearOperator as TaggedLinearOperator,
    TangentLinearOperator as TangentLinearOperator,
    tridiagonal as tridiagonal,
    TridiagonalLinearOperator as TridiagonalLinearOperator,
)
from ._solution import RESULTS as RESULTS, Solution as Solution
from ._solve import (
    AbstractLinearSolver as AbstractLinearSolver,
    AutoLinearSolver as AutoLinearSolver,
    linear_solve as linear_solve,
)
from ._solver import (
    BiCGStab as BiCGStab,
    CG as CG,
    Cholesky as Cholesky,
    Diagonal as Diagonal,
    GMRES as GMRES,
    LU as LU,
    NormalCG as NormalCG,
    QR as QR,
    SVD as SVD,
    Triangular as Triangular,
    Tridiagonal as Tridiagonal,
)
from ._tags import (
    diagonal_tag as diagonal_tag,
    lower_triangular_tag as lower_triangular_tag,
    negative_semidefinite_tag as negative_semidefinite_tag,
    positive_semidefinite_tag as positive_semidefinite_tag,
    symmetric_tag as symmetric_tag,
    transpose_tags as transpose_tags,
    transpose_tags_rules as transpose_tags_rules,
    tridiagonal_tag as tridiagonal_tag,
    unit_diagonal_tag as unit_diagonal_tag,
    upper_triangular_tag as upper_triangular_tag,
)


__version__ = importlib.metadata.version("lineax")
