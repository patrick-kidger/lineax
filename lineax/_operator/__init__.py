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

from .base import (
    AbstractLinearOperator as AbstractLinearOperator,
    conj as conj,
    diagonal as diagonal,
    first_column as first_column,
    has_unit_diagonal as has_unit_diagonal,
    is_circulant as is_circulant,
    is_diagonal as is_diagonal,
    is_lower_triangular as is_lower_triangular,
    is_negative_semidefinite as is_negative_semidefinite,
    is_positive_semidefinite as is_positive_semidefinite,
    is_symmetric as is_symmetric,
    is_tridiagonal as is_tridiagonal,
    is_upper_triangular as is_upper_triangular,
    linearise as linearise,
    materialise as materialise,
    max_rank as max_rank,
    tridiagonal as tridiagonal,
)
from .binary import (
    AddLinearOperator as AddLinearOperator,
    ComposedLinearOperator as ComposedLinearOperator,
)
from .core import (
    FunctionLinearOperator as FunctionLinearOperator,
    JacobianLinearOperator as JacobianLinearOperator,
    MatrixLinearOperator as MatrixLinearOperator,
    PyTreeLinearOperator as PyTreeLinearOperator,
)
from .structured import (
    CirculantLinearOperator as CirculantLinearOperator,
    DiagonalLinearOperator as DiagonalLinearOperator,
    IdentityLinearOperator as IdentityLinearOperator,
    TridiagonalLinearOperator as TridiagonalLinearOperator,
)
from .wrapper import (
    DivLinearOperator as DivLinearOperator,
    MulLinearOperator as MulLinearOperator,
    NegLinearOperator as NegLinearOperator,
    TaggedLinearOperator as TaggedLinearOperator,
    TangentLinearOperator as TangentLinearOperator,
)
