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

import dataclasses
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ._operator import AbstractLinearOperator


class _HasRepr:
    def __init__(self, string: str):
        self.string = string

    def __repr__(self):
        return self.string


symmetric_tag = _HasRepr("symmetric_tag")
diagonal_tag = _HasRepr("diagonal_tag")
tridiagonal_tag = _HasRepr("tridiagonal_tag")
unit_diagonal_tag = _HasRepr("unit_diagonal_tag")
lower_triangular_tag = _HasRepr("lower_triangular_tag")
upper_triangular_tag = _HasRepr("upper_triangular_tag")
positive_semidefinite_tag = _HasRepr("positive_semidefinite_tag")
negative_semidefinite_tag = _HasRepr("negative_semidefinite_tag")


@dataclasses.dataclass(frozen=True)
class MaxRankTag:
    """Marks that an operator's rank is no more than the value given in its argument
    (`MaxRankTag(r)`). Use [`lineax.max_rank`][] to query the bound.

    `MaxRankTag` is preserved through transposition and inversion (rank is invariant
    under both). It composes through `@` as `min(rank_A, rank_B)` and through `+` as
    `min(rank_A + rank_B, in_size, out_size)`.

    `MaxRankTag(0)` is valid, and represents the zero operator.

    !!! Example

        ```python
        k, n = 5, 100
        U  = lx.MatrixLinearOperator(jnp.zeros((n, k)), lx.MaxRankTag(k))
        C  = lx.MatrixLinearOperator(jnp.zeros((k, k)), lx.MaxRankTag(k))
        Vt = lx.MatrixLinearOperator(jnp.zeros((k, n)), lx.MaxRankTag(k))

        update = U @ C @ Vt
        assert lx.max_rank(update) == k   # propagated automatically through composition
        ```

    An operator is considered rank-deficient if
    `lx.max_rank(operator) < min(operator.in_size(), operator.out_size())`. Full-rank
    solvers (e.g. `lx.AutoLinearSolver(well_posed=None/True)`) will raise a `ValueError`
    if asked to solve a rank-deficient system. Rank-deficient solvers MAY make internal
    optimisations based on [`lineax.max_rank`][]. For example, tagging `MaxRankTag(r)`
    and solving with [`lineax.SVD`][] will truncate to the `r` largest singular values
    after decomposition. As such, correctness may be impacted to the extent that an
    operator's actual rank exceeds `max_rank` (in exactly the same way that specifying
    an overly high `rcond` in the solver might).

    !!! info

        Any internal optimisations made by rank-deficient direct solvers are essentially
        [continuous retraction mappings](https://en.wikipedia.org/wiki/Retraction_(topology))
        of the operator A to another operator Â with rank `<=r`. Therefore, if an
        operator A exceeds `max_rank` (e.g. due to floating point roundoff error, or
        an attempt to obtain a low rank approximation of a full rank matrix), the
        solver solves against Â instead of A. In the case of SVD's singular value
        truncation the retraction mapping is the Frobenius norm projection.

        When differentiating a rank-deficient solve, the tangent dA is orthogonally
        projected onto the tangent space of the rank-r locus at Â (this is what the
        [Moore-Penrose pseudoinverse derivative](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Derivative)
        computes). Consequently derivatives are also accurate to the same degree
        that A is approximately rank r.

    **Arguments:**

    - `r`: non-negative integer upper bound on the rank.
    """

    r: int

    def __post_init__(self):
        if not isinstance(self.r, int) or self.r < 0:
            raise ValueError(
                f"MaxRankTag.r must be a non-negative integer, got {self.r!r}"
            )

    def __repr__(self):
        return f"max_rank_tag({self.r})"


def tags_from_checks(operator: "AbstractLinearOperator") -> frozenset[object]:
    """Inspects an operator using all standard property checks and
    returns a frozenset of the tags that apply to it.

    This is the canonical way to collect the full set of tags for any operator,
    regardless of whether that operator stores its properties as an explicit `.tags`
    frozenset or encodes them structurally (e.g. a `DiagonalLinearOperator`
    is always diagonal).

    **Arguments:**

    - `operator`: the linear operator to inspect.

    **Returns:**

    A `frozenset` of tags.
    """
    # Lazy import to avoid a circular dependency: _operator.py imports _tags.py
    # at module level, so we defer the reverse import to call time when both
    # modules are fully initialised.
    from ._operator import (
        has_unit_diagonal,
        is_diagonal,
        is_lower_triangular,
        is_negative_semidefinite,
        is_positive_semidefinite,
        is_symmetric,
        is_tridiagonal,
        is_upper_triangular,
        max_rank,
    )

    tags: set[object] = {
        tag
        for check, tag in [
            (is_symmetric, symmetric_tag),
            (is_diagonal, diagonal_tag),
            (is_lower_triangular, lower_triangular_tag),
            (is_upper_triangular, upper_triangular_tag),
            (is_positive_semidefinite, positive_semidefinite_tag),
            (is_negative_semidefinite, negative_semidefinite_tag),
            (has_unit_diagonal, unit_diagonal_tag),
            (is_tridiagonal, tridiagonal_tag),
        ]
        if check(operator)
    }
    dim_bound = min(operator.in_size(), operator.out_size())
    mr = max_rank(operator)
    # verify that adding a max rank tag wouldn't be redundant
    if mr < dim_bound:
        tags.add(MaxRankTag(mr))
    return frozenset(tags)


transpose_tags_rules = []


for tag in (
    symmetric_tag,
    unit_diagonal_tag,
    diagonal_tag,
    positive_semidefinite_tag,
    negative_semidefinite_tag,
    tridiagonal_tag,
):

    @transpose_tags_rules.append
    def _(tags: frozenset[object], tag=tag):
        if tag in tags:
            return tag


@transpose_tags_rules.append
def _(tags: frozenset[object]):
    if lower_triangular_tag in tags:
        return upper_triangular_tag


@transpose_tags_rules.append
def _(tags: frozenset[object]):
    if upper_triangular_tag in tags:
        return lower_triangular_tag


@transpose_tags_rules.append
def _(tags: frozenset[object]):
    rank_tags = [t for t in tags if isinstance(t, MaxRankTag)]
    if rank_tags:
        # drop redundant tags
        return min(rank_tags, key=lambda t: t.r)


def transpose_tags(tags: frozenset[object]):
    """Lineax uses "tags" to declare that a particular linear operator exhibits some
    property, e.g. symmetry.

    This function takes in a collection of tags representing a linear operator, and
    returns a collection of tags that should be associated with the transpose of that
    linear operator.

    **Arguments:**

    - `tags`: a `frozenset` of tags.

    **Returns:**

    A `frozenset` of tags.
    """
    if symmetric_tag in tags:
        return tags
    new_tags = []
    for rule in transpose_tags_rules:
        out = rule(tags)
        if out is not None:
            new_tags.append(out)
    return frozenset(new_tags)


invert_tags_rules = []


for tag in (
    symmetric_tag,
    diagonal_tag,
    lower_triangular_tag,
    upper_triangular_tag,
    positive_semidefinite_tag,
    negative_semidefinite_tag,
):

    @invert_tags_rules.append
    def _(tags: frozenset[object], tag=tag):
        if tag in tags:
            return tag


@invert_tags_rules.append
def _(tags: frozenset[object]):
    if unit_diagonal_tag in tags and (
        diagonal_tag in tags
        or lower_triangular_tag in tags
        or upper_triangular_tag in tags
    ):
        return unit_diagonal_tag


@invert_tags_rules.append
def _(tags: frozenset[object]):
    rank_tags = [t for t in tags if isinstance(t, MaxRankTag)]
    if rank_tags:
        # drop redundant tags
        return min(rank_tags, key=lambda t: t.r)


# tridiagonal_tag intentionally absent: inverse of tridiagonal matrix generally dense.


def invert_tags(tags: frozenset[object]) -> frozenset[object]:
    """Lineax uses "tags" to declare that a particular linear operator exhibits some
    property, e.g. symmetry.

    This function takes in a collection of tags representing a linear operator, and
    returns a collection of tags that should be associated with the (pseudo)inverse
    of that linear operator.

    Most structural properties are preserved by inversion (symmetric, diagonal,
    triangular, positive/negative semidefinite).  Notable exceptions:

    - `tridiagonal_tag` is **not** preserved — the inverse of a tridiagonal matrix
      is generally dense.
    - `unit_diagonal_tag` is only preserved when the operator is also diagonal or
      triangular.

    **Arguments:**

    - `tags`: a `frozenset` of tags.

    **Returns:**

    A `frozenset` of tags.
    """
    new_tags = []
    for rule in invert_tags_rules:
        out = rule(tags)
        if out is not None:
            new_tags.append(out)
    return frozenset(new_tags)
