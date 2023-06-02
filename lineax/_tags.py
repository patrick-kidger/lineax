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
