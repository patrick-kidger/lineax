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

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest


# ---------------------------------------------------------------------------
# MaxRankTag semantics
# ---------------------------------------------------------------------------


def test_max_rank_tag_equality():
    assert lx.MaxRankTag(5) == lx.MaxRankTag(5)
    assert lx.MaxRankTag(5) != lx.MaxRankTag(3)


def test_max_rank_tag_hashable_frozenset_dedup():
    s = frozenset({lx.MaxRankTag(5), lx.MaxRankTag(5)})
    assert len(s) == 1


def test_max_rank_tag_zero_valid():
    tag = lx.MaxRankTag(0)
    assert tag.r == 0


def test_max_rank_tag_negative_raises():
    with pytest.raises(ValueError):
        lx.MaxRankTag(-1)


def test_max_rank_tag_repr():
    assert repr(lx.MaxRankTag(7)) == "max_rank_tag(7)"


# ---------------------------------------------------------------------------
# Basic operator dispatch
# ---------------------------------------------------------------------------


def test_max_rank_matrix_no_tag():
    op = lx.MatrixLinearOperator(jnp.eye(4))
    assert lx.max_rank(op) == 4  # min(4, 4)


def test_max_rank_matrix_rectangular_no_tag():
    op = lx.MatrixLinearOperator(jnp.zeros((6, 3)))
    assert lx.max_rank(op) == 3  # min(6, 3)


def test_max_rank_matrix_with_tag():
    op = lx.MatrixLinearOperator(jnp.zeros((10, 10)), lx.MaxRankTag(3))
    assert lx.max_rank(op) == 3


def test_max_rank_matrix_tag_capped_by_dimension():
    # Tag claims rank 20 but matrix is only 4×4 — dimension wins.
    op = lx.MatrixLinearOperator(jnp.zeros((4, 4)), lx.MaxRankTag(20))
    assert lx.max_rank(op) == 4


def test_max_rank_tagged_operator():
    inner = lx.MatrixLinearOperator(jnp.zeros((8, 8)))
    op = lx.TaggedLinearOperator(inner, lx.MaxRankTag(2))
    assert lx.max_rank(op) == 2


def test_max_rank_tagged_operator_narrows_inner():
    inner = lx.MatrixLinearOperator(jnp.zeros((8, 8)), lx.MaxRankTag(5))
    op = lx.TaggedLinearOperator(inner, lx.MaxRankTag(3))
    assert lx.max_rank(op) == 3


def test_max_rank_identity():
    import jax

    struct = jax.ShapeDtypeStruct((5,), jnp.float32)
    op = lx.IdentityLinearOperator(input_structure=struct)
    assert lx.max_rank(op) == 5


def test_max_rank_diagonal():
    op = lx.DiagonalLinearOperator(jnp.ones(6))
    assert lx.max_rank(op) == 6


# ---------------------------------------------------------------------------
# Composition rules
# ---------------------------------------------------------------------------


def test_max_rank_composed_both_tagged():
    # n×k @ k×n: min(k, k) = k
    k, n = 3, 10
    U = lx.MatrixLinearOperator(jnp.zeros((n, k)), lx.MaxRankTag(k))
    Vt = lx.MatrixLinearOperator(jnp.zeros((k, n)), lx.MaxRankTag(k))
    assert lx.max_rank(U @ Vt) == k


def test_max_rank_composed_one_tagged():
    k, n = 3, 10
    U = lx.MatrixLinearOperator(jnp.zeros((n, k)), lx.MaxRankTag(k))
    A = lx.MatrixLinearOperator(jnp.zeros((k, n)))  # no tag → min(k,n)=k
    assert lx.max_rank(U @ A) == k


def test_max_rank_composed_chain():
    k, n = 5, 100
    U = lx.MatrixLinearOperator(jnp.zeros((n, k)), lx.MaxRankTag(k))
    C = lx.MatrixLinearOperator(jnp.zeros((k, k)), lx.MaxRankTag(k))
    Vt = lx.MatrixLinearOperator(jnp.zeros((k, n)), lx.MaxRankTag(k))
    assert lx.max_rank(U @ C @ Vt) == k


def test_max_rank_add_both_tagged():
    op1 = lx.MatrixLinearOperator(jnp.zeros((6, 6)), lx.MaxRankTag(2))
    op2 = lx.MatrixLinearOperator(jnp.zeros((6, 6)), lx.MaxRankTag(1))
    assert lx.max_rank(op1 + op2) == 3  # min(2+1, 6)


def test_max_rank_add_capped_by_dimension():
    op1 = lx.MatrixLinearOperator(jnp.zeros((4, 4)), lx.MaxRankTag(3))
    op2 = lx.MatrixLinearOperator(jnp.zeros((4, 4)), lx.MaxRankTag(3))
    assert lx.max_rank(op1 + op2) == 4  # min(3+3=6, 4) = 4


def test_max_rank_add_one_untagged():
    op1 = lx.MatrixLinearOperator(jnp.zeros((5, 5)), lx.MaxRankTag(2))
    op2 = lx.MatrixLinearOperator(jnp.zeros((5, 5)))  # no tag → 5
    assert lx.max_rank(op1 + op2) == 5  # min(2+5, 5) = 5


# ---------------------------------------------------------------------------
# Scalar multiplication
# ---------------------------------------------------------------------------


def test_max_rank_scalar_mul_propagates():
    op = lx.MatrixLinearOperator(jnp.zeros((5, 5)), lx.MaxRankTag(2))
    assert lx.max_rank(3.0 * op) == 2


def test_max_rank_scalar_zero_gives_rank_zero():
    op = lx.MatrixLinearOperator(jnp.zeros((5, 5)), lx.MaxRankTag(2))
    assert lx.max_rank(0 * op) == 0


def test_max_rank_neg_propagates():
    op = lx.MatrixLinearOperator(jnp.zeros((5, 5)), lx.MaxRankTag(2))
    assert lx.max_rank(-op) == 2


def test_max_rank_div_propagates():
    op = lx.MatrixLinearOperator(jnp.zeros((5, 5)), lx.MaxRankTag(2))
    assert lx.max_rank(op / 2.0) == 2


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


def test_max_rank_transpose_preserves_tag():
    op = lx.MatrixLinearOperator(jnp.zeros((10, 3)), lx.MaxRankTag(3))
    assert lx.max_rank(op.T) == 3


def test_max_rank_tagged_operator_transpose():
    inner = lx.MatrixLinearOperator(jnp.zeros((8, 8)))
    op = lx.TaggedLinearOperator(inner, lx.MaxRankTag(4))
    assert lx.max_rank(op.T) == 4


# ---------------------------------------------------------------------------
# Invert tags
# ---------------------------------------------------------------------------


def test_max_rank_invert_tags_preserves_bound():
    from lineax._tags import invert_tags

    tags = frozenset({lx.MaxRankTag(3)})
    result = invert_tags(tags)
    assert lx.MaxRankTag(3) in result


def test_max_rank_invert_tags_takes_min():
    from lineax._tags import invert_tags

    tags = frozenset({lx.MaxRankTag(5), lx.MaxRankTag(2)})
    result = invert_tags(tags)
    rank_tags = [t for t in result if isinstance(t, lx.MaxRankTag)]
    assert len(rank_tags) == 1
    assert rank_tags[0].r == 2


def test_max_rank_invert_tags_absent_when_no_rank_tag():
    from lineax._tags import invert_tags

    tags = frozenset({lx.symmetric_tag})
    result = invert_tags(tags)
    assert not any(isinstance(t, lx.MaxRankTag) for t in result)


# ---------------------------------------------------------------------------
# SVD truncation
# ---------------------------------------------------------------------------


def test_svd_truncates_state_to_max_rank():
    # A genuinely rank-2 matrix, declared rank 2: SVD state is truncated to 2
    # components, and the solution matches the untruncated solve.
    u = jax.random.normal(jax.random.PRNGKey(0), (10, 2))
    v = jax.random.normal(jax.random.PRNGKey(1), (10, 2))
    matrix = u @ v.T
    solver = lx.SVD()

    plain = lx.MatrixLinearOperator(matrix)
    (u_full, s_full, vt_full), _ = solver.init(plain, {})
    assert s_full.shape == (10,)

    tagged = lx.MatrixLinearOperator(matrix, lx.MaxRankTag(2))
    (u_t, s_t, vt_t), _ = solver.init(tagged, {})
    assert u_t.shape == (10, 2)
    assert s_t.shape == (2,)
    assert vt_t.shape == (2, 10)

    vector = jnp.arange(10.0) + 0.5
    x_plain = lx.linear_solve(plain, vector, solver).value
    x_tagged = lx.linear_solve(tagged, vector, solver).value
    assert jnp.allclose(x_plain, x_tagged)


def test_svd_raises_when_max_rank_too_small():
    # A genuinely rank-3 matrix declared rank 2: truncation would discard a
    # singular value above the rcond threshold, so the solve must raise.
    u = jax.random.normal(jax.random.PRNGKey(0), (10, 3))
    v = jax.random.normal(jax.random.PRNGKey(1), (10, 3))
    matrix = u @ v.T
    operator = lx.MatrixLinearOperator(matrix, lx.MaxRankTag(2))
    vector = jnp.arange(10.0) + 0.5
    with pytest.raises(eqx.EquinoxRuntimeError):
        lx.linear_solve(operator, vector, lx.SVD())
