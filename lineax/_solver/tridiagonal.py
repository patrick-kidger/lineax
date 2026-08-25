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

from typing import Any, TypeAlias

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._misc import unit_phase
from .._operator import AbstractLinearOperator, is_tridiagonal, tridiagonal
from .._solution import RESULTS
from .base import AbstractDirectLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_TridiagonalState: TypeAlias = tuple[tuple[Array, Array, Array], PackedStructures]


class Tridiagonal(AbstractDirectLinearSolver[_TridiagonalState]):
    """Tridiagonal solver for linear systems, uses the LAPACK/cusparse implementation
    of Gaussian elimination with partial pivotting (which increases stability).
    ."""

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`Tridiagonal` may only be used for linear solves with square matrices"
            )
        if not is_tridiagonal(operator):
            raise ValueError(
                "`Tridiagonal` may only be used for linear solves with tridiagonal "
                "matrices"
            )
        return tridiagonal(operator), pack_structures(operator)

    def compute(
        self,
        state: _TridiagonalState,
        vector,
        options,
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        (diagonal, lower_diagonal, upper_diagonal), packed_structures = state
        del state, options
        vector = ravel_vector(vector, packed_structures)

        solution = lax.linalg.tridiagonal_solve(
            jnp.append(0.0, lower_diagonal),
            diagonal,
            jnp.append(upper_diagonal, 0.0),
            vector[:, None],
        ).flatten()

        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _TridiagonalState, options: dict[str, Any]):
        (diagonal, lower_diagonal, upper_diagonal), packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_diagonals = (diagonal, upper_diagonal, lower_diagonal)
        transpose_state = (transpose_diagonals, transposed_packed_structures)
        return transpose_state, options

    def conj(self, state: _TridiagonalState, options: dict[str, Any]):
        (diagonal, lower_diagonal, upper_diagonal), packed_structures = state
        conj_diagonals = (diagonal.conj(), lower_diagonal.conj(), upper_diagonal.conj())
        conj_state = (conj_diagonals, packed_structures)
        return conj_state, options

    def slogdet(
        self, state: _TridiagonalState, options: dict[str, Any]
    ) -> tuple[Array, Array]:
        del options
        (diagonal, lower_diagonal, upper_diagonal), _ = state
        if diagonal.shape[0] == 1:
            return unit_phase(diagonal[0]), jnp.log(jnp.abs(diagonal[0]))
        # Two evaluations of the same three-term minor recurrence, differing only in how
        # they associate it: sequentially, which is optimal on CPU, or as a tree, which
        # is what GPUs need. See the block comment below the class.
        return lax.platform_dependent(
            diagonal,
            lower_diagonal,
            upper_diagonal,
            cpu=_slogdet_scan,
            default=_slogdet_associative_reduce,
        )

    def assume_full_rank(self):
        return True


Tridiagonal.__init__.__doc__ = """**Arguments:**

Nothing.
"""


# ----------------------------------------------------------------------------------
# Determinants.
#
# Both implementations evaluate the same three-term recurrence on the leading
# principal minors D_i = det(A[:i+1, :i+1]):
#
#     D_i = d_i D_{i-1} - l_{i-1} u_{i-1} D_{i-2},      D_{-1} = 1, D_0 = d_0
#
# with det(A) = D_{n-1}. This is division-free, unlike the LU-pivot recurrence
# p_i = d_i - l_{i-1} u_{i-1} / p_{i-1}, which divides by zero as soon as any *leading*
# minor is singular -- even when A itself is invertible, e.g. [[0, 1], [1, 0]].
#
# They differ only in how the recurrence is associated:
#
#   * `_slogdet_scan` walks it in order, which is optimal on CPU.
#   * `_slogdet_associative_reduce` rewrites it as a product of 2x2 transfer
#     matrices and reassociates that product into a balanced tree, which is what GPUs
#     need: there, each
#     `lax.scan` iteration costs a kernel launch (~8us) whatever the work inside it,
#     so the sequential form costs ~8us * n / block: on an A100 in float64 that is 15x
#     slower at n = 512, 199x at n = 8192, and 3377x at n = 131072.
#
# Both names refer to how the product is associated, not to the JAX primitives used:
# the tree runs a short `lax.scan` within each chunk too. It is a *reduction* rather
# than a scan because only the final product is wanted, not every prefix, which is what
# makes it cheaper than `lax.associative_scan`; and it is built by an iterative loop
# over levels, not by recursion.
#
# `Tridiagonal.slogdet` picks between them with `lax.platform_dependent`. `lx.slogdet`'s
# JVP rule differentiates that, so each platform also reverses the implementation it
# uses for the primal, which is what you want on both. On GPU the tree's primal
# advantage decides it. On CPU the scan wins on both counts: in float64 at n = 8192 its
# primal is 85us against 171us, and its backward pass costs a further 1.4-1.8x against
# 3.9-12.3x for the tree (reversing a scan is a second serial pass over stored
# residuals; reversing a tree is another tree, but over much more data).
# ----------------------------------------------------------------------------------

# Number of raw recurrence steps between renormalisations. Bigger blocks amortise the
# renormalisation over more steps; smaller blocks let the minors decay further before a
# block underflows. A block underflows once the minors decay past float range within
# it, i.e. roughly when R * K > 308 for an operator whose entries span 10**-R, so this
# caps the tolerated grading. The heuristic is accurate for K >= 8 and optimistic below
# it, where the tree's own ceiling near 68 decades binds first instead. At K = 4 both
# implementations land around 60 decades -- the exact figure moves by a few either way
# with the draw -- so the GPU path costs no tolerance relative to the scan.
#
# Past that limit neither path fails cleanly. A block's minors decay into the denormal
# range before the renormalisation sees them, losing precision but staying finite, so
# above the limit there is a band of R where the result is silently wrong rather than
# -inf: at K = 4, `_slogdet_scan` is wrong by up to 1.3e-3 relative in `lad` -- a factor
# of e**14 in |det| -- for R in 66..79 before underflowing at 80. This is inherent to
# renormalising in blocks and is not specific to K = 4: K = 2 has the same band, at
# roughly double the R.
#
# The binding quantity is the exponent range, so float32 divides every figure by about
# eight: at K = 4 the limit is 8 decades, silently wrong at 9, underflow from 10.
#
# 4 is the largest K whose limit clears the 40 decades that
# `test_tridiagonal_slogdet_graded` pins -- 8 drops it to 38 and fails outright -- and
# the smallest that beats LAPACK `gttrf` on CPU at every size (1.1-1.5x unbatched,
# 2.2-3.0x batched). It costs nothing on GPU, where the tree's launch count is
# `block + log_radix(n / block)` rather than `n / block`. `benchmarks/determinant_speeds
# .py` regenerates every figure quoted here.
_SLOGDET_BLOCK = 4

# Fan-in of the tree: how many chunk transfer matrices are multiplied together between
# renormalisations. Only `_slogdet_associative_reduce` has a tree, so this affects
# GPU only.
#
# On well-scaled operators the radix barely matters: 2 through 32 agree to within a
# factor of 6 in relative error, even on near-defective ones where the transfer
# matrices are close to rank one. What it does cost is grading tolerance, and there it
# is a cliff rather than a gradient, for the same reason `_SLOGDET_BLOCK` is -- a group
# spans `radix` times as many steps between renormalisations:
#
#     radix:            2     4     8    16    32
#     tolerated R:     66    66    44    26    18
#
# 4 is the largest radix that costs nothing: it holds whatever the block length allows,
# and beyond it the tolerance halves each time -- at 16, an operator
# whose entries merely span 30 decades already comes back non-finite. That is worth
# more than the ~15% 8 would buy (A100, float64, n = 131072: 65, 50, 43, 49, 53us of
# device time for 2, 4, 8, 16, 32; 8 is the fastest, 2 reliably the slowest, since
# fewer matrices per level means more levels and so more kernel launches).
#
# `benchmarks/determinant_speeds.py` regenerates all of this. Note when re-measuring
# that `jax.jit` caches on function identity, so a sweep that jits the same function
# object while mutating this constant silently measures the first setting every time.
_SLOGDET_RADIX = 4


def _prescale(
    diagonal: Array, lower_diagonal: Array, upper_diagonal: Array
) -> tuple[Array, Array, Array]:
    """Rescale the operator by a power of two so that every entry is bounded by 1.

    The minors grow or decay by roughly one entry-magnitude per step, so for an operator
    whose entries are far from unit scale they leave float range fast. Normalising the
    whole operator first uses det(A) = sigma**n det(A / sigma), i.e.

        logabsdet(A) = n log(sigma) + logabsdet(A / sigma)

    Choosing sigma as a power of two makes the rescale exact, and choosing it to bound
    both scaled arrays by 1 bounds the growth to |D_i| <= |D_{i-1}| + |D_{i-2}|, i.e. at
    most 2**block per block -- so the recurrence can no longer overflow at all. This is
    a single reduction, outside the serial loop.

    The rescale itself gives up in the outermost binade at each end of the float range,
    where `frexp`/`ldexp` cannot represent `sigma`: an operator whose largest entry is
    at least `2**1023` (`2**127` in float32) needs a subnormal `sigma_inv`, and one
    whose entries are all subnormal has no exponent for `frexp` to report. Either way
    the result is non-finite or `-inf` rather than silently wrong, and the platforms
    disagree about which.

    Returns the scaled diagonal, the scaled `coupling` l_i u_i, and log2(sigma).
    """
    dtype = diagonal.dtype
    real_dtype = jnp.finfo(dtype).dtype
    zero = jnp.zeros((), real_dtype)
    # The recurrence's coupling scale is `sqrt(|l u|)`, but forming `lower * upper` to
    # get it overflows to `inf` (and thence `nan`) once |entries| exceed
    # sqrt(max_float) -- only ~1.8e19 in float32, precisely the badly-scaled operators
    # this rescale exists to handle. Take the square roots separately instead: that
    # never forms the product, and unlike `max(|l|, |u|)` it stays correct when the two
    # off-diagonals are wildly different sizes (`l = 1e77`, `u = 1e-77` has coupling 1).
    magnitude = jnp.maximum(
        jnp.max(jnp.abs(diagonal), initial=zero),
        jnp.sqrt(jnp.max(jnp.abs(lower_diagonal), initial=zero))
        * jnp.sqrt(jnp.max(jnp.abs(upper_diagonal), initial=zero)),
    )
    _, exponent = jnp.frexp(magnitude)
    sigma_inv = jnp.ldexp(jnp.ones((), real_dtype), -exponent).astype(dtype)
    coupling = (lower_diagonal * sigma_inv) * (upper_diagonal * sigma_inv)
    return diagonal * sigma_inv, coupling, exponent


def _pad_steps(
    diagonal: Array, coupling: Array, block: int
) -> tuple[Array, Array, int]:
    """Pad the `n - 1` recurrence steps up to a whole number of blocks.

    The padding steps are `(d, coupling) = (1, 0)`, which map D_i -> D_{i-1} and so
    leave the value alone. They go at the very end.
    """
    dtype = diagonal.dtype
    steps = diagonal.shape[0] - 1
    pad = (-steps) % block
    if pad:
        diagonal = jnp.concatenate([diagonal[1:], jnp.ones((pad,), dtype)])
        coupling = jnp.concatenate([coupling, jnp.zeros((pad,), dtype)])
    else:
        diagonal = diagonal[1:]
    return diagonal, coupling, (steps + pad) // block


def _slogdet_scan(
    diagonal: Array, lower_diagonal: Array, upper_diagonal: Array
) -> tuple[Array, Array]:
    """Walk the minor recurrence in order. Optimal on CPU."""
    n = diagonal.shape[0]
    dtype = diagonal.dtype
    real_dtype = jnp.finfo(dtype).dtype
    block = _SLOGDET_BLOCK
    diagonal, coupling, exponent = _prescale(diagonal, lower_diagonal, upper_diagonal)

    def unit_step(carry, args):
        prev, curr = carry
        d_i, coupling_i = args
        return (curr, d_i * curr - coupling_i * prev), None

    def block_step(carry, args):
        (prev, curr), _ = lax.scan(unit_step, carry, args, unroll=block)
        # `lad` is invariant to `scale`: it divides the minors and is added back as
        # `log(scale)`, so the two contributions cancel exactly. Differentiating it
        # anyway costs a `1/scale**2` in reverse mode, which overflows once a block's
        # scale drops below sqrt(smallest normal) -- halving the grading a *gradient*
        # tolerates relative to the primal. Stopping it is exact, not an approximation.
        scale = lax.stop_gradient(jnp.maximum(jnp.abs(prev), jnp.abs(curr)))
        # A singular A drives both minors to exactly zero. Dividing by `scale` would
        # turn that into `nan`; holding the pair at zero instead lets the `log(scale)`
        # sum absorb the `-inf` and yields `(sign, lad) = (0, -inf)`, matching
        # `jnp.linalg.slogdet` on a singular input.
        nonzero = jnp.where(scale == 0, 1.0, scale)
        # Reciprocal-and-multiply rather than two divides: measurably faster, and
        # `nonzero >=` the larger minor keeps the reciprocal in range.
        inv_scale = (1.0 / nonzero).astype(dtype)
        # The scales are emitted rather than accumulated so that their logs happen in
        # one vectorised pass instead of once per step of the serial loop.
        return (prev * inv_scale, curr * inv_scale), scale

    diagonal_rest, coupling_rest, num_blocks = _pad_steps(diagonal, coupling, block)

    # The initial pair is (D_{-1}, D_0) = (1, d_0), which needs no renormalising:
    # `|d_0| <= 1` after the prescale.
    init = (jnp.ones((), dtype), diagonal[0])
    (_, det), scales = lax.scan(
        block_step,
        init,
        (
            diagonal_rest.reshape(num_blocks, block),
            coupling_rest.reshape(num_blocks, block),
        ),
    )
    lad = (
        n * exponent.astype(real_dtype) * jnp.log(jnp.array(2.0, real_dtype))
        + jnp.sum(jnp.log(scales))
        + jnp.log(jnp.abs(det))
    )
    return unit_phase(det), lad


# The same recurrence, written as a product of 2x2 transfer matrices:
#
#     (D_{i-1}, D_i) = M_i (D_{i-2}, D_{i-1}),    M_i = [[0, 1], [-l u, d_i]]
#
# so det(A) is the second component of (M_{n-1} ... M_1) (1, d_0). Matrix products
# associate, so the product can be evaluated as a tree of depth log(n) rather than a
# chain of length n. The price is 4 multiplies per step instead of 2 (a chunk must
# propagate both basis vectors, not knowing its incoming pair), which is irrelevant on
# a GPU that was launch-bound anyway.
#
# The matrices are carried as four separate arrays rather than one `(m, 2, 2)` array:
# `a @ b` on stacked 2x2s lowers to a batched `dot_general`, i.e. an unfusable cuBLAS
# call. At n = 2048 x batch 4096 in float64 that costs 25.7ms against 2.9-3.5ms for the
# component-wise form, a 7-9x difference.


def _mul(hi: tuple, lo: tuple) -> tuple:
    """Product of two 2x2 matrices held as component tuples: `hi @ lo`."""
    a1, b1, c1, d1 = hi
    a2, b2, c2, d2 = lo
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
    )


def _normalise(m: tuple, log_scale: Array) -> tuple[tuple, Array]:
    """Divide out the largest entry, accumulating its log."""
    a, b, c, d = m
    # Exactly as in `_slogdet_scan`: the scale cancels, so stop it rather than
    # pay a `1/scale**2` in reverse mode. Here that is a real trade rather than a free
    # win -- it raises the grading a gradient tolerates from ~52 decades to ~64, so
    # that the gradient matches the primal, but costs 1.6x in gradient time (A100,
    # float64, n = 131072: 128us against 210us). Deleting the `stop_gradient` takes the
    # speed back and reinstates the cliff. Normalising by an exact power of two instead,
    # whose `frexp` exponent has a structurally zero derivative, gives the same
    # tolerance for the same 1.6x, so the cost is inherent rather than an artifact.
    scale = lax.stop_gradient(
        jnp.maximum(
            jnp.maximum(jnp.abs(a), jnp.abs(b)), jnp.maximum(jnp.abs(c), jnp.abs(d))
        )
    )
    # As in `_slogdet_scan`: hold an exactly-singular block at zero rather than
    # producing `nan`, and let the `-inf` appear in the final log.
    inv = (1.0 / jnp.where(scale == 0, 1.0, scale)).astype(a.dtype)
    return (a * inv, b * inv, c * inv, d * inv), log_scale + jnp.log(scale)


def _reduce_level(m: tuple, log_scale: Array, size: int, radix: int):
    """Multiply groups of `radix` neighbouring matrices, renormalising once per group.

    Renormalising once per *group* rather than once per product matters: dividing by
    the largest entry rounds all four entries, and for a near-defective operator (the
    discrete Laplacian being the standard example) the transfer matrices are close to
    rank 1, so that rounding lands in the small entries which carry the answer and is
    then amplified by every later product. Grouping keeps the count of roundings down.
    """
    rem = -size % radix
    if rem:
        a, b, c, d = m
        one = jnp.ones((rem,), a.dtype)
        zero = jnp.zeros((rem,), a.dtype)
        # Identity matrices at the high end leave the product unchanged.
        m = (
            jnp.concatenate([a, one]),
            jnp.concatenate([b, zero]),
            jnp.concatenate([c, zero]),
            jnp.concatenate([d, one]),
        )
        log_scale = jnp.concatenate([log_scale, jnp.zeros((rem,), log_scale.dtype)])
        size += rem
    groups = size // radix
    m = tuple(x.reshape(groups, radix) for x in m)
    log_scale = log_scale.reshape(groups, radix).sum(axis=1)
    acc = tuple(x[:, 0] for x in m)
    for j in range(1, radix):
        # Matrix j applies *after* matrix j - 1, so it multiplies on the left.
        acc = _mul(tuple(x[:, j] for x in m), acc)
    acc, log_scale = _normalise(acc, log_scale)
    return acc, log_scale, groups


def _slogdet_associative_reduce(
    diagonal: Array, lower_diagonal: Array, upper_diagonal: Array
) -> tuple[Array, Array]:
    """Evaluate the transfer-matrix product with a tree. What GPUs want.

    The `n - 1` steps are cut into chunks of `_SLOGDET_BLOCK`; each chunk accumulates
    its own transfer matrix sequentially (one unrolled scan, all chunks in parallel),
    renormalising once at the end, and the chunk matrices are then combined by a
    radix-`_SLOGDET_RADIX` tree. Total kernel launches are `block + log_radix(n/block)`
    rather than `n / block`.
    """
    n = diagonal.shape[0]
    dtype = diagonal.dtype
    real_dtype = jnp.finfo(dtype).dtype
    block = _SLOGDET_BLOCK
    diagonal, coupling, exponent = _prescale(diagonal, lower_diagonal, upper_diagonal)
    diagonal_rest, coupling_rest, n_chunks = _pad_steps(diagonal, coupling, block)
    # (n_chunks, block) -> (block, n_chunks): scan the sequential axis, with the chunks
    # vectorised across it.
    d_steps = diagonal_rest.reshape(n_chunks, block).T
    c_steps = coupling_rest.reshape(n_chunks, block).T

    def unit_step(carry, args):
        # One step maps the chunk's matrix [[a, b], [c, d]] to
        # [[c, d], [d_i c - k_i a, d_i d - k_i b]].
        a, b, c, d = carry
        d_i, k_i = args
        return (c, d, d_i * c - k_i * a, d_i * d - k_i * b), None

    ones = jnp.ones((n_chunks,), dtype)
    zeros = jnp.zeros((n_chunks,), dtype)
    m, _ = lax.scan(
        unit_step, (ones, zeros, zeros, ones), (d_steps, c_steps), unroll=block
    )
    m, log_scale = _normalise(m, jnp.zeros((n_chunks,), real_dtype))

    size = n_chunks
    while size > 1:
        m, log_scale, size = _reduce_level(
            m, log_scale, size, min(_SLOGDET_RADIX, size)
        )

    # Apply the total product to (D_{-1}, D_0) = (1, d_0); the second component is
    # det(A / sigma). `|d_0| <= 1` after the prescale, so this needs no renormalising.
    _, _, c, d = (x[0] for x in m)
    det = c + d * diagonal[0]
    lad = (
        n * exponent.astype(real_dtype) * jnp.log(jnp.array(2.0, real_dtype))
        + log_scale[0]
        + jnp.log(jnp.abs(det))
    )
    return unit_phase(det), lad
