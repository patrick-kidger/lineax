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

import abc
import functools as ft
from typing import Any, Generic, Optional, TypeVar
from typing_extensions import TypeAlias

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.core
import jax.interpreters.ad as ad
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, PyTree

from ._custom_types import sentinel
from ._misc import inexact_asarray
from ._operator import (
    AbstractLinearOperator,
    IdentityLinearOperator,
    is_diagonal,
    is_lower_triangular,
    is_negative_semidefinite,
    is_positive_semidefinite,
    is_tridiagonal,
    is_upper_triangular,
    linearise,
    TangentLinearOperator,
)
from ._solution import RESULTS, Solution


#
# _linear_solve_p
#


def _to_shapedarray(x):
    if isinstance(x, jax.ShapeDtypeStruct):
        return jax.core.ShapedArray(x.shape, x.dtype)
    else:
        return x


def _to_struct(x):
    if isinstance(x, jax.core.ShapedArray):
        return jax.ShapeDtypeStruct(x.shape, x.dtype)
    elif isinstance(x, jax.core.AbstractValue):
        raise NotImplementedError(
            "`lineax.linear_solve` only supports working with JAX arrays; not "
            f"other abstract values. Got abstract value {x}."
        )
    else:
        return x


def _assert_false(x):
    assert False


def _is_none(x):
    return x is None


def _sum(*args):
    return sum(args)


def _linear_solve_impl(_, state, vector, options, solver, throw, *, check_closure):
    out = solver.compute(state, vector, options)
    if check_closure:
        out = eqxi.nontraceable(
            out, name="lineax.linear_solve with respect to a closed-over value"
        )
    solution, result, stats = out
    has_nonfinites = jnp.any(
        jnp.stack(
            [jnp.any(jnp.invert(jnp.isfinite(x))) for x in jtu.tree_leaves(solution)]
        )
    )
    result = RESULTS.where(
        (result == RESULTS.successful) & has_nonfinites,
        RESULTS.singular,
        result,
    )
    if throw:
        solution, result, stats = result.error_if(
            (solution, result, stats),
            result != RESULTS.successful,
        )
    return solution, result, stats


@eqxi.filter_primitive_def
def _linear_solve_abstract_eval(operator, state, vector, options, solver, throw):
    state, vector, options, solver = jtu.tree_map(
        _to_struct, (state, vector, options, solver)
    )
    out = eqx.filter_eval_shape(
        _linear_solve_impl,
        operator,
        state,
        vector,
        options,
        solver,
        throw,
        check_closure=False,
    )
    out = jtu.tree_map(_to_shapedarray, out)
    return out


@eqxi.filter_primitive_jvp
def _linear_solve_jvp(primals, tangents):
    operator, state, vector, options, solver, throw = primals
    t_operator, t_state, t_vector, t_options, t_solver, t_throw = tangents
    jtu.tree_map(_assert_false, (t_state, t_options, t_solver, t_throw))
    del t_state, t_options, t_solver, t_throw

    # Note that we pass throw=True unconditionally to all the tangent solves, as there
    # is nowhere we can pipe their error to.
    # This is the primal solve so we can respect the original `throw`.
    solution, result, stats = eqxi.filter_primitive_bind(
        linear_solve_p, operator, state, vector, options, solver, throw
    )

    #
    # Consider the primal problem of linearly solving for x in Ax=b.
    # Let ^ denote pseudoinverses, ᵀ denote transposes, and ' denote tangents.
    # The linear_solve routine returns specifically the pseudoinverse solution, i.e.
    #
    # x = A^b
    #
    # Therefore x' = A^'b + A^b'
    #
    # Now A^' = -A^A'A^ + A^A^ᵀAᵀ'(I - AA^) + (I - A^A)Aᵀ'A^ᵀA^
    #
    # (Source: https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Derivative)
    #
    # This results in:
    #
    # x' = A^(-A'x + A^ᵀAᵀ'(b - Ax) - Ay + b') + y
    #
    # where
    #
    # y = Aᵀ'A^ᵀx
    #
    # note that if A has linearly independent columns, then the y - A^Ay
    # term disappears and gives
    #
    # x' = A^(-A'x + A^ᵀAᵀ'(b - Ax) + b')
    #
    # and if A has linearly independent rows, then the A^A^ᵀAᵀ'(b - Ax) term
    # disappears giving:
    #
    # x' = A^(-A'x - Ay + b') + y
    #
    # if A has linearly independent rows and columns, then A is nonsingular and
    #
    # x' = A^(-A'x + b')

    vecs = []
    sols = []
    if any(t is not None for t in jtu.tree_leaves(t_vector, is_leaf=_is_none)):
        # b' term
        vecs.append(
            jtu.tree_map(eqxi.materialise_zeros, vector, t_vector, is_leaf=_is_none)
        )
    if any(t is not None for t in jtu.tree_leaves(t_operator, is_leaf=_is_none)):
        t_operator = TangentLinearOperator(operator, t_operator)
        t_operator = linearise(t_operator)  # optimise for matvecs
        # -A'x term
        vec = (-t_operator.mv(solution) ** ω).ω
        vecs.append(vec)
        allow_dependent_rows = solver.allow_dependent_rows(operator)
        allow_dependent_columns = solver.allow_dependent_columns(operator)
        if allow_dependent_rows or allow_dependent_columns:
            operator_transpose = operator.transpose()
            t_operator_transpose = t_operator.transpose()
            state_transpose, options_transpose = solver.transpose(state, options)
        if allow_dependent_rows:
            lst_sqr_diff = (vector**ω - operator.mv(solution) ** ω).ω
            tmp = t_operator_transpose.mv(lst_sqr_diff)  # pyright: ignore
            tmp, _, _ = eqxi.filter_primitive_bind(
                linear_solve_p,
                operator_transpose,  # pyright: ignore
                state_transpose,  # pyright: ignore
                tmp,
                options_transpose,  # pyright: ignore
                solver,
                True,
            )
            vecs.append(tmp)

        if allow_dependent_columns:
            tmp1, _, _ = eqxi.filter_primitive_bind(
                linear_solve_p,
                operator_transpose,  # pyright: ignore
                state_transpose,  # pyright:ignore
                solution,
                options_transpose,  # pyright: ignore
                solver,
                True,
            )
            tmp2 = t_operator_transpose.mv(tmp1)  # pyright: ignore
            # tmp2 is the y term
            tmp3 = operator.mv(tmp2)
            tmp4 = (-(tmp3**ω)).ω
            # tmp4 is the Ay term
            vecs.append(tmp4)
            sols.append(tmp2)
    vecs = jtu.tree_map(_sum, *vecs)
    # the A^ term at the very beginning
    sol, _, _ = eqxi.filter_primitive_bind(
        linear_solve_p, operator, state, vecs, options, solver, True
    )
    sols.append(sol)
    t_solution = jtu.tree_map(_sum, *sols)

    out = solution, result, stats
    t_out = (
        t_solution,
        jtu.tree_map(lambda _: None, result),
        jtu.tree_map(lambda _: None, stats),
    )
    return out, t_out


def _is_undefined(x):
    return isinstance(x, ad.UndefinedPrimal)


def _assert_defined(x):
    assert not _is_undefined(x)


def _keep_undefined(v, ct):
    if _is_undefined(v):
        return ct
    else:
        return None


@eqxi.filter_primitive_transpose(materialise_zeros=True)  # pyright: ignore
def _linear_solve_transpose(inputs, cts_out):
    cts_solution, _, _ = cts_out
    operator, state, vector, options, solver, _ = inputs
    jtu.tree_map(
        _assert_defined, (operator, state, options, solver), is_leaf=_is_undefined
    )
    operator_transpose = operator.transpose()
    state_transpose, options_transpose = solver.transpose(state, options)
    cts_vector, _, _ = eqxi.filter_primitive_bind(
        linear_solve_p,
        operator_transpose,
        state_transpose,
        cts_solution,
        options_transpose,
        solver,
        True,  # throw=True unconditionally: nowhere to pipe result to.
    )
    cts_vector = jtu.tree_map(
        _keep_undefined, vector, cts_vector, is_leaf=_is_undefined
    )
    operator_none = jtu.tree_map(lambda _: None, operator)
    state_none = jtu.tree_map(lambda _: None, state)
    options_none = jtu.tree_map(lambda _: None, options)
    solver_none = jtu.tree_map(lambda _: None, solver)
    throw_none = None
    return operator_none, state_none, cts_vector, options_none, solver_none, throw_none


# Call with `check_closure=False` so that the autocreated vmap rule works.
linear_solve_p = eqxi.create_vprim(
    "linear_solve",
    eqxi.filter_primitive_def(ft.partial(_linear_solve_impl, check_closure=False)),
    _linear_solve_abstract_eval,
    _linear_solve_jvp,
    _linear_solve_transpose,
)
# Then rebind so that the impl rule catches leaked-in tracers.
linear_solve_p.def_impl(
    eqxi.filter_primitive_def(ft.partial(_linear_solve_impl, check_closure=True))
)
eqxi.register_impl_finalisation(linear_solve_p)


#
# linear_solve
#


_SolverState = TypeVar("_SolverState")


class AbstractLinearSolver(eqx.Module, Generic[_SolverState]):
    """Abstract base class for all linear solvers."""

    @abc.abstractmethod
    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _SolverState:
        """Do any initial computation on just the `operator`.

        For example, an LU solver would compute the LU decomposition of the operator
        (and this does not require knowing the vector yet).

        It is common to need to solve the linear system `Ax=b` multiple times in
        succession, with the same operator `A` and multiple vectors `b`. This method
        improves efficiency by making it possible to re-use the computation performed
        on just the operator.

        !!! Example

            ```python
            operator = lx.MatrixLinearOperator(...)
            vector1 = ...
            vector2 = ...
            solver = lx.LU()
            state = solver.init(operator, options={})
            solution1 = lx.linear_solve(operator, vector1, solver, state=state)
            solution2 = lx.linear_solve(operator, vector2, solver, state=state)
            ```

        **Arguments:**

        - `operator`: a linear operator.
        - `options`: a dictionary of any extra options that the solver may wish to
            accept.

        **Returns:**

        A PyTree of arbitrary Python objects.
        """

    @abc.abstractmethod
    def compute(
        self, state: _SolverState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        """Solves a linear system.

        **Arguments:**

        - `state`: as returned from [`lineax.AbstractLinearSolver.init`][].
        - `vector`: the vector to solve against.
        - `options`: a dictionary of any extra options that the solver may wish to
            accept. For example, [`lineax.CG`][] accepts a `preconditioner` option.

        **Returns:**

        A 3-tuple of:

        - The solution to the linear system.
        - An integer indicating the success or failure of the solve. This is an integer
            which may be converted to a human-readable error message via
            `lx.RESULTS[...]`.
        - A dictionary of an extra statistics about the solve, e.g. the number of steps
            taken.
        """

    @abc.abstractmethod
    def allow_dependent_columns(self, operator: AbstractLinearOperator) -> bool:
        """Does this method ever produce non-NaN outputs for operators with linearly
        dependent columns? (Even if only sometimes.)

        If `True` then a more expensive backward pass is needed, to account for the
        extra generality.

        If you do not need to autodifferentiate through a custom linear solver then you
        simply define this method as
        ```python
        class MyLinearSolver(AbstractLinearsolver):
            def allow_dependent_columns(self, operator):
                raise NotImplementedError
        ```

        **Arguments:**

        - `operator`: a linear operator.

        **Returns:**

        Either `True` or `False`.
        """

    @abc.abstractmethod
    def allow_dependent_rows(self, operator: AbstractLinearOperator) -> bool:
        """Does this method ever produce non-NaN outputs for operators with
        linearly dependent rows? (Even if only sometimes)

        If `True` then a more expensive backward pass is needed, to account for the
        extra generality.

        If you do not need to autodifferentiate through a custom linear solver then you
        simply define this method as
        ```python
        class MyLinearSolver(AbstractLinearsolver):
            def allow_dependent_rows(self, operator):
                raise NotImplementedError
        ```

        **Arguments:**

        - `operator`: a linear operator.

        **Returns:**

        Either `True` or `False`.
        """

    @abc.abstractmethod
    def transpose(
        self, state: _SolverState, options: dict[str, Any]
    ) -> tuple[_SolverState, dict[str, Any]]:
        """Transposes the result of [`lineax.AbstractLinearSolver.init`][].

        That is, it should be the case that
        ```python
        state_transpose, _ = solver.transpose(solver.init(operator, options), options)
        state_transpose2 = solver.init(operator.T, options)
        ```
        must be identical to each other.

        It is relatively common (in particular when differentiating through a linear
        solve) to need to solve both `Ax = b` and `A^T x = b`. This method makes it
        possible to avoid computing both `solver.init(operator)` and
        `solver.init(operator.T)` if one can be cheaply computed from the other.

        **Arguments:**

        - `state`: as returned from `solver.init`.
        - `options`: any extra options that were passed to `solve.init`.

        **Returns:**

        A 2-tuple of:

        - The state of the transposed operator.
        - The options for the transposed operator.
        """


_qr_token = eqxi.str2jax("qr_token")
_diagonal_token = eqxi.str2jax("diagonal_token")
_well_posed_diagonal_token = eqxi.str2jax("well_posed_diagonal_token")
_tridiagonal_token = eqxi.str2jax("tridiagonal_token")
_triangular_token = eqxi.str2jax("triangular_token")
_cholesky_token = eqxi.str2jax("cholesky_token")
_lu_token = eqxi.str2jax("lu_token")
_svd_token = eqxi.str2jax("svd_token")


# Ugly delayed import because we have the dependency chain
# linear_solve -> AutoLinearSolver -> {Cholesky,...} -> AbstractLinearSolver
# but we want linear_solver and AbstractLinearSolver in the same file.
def _lookup(token) -> AbstractLinearSolver:
    from . import _solver

    # pyright doesn't know that these keys are hashable
    _lookup_dict = {
        _qr_token: _solver.QR(),  # pyright: ignore
        _diagonal_token: _solver.Diagonal(),  # pyright: ignore
        _well_posed_diagonal_token: _solver.Diagonal(  # pyright: ignore
            well_posed=True
        ),
        _tridiagonal_token: _solver.Tridiagonal(),  # pyright: ignore
        _triangular_token: _solver.Triangular(),  # pyright: ignore
        _cholesky_token: _solver.Cholesky(),  # pyright: ignore
        _lu_token: _solver.LU(),  # pyright: ignore
        _svd_token: _solver.SVD(),  # pyright: ignore
    }
    return _lookup_dict[token]


_AutoLinearSolverState: TypeAlias = tuple[Any, Any]


class AutoLinearSolver(AbstractLinearSolver[_AutoLinearSolverState]):
    """Automatically determines a good linear solver based on the structure of the
    operator.

    - If `well_posed=True`:
        - If the operator is diagonal, then use [`lineax.Diagonal`][].
        - If the operator is tridiagonal, then use [`lineax.Tridiagonal`][].
        - If the operator is triangular, then use [`lineax.Triangular`][].
        - If the matrix is positive or negative definite, then use
            [`lineax.Cholesky`][].
        - Else use [`lineax.LU`][].

    This is a good choice if you want to be certain that an error is raised for
    ill-posed systems.

    - If `well_posed=False`:
        - If the operator is diagonal, then use [`lineax.Diagonal`][].
        - Else use [`lineax.SVD`][].

    This is a good choice if you want to be certain that you can handle ill-posed
    systems.

    - If `well_posed=None`:
        - If the operator is non-square, then use [`lineax.QR`][].
        - If the operator is diagonal, then use [`lineax.Diagonal`][].
        - If the operator is tridiagonal, then use [`lineax.Tridiagonal`][].
        - If the operator is triangular, then use [`lineax.Triangular`][].
        - If the matrix is positive or negative definite, then use
            [`lineax.Cholesky`][].
        - Else, use [`lineax.LU`][].

    This is a good choice if your primary concern is computational efficiency. It will
    handle ill-posed systems as long as it is not computationally expensive to do so.
    """

    well_posed: Optional[bool]

    def _select_solver(self, operator: AbstractLinearOperator):
        if self.well_posed is True:
            if operator.in_size() != operator.out_size():
                raise ValueError(
                    "Cannot use `AutoLinearSolver(well_posed=True)` with a non-square "
                    "operator. If you are trying solve a least-squares problem then "
                    "you should pass `solver=AutoLinearSolver(well_posed=False)`. By "
                    "default `lineax.linear_solve` assumes that the operator is "
                    "square and nonsingular."
                )
            if is_diagonal(operator):
                token = _well_posed_diagonal_token
            elif is_tridiagonal(operator):
                token = _tridiagonal_token
            elif is_lower_triangular(operator) or is_upper_triangular(operator):
                token = _triangular_token
            elif is_positive_semidefinite(operator) or is_negative_semidefinite(
                operator
            ):
                token = _cholesky_token
            else:
                token = _lu_token
        elif self.well_posed is False:
            if is_diagonal(operator):
                token = _diagonal_token
            else:
                # TODO: use rank-revealing QR instead.
                token = _svd_token
        elif self.well_posed is None:
            if operator.in_size() != operator.out_size():
                token = _qr_token
            elif is_diagonal(operator):
                token = _diagonal_token
            elif is_tridiagonal(operator):
                token = _tridiagonal_token
            elif is_lower_triangular(operator) or is_upper_triangular(operator):
                token = _triangular_token
            elif is_positive_semidefinite(operator) or is_negative_semidefinite(
                operator
            ):
                token = _cholesky_token
            else:
                token = _lu_token
        else:
            raise ValueError(f"Invalid value `well_posed={self.well_posed}`.")
        return token

    def select_solver(self, operator: AbstractLinearOperator) -> AbstractLinearSolver:
        """Check which solver that [`lineax.AutoLinearSolver`][] will dispatch to.

        **Arguments:**

        - `operator`: a linear operator.

        **Returns:**

        The linear solver that will be used.
        """
        return _lookup(self._select_solver(operator))

    def init(self, operator, options) -> _AutoLinearSolverState:
        token = self._select_solver(operator)
        return token, _lookup(token).init(operator, options)

    def compute(
        self,
        state: _AutoLinearSolverState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        token, state = state
        solver = _lookup(token)
        solution, result, _ = solver.compute(state, vector, options)
        return solution, result, {}

    def transpose(self, state: _AutoLinearSolverState, options: dict[str, Any]):
        token, state = state
        solver = _lookup(token)
        transpose_state, transpose_options = solver.transpose(state, options)
        transpose_state = (token, transpose_state)
        return transpose_state, transpose_options

    def allow_dependent_columns(self, operator: AbstractLinearOperator) -> bool:
        token = self._select_solver(operator)
        return _lookup(token).allow_dependent_columns(operator)

    def allow_dependent_rows(self, operator: AbstractLinearOperator) -> bool:
        token = self._select_solver(operator)
        return _lookup(token).allow_dependent_rows(operator)


AutoLinearSolver.__init__.__doc__ = """**Arguments:**

- `well_posed`: whether to only handle well-posed systems or not, as discussed above.
"""


# TODO(kidger): gmres, bicgstab
# TODO(kidger): support auxiliary outputs
@eqx.filter_jit
def linear_solve(
    operator: AbstractLinearOperator,
    vector: PyTree[ArrayLike],
    solver: AbstractLinearSolver = AutoLinearSolver(well_posed=True),
    *,
    options: Optional[dict[str, Any]] = None,
    state: PyTree[Any] = sentinel,
    throw: bool = True,
) -> Solution:
    r"""Solves a linear system.

    Given an operator represented as a matrix $A$, and a vector $b$: if the operator is
    square and nonsingular (so that the problem is well-posed), then this returns the
    usual solution $x$ to $Ax = b$, defined as $A^{-1}b$.

    If the operator is overdetermined, then this either returns the least-squares
    solution $\min_x \| Ax - b \|_2$, or throws an error. (Depending on the choice of
    solver.)

    If the operator is underdetermined, then this either returns the minimum-norm
    solution $\min_x \|x\|_2 \text{ subject to } Ax = b$, or throws an error. (Depending
    on the choice of solver.)

    !!! info

        This function is equivalent to either `numpy.linalg.solve`, or to its
        generalisation `numpy.linalg.lstsq`, depending on the choice of solver.

    The default solver is `lineax.AutoLinearSolver(well_posed=True)`. This
    automatically selects a solver depending on the structure (e.g. triangular) of your
    problem, and will throw an error if your system is overdetermined or
    underdetermined.

    Use `lineax.AutoLinearSolver(well_posed=False)` if your system is known to be
    overdetermined or underdetermined (although handling this case implies greater
    computational cost).

    !!! tip

        These three kinds of solution to a linear system are collectively known as the
        "pseudoinverse solution" to a linear system. That is, given our matrix $A$, let
        $A^\dagger$ denote the
        [Moore--Penrose pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)
        of $A$. Then the usual/least-squares/minimum-norm solution are all equal to
        $A^\dagger b$.

    **Arguments:**

    - `operator`: a linear operator. This is the '$A$' in '$Ax = b$'.

        Most frequently this operator is simply represented as a JAX matrix (i.e. a
        rank-2 JAX array), but any [`lineax.AbstractLinearOperator`][] is supported.

        Note that if it is a matrix, then it should be passed as an
        [`lineax.MatrixLinearOperator`][], e.g.
        ```python
        matrix = jax.random.normal(key, (5, 5))  # JAX array of shape (5, 5)
        operator = lx.MatrixLinearOperator(matrix)  # Wrap into a linear operator
        solution = lx.linear_solve(operator, ...)
        ```
        rather than being passed directly.

    - `vector`: the vector to solve against. This is the '$b$' in '$Ax = b$'.

    - `solver`: the solver to use. Should be any [`lineax.AbstractLinearSolver`][].
        The default is [`lineax.AutoLinearSolver`][] which behaves as discussed
        above.

        If the operator is overdetermined or underdetermined , then passing
        [`lineax.SVD`][] is typical.

    - `options`: Individual solvers may accept additional runtime arguments; for example
        [`lineax.CG`][] allows for specifying a preconditioner. See each individual
        solver's documentation for more details. Keyword only argument.

    - `state`: If performing multiple linear solves with the same operator, then it is
        possible to save re-use some computation between these solves, and to pass the
        result of any intermediate computation in as this argument. See
        [`lineax.AbstractLinearSolver.init`][] for more details. Keyword only
        argument.

    - `throw`: How to report any failures. (E.g. an iterative solver running out of
        steps, or a well-posed-only solver being run with a singular operator.)

        If `True` then a failure will raise an error. Note that errors are only reliably
        raised on CPUs. If on GPUs then the error may only be printed to stderr, whilst
        on TPUs then the behaviour is undefined.

        If `False` then the returned solution object will have a `result` field
        indicating whether any failures occured. (See [`lineax.Solution`][].)

        Keyword only argument.

    **Returns:**

    An [`lineax.Solution`][] object containing the solution to the linear system.
    """  # noqa: E501

    if eqx.is_array(operator):
        raise ValueError(
            "`lineax.linear_solve(operator=...)` should be an "
            "`AbstractLinearOperator`, not a raw JAX array. If you are trying to pass "
            "a matrix then this should be passed as "
            "`lineax.MatrixLinearOperator(matrix)`."
        )
    if options is None:
        options = {}
    vector = jtu.tree_map(inexact_asarray, vector)
    vector_struct = jax.eval_shape(lambda: vector)
    operator_out_structure = operator.out_structure()
    # `is` to handle tracers
    if eqx.tree_equal(vector_struct, operator_out_structure) is not True:
        raise ValueError(
            "Vector and operator structures do not match. Got a vector with structure "
            f"{vector_struct} and an operator with out-structure "
            f"{operator_out_structure}"
        )
    if isinstance(operator, IdentityLinearOperator):
        return Solution(
            value=vector,
            result=RESULTS.successful,
            state=state,
            stats={},
        )
    if state == sentinel:
        state = solver.init(operator, options)
        dynamic_state, static_state = eqx.partition(state, eqx.is_array)
        dynamic_state = lax.stop_gradient(dynamic_state)
        state = eqx.combine(dynamic_state, static_state)

    state = eqxi.nondifferentiable(state, name="`lineax.linear_solve(..., state=...)`")
    options = eqxi.nondifferentiable(
        options, name="`lineax.linear_solve(..., options=...)`"
    )
    solver = eqxi.nondifferentiable(
        solver, name="`lineax.linear_solve(..., solver=...)`"
    )
    solution, result, stats = eqxi.filter_primitive_bind(
        linear_solve_p, operator, state, vector, options, solver, throw
    )
    # TODO: prevent forward-mode autodiff through stats
    stats = eqxi.nondifferentiable_backward(stats)
    return Solution(value=solution, result=result, state=state, stats=stats)
