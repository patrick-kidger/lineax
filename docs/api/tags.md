# Tags

Lineax offers a way to "tag" linear operators as exhibiting certain properties, e.g. that they are positive semidefinite.

If a linear operator is known to have a particular property, then this can be used to dispatch to a more efficient implementation, e.g. when solving a linear system.

Generally speaking, tags are an *optional* tool that can be used to improve your run time and/or compile time, by statically telling the linear solvers what properties they may assume about your system. However, if misused then you may find that the wrong result is silently returned.

In this way they are analogous to flags like `scipy.linalg.solve(..., assume_a="pos")`.

!!! Example

    ```python
    # Some rank-2 JAX array.
    matrix = ...
    # Some rank-1 JAX array.
    vector = ...

    # Declare that this matrix is positive semidefinite.
    operator = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)

    # This tag is used to dispatch to a maximally-efficient linear solver.
    # In this case, a Cholesky solver is used:
    solution = lx.linear_solve(operator, vector)

    # Whether operators are tagged can be checked:
    assert lx.is_positive_semidefinite(operator)
    ```

!!! Warning

    Be careful, only the tag is actually checked, not the actual value of the matrix:
    ```python
    # Not a positive semidefinite matrix
    matrix = jax.numpy.array([[1, 2], [3, 4]])

    operator = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
    lx.is_positive_semidefinite(operator)  # True
    lx.linear_solve(operator, vector)  # Returns the wrong solution!
    ```

Of the built-in operators: [`lineax.MatrixLinearOperator`][], [`lineax.PyTreeLinearOperator`][], [`lineax.JacobianLinearOperator`][], [`lineax.FunctionLinearOperator`][], [`lineax.TaggedLinearOperator`][] directly support a `tags` argument that mark them as having certain characteristics:
```python
operator = lx.MatrixLinearOperator(matrix, lx.symmetric_tag)
```

You can pass multiple tags at once:
```python
operator = lx.MatrixLinearOperator(matrix, (lx.symmetric_tag, lx.unit_diagonal_tag))
```

Other linear operators can be wrapped into a [`lineax.TaggedLinearOperator`][] if necessary:
```python
operator = lx.MatrixLinearOperator(...)
symmetric_operator = operator + operator.T
lx.is_symmetric(symmetric_operator)  # False
symmetric_operator = lx.TaggedLinearOperator(symmetric_operator, lx.symmetric_tag)
lx.is_symmetric(symmetric_operator)  # True
```

Some linear operators are known to exhibit certain properties by construction, and need no additional tags:
```python
lx.is_symmetric(lx.DiagonalLinearOperator(...))  # True
lx.is_positive_semidefinite(lx.IdentityLinearOperator(...))  # True
```

## List of available tags

::: lineax.symmetric_tag

Marks that an operator is symmetric. (As a matrix, $A = A^\intercal$.)

---

::: lineax.diagonal_tag

Marks than an operator is diagonal. (As a matrix, it must have zeros in the off-diagonal entries.)

For example, the default solver for [`lineax.linear_solve`][] uses this to dispatch to [`lineax.Diagonal`][] as the solver.

---

::: lineax.unit_diagonal_tag

Marks than an operator has $1$ for every diagonal element. (As a matrix $A$, then it must have $A_{ii} = 1$ for all $i$.) Note that the whole matrix need not be diagonal.

For example, [`lineax.Triangular`][] uses this to cheapen its solve.

---

::: lineax.lower_triangular_tag

Marks that an operator is lower triangular. (As a matrix $A$, then it must have $A_{ij} = 0 for all $i < j$.) Note that the diagonal may still have nonzero entries.

For example, the default solver for [`lineax.linear_solve`][] uses this to dispatch to [`lineax.Triangular`][] as the solver.

---

::: lineax.upper_triangular_tag

Marks that an operator is upper triangular. (As a matrix $A$, then it must have $A_{ij} = 0 for all $i > j$.) Note that the diagonal may still have nonzero entries.

For example, the default solver for [`lineax.linear_solve`][] uses this to dispatch to [`lineax.Triangular`][] as the solver.

---

::: lineax.positive_semidefinite_tag

Marks than operator is positive **semidefinite**.

For example, the default solver for [`lineax.linear_solve`][] uses this to dispatch to [`lineax.Cholesky`][] as the solver.

If you wish to mark that an operator is specifically postive **definite** then combine this with [`lineax.nonsingular_tag`].

---

::: lineax.negative_semidefinite_tag

Marks than operator is negative **semidefinite**.

For example, the default solver for [`lineax.linear_solve`][] uses this to dispatch to [`lineax.Cholesky`][] as the solver.

If you wish to mark that an operator is specifically postive **definite** then combine this with [`lineax.nonsingular_tag`].
