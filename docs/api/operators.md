# Linear operators

We often talk about solving a linear system $Ax = b$, where $A \in \mathbb{R}^{n \times m}$ is a matrix, $b \in \mathbb{R}^n$ is a vector, and $x \in \mathbb{R}^m$ is our desired solution.

The linear operators described on this page are ways of describing the matrix $A$. The simplest is [`lineax.MatrixLinearOperator`][], which simply holds the matrix $A$ directly.

Meanwhile if $A$ is diagonal, then there is also [`lineax.DiagonalLinearOperator`][]: for efficiency this only stores the diagonal of $A$.

Or, perhaps we only have a function $F : \mathbb{R}^m \to \mathbb{R}^n$ such that $F(x) = Ax$. Whilst we could use $F$ to materialise the whole matrix $A$ and then store it in a [`lineax.MatrixLinearOperator`][], that may be very memory intensive. Instead, we may prefer to use [`lineax.FunctionLinearOperator`][]. Many linear solvers (e.g. [`lineax.CG`][]) only use matrix-vector products, and this means we can avoid ever needing to materialise the whole matrix $A$.

??? abstract "`lineax.AbstractLinearOperator`"

    ::: lineax.AbstractLinearOperator
        options:
            members:
                - mv
                - as_matrix
                - transpose
                - in_structure
                - out_structure
                - in_size
                - out_size

::: lineax.MatrixLinearOperator
    options:
        members:
            - __init__

---

::: lineax.DiagonalLinearOperator
    options: 
        members: 
            - __init__

---

::: lineax.TridiagonalLinearOperator
    options:
        members:
            - __init__

---

::: lineax.PyTreeLinearOperator
    options:
        members:
            - __init__

---

::: lineax.JacobianLinearOperator
    options:
        members:
            - __init__

---

::: lineax.FunctionLinearOperator
    options:
        members:
            - __init__

---

::: lineax.IdentityLinearOperator
    options:
        members:
            - __init__

---

::: lineax.TaggedLinearOperator
    options:
        members:
            - __init__

---

## Sparse Linear Operators

Lineax supports two flavours of sparse representations of linear operators by wrapping implementations from the `jax.experimental.sparse` module. Representations of sparse operators can be created from a dense matrix, or directly from values and row and column indices in a given format. In either case, Lineax only ever sees the existing JAX representation of the sparse operator.
Note that the API of `jax.experimental.sparse` is still subject to change, and the API offered here may therefore change as well.

!!! Example
    
    ```python
    import lineax as lx
    from jax.experimental.sparse import CSR

    matrix = ...  # Some 2D matrix
    operator = lx.CSLinearOperator(CSR.from_dense(matrix))

    # Or directly create it with values, indices and offsets
    values = ...
    indices = ...
    offsets = ...
    shape = ...

    operator2 = lx.CSLinearOperator(CSR((values, indices, offsets), shape=shape))
    ```



::: lineax.CSLinearOperator
    options:
        members:
            - __init__

---

::: lineax.COOLinearOperator
    options:
        members:
            - __init__
