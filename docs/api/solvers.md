# Solvers

If you're not sure what to use, then pick [`lineax.AutoLinearSolver`][] and it will automatically dispatch to an efficient solver depending on what structure your linear operator is declared to exhibit. (See the [tags](./tags.md) page.)

??? abstract "`lineax.AbstractLinearSolver`"

    ::: lineax.AbstractLinearSolver
        options:
            members:
                - init
                - compute
                - transpose
                - conj
                - assume_full_rank

::: lineax.AutoLinearSolver
    options:
        members:
            - __init__
            - select_solver

---

::: lineax.LU
    options:
        members:
            - __init__

## Least squares solvers

These are capable of solving ill-posed linear problems.

::: lineax.QR
    options:
        members:
            - __init__

---

::: lineax.SVD
    options:
        members:
            - __init__

---

::: lineax.Normal
    options:
        members:
            - __init__

---

::: lineax.LSMR
    options:
        members:
            - __init__


### Diagonal

In addition to these, [`lineax.Diagonal`][] with `well_posed=False` (below) also supports ill-posed problems.

## Iterative solvers

These solvers use only matrix-vector products, and do not require instantiating the whole matrix. This makes them good when used alongside e.g. [`lineax.JacobianLinearOperator`][] or [`lineax.FunctionLinearOperator`][], which only provide matrix-vector products.

!!! warning

    Note that [`lineax.BiCGStab`][] and [`lineax.GMRES`][] may fail to converge on some (typically non-sparse) problems.

::: lineax.CG
    options:
        members:
            - __init__

---

::: lineax.BiCGStab
    options:
        members:
            - __init__

---

::: lineax.GMRES
    options:
        members:
            - __init__

### LSMR

In addition to these, [`lineax.LSMR`][] (above) is also an iterative method.

## Structure-exploiting solvers

These require special structure in the operator. (And will throw an error if passed an operator without that structure.) In return, they are able to solve the linear problem much more efficiently.

::: lineax.Cholesky
    options:
        members:
            - __init__

---

::: lineax.Diagonal
    options:
        members:
            - __init__

---

::: lineax.Triangular
    options:
        members:
            - __init__

---

::: lineax.Tridiagonal
    options:
        members:
            - __init__

### CG

In addition to these, [`lineax.CG`][] also requires special structure (positive or negative definiteness).
