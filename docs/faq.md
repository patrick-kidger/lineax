# FAQ

## How does this differ from `jax.numpy.solve`, `jax.scipy.{...}` etc.?

Lineax offers several improvements. Most notably:

- Several new solvers. For example, [`lineax.QR`][] has no counterpart in core JAX. (And it is much faster than `jax.numpy.linalg.lstsq`, which is the closest equivalent, and uses an SVD decomposition instead.)

- Several new operators. For example, [`lineax.JacobianLinearOperator`][] has no counterpart in core JAX.

- A consistent API. The built-in JAX operations all differ from each other slightly, and are split across `jax.numpy`, `jax.scipy`, and `jax.scipy.sparse`.

- Numerically stable gradients. The existing JAX implementations will sometimes return `NaN`s!

- Some faster compile times and run times in a few places.

Most of these are because JAX aims to mimic the existing NumPy/SciPy APIs. (I.e. it's not JAX's fault that it doesn't take the approach that Lineax does!)

## Why does a gradient with respect to an operator not represent the matrix gradient?

JAX differentiates with respect to the array leaves of a PyTree. A gradient with respect to an operator therefore contains derivatives with respect to the arrays used to construct that operator, in the same PyTree structure. It does not in general represent a new linear operator: calling methods such as `.as_matrix()` on it can give misleading results.

For example, the only differentiable parameter of a scaled identity operator is its scalar scale:

```python
import jax
import jax.numpy as jnp
import lineax as lx


def trace(operator):
    return jnp.trace(operator.as_matrix())


struct = jax.ShapeDtypeStruct((3,), jnp.float32)
operator = jnp.array(2.0) * lx.IdentityLinearOperator(struct)
gradient = jax.grad(trace)(operator)
print(jax.tree.leaves(gradient))  # A single scalar derivative: 3.
```

The derivative with respect to the scale is `3`, since the trace is three times the scale. Interpreting the returned PyTree as an operator would instead turn that derivative into `3 * I`, which is not the derivative with respect to the entries of a dense matrix.

If you want derivatives with respect to a dense matrix's entries, differentiate a function whose input is that matrix instead:

```python
matrix = operator.as_matrix()
print(jax.grad(jnp.trace)(matrix))  # The 3-by-3 identity matrix.
```

The same distinction applies to composite operators and to gradients returned by `equinox.filter_grad`.

## How do I represent a {lower, upper} triangular matrix?

Typically: create a full matrix, with the {lower, upper} part containing your values, and the converse {upper, lower} part containing all zeros. Then use, e.g., `operator = lx.MatrixLinearOperator(matrix, lx.lower_triangular_tag)`.

This is the most efficient way to store a triangular matrix in JAX's ndarray-based programming model.

## What about other operations from linear algebra? (Determinants, eigenvalues, etc.)

See [`jax.numpy.linalg`](https://jax.readthedocs.io/en/latest/jax.numpy.html#module-jax.numpy.linalg) and [`jax.scipy.linalg`](https://jax.readthedocs.io/en/latest/jax.scipy.html#module-jax.scipy.linalg).

## How do I solve multiple systems of equations (i.e. `AX = B`)?

Solvers implemented in Lineax target single systems of linear equations (i.e., `Ax = b`), however, using `jax.vmap` or `equinox.filter_vmap`, it can solve multiple systems with minimal effort.

```python
multi_linear_solve = eqx.filter_vmap(lx.linear_solve, in_axes=(None, 1))
#  or    
multi_linear_solve = jax.vmap(lx.linear_solve, in_axes=(None, 1))
```
