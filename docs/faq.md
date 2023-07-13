# FAQ

## How does this differ to `jax.numpy.solve`, `jax.scipy.{...}` etc.?

Lineax offers several improvements. Most notably:

- Several new solvers. For example, [`lineax.QR`][] has no counterpart in core JAX. (And is much faster than `jax.numpy.linalg.lstsq`, which is the closest equivalent, and uses an SVD decomposition instead.)

- Several new operators. For example, [`lineax.JacobianLinearOperator`][] has no counterpart in core JAX.

- A consistent API. The built-in JAX operations all differ from each other slightly, and are split across `jax.numpy`, `jax.scipy`, and `jax.scipy.sparse`.

- Numerically stable gradients. The existing JAX implementations will sometimes return `NaN`s!

- Some faster compile times and run times in a few places.

Most of these are because JAX aims to mimc the existing NumPy/SciPy APIs. (I.e. it's not JAX's fault that it doesn't take the approach that Lineax does!)

## What about other operations from linear algebra? (Determinants, eigenvalues etc.)

See [`jax.numpy.linalg`](https://jax.readthedocs.io/en/latest/jax.numpy.html#module-jax.numpy.linalg) and [`jax.scipy.linalg`](https://jax.readthedocs.io/en/latest/jax.scipy.html#module-jax.scipy.linalg).

## How do I solve multiple systems of equations (i.e. `AX = B`)?

Solvers implemented in Lineax target single systems of linear equations (i.e. `Ax = b`), however using `jax.vmap` or `equinox.filter_vmap`, it can solve multiple systems with minimal effort.

```python
multi_linear_solve = eqx.filter_vmap(lx.linear_solve, in_axes=(None, 1))
#  or    
multi_linear_solve = jax.vmap(lx.linear_solve, in_axes=(None, 1))
```
