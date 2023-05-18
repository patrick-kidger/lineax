<h1 align='center'>Lineax</h1>

Lineax is a [JAX](https://github.com/google/jax) library for linear solves and
linear least squares.

Features include:
- PyTree-valued matrices and vectors;
- General linear operators for Jacobians, transposes, etc.;
- Efficient linear least squares (e.g. QR solvers);
- Numerically stable gradients through linear least squares;
- Support for structured (e.g. symmetric) matrices;
- Improved compilation times;
- Improved runtime of some algorithms;
- All the benefits of working with JAX: autodiff, autoparallism, GPU/TPU support etc.

## Installation

```bash
pip install lineax
```

Requires Python 3.9+ and JAX 0.4.8+.

## Quick example

Lineax can solve a least squares problem with an explicit matrix operator:

```python
import jax.random as jr
import lineax as lx

matrix_key, vector_key = jr.split(jr.PRNGKey(0))
matrix = jr.normal(matrix_key, (10, 8))
vector = jr.normal(vector_key, (10,))
operator = lx.MatrixLinearOperator(matrix)
solution = lx.linear_solve(operator, vector, lx.QR())
```

or Lineax can solve a problem without ever materializing a matrix, as done in this
quadratic solve:

```python
import jax
import lineax as lx

key = jax.random.PRNGKey(0)
y = jax.random.normal(key, (10,))

def quadratic_fn(y, args):
  return jax.numpy.sum((y - 1)**2)

gradient_fn = jax.grad(quadratic_fn)
hessian = lx.JacobianLinearOperator(gradient_fn, y, tags=lx.positive_semidefinite_tag)
out = lx.linear_solve(hessian, gradient_fn(y, args=None))
minimum = y - out.value
```

## See also

Neural Networks: [Equinox](https://github.com/patrick-kidger/equinox).

Numerical differential equation solvers: [Diffrax](https://github.com/patrick-kidger/diffrax).

Type annotations and runtime checking for PyTrees and shape/dtype of JAX arrays: [jaxtyping](https://github.com/google/jaxtyping).

Computer vision models: [Eqxvision](https://github.com/paganpasta/eqxvision).

SymPy<->JAX conversion; train symbolic expressions via gradient descent: [sympy2jax](https://github.com/google/sympy2jax).
