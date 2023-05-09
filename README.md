<h1 align='center'>Lineax</h1>

Lineax is a [JAX](https://github.com/google/jax) library for linear solvers.

Features include:

- A sophisticated handling of linear solvers:
  - PyTree-valued matrices and vectors;
  - Linear operators for Jacobians, transposes, etc.;
  - Support for structured (e.g. symmetric) matrices;
  - Efficient linear least squares (e.g. QR solvers);
  - Numerically stable gradients through linear least squares;
  - Improved compilation times.
- All the benefits of working with JAX: autodiff, autoparallism, GPU/TPU support etc.

## Installation

```bash
pip install lineax
```

Requires Python 3.9+ and JAX 0.4.8+.

## Quick example

```python
TODO
```

## See also

Neural Networks: [Equinox](https://github.com/patrick-kidger/equinox).

Numerical differential equation solvers: [Diffrax](https://github.com/patrick-kidger/diffrax).

Type annotations and runtime checking for PyTrees and shape/dtype of JAX arrays: [jaxtyping](https://github.com/google/jaxtyping).

Computer vision models: [Eqxvision](https://github.com/paganpasta/eqxvision).

SymPy<->JAX conversion; train symbolic expressions via gradient descent: [sympy2jax](https://github.com/google/sympy2jax).
