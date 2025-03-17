<h1 align='center'>Lineax</h1>

Lineax is a [JAX](https://github.com/google/jax) library for linear solves and linear least squares. That is, Lineax provides routines that solve for $x$ in $Ax = b$. (Even when $A$ may be ill-posed or rectangular.)

Features include:
- PyTree-valued matrices and vectors;
- General linear operators for Jacobians, transposes, etc.;
- Efficient linear least squares (e.g. QR solvers);
- Numerically stable gradients through linear least squares;
- Support for structured (e.g. symmetric) matrices;
- Improved compilation times;
- Improved runtime of some algorithms;
- Support for both real-valued and complex-valued inputs;
- All the benefits of working with JAX: autodiff, autoparallelism, GPU/TPU support, etc.

## Installation

```bash
pip install lineax
```

Requires Python 3.10+, JAX 0.4.38+, and [Equinox](https://github.com/patrick-kidger/equinox) 0.11.10+.

## Documentation

Available at [https://docs.kidger.site/lineax](https://docs.kidger.site/lineax).

## Quick examples

Lineax can solve a least squares problem with an explicit matrix operator:

```python
import jax.random as jr
import lineax as lx

matrix_key, vector_key = jr.split(jr.PRNGKey(0))
matrix = jr.normal(matrix_key, (10, 8))
vector = jr.normal(vector_key, (10,))
operator = lx.MatrixLinearOperator(matrix)
solution = lx.linear_solve(operator, vector, solver=lx.QR())
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
solver = lx.CG(rtol=1e-6, atol=1e-6)
out = lx.linear_solve(hessian, gradient_fn(y, args=None), solver)
minimum = y - out.value
```

## Citation

If you found this library to be useful in academic work, then please cite: ([arXiv link](https://arxiv.org/abs/2311.17283))

```bibtex
@article{lineax2023,
    title={Lineax: unified linear solves and linear least-squares in JAX and Equinox},
    author={Jason Rader and Terry Lyons and Patrick Kidger},
    journal={
        AI for science workshop at Neural Information Processing Systems 2023,
        arXiv:2311.17283
    },
    year={2023},
}
```

(Also consider starring the project on GitHub.)

## See also: other libraries in the JAX ecosystem

**Always useful**  
[Equinox](https://github.com/patrick-kidger/equinox): neural networks and everything not already in core JAX!  
[jaxtyping](https://github.com/patrick-kidger/jaxtyping): type annotations for shape/dtype of arrays.  

**Deep learning**  
[Optax](https://github.com/deepmind/optax): first-order gradient (SGD, Adam, ...) optimisers.  
[Orbax](https://github.com/google/orbax): checkpointing (async/multi-host/multi-device).  
[Levanter](https://github.com/stanford-crfm/levanter): scalable+reliable training of foundation models (e.g. LLMs).  
[paramax](https://github.com/danielward27/paramax): parameterizations and constraints for PyTrees.  

**Scientific computing**  
[Diffrax](https://github.com/patrick-kidger/diffrax): numerical differential equation solvers.  
[Optimistix](https://github.com/patrick-kidger/optimistix): root finding, minimisation, fixed points, and least squares.  
[BlackJAX](https://github.com/blackjax-devs/blackjax): probabilistic+Bayesian sampling.  
[sympy2jax](https://github.com/patrick-kidger/sympy2jax): SymPy<->JAX conversion; train symbolic expressions via gradient descent.  
[PySR](https://github.com/milesCranmer/PySR): symbolic regression. (Non-JAX honourable mention!)  

**Awesome JAX**  
[Awesome JAX](https://github.com/n2cholas/awesome-jax): a longer list of other JAX projects.  
