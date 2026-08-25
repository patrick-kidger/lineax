# Determinants

These are the main entry points for computing determinants and log-determinants of linear operators.

Any [`lineax.AbstractDirectLinearSolver`][] (or [`lineax.Normal`][] wrapping one) may be used. The solver's factorisation is reused: if `lx.slogdet` and `lx.linear_solve` are called for the same operator inside a single `jax.jit`, XLA will CSE the factorisation so it is only computed once.

::: lineax.slogdet

---

::: lineax.determinant
