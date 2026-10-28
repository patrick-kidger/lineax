# Determinants

These are the main entry points for computing determinants and log-determinants of linear operators.

Any [`lineax.AbstractDirectLinearSolver`][] (or [`lineax.Normal`][] wrapping one) may be used. The solver's factorisation is reused: if `lx.slogdet` and `lx.linear_solve` are called for the same operator inside a single `jax.jit`, XLA will CSE the factorisation so it is only computed once.

The derivative is `d log|det A| = Re trace(A^{-1} dA)`, evaluated as [`lineax.trace`][] of the composed operator so that structured operators (e.g. diagonal ones) dispatch to a cheap rule automatically; some solvers instead differentiate their own log-determinant where that is cheaper still.

::: lineax.slogdet

---

::: lineax.determinant
