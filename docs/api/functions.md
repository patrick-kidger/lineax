# Functions on linear operators

We define a number of functions on [linear operators](./operators.md).

## Computational changes

These do not change the mathematical meaning of the operator; they simply change how it is stored computationally. (E.g. to materialise the whole operator.)

::: lineax.linearise

---

::: lineax.materialise

## Extract information from the operator

::: lineax.diagonal

---

::: lineax.tridiagonal

## Test the operator to see if it exhibits a certain property

Note that these do *not* inspect the values of the operator -- instead, they use typically use [tags](./tags.md). (Or in some cases, just the type of the operator: e.g. `is_diagonal(DiagonalLinearOperator(...)) == True`.)

::: lineax.has_unit_diagonal

---

::: lineax.is_diagonal

---

::: lineax.is_tridiagonal

---

::: lineax.is_lower_triangular

---

::: lineax.is_upper_triangular

---

::: lineax.is_positive_semidefinite

---

::: lineax.is_negative_semidefinite

---

::: lineax.is_symmetric
