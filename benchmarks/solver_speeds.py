# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools as ft
import sys
import timeit

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import lineax as lx


sys.path.append("../tests")
from helpers import construct_matrix, has_tag  # pyright: ignore[reportMissingImports]


getkey = eqxi.GetKey()


def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8):
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


jax.config.update("jax_enable_x64", True)

if jax.config.jax_enable_x64:  # pyright: ignore
    tol = 1e-12
else:
    tol = 1e-6


def base_wrapper(a, b, solver):
    op = lx.MatrixLinearOperator(
        a,
        (
            lx.positive_semidefinite_tag,
            lx.symmetric_tag,
            lx.diagonal_tag,
            lx.tridiagonal_tag,
        ),
    )
    out = lx.linear_solve(op, b, solver, throw=False)
    return out.value


def jax_svd(a, b):
    out, _, _, _ = jnp.linalg.lstsq(a, b)  # pyright: ignore
    return out


def jax_gmres(a, b):
    out, _ = jsp.sparse.linalg.gmres(a, b, tol=tol)
    return out


def jax_bicgstab(a, b):
    out, _ = jsp.sparse.linalg.bicgstab(a, b, tol=tol)
    return out


def jax_cg(a, b):
    out, _ = jsp.sparse.linalg.cg(a, b, tol=tol)
    return out


def jax_lu(matrix, vector):
    return jsp.linalg.lu_solve(jsp.linalg.lu_factor(matrix), vector)


def jax_cholesky(matrix, vector):
    return jsp.linalg.cho_solve(jsp.linalg.cho_factor(matrix), vector)


def jax_tridiagonal(matrix, vector):
    dl = jnp.append(0.0, matrix.diagonal(-1))
    d = matrix.diagonal(0)
    du = jnp.append(matrix.diagonal(1), 0.0)
    return jax.lax.linalg.tridiagonal_solve(dl, d, du, vector[:, None])[:, 0]


named_solvers = [
    ("LU", "LU", lx.LU(), jax_lu, ()),
    ("QR", "SVD", lx.QR(), jax_svd, ()),
    ("SVD", "SVD", lx.SVD(), jax_svd, ()),
    (
        "Cholesky",
        "Cholesky",
        lx.Cholesky(),
        jax_cholesky,
        lx.positive_semidefinite_tag,
    ),
    ("Diagonal", "None", lx.Diagonal(), None, lx.diagonal_tag),
    (
        "Tridiagonal",
        "Tridiagonal",
        lx.Tridiagonal(),
        jax_tridiagonal,
        lx.tridiagonal_tag,
    ),
    (
        "CG",
        "CG",
        lx.CG(atol=tol, rtol=tol, stabilise_every=None),
        jax_cg,
        lx.positive_semidefinite_tag,
    ),
    (
        "GMRES",
        "GMRES",
        lx.GMRES(atol=1, rtol=1),
        jax_gmres,
        (),
    ),
    (
        "BiCGStab",
        "BiCGStab",
        lx.BiCGStab(atol=tol, rtol=tol),
        jax_bicgstab,
        (),
    ),
]


def create_problem(solver, tags, size=3):
    (matrix,) = construct_matrix(getkey, solver, tags, size=size)
    true_x = jr.normal(getkey(), (size,))
    b = matrix @ true_x
    return matrix, true_x, b


def create_easy_iterative_problem(size, tags):
    matrix = jr.normal(getkey(), (size, size)) / size + 2 * jnp.eye(size)
    true_x = jr.normal(getkey(), (size,))
    if has_tag(tags, lx.positive_semidefinite_tag):
        matrix = matrix.T @ matrix
    b = matrix @ true_x
    return matrix, true_x, b


def test_solvers(vmap_size, mat_size):
    for lx_name, jax_name, _lx_solver, jax_solver, tags in named_solvers:
        lx_solver = ft.partial(base_wrapper, solver=_lx_solver)
        if vmap_size == 1:
            if isinstance(_lx_solver, (lx.CG, lx.GMRES, lx.BiCGStab)):
                matrix, true_x, b = create_easy_iterative_problem(mat_size, tags)
            else:
                matrix, true_x, b = create_problem(lx_solver, tags, size=mat_size)
        else:
            if isinstance(_lx_solver, (lx.CG, lx.GMRES, lx.BiCGStab)):
                matrix, true_x, b = eqx.filter_vmap(
                    create_easy_iterative_problem,
                    axis_size=vmap_size,
                    out_axes=eqx.if_array(0),
                )(mat_size, tags)
            else:
                matrix, true_x, b = create_problem(lx_solver, tags, size=mat_size)
                _create_problem = ft.partial(create_problem, size=mat_size)
                matrix, true_x, b = eqx.filter_vmap(
                    _create_problem, axis_size=vmap_size, out_axes=eqx.if_array(0)
                )(lx_solver, tags)

            lx_solver = jax.vmap(lx_solver)
            if jax_solver is not None:
                jax_solver = jax.vmap(jax_solver)

        lx_solver = jax.jit(lx_solver)
        bench_lx = ft.partial(lx_solver, matrix, b)

        if vmap_size == 1:
            batch_msg = "problem"
        else:
            batch_msg = f"batch of {vmap_size} problems"

        lx_soln = bench_lx()
        if tree_allclose(lx_soln, true_x, atol=1e-4, rtol=1e-4):
            lx_solve_time = timeit.timeit(bench_lx, number=1)

            print(
                f"Lineax's {lx_name} solved {batch_msg} of "
                f"size {mat_size} in {lx_solve_time} seconds."
            )
        else:
            fail_time = timeit.timeit(bench_lx, number=1)
            err = jnp.abs(lx_soln - true_x).max()
            print(
                f"Lineax's {lx_name} failed to solve {batch_msg} of "
                f"size {mat_size} with error {err} in {fail_time} seconds"
            )
        if jax_solver is None:
            print("JAX has no equivalent solver. \n")

        else:
            jax_solver = jax.jit(jax_solver)
            bench_jax = ft.partial(jax_solver, matrix, b)
            jax_soln = bench_jax()
            if tree_allclose(jax_soln, true_x, atol=1e-4, rtol=1e-4):
                jax_solve_time = timeit.timeit(bench_jax, number=1)
                print(
                    f"JAX's {jax_name} solved {batch_msg} of "
                    f"size {mat_size} in {jax_solve_time} seconds. \n"
                )
            else:
                fail_time = timeit.timeit(bench_jax, number=1)
                err = jnp.abs(jax_soln - true_x).max()
                print(
                    f"JAX's {jax_name} failed to solve {batch_msg} of "
                    f"size {mat_size} with error {err} in {fail_time} seconds. \n"
                )


for vmap_size, mat_size in [(1, 50), (1000, 50)]:
    test_solvers(vmap_size, mat_size)
