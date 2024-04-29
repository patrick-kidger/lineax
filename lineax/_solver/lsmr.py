"""Implementation adapted from SciPy, with BSD license:

Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

from collections.abc import Callable
from typing import Any, Optional
from typing_extensions import TypeAlias

import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, PyTree

from .._norm import two_norm
from .._operator import AbstractLinearOperator, conj
from .._solution import RESULTS
from .._solve import AbstractLinearSolver


_LSMRState: TypeAlias = AbstractLinearOperator


class LSMR(AbstractLinearSolver[_LSMRState], strict=True):
    """LSMR solver for linear systems.

    This solver can handle any operator, even nonsquare or singular ones. In these
    cases it will return the pseudoinverse solution to the linear system.

    Similar to `scipy.sparse.linalg.lsmr`.

    This supports the following `options` (as passed to
    `lx.linear_solve(..., options=...)`).

    - `y0`: The initial estimate of the solution to the linear system. Defaults to all
        zeros.
    - `damp`: Damping factor for regularized least-squares. LSMR solves the regularized
        least-squares problem:

            min ||A y - b||_2 + damp*||y||_2

        where damp is a scalar. If damp is None or 0, the system is solved without
        regularization. Default is 0.
    """

    atol: float
    btol: float
    norm: Callable = two_norm
    max_steps: Optional[int] = None
    conlim: float = 1e8

    def __check_init__(self):
        if isinstance(self.atol, (int, float)) and self.atol < 0:
            raise ValueError("Tolerances must be non-negative.")
        if isinstance(self.btol, (int, float)) and self.btol < 0:
            raise ValueError("Tolerances must be non-negative.")

        if isinstance(self.atol, (int, float)) and isinstance(self.btol, (int, float)):
            if self.atol == 0 and self.btol == 0 and self.max_steps is None:
                raise ValueError(
                    "Must specify `atol`, `btol`, or `max_steps` (or some combination "
                    "of all three)."
                )
        if self.conlim is None:
            self.conlim = 1e8

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        return operator

    def compute(
        self,
        state: _LSMRState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        operator = state
        x = options.get("y0", None)
        damp = options.get("damp", 0.0)

        m, n = operator.out_size(), operator.in_size()
        # number of singular values
        min_dim = min([m, n])
        if self.max_steps is None:
            max_steps = min_dim * 10  # for consistency with other iterative solvers
        else:
            max_steps = self.max_steps

        if x is None:
            x = jtu.tree_map(jnp.zeros_like, operator.in_structure())

        dtype = jnp.result_type(
            *jtu.tree_leaves(vector),
            *jtu.tree_leaves(x),
            *jtu.tree_leaves(operator.in_structure()),
        )

        b = vector
        u = (ω(b) - ω(operator.mv(x))).ω
        normb = self.norm(b)
        beta = self.norm(u)

        def beta_nonzero(beta, u):
            u = (ω(u) / lax.select(beta == 0.0, 1.0, beta).astype(dtype)).ω
            v = conj(operator).T.mv(u)
            alpha = self.norm(v)
            return u, v, alpha

        def beta_zero(beta, u):
            v = jtu.tree_map(jnp.zeros_like, operator.in_structure())
            alpha = 0.0
            return u, v, alpha

        u, v, alpha = lax.cond(beta == 0.0, beta_zero, beta_nonzero, beta, u)
        v = (ω(v) / lax.select(alpha == 0.0, 1.0, alpha).astype(dtype)).ω

        h = v
        hbar = jtu.tree_map(jnp.zeros_like, operator.in_structure())

        loop_state_vecs = (x, u, v, h, hbar)

        # Initialize variables for 1st iteration.
        # generally, latin letters (b, x, u, v, h etc) are vectors that may be complex
        # greek letters (alpha, beta, rho, zeta etc) are scalars that are always real
        itn = 0
        zetabar = alpha * beta
        alphabar = alpha
        rho = 1.0
        rhobar = 1.0
        cbar = 1.0
        sbar = 0.0

        loop_state_main = (itn, alpha, beta, zetabar, alphabar, rho, rhobar, cbar, sbar)

        # Initialize variables for estimation of ||r||.
        betadd = beta
        betad = 0.0
        rhodold = 1.0
        tautildeold = 0.0
        thetatilde = 0.0
        zeta = 0.0
        delta = 0.0

        loop_state_r_est = (
            betadd,
            betad,
            rhodold,
            tautildeold,
            thetatilde,
            zeta,
            delta,
        )

        # Initialize variables for estimation of ||A|| and cond(A)
        normA2 = alpha**2
        maxrbar = 0.0
        minrbar = 1e100
        normA = jnp.sqrt(normA2)
        condA = 1.0

        loop_state_anorm = (normA2, maxrbar, minrbar, normA, condA)

        # Items for use in stopping rules, normb set earlier
        istop = 0
        ctol = lax.select(
            self.conlim > 0, 1.0 / lax.select(self.conlim > 0, self.conlim, 1.0), 0.0
        )
        normr = beta
        normAr = alpha * beta

        loop_state_stopping = (istop, normr, normAr, normb)

        loop_state = (
            loop_state_main,
            loop_state_r_est,
            loop_state_anorm,
            loop_state_stopping,
            loop_state_vecs,
        )

        def condfun(loop_state):
            (
                loop_state_main,
                loop_state_r_est,
                loop_state_anorm,
                loop_state_stopping,
                loop_state_vecs,
            ) = loop_state
            (
                itn,
                alpha,
                beta,
                zetabar,
                alphabar,
                rho,
                rhobar,
                cbar,
                sbar,
            ) = loop_state_main
            (
                betadd,
                betad,
                rhodold,
                tautildeold,
                thetatilde,
                zeta,
                delta,
            ) = loop_state_r_est
            (normA2, maxrbar, minrbar, normA, condA) = loop_state_anorm
            (istop, normr, normAr, normb) = loop_state_stopping
            (x, u, v, h, hbar) = loop_state_vecs

            return istop == 0

        def bodyfun(loop_state):
            # unpack everything. Maybe cleaner to use a dict or struct?
            (
                loop_state_main,
                loop_state_r_est,
                loop_state_anorm,
                loop_state_stopping,
                loop_state_vecs,
            ) = loop_state
            (
                itn,
                alpha,
                beta,
                zetabar,
                alphabar,
                rho,
                rhobar,
                cbar,
                sbar,
            ) = loop_state_main
            (
                betadd,
                betad,
                rhodold,
                tautildeold,
                thetatilde,
                zeta,
                delta,
            ) = loop_state_r_est
            (normA2, maxrbar, minrbar, normA, condA) = loop_state_anorm
            (istop, normr, normAr, normb) = loop_state_stopping
            (x, u, v, h, hbar) = loop_state_vecs

            itn = itn + 1

            # Perform the next step of the bidiagonalization to obtain the
            # next  beta, u, alpha, v.  These satisfy the relations
            #         beta*u  =  A@v   -  alpha*u,
            #        alpha*v  =  A'@u  -  beta*v.

            u = (ω(u) * -alpha.astype(dtype)).ω
            u = (ω(u) + ω(operator.mv(v))).ω
            beta = self.norm(u)

            def beta_nonzero(alpha, beta, u, v):
                u = (ω(u) / lax.select(beta == 0.0, 1.0, beta).astype(dtype)).ω
                v = (ω(v) * -beta.astype(dtype)).ω
                v = (ω(v) + ω(conj(operator).T.mv(u))).ω
                alpha = self.norm(v)
                v = (ω(v) / lax.select(alpha == 0.0, 1.0, alpha).astype(dtype)).ω
                return alpha, beta, u, v

            def beta_zero(alpha, beta, u, v):
                return alpha, beta, u, v

            alpha, beta, u, v = lax.cond(
                beta == 0, beta_zero, beta_nonzero, alpha, beta, u, v
            )
            # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

            # Construct rotation Qhat_{k,2k+1}.
            chat, shat, alphahat = self._givens(alphabar, damp)

            # Use a plane rotation (Q_i) to turn B_i to R_i
            rhoold = rho
            c, s, rho = self._givens(alphahat, beta)
            thetanew = s * alpha
            alphabar = c * alpha

            # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar
            rhobarold = rhobar
            zetaold = zeta
            thetabar = sbar * rho
            rhotemp = cbar * rho
            cbar, sbar, rhobar = self._givens(cbar * rho, thetanew)
            zeta = cbar * zetabar
            zetabar = -sbar * zetabar

            # Update h, h_hat, x.
            hbar = (ω(hbar) * -(thetabar * rho / (rhoold * rhobarold)).astype(dtype)).ω
            hbar = (ω(hbar) + ω(h)).ω
            x = (ω(x) + (zeta / (rho * rhobar)).astype(dtype) * ω(hbar)).ω
            h = (ω(h) * -(thetanew / rho).astype(dtype)).ω
            h = (ω(h) + ω(v)).ω

            # Estimate of ||r||.
            # Apply rotation Qhat_{k,2k+1}.
            betaacute = chat * betadd
            betacheck = -shat * betadd
            # Apply rotation Q_{k,k+1}.
            betahat = c * betaacute
            betadd = -s * betaacute

            # Apply rotation Qtilde_{k-1}.
            # betad = betad_{k-1} here.
            thetatildeold = thetatilde
            ctildeold, stildeold, rhotildeold = self._givens(rhodold, thetabar)
            thetatilde = stildeold * rhobar
            rhodold = ctildeold * rhobar
            betad = -stildeold * betad + ctildeold * betahat

            # betad   = betad_k here.
            # rhodold = rhod_k  here.

            tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
            taud = (zeta - thetatilde * tautildeold) / rhodold
            delta = delta + betacheck**2
            normr = jnp.sqrt(delta + (betad - taud) ** 2 + betadd**2)

            # Estimate ||A||.
            normA2 = normA2 + beta**2
            normA = jnp.sqrt(normA2)
            normA2 = normA2 + alpha**2

            # Estimate cond(A).
            maxrbar = jnp.maximum(maxrbar, rhobarold)
            minrbar = lax.select(itn > 1, jnp.minimum(minrbar, rhobarold), minrbar)
            condA = jnp.maximum(maxrbar, rhotemp) / jnp.minimum(minrbar, rhotemp)

            # Compute norms for convergence testing.
            normAr = jnp.abs(zetabar)
            normx = self.norm(x)

            # Now use these norms to estimate certain other quantities,
            # some of which will be small near a solution.
            test1 = normr / normb
            test2 = lax.select((normA * normr) != 0, normAr / (normA * normr), jnp.inf)
            test3 = 1 / condA
            t1 = test1 / (1.0 + normA * normx / normb)
            rtol = self.btol + self.atol * normA * normx / normb

            # bitmask for istop for possibly multiple conditions?

            # x is an approximate solution to A@x = b, according to atol and btol.
            istop += jnp.array(test1 <= rtol).astype(jnp.int32) * 2**1
            # x approximately solves the least-squares problem according to atol.
            istop += jnp.array(test2 <= self.atol).astype(jnp.int32) * 2**2
            # cond(A) seems to be greater than conlim
            istop += jnp.array(test3 <= ctol).astype(jnp.int32) * 2**3

            # The following tests guard against extremely small values of atol, btol
            # or ctol.  (The user may have set any or all of the parameters atol, btol,
            # conlim  to 0.) The effect is equivalent to the normal tests using
            # atol = eps,  btol = eps,  conlim = 1/eps.

            # x is an approximate solution to A@x = b, according to atol=btol=eps.
            istop += jnp.array(1 + t1 <= 1).astype(jnp.int32) * 2**4
            # x approximately solves the least-squares problem according to atol=eps.
            istop += jnp.array(1 + test2 <= 1).astype(jnp.int32) * 2**5
            # cond(A) seems to be greater than 1/eps
            istop += jnp.array(1 + test3 <= 1).astype(jnp.int32) * 2**6

            # maxiter exceeded
            istop += jnp.array(itn >= max_steps).astype(jnp.int32) * 2**7

            loop_state_vecs = (x, u, v, h, hbar)
            loop_state_stopping = (istop, normr, normAr, normb)
            loop_state_anorm = (normA2, maxrbar, minrbar, normA, condA)
            loop_state_r_est = (
                betadd,
                betad,
                rhodold,
                tautildeold,
                thetatilde,
                zeta,
                delta,
            )
            loop_state_main = (
                itn,
                alpha,
                beta,
                zetabar,
                alphabar,
                rho,
                rhobar,
                cbar,
                sbar,
            )
            loop_state = (
                loop_state_main,
                loop_state_r_est,
                loop_state_anorm,
                loop_state_stopping,
                loop_state_vecs,
            )
            return loop_state

        loop_state = lax.while_loop(condfun, bodyfun, loop_state)
        (
            loop_state_main,
            loop_state_r_est,
            loop_state_anorm,
            loop_state_stopping,
            loop_state_vecs,
        ) = loop_state
        (
            itn,
            alpha,
            beta,
            zetabar,
            alphabar,
            rho,
            rhobar,
            cbar,
            sbar,
        ) = loop_state_main
        (
            betadd,
            betad,
            rhodold,
            tautildeold,
            thetatilde,
            zeta,
            d,
        ) = loop_state_r_est
        (normA2, maxrbar, minrbar, normA, condA) = loop_state_anorm
        (istop, normr, normAr, normb) = loop_state_stopping
        (x, u, v, h, hbar) = loop_state_vecs

        stats = {
            "num_steps": itn,
            "istop": istop,
            "norm_r": normr,
            "norm_Ar": normAr,
            "norm_A": normA,
            "cond_A": condA,
            "norm_x": self.norm(x),
        }

        # TODO: return actual status when failed
        result = RESULTS.where(
            istop < 2**6,
            RESULTS.successful,
            RESULTS.max_steps_reached,
        )

        return x, result, stats

    def _givens(self, a, b):
        """Stable implementation of Givens rotation, from [1]_

        finds c, s, r such that

        |c  -s|[a| = |r|
        [s   c|[b|   |0|

        r = sqrt(a^2 + b^2)

        Assumes a, b are real.

        References
        ----------
        .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
            and Least-Squares Problems", Dissertation,
            http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf

        """

        def bzero(a, b):
            return jnp.sign(a), 0.0, abs(a)

        def azero(a, b):
            return 0.0, jnp.sign(b), abs(b)

        def b_gt_a(a, b):
            tau = a / b
            s = jnp.sign(b) / jnp.sqrt(1 + tau * tau)
            c = s * tau
            r = b / s
            return c, s, r

        def a_ge_b(a, b):
            tau = b / a
            c = jnp.sign(a) / jnp.sqrt(1 + tau * tau)
            s = c * tau
            r = a / c
            return c, s, r

        def either_zero(a, b):
            return lax.cond(b == 0, bzero, azero, a, b)

        def both_nonzero(a, b):
            return lax.cond(abs(b) > abs(a), b_gt_a, a_ge_b, a, b)

        return lax.cond((a == 0) | (b == 0), either_zero, both_nonzero, a, b)

    def transpose(self, state: _LSMRState, options: dict[str, Any]):
        del options
        operator = state
        transpose_options = {}
        return operator.transpose(), transpose_options

    def conj(self, state: _LSMRState, options: dict[str, Any]):
        del options
        operator = state
        conj_options = {}
        return conj(operator), conj_options

    def allow_dependent_rows(self, operator):
        return True

    def allow_dependent_columns(self, operator):
        return True


LSMR.__init__.__doc__ = r"""**Arguments:**

- `atol`: Relative tolerance (relative to norm(A*x)) for terminating solve.
- `btol`: Relative tolerance (relative to norm(b)) for terminating solve.
- `norm`: The norm to use when computing whether the error falls within the tolerance.
    Defaults to the two norm.
- `max_steps`: The maximum number of iterations to run the solver for. If more steps
    than this are required, then the solve is halted with a failure.
- `conlim`: The solver terminates if an estimate of cond(A) exceeds conlim. For
    compatible systems Ax = b, conlim could be as large as 1.0e+12 (say). For
    least-squares problems, conlim should be less than 1.0e+8. If conlim is None,
    the default value is 1e+8. Maximum precision can be obtained by setting
    atol = btol = conlim = 0, but the number of iterations may then be excessive.
    Default is 1e8.
"""
