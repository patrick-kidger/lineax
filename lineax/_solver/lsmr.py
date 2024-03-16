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


import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω

from .._norm import two_norm
from .._operator import conj
from .._solution import RESULTS
from .._solve import AbstractLinearSolver


def givens(a, b):
    """Stable implementation of Givens rotation, from [1]_

    finds c, s such that

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


class LSMR(AbstractLinearSolver):
    atol: float = 1e-6
    btol: float = 1e-6
    max_steps: int = None
    norm = two_norm
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

    def init(self, operator, options):
        return operator

    def compute(self, state, vector, options):
        operator = state
        x0 = options.get("x0", None)
        damp = options.get("damp", 0.0)

        m, n = operator.out_size(), operator.in_size()
        # stores the num of singular values
        minDim = min([m, n])
        if self.max_steps is None:
            max_steps = minDim
        else:
            max_steps = self.max_steps

        if x0 is None:
            x = jtu.tree_map(jnp.zeros_like, operator.in_structure())

        b = vector
        u = b - operator.mv(x)
        normb = self.norm(b)
        beta = self.norm(u)

        def beta_gt0(beta, u):
            u = (ω(u) * (1 / beta)).ω
            v = operator.T.mv(u)
            alpha = self.norm(v)
            return u, v, alpha

        def beta_le0(beta, u):
            v = jtu.tree_map(jnp.zeros_like, operator.in_structure())
            alpha = 0.0
            return u, v, alpha

        u, v, alpha = lax.cond(beta > 0, beta_gt0, beta_le0, beta, u)
        v = jnp.where(alpha > 0, v / alpha, v)

        h = v.copy()
        hbar = jtu.tree_map(jnp.zeros_like, operator.in_structure())

        state_vecs = (x, u, v, h, hbar)

        # Initialize variables for 1st iteration.
        itn = 0
        zetabar = alpha * beta
        alphabar = alpha
        rho = 1.0
        rhobar = 1.0
        cbar = 1.0
        sbar = 0.0

        state_main = (itn, alpha, beta, zetabar, alphabar, rho, rhobar, cbar, sbar)

        # Initialize variables for estimation of ||r||.
        betadd = beta
        betad = 0.0
        rhodold = 1.0
        tautildeold = 0.0
        thetatilde = 0.0
        zeta = 0.0
        d = 0.0

        state_r_est = (betadd, betad, rhodold, tautildeold, thetatilde, zeta, d)

        # Initialize variables for estimation of ||A|| and cond(A)
        normA2 = alpha**2
        maxrbar = 0.0
        minrbar = 1e100
        normA = jnp.sqrt(normA2)
        condA = 1.0

        state_anorm = (normA2, maxrbar, minrbar, normA, condA)

        # Items for use in stopping rules, normb set earlier
        istop = 0
        ctol = 0.0
        if self.conlim > 0:
            ctol = 1 / self.conlim
        normr = beta

        state_stopping = (istop, normr, normb)
        # TODO: implement this early exit?
        #         normar = alpha * beta
        #         if normar == 0:
        #             return x, istop, itn, normr, normar, normA, condA, normx

        #         if normb == 0:
        #             x[()] = 0
        #             return x, istop, itn, normr, normar, normA, condA, normx

        state = (state_main, state_r_est, state_anorm, state_stopping, state_vecs)

        def condfun(state):
            state_main, state_r_est, state_anorm, state_stopping, state_vecs = state
            (itn, alpha, beta, zetabar, alphabar, rho, rhobar, cbar, sbar) = state_main
            (betadd, betad, rhodold, tautildeold, thetatilde, zeta, d) = state_r_est
            (normA2, maxrbar, minrbar, normA, condA) = state_anorm
            (istop, normr, normb) = state_stopping
            (x, u, v, h, hbar) = state_vecs

            # Test for convergence.

            # Compute norms for convergence testing.
            normar = abs(zetabar)
            normx = self.norm(x)

            # Now use these norms to estimate certain other quantities,
            # some of which will be small near a solution.
            test1 = normr / normb
            test2 = jnp.where((normA * normr) != 0, normar / (normA * normr), jnp.inf)
            test3 = 1 / condA
            t1 = test1 / (1 + normA * normx / normb)
            rtol = self.btol + self.atol * normA * normx / normb

            # The following tests guard against extremely small values of
            # atol, btol or ctol.  (The user may have set any or all of
            # the parameters atol, btol, conlim  to 0.)
            # The effect is equivalent to the normAl tests using
            # atol = eps,  btol = eps,  conlim = 1/eps.

            # bitmask for istop for possibly multiple conditions?
            istop += (itn >= max_steps) * 2**7
            istop += (1 + test3 <= 1) * 2**6
            istop += (1 + test2 <= 1) * 2**5
            istop += (1 + t1 <= 1) * 2**4
            # Allow for tolerances set by the user.
            istop += (test3 <= ctol) * 2**3
            istop += (test2 <= self.atol) * 2**2
            istop += (test1 <= rtol) * 2**1

            return istop == 0

        def bodyfun(state):
            # unpack everything. Maybe cleaner to use a dict or struct?
            (state_main, state_r_est, state_anorm, state_stopping, state_vecs) = state
            (itn, alpha, beta, zetabar, alphabar, rho, rhobar, cbar, sbar) = state_main
            (betadd, betad, rhodold, tautildeold, thetatilde, zeta, d) = state_r_est
            (normA2, maxrbar, minrbar, normA, condA) = state_anorm
            (istop, normr, normb) = state_stopping
            (x, u, v, h, hbar) = state_vecs

            itn = itn + 1

            # Perform the next step of the bidiagonalization to obtain the
            # next  beta, u, alpha, v.  These satisfy the relations
            #         beta*u  =  A@v   -  alpha*u,
            #        alpha*v  =  A'@u  -  beta*v.

            u = (ω(u) * -alpha).ω
            u = (ω(u) + ω(operator.mv(v))).ω
            beta = self.norm(u)

            def beta_gt0(alpha, beta, u, v):
                u = (ω(u) * (1 / beta)).ω
                v = (ω(v) * -beta).ω
                v = (ω(v) + operator.T.mv(u)).ω
                alpha = self.norm(v)
                v = lax.cond(alpha > 0, lambda x: (ω(x) / alpha).ω, lambda x: x, v)
                return alpha, beta, u, v

            def beta_le0(alpha, beta, u, v):
                return alpha, beta, u, v

            alpha, beta, u, v = lax.cond(
                beta > 0, beta_gt0, beta_le0, alpha, beta, u, v
            )
            # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

            # Construct rotation Qhat_{k,2k+1}.
            chat, shat, alphahat = givens(alphabar, damp)

            # Use a plane rotation (Q_i) to turn B_i to R_i
            rhoold = rho
            c, s, rho = givens(alphahat, beta)
            thetanew = s * alpha
            alphabar = c * alpha

            # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar
            rhobarold = rhobar
            zetaold = zeta
            thetabar = sbar * rho
            rhotemp = cbar * rho
            cbar, sbar, rhobar = givens(cbar * rho, thetanew)
            zeta = cbar * zetabar
            zetabar = -sbar * zetabar

            # Update h, h_hat, x.
            hbar = (ω(hbar) * -(thetabar * rho / (rhoold * rhobarold))).ω
            hbar = (ω(hbar) + ω(h)).ω
            x = (ω(x) + (zeta / (rho * rhobar)) * ω(hbar)).ω
            h = (ω(h) * -(thetanew / rho)).ω
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
            ctildeold, stildeold, rhotildeold = givens(rhodold, thetabar)
            thetatilde = stildeold * rhobar
            rhodold = ctildeold * rhobar
            betad = -stildeold * betad + ctildeold * betahat

            # betad   = betad_k here.
            # rhodold = rhod_k  here.

            tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
            taud = (zeta - thetatilde * tautildeold) / rhodold
            d = d + betacheck * betacheck
            normr = jnp.sqrt(d + (betad - taud) ** 2 + betadd * betadd)

            # Estimate ||A||.
            normA2 = normA2 + beta**2
            normA = jnp.sqrt(normA2)
            normA2 = normA2 + alpha**2

            # Estimate cond(A).
            maxrbar = jnp.maximum(maxrbar, rhobarold)
            minrbar = jnp.where(itn > 1, jnp.minimum(minrbar, rhobarold), minrbar)
            condA = jnp.maximum(maxrbar, rhotemp) / jnp.minimum(minrbar, rhotemp)

            state_vecs = (x, u, v, h, hbar)
            state_stopping = (istop, normr, normb)
            state_anorm = (normA2, maxrbar, minrbar, normA, condA)
            state_r_est = (betadd, betad, rhodold, tautildeold, thetatilde, zeta, d)
            state_main = (itn, alpha, beta, zetabar, alphabar, rho, rhobar, cbar, sbar)
            state = (state_main, state_r_est, state_anorm, state_stopping, state_vecs)
            return state

        state = lax.while_loop(condfun, bodyfun, state)

        (state_main, state_r_est, state_anorm, state_stopping, state_vecs) = state
        (itn, alpha, beta, zetabar, alphabar, rho, rhobar, cbar, sbar) = state_main
        (betadd, betad, rhodold, tautildeold, thetatilde, zeta, d) = state_r_est
        (normA2, maxrbar, minrbar, normA, condA) = state_anorm
        (istop, normr, normb) = state_stopping
        (x, u, v, h, hbar) = state_vecs

        normar = abs(zetabar)
        normx = self.norm(x)
        stats = {
            "num_steps": itn,
            "norm_r": normr,
            "norm_Ar": normar,
            "norm_A": normA,
            "cond_A": condA,
            "norm_x": normx,
        }

        # TODO: return actual status when failed
        result = RESULTS.where(
            istop == 0,
            RESULTS.successful,
            RESULTS.max_steps_reached,
        )

        return x, result, stats

    def transpose(self, state, options):
        del options
        operator = state
        transpose_options = {}
        return operator.transpose(), transpose_options

    def conj(self, state, options):
        del options
        operator = state
        conj_options = {}
        return conj(operator), conj_options

    def allow_dependent_rows(self, operator):
        return True

    def allow_dependent_columns(self, operator):
        return True
