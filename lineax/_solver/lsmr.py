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
from typing import Any, TypeAlias

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


class LSMR(AbstractLinearSolver[_LSMRState]):
    """LSMR solver for linear systems.

    This solver can handle any operator, even nonsquare or singular ones. In these
    cases it will return the pseudoinverse solution to the linear system.

    Similar to `scipy.sparse.linalg.lsmr`.

    This supports the following `options` (as passed to
    `lx.linear_solve(..., options=...)`).

    - `y0`: The initial estimate of the solution to the linear system. Defaults to all
        zeros.
    """

    rtol: float
    atol: float
    norm: Callable = two_norm
    max_steps: int | None = None
    conlim: float = 1e8

    def __check_init__(self):
        if isinstance(self.rtol, (int, float)) and self.rtol < 0:
            raise ValueError("Tolerances must be non-negative.")
        if isinstance(self.atol, (int, float)) and self.atol < 0:
            raise ValueError("Tolerances must be non-negative.")
        if isinstance(self.conlim, (int, float)) and self.conlim < 0:
            raise ValueError("Tolerances must be non-negative.")

        if isinstance(self.atol, (int, float)) and isinstance(self.rtol, (int, float)):
            if self.atol == 0 and self.rtol == 0 and self.max_steps is None:
                raise ValueError(
                    "Must specify `atol`, `rtol`, or `max_steps` (or some combination "
                    "of all three)."
                )

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
        # damp is not supported at this time.
        #  damp = options.get("damp", 0.0)
        damp = 0.0
        has_scale = not (
            isinstance(self.atol, (int, float))
            and isinstance(self.rtol, (int, float))
            and self.atol == 0
            and self.rtol == 0
        )

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

        # Initialize variables for 1st iteration.
        # generally, latin letters (b, x, u, v, h etc) are vectors that may be complex
        # greek letters (alpha, beta, rho, zeta etc) are scalars that are always real
        loop_state = dict(
            # vectors
            x=x,
            u=u,
            v=v,
            h=h,
            hbar=hbar,
            # main loop variables
            itn=0,
            alpha=alpha,
            beta=beta,
            zetabar=alpha * beta,
            alphabar=alpha,
            rho=1.0,
            rhobar=1.0,
            cbar=1.0,
            sbar=0.0,
            # loop variables for estimation of ||r||.
            betadd=beta,
            betad=0.0,
            rhodold=1.0,
            tautildeold=0.0,
            thetatilde=0.0,
            zeta=0.0,
            delta=0.0,
            # variables for estimation of ||A|| and cond(A)
            normA2=alpha**2,
            maxrbar=0.0,
            minrbar=1e100,
            condA=1.0,
            # variables for use in stopping rules
            istop=0,
            normr=beta,
            normAr=alpha * beta,
        )

        def condfun(loop_state):
            return loop_state["istop"] == 0

        def bodyfun(loop_state):
            st = loop_state  # to avoid writing out loop_state every time
            st["itn"] = st["itn"] + 1

            # Perform the next step of the bidiagonalization to obtain the
            # next  beta, u, alpha, v.  These satisfy the relations
            #         beta*u  =  A@v   -  alpha*u,
            #        alpha*v  =  A'@u  -  beta*v.

            st["u"] = (ω(st["u"]) * -st["alpha"].astype(dtype)).ω
            st["u"] = (ω(st["u"]) + ω(operator.mv(st["v"]))).ω
            st["beta"] = self.norm(st["u"])

            def beta_nonzero(alpha, beta, u, v):
                u = (ω(u) / lax.select(beta == 0.0, 1.0, beta).astype(dtype)).ω
                v = (ω(v) * -beta.astype(dtype)).ω
                v = (ω(v) + ω(conj(operator).T.mv(u))).ω
                alpha = self.norm(v)
                v = (ω(v) / lax.select(alpha == 0.0, 1.0, alpha).astype(dtype)).ω
                return alpha, beta, u, v

            def beta_zero(alpha, beta, u, v):
                return alpha, beta, u, v

            st["alpha"], st["beta"], st["u"], st["v"] = lax.cond(
                st["beta"] == 0,
                beta_zero,
                beta_nonzero,
                st["alpha"],
                st["beta"],
                st["u"],
                st["v"],
            )
            # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

            # Construct rotation Qhat_{k,2k+1}.
            chat, shat, alphahat = self._givens(st["alphabar"], damp)

            # Use a plane rotation (Q_i) to turn B_i to R_i
            rhoold = st["rho"]
            c, s, st["rho"] = self._givens(alphahat, st["beta"])
            thetanew = s * st["alpha"]
            st["alphabar"] = c * st["alpha"]

            # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar
            rhobarold = st["rhobar"]
            zetaold = st["zeta"]
            thetabar = st["sbar"] * st["rho"]
            rhotemp = st["cbar"] * st["rho"]
            st["cbar"], st["sbar"], st["rhobar"] = self._givens(
                st["cbar"] * st["rho"], thetanew
            )
            st["zeta"] = st["cbar"] * st["zetabar"]
            st["zetabar"] = -st["sbar"] * st["zetabar"]

            # Update h, h_hat, x.
            st["hbar"] = (
                ω(st["hbar"])
                * -(thetabar * st["rho"] / (rhoold * rhobarold)).astype(dtype)
            ).ω
            st["hbar"] = (ω(st["hbar"]) + ω(st["h"])).ω
            st["x"] = (
                ω(st["x"])
                + (st["zeta"] / (st["rho"] * st["rhobar"])).astype(dtype)
                * ω(st["hbar"])
            ).ω
            st["h"] = (ω(st["h"]) * -(thetanew / st["rho"]).astype(dtype)).ω
            st["h"] = (ω(st["h"]) + ω(st["v"])).ω

            # Estimate of ||r||.
            # Apply rotation Qhat_{k,2k+1}.
            betaacute = chat * st["betadd"]
            betacheck = -shat * st["betadd"]
            # Apply rotation Q_{k,k+1}.
            betahat = c * betaacute
            st["betadd"] = -s * betaacute

            # Apply rotation Qtilde_{k-1}.
            # betad = betad_{k-1} here.
            thetatildeold = st["thetatilde"]
            ctildeold, stildeold, rhotildeold = self._givens(st["rhodold"], thetabar)
            st["thetatilde"] = stildeold * loop_state["rhobar"]
            st["rhodold"] = ctildeold * st["rhobar"]
            st["betad"] = -stildeold * st["betad"] + ctildeold * betahat

            # betad   = betad_k here.
            # rhodold = rhod_k  here.

            loop_state["tautildeold"] = (
                zetaold - thetatildeold * st["tautildeold"]
            ) / rhotildeold
            taud = (st["zeta"] - st["thetatilde"] * st["tautildeold"]) / st["rhodold"]
            st["delta"] = st["delta"] + betacheck**2
            st["normr"] = jnp.sqrt(
                st["delta"] + (st["betad"] - taud) ** 2 + st["betadd"] ** 2
            )

            # Estimate ||A||.
            st["normA2"] = st["normA2"] + st["beta"] ** 2
            normA = jnp.sqrt(st["normA2"])
            st["normA2"] = st["normA2"] + st["alpha"] ** 2

            # Estimate cond(A).
            st["maxrbar"] = jnp.maximum(st["maxrbar"], rhobarold)
            st["minrbar"] = lax.select(
                st["itn"] > 1, jnp.minimum(st["minrbar"], rhobarold), st["minrbar"]
            )
            st["condA"] = jnp.maximum(st["maxrbar"], rhotemp) / jnp.minimum(
                st["minrbar"], rhotemp
            )

            # Compute norms for convergence testing.
            st["normAr"] = jnp.abs(st["zetabar"])
            normx = self.norm(st["x"])

            well_posed_tol = self.atol + self.rtol * (normA * normx + normb)
            least_squares_tol = self.atol + self.rtol * (normA * st["normr"])
            # maxiter exceeded
            st["istop"] = lax.select(st["itn"] >= max_steps, 4, st["istop"])
            # cond(A) seems to be greater than conlim
            st["istop"] = lax.select(st["condA"] > self.conlim, 3, st["istop"])
            # x solves the least-squares problem according to atol and rtol.
            st["istop"] = lax.select(st["normAr"] < least_squares_tol, 2, st["istop"])
            # x is a solution to A@x = b, according to atol and rtol.
            st["istop"] = lax.select(st["normr"] < well_posed_tol, 1, st["istop"])
            return st

        loop_state = lax.while_loop(condfun, bodyfun, loop_state)

        stats = {
            "num_steps": loop_state["itn"],
            "istop": loop_state["istop"],
            "norm_r": loop_state["normr"],
            "norm_Ar": loop_state["normAr"],
            "norm_A": jnp.sqrt(loop_state["normA2"]),
            "cond_A": loop_state["condA"],
            "norm_x": self.norm(loop_state["x"]),
        }
        if (self.max_steps is None) or (max_steps < self.max_steps):
            result = RESULTS.where(
                loop_state["itn"] == max_steps,
                RESULTS.singular,
                RESULTS.successful,
            )
        else:
            result = RESULTS.where(
                loop_state["itn"] == max_steps,
                RESULTS.max_steps_reached if has_scale else RESULTS.successful,
                RESULTS.successful,
            )

        result = RESULTS.where(loop_state["istop"] < 3, RESULTS.successful, result)
        result = RESULTS.where(loop_state["istop"] == 3, RESULTS.conlim, result)

        return loop_state["x"], result, stats

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
        assert not jnp.iscomplexobj(a)
        assert not jnp.iscomplexobj(b)

        def bzero(a, b):
            return jnp.sign(a), 0.0, jnp.abs(a)

        def azero(a, b):
            return 0.0, jnp.sign(b), jnp.abs(b)

        def b_gt_a(a, b):
            tau = a / lax.select(b == 0.0, 1.0, b)
            s = jnp.sign(b) / jnp.sqrt(1.0 + tau**2)
            c = s * tau
            r = b / lax.select(s == 0.0, 1.0, s)
            return c, s, r

        def a_ge_b(a, b):
            tau = b / lax.select(a == 0.0, 1.0, a)
            c = jnp.sign(a) / jnp.sqrt(1.0 + tau**2)
            s = c * tau
            r = a / lax.select(c == 0.0, 1.0, c)
            return c, s, r

        def either_zero(a, b):
            return lax.cond(b == 0.0, bzero, azero, a, b)

        def both_nonzero(a, b):
            return lax.cond(jnp.abs(b) > jnp.abs(a), b_gt_a, a_ge_b, a, b)

        return lax.cond((a == 0.0) | (b == 0.0), either_zero, both_nonzero, a, b)

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

    def assume_full_rank(self):
        return False


LSMR.__init__.__doc__ = r"""**Arguments:**

- `rtol`: Relative tolerance for terminating solve.
- `atol`: Absolute tolerance for terminating solve.
- `norm`: The norm to use when computing whether the error falls within the tolerance.
    Defaults to the two norm.
- `max_steps`: The maximum number of iterations to run the solver for. If more steps
    than this are required, then the solve is halted with a failure.
- `conlim`: The solver terminates if an estimate of cond(A) exceeds conlim. For
    compatible systems Ax = b, conlim could be as large as 1.0e+12 (say). For
    least-squares problems, conlim should be less than 1.0e+8. If conlim is None,
    the default value is 1e+8. Maximum precision can be obtained by setting
    atol = rtol = 0, conlim = np.inf, but the number of iterations may then be
    excessive. Default is 1e8.
"""
