import jax
import jax.numpy as jnp

import lineax._misc as lx_misc


def test_inexact_asarray_no_copy():
    x = jnp.array([1.0])
    assert lx_misc.inexact_asarray(x) is x
    y = jnp.array([1.0, 2.0])
    assert jax.vmap(lx_misc.inexact_asarray)(y) is y


# See JAX issue #15676
def test_inexact_asarray_jvp():
    p, t = jax.jvp(lx_misc.inexact_asarray, (1.0,), (2.0,))
    assert type(p) is not float
    assert type(t) is not float
