import jax
import jax.numpy as jnp
import numpy as np

from numpyro.distributions.util import lazy_property
from functools import partial


@partial(jax.jit, static_argnums=(5,))
def P_Matern(k, n, sigma, rho, c=1e-20, k_zero=None):
    '''
    Power spectrum for the Matern covariance kernel.

    The covariance follows:
    C_n(d) = [(sigma^2 * 2^(1-n)) / Gamma(n)] * [sqrt(2*n) * d/rho]^n * K_n(sqrt(2*n) * d/rho)

    And the (2D) power spectrum follows:
    P(k) = sigma^2 * 4 * pi * n * (2 *n/rho^2)^n * [(2 *n/rho^2) + k^2]^-(n + 1)

    https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function

    parameters
    ----------
    k: spatial frequency
    n: "clumpy-ness" factor, smaller values lead to more small scale
        variation
    sigma: amplitude
    rho: large scale correlation length
    c: small scale power
    '''
    r = (2 * n / rho**2)
    norm = sigma**2 * 4 * jnp.pi * n * jnp.power(r, n)
    P = norm * jnp.power(r + k**2, -(n + 1))  # + c
    if k_zero is not None:
        P = jnp.where(k == 0, k_zero, P)
    return P


@partial(jax.jit, static_argnums=(5,))
def P_Wolf(k, n, sigma_ls, sigma_ss, r, k_zero=None):
    '''
    Power spectrum Wolfgang came up with

    P(k) = sigma_ss^2 + sigma_ls^2 * r^n * (r + k)^-n

    parameters
    ----------
    k: spatial frequency
    n: "clumpy-ness" factor, smaller values lead to more small scale
        variation
    sigma_ls: large scale amplitude
    sigma_ss: small scale amplitude
    r: large scale correlation length
    '''
    P = sigma_ss**2 + sigma_ls**2 * jnp.power(r, n) * jnp.power(r + k, -n)
    if k_zero is not None:
        P = jnp.where(k == 0, k_zero, P)
    return P


class K_grid():
    def __init__(self, shape, scale=1):
        '''
        Helper class for constructing a K grid for 2d FFTs or rFFTs.

        parameters
        ----------
        shape: tuple containing the shape of the image being FFTd
            (Ny, Nx)
        scale: (optional) physical size for each pixel, default=1
            (e.g. all sizes are in units of pixels)
        '''
        self.Ny, self.Nx = shape
        self.scale = scale

    @lazy_property
    def rk(self):
        '''Construct the *independent* spatial frequencies for a 2D real-FFT'''
        kx = 2 * np.pi * np.fft.rfftfreq(self.Nx, d=self.scale)
        ky = 2 * np.pi * np.fft.fftfreq(self.Ny, d=self.scale)
        return np.sqrt(ky.reshape(-1, 1)**2 + kx**2)

    @lazy_property
    def k(self):
        '''Construct all the spatial frequencies for a 2D FFT'''
        kx = 2 * np.pi * np.fft.fftfreq(self.Nx, d=self.scale)
        ky = 2 * np.pi * np.fft.fftfreq(self.Ny, d=self.scale)
        return np.sqrt(ky.reshape(-1, 1)**2 + kx**2)


@partial(jax.jit, static_argnums=(1,))
def odd_pack(values, N):
    '''Pack the value for an odd-by-odd FFT'''
    n1 = N//2 + 1
    thin_real = jax.lax.dynamic_slice(values, (0, 1), (N, n1 - 1))
    thin_imag = jnp.flip(jax.lax.dynamic_slice(values, (0, n1), (N, n1 - 1)), axis=1)

    first_real_slice = jax.lax.dynamic_slice(values, (1, 0), (n1 - 1, 1))
    first_real = jnp.vstack([
        2 * values[0, 0].reshape(1, 1),
        first_real_slice,
        jnp.flip(first_real_slice, axis=0)
    ])

    first_imag_slice = jax.lax.dynamic_slice(values, (n1, 0), (n1 - 1, 1))
    first_imag = jnp.vstack([
        jnp.zeros((1, 1)),
        -jnp.flip(first_imag_slice, axis=0),
        first_imag_slice,
    ])

    fft_real = jnp.hstack([first_real[:thin_real.shape[0]], thin_real])
    fft_imag = jnp.hstack([first_imag[:thin_imag.shape[0]], thin_imag])
    return fft_real + 1j * fft_imag


@partial(jax.jit, static_argnums=(1,))
def odd_unpack(values, N):
    n1 = N//2 + 1
    thin_slice = jax.lax.dynamic_slice(values, (0, 1), (N, n1 - 1))
    first_slice = jax.lax.dynamic_slice(values, (1, 0), (n1 - 1, 1))
    first = jnp.vstack([
        0.5 * values[0, 0].real.reshape(1, 1),
        first_slice.real,
        -jnp.flip(first_slice.imag, axis=0)
    ])

    delta = first.shape[0] - thin_slice.shape[0]
    thin_slice = jnp.pad(thin_slice, ((0, delta), (0, 0)))

    return jax.lax.dynamic_slice(
        jnp.hstack([
            first,
            thin_slice.real,
            jnp.flip(thin_slice.imag, axis=1)
        ]),
        (0, 0),
        (N, N)
    )


@partial(jax.jit, static_argnums=(1,))
def even_pack(values, N):
    '''Pack the value for an even-by-even FFT'''
    n1 = N//2 + 1
    thin_real = jax.lax.dynamic_slice(values, (0, 1), (N, n1 - 2))
    thin_imag = jnp.flip(jax.lax.dynamic_slice(values, (0, n1), (N, n1 - 2)), axis=1)

    first_real_slice = jax.lax.dynamic_slice(values, (1, 0), (n1 - 2, 1))
    first_real = jnp.vstack([
        2 * jax.lax.dynamic_slice(values, (0, 0), (1, 1)),
        first_real_slice,
        2 * jax.lax.dynamic_slice(values, (n1 - 1, 0), (1, 1)),
        jnp.flip(first_real_slice, axis=0)
    ])

    last_real_slice = jax.lax.dynamic_slice(values, (1, n1 - 1), (n1 - 2, 1))
    last_real = jnp.vstack([
        2 * jax.lax.dynamic_slice(values, (0, n1 - 1), (1, 1)),
        last_real_slice,
        2 * jax.lax.dynamic_slice(values, (n1 - 1, n1 - 1), (1, 1)),
        jnp.flip(last_real_slice, axis=0)
    ])

    first_imag_slice = jax.lax.dynamic_slice(values, (n1, 0), (n1 - 2, 1))
    first_imag = jnp.vstack([
        jnp.zeros((1, 1)),
        -jnp.flip(first_imag_slice, axis=0),
        jnp.zeros((1, 1)),
        first_imag_slice
    ])

    last_imag_slice = jax.lax.dynamic_slice(values, (n1, n1 - 1), (n1 - 2, 1))
    last_imag = jnp.vstack([
        jnp.zeros((1, 1)),
        -jnp.flip(last_imag_slice, axis=0),
        jnp.zeros((1, 1)),
        last_imag_slice
    ])

    delta = thin_real.shape[0] - first_real.shape[0]
    first_real = jnp.pad(first_real, ((0, delta), (0, 0)))
    last_real = jnp.pad(last_real, ((0, delta), (0, 0)))
    first_imag = jnp.pad(first_imag, ((0, delta), (0, 0)))
    last_imag = jnp.pad(last_imag, ((0, delta), (0, 0)))

    fft_real = jnp.hstack([first_real, thin_real, last_real])
    fft_imag = jnp.hstack([first_imag, thin_imag, last_imag])
    return fft_real + 1j * fft_imag


@partial(jax.jit, static_argnums=(1,))
def even_unpack(values, N):
    n1 = N//2 + 1
    thin_slice = jax.lax.dynamic_slice(values, (0, 1), (N, n1 - 2))
    first_slice = jax.lax.dynamic_slice(values, (1, 0), (n1 - 2, 1))
    first = jnp.vstack([
        0.5 * values[0, 0].real.reshape(1, 1),
        first_slice.real,
        0.5 * values[n1 - 1, 0].real.reshape(1, 1),
        -jnp.flip(first_slice.imag, axis=0)
    ])
    last_slice = jax.lax.dynamic_slice(values, (1, n1 - 1), (n1 - 2, 1))
    last = jnp.vstack([
        0.5 * values[0, n1 - 1].real.reshape(1, 1),
        last_slice.real,
        0.5 * values[n1 - 1, n1 - 1].real.reshape(1, 1),
        -jnp.flip(last_slice.imag, axis=0)
    ])

    delta = thin_slice.shape[0] - first.shape[0]
    first = jnp.pad(first, ((0, delta), (0, 0)))
    last = jnp.pad(last, ((0, delta), (0, 0)))

    out = jnp.hstack([
        first,
        thin_slice.real,
        last,
        jnp.flip(thin_slice.imag, axis=1)
    ])

    delta = N - out.shape[1]
    return jnp.pad(out, ((0, 0), (0, delta)))


@jax.jit
def pack_fft_values(values):
    '''
    Take values in a NxN array and re-arange them into
    the shape of the output of rFFT on an array of the
    same size by adding in the correct symmetries that
    come from a real image input.

    Doing this on a grid of values drawn from a standard
    Normal will result in the rFFT of a white noise image.

    This is used for drawing a white noise FFT in Fourier
    space so it can be "colored" by a power spectrum
    directly.

    parameters
    ----------
    values: The values to be packed, must be a square array
    '''
    Ny, Nx = values.shape
    assert Ny == Nx, 'Input array must be square'
    even_pack_n = partial(even_pack, N=Nx)
    odd_pack_n = partial(odd_pack, N=Nx)
    return jax.lax.cond(
        Nx % 2 == 0,
        even_pack_n,
        odd_pack_n,
        jnp.sqrt(0.5) * values
    )


vpack_fft_values = jax.vmap(pack_fft_values)


@jax.jit
def unpack_fft_values(values):
    Ny, _ = values.shape
    even_unpack_n = partial(even_unpack, N=Ny)
    odd_unpack_n = partial(odd_unpack, N=Ny)
    return jax.lax.cond(
        Ny % 2 == 0,
        even_unpack_n,
        odd_unpack_n,
        jnp.sqrt(2.0) * values
    )


vunpack_fft_values = jax.vmap(unpack_fft_values)
