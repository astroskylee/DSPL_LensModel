from __future__ import annotations

import inspect
import pickle
import warnings
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.infer as infer
import numpyro.infer.autoguide as autoguide
import optax
import xarray as xr
from astropy.io import fits
from corner import corner
from jax import lax
from matplotlib.gridspec import GridSpec
from numpyro import distributions as dist
from numpyro.handlers import condition
from scipy.optimize import least_squares
from scipy.special import roots_legendre

from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.noise import Noise
from herculens.Instrument.psf import PSF
from herculens.LightModel.light_model import LightModel
from herculens.MassModel import mass_model_base
from herculens.MassModel.mass_model import MassModel
from herculens.PointSourceModel.point_source_model import PointSourceModel
from jax_lensing_profiles.MassModel.Profiles.CuspyNFW_ellipse_kappa import CuspyNFW_3D_fn
from jax_lensing_profiles.MassModel.Profiles.MGE import MGE
from lens_images_extension import LensImageExtension
from power_spectrum_prior import K_grid, P_Matern, pack_fft_values


class Plot:
    @staticmethod
    def sanitize_label(label):
        return (
            str(label)
            .replace(' ', '_')
            .replace('(', '')
            .replace(')', '')
            .replace('/', '_')
        )

    @staticmethod
    def plot_loss(losses, max_iterations, ax=None, axins=None, inset=True, **kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=(15, 3.5))
        ax.plot(losses, **kwargs)
        ax.set_yscale('asinh')

        if inset and axins is None:
            axins = ax.inset_axes([0.3, 0.5, 0.64, 0.45])
        n_end = max_iterations // 3
        x_plot = np.linspace(max_iterations - n_end, max_iterations, n_end)
        if inset:
            axins.plot(x_plot, losses[max_iterations - n_end:], **kwargs)
            ax.indicate_inset_zoom(axins, edgecolor='k')
        return ax


class ResumeInit:
    @staticmethod
    def select_init_values(params, allowed_keys):
        return {k: params[k] for k in allowed_keys if k in params}

    @staticmethod
    def stack_or_none(*xs):
        first = xs[0]
        return None if first is None else jnp.stack(xs)

    @staticmethod
    def stack_dicts(dict_list):
        return jax.tree.map(lambda *xs: jnp.stack(xs), *dict_list)

    @staticmethod
    def existing_batch_indices(output_dir, suffix_hmc):
        prefix = 'WFI2033_'
        indices = []
        for path in Path(output_dir).glob(f'WFI2033_[0-9]*{suffix_hmc}.nc'):
            name = path.name
            idx_str = name[len(prefix):name.index(suffix_hmc)]
            indices.append(int(idx_str))
        return sorted(indices)

    @staticmethod
    def save_resume_state(path, state):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as fh:
            pickle.dump(jax.device_get(state), fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_resume_state(path):
        with Path(path).open('rb') as fh:
            return pickle.load(fh)

    @staticmethod
    @jax.jit
    def get_value_from_index(xs, i):
        i = jnp.asarray(i)
        return jax.tree.map(lambda x: x[i], xs)

    @staticmethod
    def init_to_value_or_defer(site=None, values=None, defer=numpyro.infer.init_to_median):
        if values is None:
            values = {}
        if site is None:
            return partial(ResumeInit.init_to_value_or_defer, values=values, defer=defer)

        if site["type"] == "sample" and not site["is_observed"]:
            if site["name"] in values:
                return values[site["name"]]
            return defer(site)


class Geometry:
    @staticmethod
    def get_pixel_grid(data, pix_scale, ss=1):
        ny, nx = data.shape
        ny *= ss
        nx *= ss
        pix_scale /= ss
        half_size_x = nx * pix_scale / 2
        half_size_y = ny * pix_scale / 2
        ra_at_xy_0 = -half_size_x + pix_scale / 2
        dec_at_xy_0 = -half_size_y + pix_scale / 2
        transform_pix2angle = pix_scale * np.eye(2)
        kwargs_pixel = {
            'nx': nx,
            'ny': ny,
            'ra_at_xy_0': ra_at_xy_0,
            'dec_at_xy_0': dec_at_xy_0,
            'transform_pix2angle': transform_pix2angle,
        }
        pixel_grid = PixelGrid(**kwargs_pixel)
        xgrid, ygrid = pixel_grid.pixel_coordinates
        x_axis = xgrid[0]
        y_axis = ygrid[:, 0]
        extent = pixel_grid.extent
        return pixel_grid, xgrid, ygrid, x_axis, y_axis, extent, nx, ny

    @staticmethod
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def reduced_distance(a, i, j):
        u = jax.lax.dynamic_slice_in_dim(a, i, 1)
        v = jax.lax.dynamic_slice_in_dim(a, j, 1)
        return jnp.linalg.norm(u - v, ord=2)

    @staticmethod
    def reduced_distance_matrix(a):
        i, j = jnp.triu_indices(a.shape[0], k=1)
        return Geometry.reduced_distance(a, i, j)


class Mass:
    @staticmethod
    def scale_theta_E_from_g2(theta_E_g2, target_theta_mean, g2_theta_mean):
        return theta_E_g2 * target_theta_mean / g2_theta_mean

    @staticmethod
    def SIS(
        plate_name,
        param_name,
        origin,
        theta_low=0.0,
        theta_high=0.01,
        theta_mean=None,
        theta_sigma=None,
    ):
        with numpyro.plate(f'{plate_name} scalers - [1]', 1):
            if theta_mean is not None and theta_sigma is not None:
                theta_E = numpyro.sample(
                    f'theta_E_{param_name}',
                    dist.Normal(theta_mean, theta_sigma),
                )
            else:
                theta_E = numpyro.sample(
                    f'theta_E_{param_name}',
                    dist.Uniform(theta_low, theta_high),
                )
            center_0 = numpyro.deterministic(f'center_1_{param_name}', jnp.array([origin[0]]))
            center_1 = numpyro.deterministic(f'center_2_{param_name}', jnp.array([origin[1]]))

        return [{
            'theta_E': theta_E[0],
            'center_x': center_0[0],
            'center_y': center_1[0],
        }]

    @staticmethod
    def EPL_with_shear(
        plate_name,
        param_name,
        theta_low,
        theta_high,
        gamma_low=1.6,
        gamma_high=2.4,
        e_low=-0.2,
        e_high=0.2,
        center_low=-0.3,
        center_high=0.3,
        shear_low=-0.15,
        shear_high=0.15,
    ):
        with numpyro.plate(f'{plate_name} scalers - [1]', 1):
            theta_e = numpyro.sample(f'theta_E_{param_name}', dist.Uniform(theta_low, theta_high))
            gamma = numpyro.sample(f'gamma_{param_name}', dist.Uniform(gamma_low, gamma_high))
        with numpyro.plate(f'{plate_name} vectors - [2]', 2):
            ellipticity = numpyro.sample(
                f'e_{param_name}',
                dist.TruncatedNormal(0.0, 0.15, low=e_low, high=e_high),
            )
            shear = numpyro.sample(f'gamma_sheer_{param_name}', dist.Uniform(shear_low, shear_high))
        center = numpyro.sample(
            f'center_{param_name}',
            dist.TruncatedNormal(0.0, 0.08, low=center_low, high=center_high).expand([2]),
        )
        return [{
            'theta_E': theta_e[0],
            'gamma': gamma[0],
            'e1': ellipticity[0],
            'e2': ellipticity[1],
            'center_x': center[0],
            'center_y': center[1],
        }, {
            'gamma1': shear[0],
            'gamma2': shear[1],
            'ra_0': center[0],
            'dec_0': center[1],
        }]

    @staticmethod
    def params2kwargs_EPL_with_shear(params, param_name):
        return [{
            'theta_E': params[f'theta_E_{param_name}'][0],
            'gamma': params[f'gamma_{param_name}'][0],
            'e1': params[f'e_{param_name}'][0],
            'e2': params[f'e_{param_name}'][1],
            'center_x': params[f'center_{param_name}'][0],
            'center_y': params[f'center_{param_name}'][1],
        }, {
            'gamma1': params[f'gamma_sheer_{param_name}'][0],
            'gamma2': params[f'gamma_sheer_{param_name}'][1],
            'ra_0': params[f'center_{param_name}'][0],
            'dec_0': params[f'center_{param_name}'][1],
        }]

    @staticmethod
    def params2kwargs_SIS(params, param_name, fallback_origin=None):
        if f'center_1_{param_name}' in params and f'center_2_{param_name}' in params:
            center_x = params[f'center_1_{param_name}'][0]
            center_y = params[f'center_2_{param_name}'][0]
        elif fallback_origin is not None:
            center_x, center_y = fallback_origin
        else:
            raise KeyError(f'Missing SIS center sites for {param_name}.')
        return [{
            'theta_E': params[f'theta_E_{param_name}'][0],
            'center_x': center_x,
            'center_y': center_y,
        }]

    @staticmethod
    def GNFW_w_shear(
        plate_name,
        param_name,
        gamma_in_up=2,
        gamma_in_low=0.5,
        Rs_high=None,
        Rs_low=None,
        Rs_mean=None,
        Rs_std=None,
        Rs_value=None,
        e_low=-0.2,
        e_high=0.2,
        center_x=None,
        center_y=None,
        kappa_s_low=0.0,
        kappa_s_high=1,
        sph=False,
        gamma_sheer_low=-0.2,
        gamma_sheer_high=0.2,
        gamma_sheer_value=None,
    ):
        if Rs_value is not None:
            Rs = numpyro.deterministic(f'Rs_{param_name}', jnp.float64(Rs_value))
        elif Rs_low is not None:
            Rs = numpyro.sample(f'Rs_{param_name}', dist.Uniform(Rs_low, Rs_high))
        elif Rs_mean is not None:
            Rs = numpyro.sample(
                f'Rs_{param_name}',
                dist.TruncatedNormal(Rs_mean, Rs_std, low=Rs_mean - 1 * Rs_std, high=Rs_mean + 1 * Rs_std),
            )
        else:
            raise ValueError('GNFW_w_shear requires one of Rs_value, Rs_low/Rs_high, or Rs_mean/Rs_std')

        kappa_s = numpyro.sample(f'kappa_s_{param_name}', dist.Uniform(kappa_s_low, kappa_s_high))
        gamma_in = numpyro.sample(f'gammain_{param_name}', dist.Uniform(gamma_in_low, gamma_in_up))

        if gamma_sheer_value is None:
            with numpyro.plate(f'{plate_name} vectors - [2]', 2):
                gamma_sheer = numpyro.sample(
                    f'gamma_sheer_{param_name}',
                    dist.Uniform(gamma_sheer_low, gamma_sheer_high),
                )
        else:
            gamma_sheer = numpyro.deterministic(
                f'gamma_sheer_{param_name}',
                jnp.asarray(gamma_sheer_value, dtype=jnp.float64).reshape(2,),
            )

        if sph is False:
            with numpyro.plate(f'{plate_name} vectors - [2]', 2):
                e_mass = numpyro.sample(
                    f'e_{param_name}',
                    dist.TruncatedNormal(0, 0.25, low=e_low, high=e_high),
                )
        else:
            e_mass = numpyro.deterministic(f"e_{param_name}", jnp.array([0.0001, -0.0001]))

        if center_x is None:
            center = numpyro.sample(
                f'center_{param_name}',
                dist.TruncatedNormal(0, 1, low=-0.4, high=0.4).expand([2]),
            )
        else:
            center = numpyro.deterministic(f"center_{param_name}", jnp.array([center_x, center_y]))

        return [{
            'R_s': Rs,
            'gamma': gamma_in,
            'kappa_s': kappa_s,
            'e1': e_mass[0],
            'e2': e_mass[1],
            'center_x': center[0],
            'center_y': center[1],
        }, {
            'gamma1': gamma_sheer[0],
            'gamma2': gamma_sheer[1],
            'ra_0': center[0],
            'dec_0': center[1],
        }]


class Light:
    @staticmethod
    def multi_gauss_light(
        plate_name,
        param_name,
        n_gauss,
        sigma_lims,
        center_low=None,
        center_high=None,
        e_low=-0.2,
        e_high=0.2,
    ):
        sigma_low, sigma_high = sigma_lims
        if sigma_low <= 0:
            sigma_bins = jnp.linspace(sigma_low, sigma_high, n_gauss + 1)
            sigma_dist = dist.Uniform(sigma_bins[:-1], sigma_bins[1:])
        else:
            sigma_bins = jnp.logspace(jnp.log10(sigma_low), jnp.log10(sigma_high), n_gauss + 1)
            sigma_dist = dist.LogUniform(sigma_bins[:-1], sigma_bins[1:])
        with numpyro.plate(f'{plate_name} - [{n_gauss}]', n_gauss):
            amp_scale = numpyro.sample(f'A_{param_name}', dist.LogUniform(1e-5, 1e4))
            sigma = numpyro.sample(f'sigma_{param_name}', sigma_dist)
            with numpyro.plate(f'{plate_name} vectors - [2]', 2):
                ellipticity = numpyro.sample(
                    f'e_{param_name}',
                    dist.TruncatedNormal(0.0, 0.1, low=e_low, high=e_high),
                )
                if center_low is not None or center_high is not None:
                    center = numpyro.sample(
                        f'center_{param_name}',
                        dist.TruncatedNormal(0.0, 0.08, low=center_low, high=center_high),
                    )
                else:
                    center = numpyro.sample(f'center_{param_name}', dist.Normal(0.0, 0.2))
        amp = numpyro.deterministic(f'amp_{param_name}', amp_scale * sigma**2)
        return [{
            'amp': amp,
            'sigma': sigma,
            'e1': ellipticity[0],
            'e2': ellipticity[1],
            'center_x': center[0],
            'center_y': center[1],
        }]

    @staticmethod
    def params2kwargs_multi_gauss_light(params, param_name):
        return [{
            'amp': params[f'amp_{param_name}'],
            'sigma': params[f'sigma_{param_name}'],
            'e1': params[f'e_{param_name}'][0],
            'e2': params[f'e_{param_name}'][1],
            'center_x': params[f'center_{param_name}'][0],
            'center_y': params[f'center_{param_name}'][1],
        }]

    @staticmethod
    def band_from_parametric_params(params, band_key):
        return [
            Light.params2kwargs_multi_gauss_light(params, f'lens_{band_key}'),
            Light.params2kwargs_multi_gauss_light(params, f'source1_{band_key}'),
            Light.params2kwargs_multi_gauss_light(params, f'source2_{band_key}'),
        ]

    @staticmethod
    def band_from_pixelated_params(params, band_key, fixed_lens_lights):
        return [
            fixed_lens_lights[band_key],
            [PowerSpectrum.params2kwargs(params, f'source1pix_{band_key}')],
            [PowerSpectrum.params2kwargs(params, f'source2pix_{band_key}')],
        ]


class PowerSpectrum:
    K_grid = staticmethod(K_grid)

    class TruncatedWedge(dist.Distribution):
        def __init__(self, a, low, b):
            self.a = a
            self.b = b
            self.low = low
            batch_shape = jax.lax.broadcast_shapes(
                jnp.shape(a),
                jnp.shape(low),
                jnp.shape(b),
            )
            self._support = dist.constraints.interval(low, b)
            self.norm = (self.b - self.a) ** 2 - (self.low - self.a) ** 2
            super().__init__(batch_shape=batch_shape, event_shape=())

        @dist.constraints.dependent_property(is_discrete=False, event_dim=0)
        def support(self):
            return self._support

        def log_prob(self, value):
            return jnp.log(2) + jnp.log(value - self.a) - jnp.log(self.norm)

        def sample(self, key, sample_shape=()):
            shape = sample_shape + self.batch_shape
            u = jax.random.uniform(key, shape=shape, minval=0, maxval=1)
            return self.a + jnp.sqrt(self.norm * u + (self.low - self.a) ** 2)

    @staticmethod
    def matern_power_spectrum(
        plate_name,
        param_name,
        k,
        k_zero=None,
        n_high=100,
        n_value=None,
        sigma_low=1e-5,
        sigma_high=10,
        positive=True,
    ):
        with numpyro.plate(f'{plate_name} power spectrum params - [1]', 1):
            if n_value is None:
                n = numpyro.sample(
                    f'n_{param_name}',
                    PowerSpectrum.TruncatedWedge(-1, 0.0001, n_high),
                )
            else:
                n = numpyro.deterministic(f'n_{param_name}', jnp.atleast_1d(n_value))
            sigma = numpyro.sample(f'sigma_{param_name}', dist.LogUniform(sigma_low, sigma_high))
            rho = numpyro.sample(f'rho_{param_name}', dist.LogNormal(2.1, 1.1))

        P = P_Matern(k, n[0], sigma[0], rho[0], k_zero=k_zero)
        scale = jnp.sqrt(P)

        ny, nx = scale.shape
        with numpyro.plate(f'{plate_name} fft y - [{ny}]', ny):
            with numpyro.plate(f'{plate_name} fft x - [{nx}]', nx):
                pixels_wn = numpyro.sample(
                    f'pixels_wn_{param_name}',
                    dist.Normal(0, 1),
                )

        gp = jnp.fft.irfft2(
            pack_fft_values(pixels_wn * scale),
            s=scale.shape,
            norm='ortho',
        )
        if positive:
            gp = jax.nn.softplus(100 * gp) / 100.0
        pixels = numpyro.deterministic(f'pixels_{param_name}', gp)
        return {'pixels': pixels}

    @staticmethod
    def params2kwargs(params, param_name):
        return {'pixels': params[f'pixels_{param_name}']}


class Numpyro_function:
    @staticmethod
    def split_normal_logpdf(x, mean, sigma_minus, sigma_plus):
        x = jnp.asarray(x, dtype=jnp.float64)
        mean = jnp.asarray(mean, dtype=jnp.float64)
        sigma_minus = jnp.asarray(sigma_minus, dtype=jnp.float64)
        sigma_plus = jnp.asarray(sigma_plus, dtype=jnp.float64)

        sigma = jnp.where(x < mean, sigma_minus, sigma_plus)
        log_norm = jnp.log(jnp.sqrt(2.0 / jnp.pi)) - jnp.log(sigma_minus + sigma_plus)
        return log_norm - 0.5 * ((x - mean) / sigma) ** 2


class Cosmo:
    c_km_s = 299792.458

    @staticmethod
    def sample_cosmology_from_prior(stage_kwargs, cosmo_priors):
        prior_name = stage_kwargs['cosmo_prior_name']
        prior = cosmo_priors[prior_name]
        cosmo_vec = numpyro.sample(
            'cosmo_vec',
            dist.MultivariateNormal(
                loc=jnp.asarray(prior['mean_vec'], dtype=jnp.float64),
                covariance_matrix=jnp.asarray(prior['cov'], dtype=jnp.float64),
            ),
        )
        omega_m = numpyro.deterministic('omega_m_cosmo', cosmo_vec[0])
        h0 = numpyro.deterministic('H0_cosmo', cosmo_vec[1])
        return {
            'Omegam': omega_m,
            'Omegak': jnp.asarray(0.0, dtype=jnp.float64),
            'w0': jnp.asarray(-1.0, dtype=jnp.float64),
            'wa': jnp.asarray(0.0, dtype=jnp.float64),
            'h0': h0,
        }

    @staticmethod
    def func(z, Omegam, Omegak, w0, wa=0):
        Omegal = 1.0 - Omegam - Omegak
        zp1 = 1.0 + z
        de = zp1 ** (3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * z / zp1)
        Ez2 = Omegam * zp1 ** 3 + Omegak * zp1 ** 2 + Omegal * de
        return Ez2 ** -0.5

    @staticmethod
    def nth_order_quad(n=20):
        xval, weights = map(jnp.array, roots_legendre(n))
        xval = xval.reshape(-1, 1)
        weights = weights.reshape(-1, 1)

        def integrate(func, a, b, *args):
            return 0.5 * (b - a) * jnp.sum(
                weights * func(0.5 * ((b - a) * xval + (b + a)), *args),
                axis=0,
            )

        return integrate

    @staticmethod
    def integrate(func, a, b, *args, n=20):
        quad = Cosmo.nth_order_quad(n)
        return quad(func, a, b, *args)

    @staticmethod
    def Dplus(Omegak, Es, El, zs, zl):
        sqrt_ok = jnp.sqrt(jnp.abs(Omegak))
        Ds = jnp.sinh(sqrt_ok * Es) / sqrt_ok / (1 + zs)
        Dls = jnp.sinh(sqrt_ok * (Es - El)) / sqrt_ok / (1 + zs)
        Dl = jnp.sinh(sqrt_ok * El) / sqrt_ok / (1 + zl)
        return Dl, Ds, Dls

    @staticmethod
    def Dminus(Omegak, Es, El, zs, zl):
        sqrt_ok = jnp.sqrt(jnp.abs(Omegak))
        Ds = jnp.sin(sqrt_ok * Es) / sqrt_ok / (1 + zs)
        Dls = jnp.sin(sqrt_ok * (Es - El)) / sqrt_ok / (1 + zs)
        Dl = jnp.sin(sqrt_ok * El) / sqrt_ok / (1 + zl)
        return Dl, Ds, Dls

    @staticmethod
    def Dflat(Es, El, zs, zl):
        Ds = Es / (1 + zs)
        Dls = (Es - El) / (1 + zs)
        Dl = El / (1 + zl)
        return Dl, Ds, Dls

    @staticmethod
    def Dpos(Omegak, E, z):
        sqrt_ok = jnp.sqrt(jnp.abs(Omegak))
        return jnp.sinh(sqrt_ok * E) / sqrt_ok / (1 + z)

    @staticmethod
    def Dneg(Omegak, E, z):
        sqrt_ok = jnp.sqrt(jnp.abs(Omegak))
        return jnp.sin(sqrt_ok * E) / sqrt_ok / (1 + z)

    @staticmethod
    def Dzero(E, z):
        return E / (1 + z)

    @staticmethod
    def angular_diameter_distance(z, cosmology, n=20):
        Omegam = cosmology["Omegam"]
        Omegak = cosmology["Omegak"]
        w0 = cosmology["w0"]
        wa = cosmology["wa"]
        h = cosmology["h0"]
        E = Cosmo.integrate(Cosmo.func, 0, z, Omegam, Omegak, w0, wa, n=n)

        Dl = lax.cond(
            Omegak > 0,
            lambda _: Cosmo.Dpos(Omegak, E, z),
            lambda _: lax.cond(
                Omegak < 0,
                lambda _: Cosmo.Dneg(Omegak, E, z),
                lambda _: Cosmo.Dzero(E, z),
                None,
            ),
            None,
        )
        return Dl * Cosmo.c_km_s / h

    @staticmethod
    def dldsdls(zl, zs, cosmology, n=20):
        Omegam = cosmology["Omegam"]
        Omegak = cosmology["Omegak"]
        w0 = cosmology["w0"]
        wa = cosmology["wa"]
        h = cosmology["h0"]

        El = Cosmo.integrate(Cosmo.func, 0, zl, Omegam, Omegak, w0, wa, n=n)
        Es = Cosmo.integrate(Cosmo.func, 0, zs, Omegam, Omegak, w0, wa, n=n)

        Dl, Ds, Dls = lax.cond(
            Omegak > 0,
            lambda _: Cosmo.Dplus(Omegak, Es, El, zs, zl),
            lambda _: lax.cond(
                Omegak < 0,
                lambda _: Cosmo.Dminus(Omegak, Es, El, zs, zl),
                lambda _: Cosmo.Dflat(Es, El, zs, zl),
                None,
            ),
            None,
        )
        return Dl * Cosmo.c_km_s / h, Ds * Cosmo.c_km_s / h, Dls * Cosmo.c_km_s / h

    @staticmethod
    def compute_time_delay_distances(cosmology, z_lens, z_source):
        dl, ds, dls = Cosmo.dldsdls(z_lens, z_source, cosmology)
        return (1.0 + z_lens) * dl * ds / dls


def import_function(namespace=None):
    if namespace is None:
        namespace = inspect.currentframe().f_back.f_globals

    namespace.update({
        'warnings': warnings,
        'datetime': datetime,
        'Path': Path,
        'deepcopy': deepcopy,
        'np': np,
        'xr': xr,
        'jax': jax,
        'jnp': jnp,
        'numpyro': numpyro,
        'dist': dist,
        'infer': infer,
        'autoguide': autoguide,
        'optax': optax,
        'condition': condition,
        'az': az,
        'plt': plt,
        'colors': colors,
        'GridSpec': GridSpec,
        'corner': corner,
        'fits': fits,
        'least_squares': least_squares,
        'Noise': Noise,
        'PSF': PSF,
        'LightModel': LightModel,
        'MassModel': MassModel,
        'LensImageExtension': LensImageExtension,
        'PointSourceModel': PointSourceModel,
        'mass_model_base': mass_model_base,
        'CuspyNFW_3D_fn': CuspyNFW_3D_fn,
        'MGE': MGE,
        'Plot': Plot,
        'ResumeInit': ResumeInit,
        'Geometry': Geometry,
        'PowerSpectrum': PowerSpectrum,
        'Mass': Mass,
        'Numpyro_function': Numpyro_function,
        'Cosmo': Cosmo,
    })
    return namespace


def import_function_DSPL(namespace=None):
    if namespace is None:
        namespace = inspect.currentframe().f_back.f_globals

    import json
    from dataclasses import dataclass

    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    from IPython.display import display
    from matplotlib.colors import LogNorm

    from herculens.LensImage.lens_image_multiplane import MPLensImage
    from herculens.LightModel.light_model_multiplane import MPLightModel
    from herculens.MassModel.mass_model_multiplane import MPMassModel
    from svi_utils import split_scheduler

    import_function(namespace)
    jax.config.update('jax_enable_x64', True)
    numpyro.enable_x64()

    namespace.update({
        'json': json,
        'dataclass': dataclass,
        'Cutout2D': Cutout2D,
        'WCS': WCS,
        'display': display,
        'LogNorm': LogNorm,
        'MPLensImage': MPLensImage,
        'MPLightModel': MPLightModel,
        'MPMassModel': MPMassModel,
        'split_scheduler': split_scheduler,
        'Light': Light,
    })
    return namespace
