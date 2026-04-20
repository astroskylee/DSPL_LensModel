#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = Path(__file__).resolve().parent
WFI2033_DIR = ROOT / "Herculens" / "Herculensedquasar" / "WFI2033"
SLICELENS_DIR = ROOT / "Herculens" / "Slicelens"
HERCULENS_SRC = ROOT / "Herculens" / "herculens"

for path in (LOCAL_DIR, WFI2033_DIR, SLICELENS_DIR, HERCULENS_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

if hasattr(np, "dtypes") and not hasattr(np.dtypes, "StringDType"):
    np.dtypes.StringDType = lambda: np.dtypes.StrDType(0)

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
import numpyro.infer.autoguide as autoguide
import optax
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from herculens.Instrument.noise import Noise
from herculens.Instrument.psf import PSF
from herculens.LightModel.light_model import LightModel
from herculens.LightModel.light_model_multiplane import MPLightModel
from herculens.LensImage.lens_image_multiplane import MPLensImage
from herculens.MassModel.mass_model import MassModel
from herculens.MassModel.mass_model_multiplane import MPMassModel
from svi_utils import split_scheduler
from Tian_infra import Geometry, Mass, PowerSpectrum, ResumeInit

numpyro.enable_x64()


@dataclass
class BandData:
    key: str
    path: Path
    filter_name: str
    data: jnp.ndarray
    obs_mask: jnp.ndarray
    num_obs: int
    source_mask: np.ndarray
    rms: float
    pixel_scale: float
    exposure_time: float
    science_cutout_path: Path
    mask_cutout_path: Path
    psf_cutout_path: Path
    lens_image_parametric: MPLensImage
    lens_image_pixelated: MPLensImage


def estimate_background(data: np.ndarray, border: int = 6) -> tuple[float, float]:
    edge_pixels = np.concatenate(
        [
            data[:border, :].ravel(),
            data[-border:, :].ravel(),
            data[:, :border].ravel(),
            data[:, -border:].ravel(),
        ]
    )
    edge_pixels = edge_pixels[np.isfinite(edge_pixels)]
    background = float(np.median(edge_pixels))
    mad = float(np.median(np.abs(edge_pixels - background)))
    rms = 1.4826 * mad
    if not np.isfinite(rms) or rms <= 0:
        rms = float(np.std(edge_pixels))
    return background, max(rms, 1e-6)


def pixel_scale_arcsec(header) -> float:
    if "D001SCAL" in header:
        return float(header["D001SCAL"])
    cd11 = header.get("CD1_1")
    cd21 = header.get("CD2_1")
    if cd11 is not None and cd21 is not None:
        return float(np.hypot(cd11, cd21) * 3600.0)
    raise ValueError("Cannot determine pixel scale from FITS header.")


def compute_eta_fiducial(z_lens: float, z_source_1: float, z_source_2: float) -> float:
    d_s1 = cosmo.angular_diameter_distance(z_source_1)
    d_s2 = cosmo.angular_diameter_distance(z_source_2)
    d_ls1 = cosmo.angular_diameter_distance_z1z2(z_lens, z_source_1)
    d_ls2 = cosmo.angular_diameter_distance_z1z2(z_lens, z_source_2)
    return float((d_s1 * d_ls2 / (d_ls1 * d_s2)).value)


def multi_gauss_light(
    plate_name: str,
    param_name: str,
    n_gauss: int,
    sigma_lims: tuple[float, float],
    center_low: float | None = None,
    center_high: float | None = None,
    e_low: float = -0.2,
    e_high: float = 0.2,
):
    sigma_bins = jnp.logspace(jnp.log10(sigma_lims[0]), jnp.log10(sigma_lims[1]), n_gauss + 1)
    with numpyro.plate(f"{plate_name} - [{n_gauss}]", n_gauss):
        amp_scale = numpyro.sample(f"A_{param_name}", dist.LogUniform(1e-5, 1e4))
        sigma = numpyro.sample(
            f"sigma_{param_name}",
            dist.LogUniform(sigma_bins[:-1], sigma_bins[1:]),
        )
        with numpyro.plate(f"{plate_name} vectors - [2]", 2):
            ellipticity = numpyro.sample(
                f"e_{param_name}",
                dist.TruncatedNormal(0.0, 0.1, low=e_low, high=e_high),
            )
            if center_low is not None or center_high is not None:
                center = numpyro.sample(
                    f"center_{param_name}",
                    dist.TruncatedNormal(0.0, 0.08, low=center_low, high=center_high),
                )
            else:
                center = numpyro.sample(f"center_{param_name}", dist.Normal(0.0, 0.2))

    amp = numpyro.deterministic(f"amp_{param_name}", amp_scale * sigma**2)
    return [{
        "amp": amp,
        "sigma": sigma,
        "e1": ellipticity[0],
        "e2": ellipticity[1],
        "center_x": center[0],
        "center_y": center[1],
    }]


def params2kwargs_multi_gauss_light(params: dict, param_name: str) -> list[dict]:
    return [{
        "amp": params[f"amp_{param_name}"],
        "sigma": params[f"sigma_{param_name}"],
        "e1": params[f"e_{param_name}"][0],
        "e2": params[f"e_{param_name}"][1],
        "center_x": params[f"center_{param_name}"][0],
        "center_y": params[f"center_{param_name}"][1],
    }]


def epl_with_shear(
    plate_name: str,
    param_name: str,
    theta_low: float,
    theta_high: float,
    gamma_low: float = 1.6,
    gamma_high: float = 2.4,
    e_low: float = -0.2,
    e_high: float = 0.2,
    center_low: float = -0.3,
    center_high: float = 0.3,
):
    with numpyro.plate(f"{plate_name} scalers - [1]", 1):
        theta_e = numpyro.sample(f"theta_E_{param_name}", dist.Uniform(theta_low, theta_high))
        gamma = numpyro.sample(f"gamma_{param_name}", dist.Uniform(gamma_low, gamma_high))
    with numpyro.plate(f"{plate_name} vectors - [2]", 2):
        ellipticity = numpyro.sample(
            f"e_{param_name}",
            dist.TruncatedNormal(0.0, 0.15, low=e_low, high=e_high),
        )
        shear = numpyro.sample(f"gamma_sheer_{param_name}", dist.Uniform(-0.15, 0.15))
    center = numpyro.sample(
        f"center_{param_name}",
        dist.TruncatedNormal(0.0, 0.08, low=center_low, high=center_high).expand([2]),
    )
    return [{
        "theta_E": theta_e[0],
        "gamma": gamma[0],
        "e1": ellipticity[0],
        "e2": ellipticity[1],
        "center_x": center[0],
        "center_y": center[1],
    }, {
        "gamma1": shear[0],
        "gamma2": shear[1],
        "ra_0": center[0],
        "dec_0": center[1],
    }]


def params2kwargs_epl_with_shear(params: dict, param_name: str) -> list[dict]:
    return [{
        "theta_E": params[f"theta_E_{param_name}"][0],
        "gamma": params[f"gamma_{param_name}"][0],
        "e1": params[f"e_{param_name}"][0],
        "e2": params[f"e_{param_name}"][1],
        "center_x": params[f"center_{param_name}"][0],
        "center_y": params[f"center_{param_name}"][1],
    }, {
        "gamma1": params[f"gamma_sheer_{param_name}"][0],
        "gamma2": params[f"gamma_sheer_{param_name}"][1],
        "ra_0": params[f"center_{param_name}"][0],
        "dec_0": params[f"center_{param_name}"][1],
    }]


def params2kwargs_sis(
    params: dict,
    param_name: str,
    fallback_origin: tuple[float, float] | None = None,
) -> list[dict]:
    if f"center_1_{param_name}" in params and f"center_2_{param_name}" in params:
        center_x = params[f"center_1_{param_name}"][0]
        center_y = params[f"center_2_{param_name}"][0]
    elif fallback_origin is not None:
        center_x = fallback_origin[0]
        center_y = fallback_origin[1]
    else:
        raise KeyError(f"Missing SIS center sites for '{param_name}'.")
    return [{
        "theta_E": params[f"theta_E_{param_name}"][0],
        "center_x": center_x,
        "center_y": center_y,
    }]


def params2kwargs_power_spectrum(params: dict, param_name: str) -> dict:
    return {"pixels": params[f"pixels_{param_name}"]}


def build_light_model_parametric() -> MPLightModel:
    return MPLightModel(
        [
            LightModel(["MULTI_GAUSSIAN_ELLIPSE"], {}),
            LightModel(["MULTI_GAUSSIAN_ELLIPSE"], {}),
            LightModel(["MULTI_GAUSSIAN_ELLIPSE"], {}),
        ]
    )


def build_light_model_pixelated(pixel_grid_shape: int) -> MPLightModel:
    pixel_kwargs = {
        "pixel_adaptive_grid": True,
        "pixel_interpol": "fast_bilinear",
        "kwargs_pixelated": {"num_pixels": pixel_grid_shape},
    }
    return MPLightModel(
        [
            LightModel(["MULTI_GAUSSIAN_ELLIPSE"], {}),
            LightModel(["PIXELATED"], **pixel_kwargs),
            LightModel(["PIXELATED"], **pixel_kwargs),
        ]
    )


def build_mass_model() -> MPMassModel:
    return MPMassModel(
        [
            MassModel(["EPL", "SHEAR"]),
            MassModel(["SIS"]),
        ]
    )


def build_band_data(
    key: str,
    path: Path,
    science_center: SkyCoord,
    psf_center: SkyCoord | tuple[float, float],
    cutout_size: int,
    psf_cutout_size: int,
    pixel_grid_shape: int,
    source_grid_scale: float,
    cutout_output_dir: Path,
) -> BandData:
    if psf_cutout_size % 2 == 0:
        raise ValueError(f"psf_cutout_size must be odd for herculens PSF kernels, got {psf_cutout_size}.")

    with fits.open(path, memmap=True) as hdul:
        header0 = hdul[0].header
        header1 = hdul[1].header
        data = np.asarray(hdul[1].data, dtype=np.float64)
        wcs = WCS(header1)
        cutout = Cutout2D(
            data,
            science_center,
            (cutout_size, cutout_size),
            wcs=wcs,
            mode="partial",
            fill_value=np.nan,
        )
        psf_cutout = Cutout2D(
            data,
            psf_center,
            (psf_cutout_size, psf_cutout_size),
            wcs=wcs,
            mode="partial",
            fill_value=np.nan,
        )
        filter_name = header0.get("FILTER", path.stem)
        exposure_time = float(header0.get("EXPTIME", 1.0))
        pixel_scale = pixel_scale_arcsec(header0)

    background, rms = estimate_background(cutout.data)
    data_sub = np.nan_to_num(cutout.data - background, nan=0.0).astype(np.float64)
    obs_mask = np.ones_like(data_sub, dtype=bool)
    source_mask = np.ones_like(data_sub, dtype=bool)
    pixel_grid = Geometry.get_pixel_grid(data_sub, pixel_scale)[0]

    psf_background, _ = estimate_background(np.asarray(psf_cutout.data, dtype=np.float64))
    psf_kernel = np.nan_to_num(psf_cutout.data - psf_background, nan=0.0).astype(np.float64)
    psf_kernel = np.clip(psf_kernel, a_min=0.0, a_max=None)
    if not np.isfinite(psf_kernel).all() or psf_kernel.sum() <= 0:
        raise ValueError(f"Invalid PSF cutout extracted from {path}.")
    psf_kernel /= psf_kernel.sum()
    psf = PSF(psf_type="PIXEL", kernel_point_source=psf_kernel)
    noise = Noise(pixel_grid.num_pixel_axes[1], pixel_grid.num_pixel_axes[0], exposure_time=exposure_time)
    source_arc_masks = [None, source_mask, source_mask]
    source_scales = [1.0, source_grid_scale, source_grid_scale]

    cutout_output_dir.mkdir(parents=True, exist_ok=True)
    science_cutout_path = cutout_output_dir / f"{key}_science_{cutout_size}x{cutout_size}.fits"
    mask_cutout_path = cutout_output_dir / f"{key}_mask_{cutout_size}x{cutout_size}.fits"
    psf_cutout_path = cutout_output_dir / f"{key}_psf_{psf_cutout_size}x{psf_cutout_size}.fits"

    science_header = cutout.wcs.to_header()
    science_header["FILTER"] = filter_name
    science_header["EXPTIME"] = exposure_time
    science_header["PIXSCALE"] = pixel_scale
    fits.PrimaryHDU(data=data_sub.astype(np.float32), header=science_header).writeto(
        science_cutout_path,
        overwrite=True,
    )
    fits.PrimaryHDU(data=source_mask.astype(np.uint8), header=science_header).writeto(
        mask_cutout_path,
        overwrite=True,
    )
    psf_header = psf_cutout.wcs.to_header()
    psf_header["FILTER"] = filter_name
    psf_header["EXPTIME"] = exposure_time
    psf_header["PIXSCALE"] = pixel_scale
    fits.PrimaryHDU(data=psf_kernel.astype(np.float32), header=psf_header).writeto(
        psf_cutout_path,
        overwrite=True,
    )

    lens_image_parametric = MPLensImage(
        pixel_grid,
        psf,
        noise_class=noise,
        light_model_class=build_light_model_parametric(),
        mass_model_class=build_mass_model(),
        source_arc_masks=source_arc_masks,
        source_grid_scale=source_scales,
        kwargs_numerics={"supersampling_factor": 1},
    )
    lens_image_pixelated = MPLensImage(
        pixel_grid,
        psf,
        noise_class=noise,
        light_model_class=build_light_model_pixelated(pixel_grid_shape),
        mass_model_class=build_mass_model(),
        source_arc_masks=source_arc_masks,
        source_grid_scale=source_scales,
        kwargs_numerics={"supersampling_factor": 1},
    )
    return BandData(
        key=key,
        path=path,
        filter_name=filter_name,
        data=jnp.asarray(data_sub, dtype=jnp.float64),
        obs_mask=jnp.asarray(obs_mask, dtype=bool),
        num_obs=int(obs_mask.size),
        source_mask=source_mask,
        rms=rms,
        pixel_scale=pixel_scale,
        exposure_time=exposure_time,
        science_cutout_path=science_cutout_path,
        mask_cutout_path=mask_cutout_path,
        psf_cutout_path=psf_cutout_path,
        lens_image_parametric=lens_image_parametric,
        lens_image_pixelated=lens_image_pixelated,
    )


def build_parametric_model(
    bands: list[BandData],
    eta_prior: tuple[float, float],
    n_gauss_lens: int,
    n_gauss_source: int,
    theta_e_low: float,
    theta_e_high: float,
    sis_theta_low: float,
    sis_theta_high: float,
):
    sigma_lims_lens = (0.03, 1.5)
    sigma_lims_source = (0.02, 0.5)

    def model():
        eta = numpyro.sample("eta", dist.Uniform(*eta_prior))
        eta_flat = jnp.atleast_1d(eta)
        kwargs_mass_main = epl_with_shear(
            "Main lens mass",
            "main",
            theta_low=theta_e_low,
            theta_high=theta_e_high,
        )
        sis_origin = jnp.array(
            [kwargs_mass_main[0]["center_x"], kwargs_mass_main[0]["center_y"]],
            dtype=jnp.float64,
        )
        kwargs_mass_source_1 = Mass.SIS(
            "Source 1 mass",
            "s1",
            origin=sis_origin,
            theta_low=sis_theta_low,
            theta_high=sis_theta_high,
        )
        kwargs_mass = [kwargs_mass_main, kwargs_mass_source_1]

        for band in bands:
            lens_light = multi_gauss_light(
                f"{band.key} lens light",
                f"lens_{band.key}",
                n_gauss_lens,
                sigma_lims_lens,
                center_low=-0.2,
                center_high=0.2,
            )
            source_1_light = multi_gauss_light(
                f"{band.key} source 1 light",
                f"source1_{band.key}",
                n_gauss_source,
                sigma_lims_source,
                center_low=-0.3,
                center_high=0.3,
            )
            source_2_light = multi_gauss_light(
                f"{band.key} source 2 light",
                f"source2_{band.key}",
                n_gauss_source,
                sigma_lims_source,
                center_low=-0.4,
                center_high=0.4,
            )
            kwargs_light = [lens_light, source_1_light, source_2_light]
            model_image = band.lens_image_parametric.model(
                eta_flat=eta_flat,
                kwargs_mass=kwargs_mass,
                kwargs_light=kwargs_light,
            )

            rms_low = max(0.5 * band.rms, 1e-6)
            rms_high = max(2.0 * band.rms, rms_low * 1.5)
            background_rms = numpyro.sample(f"RMS_{band.key}", dist.LogUniform(rms_low, rms_high))
            model_var = band.lens_image_parametric.Noise.C_D_model(model_image, background_rms=background_rms)
            model_std = jnp.sqrt(jnp.maximum(model_var, 1e-12))
            numpyro.deterministic(f"model_{band.key}", model_image)

            with numpyro.plate(f"obs_{band.key} - [{band.num_obs}]", band.num_obs):
                numpyro.sample(
                    f"obs_{band.key}",
                    dist.Normal(model_image[band.obs_mask], model_std[band.obs_mask]),
                    obs=band.data[band.obs_mask],
                )

    return model


def build_pixelated_model(
    bands: list[BandData],
    eta_prior: tuple[float, float],
    pixel_grid_shape: int,
    fixed_lens_lights: dict[str, list[dict]],
    theta_e_low: float,
    theta_e_high: float,
    sis_theta_low: float,
    sis_theta_high: float,
):
    k_grid = PowerSpectrum.K_grid((pixel_grid_shape, pixel_grid_shape))

    def model():
        eta = numpyro.sample("eta", dist.Uniform(*eta_prior))
        eta_flat = jnp.atleast_1d(eta)
        kwargs_mass_main = epl_with_shear(
            "Main lens mass",
            "main",
            theta_low=theta_e_low,
            theta_high=theta_e_high,
        )
        sis_origin = jnp.array(
            [kwargs_mass_main[0]["center_x"], kwargs_mass_main[0]["center_y"]],
            dtype=jnp.float64,
        )
        kwargs_mass_source_1 = Mass.SIS(
            "Source 1 mass",
            "s1",
            origin=sis_origin,
            theta_low=sis_theta_low,
            theta_high=sis_theta_high,
        )
        kwargs_mass = [kwargs_mass_main, kwargs_mass_source_1]

        for band in bands:
            source_1_light = PowerSpectrum.matern_power_spectrum(
                f"{band.key} source 1 power",
                f"source1pix_{band.key}",
                k_grid.k,
                k_zero=0.0,
            )
            source_2_light = PowerSpectrum.matern_power_spectrum(
                f"{band.key} source 2 power",
                f"source2pix_{band.key}",
                k_grid.k,
                k_zero=0.0,
            )
            kwargs_light = [
                fixed_lens_lights[band.key],
                [source_1_light],
                [source_2_light],
            ]
            model_image = band.lens_image_pixelated.model(
                eta_flat=eta_flat,
                kwargs_mass=kwargs_mass,
                kwargs_light=kwargs_light,
            )

            rms_low = max(0.5 * band.rms, 1e-6)
            rms_high = max(2.0 * band.rms, rms_low * 1.5)
            background_rms = numpyro.sample(f"RMS_{band.key}", dist.LogUniform(rms_low, rms_high))
            model_var = band.lens_image_pixelated.Noise.C_D_model(model_image, background_rms=background_rms)
            model_std = jnp.sqrt(jnp.maximum(model_var, 1e-12))
            numpyro.deterministic(f"model_{band.key}", model_image)

            with numpyro.plate(f"obs_{band.key} - [{band.num_obs}]", band.num_obs):
                numpyro.sample(
                    f"obs_{band.key}",
                    dist.Normal(model_image[band.obs_mask], model_std[band.obs_mask]),
                    obs=band.data[band.obs_mask],
                )

    return model


def shared_mass_from_params(params: dict) -> list[list[dict]]:
    main_mass = params2kwargs_epl_with_shear(params, "main")
    main_origin = (main_mass[0]["center_x"], main_mass[0]["center_y"])
    return [
        main_mass,
        params2kwargs_sis(params, "s1", fallback_origin=main_origin),
    ]


def band_light_from_parametric_params(params: dict, band_key: str) -> list[list[dict]]:
    return [
        params2kwargs_multi_gauss_light(params, f"lens_{band_key}"),
        params2kwargs_multi_gauss_light(params, f"source1_{band_key}"),
        params2kwargs_multi_gauss_light(params, f"source2_{band_key}"),
    ]


def band_light_from_pixelated_params(
    params: dict,
    band_key: str,
    fixed_lens_lights: dict[str, list[dict]],
) -> list[list[dict]]:
    return [
        fixed_lens_lights[band_key],
        [params2kwargs_power_spectrum(params, f"source1pix_{band_key}")],
        [params2kwargs_power_spectrum(params, f"source2pix_{band_key}")],
    ]


def run_svi(model, rng_key, steps: int, init_values: dict | None = None):
    init_fn = infer.init_to_median(num_samples=5)
    if init_values:
        init_fn = ResumeInit.init_to_value_or_defer(values=init_values, defer=init_fn)
    guide = autoguide.AutoLowRankMultivariateNormal(model, init_loc_fn=init_fn)
    scheduler = split_scheduler(
        max_iterations=max(steps, 2),
        init_value=0.01,
        decay_rates=[0.95, 0.98],
        transition_steps=[20, 5],
        boundary=0.8,
    )
    optim = optax.adam(learning_rate=scheduler)
    svi = infer.SVI(model, guide, optim, infer.Trace_ELBO())
    result = svi.run(rng_key, steps, progress_bar=False, stable_update=True)
    return result, guide.median(result.params)


def build_stage2_init_values(parametric_median: dict, pixel_grid_shape: int, band_keys: list[str]) -> dict:
    init_values = {
        key: value
        for key, value in parametric_median.items()
        if key.startswith(("theta_E_main", "gamma_main", "e_main", "gamma_sheer_main", "center_main", "theta_E_s1", "center_1_s1", "center_2_s1", "RMS_", "eta"))
    }
    zero_grid = jnp.zeros((pixel_grid_shape, pixel_grid_shape), dtype=jnp.float64)
    for band_key in band_keys:
        for source_prefix in (f"source1pix_{band_key}", f"source2pix_{band_key}"):
            init_values[f"n_{source_prefix}"] = jnp.asarray([1.0], dtype=jnp.float64)
            init_values[f"sigma_{source_prefix}"] = jnp.asarray([0.05], dtype=jnp.float64)
            init_values[f"rho_{source_prefix}"] = jnp.asarray([1.0], dtype=jnp.float64)
            init_values[f"pixels_wn_{source_prefix}"] = zero_grid
    return init_values


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-channel DSPL SVI smoke test for Teapot data.")
    parser.add_argument("--science-ra", type=str, default="18:14:24.060")
    parser.add_argument("--science-dec", type=str, default="+67:08:15.00")
    parser.add_argument("--psf-ra", type=str, default="18:14:34.729")
    parser.add_argument("--psf-dec", type=str, default="+67:08:59.96")
    parser.add_argument("--psf-x", type=float, default=3463.9)
    parser.add_argument("--psf-y", type=float, default=1583.9)
    parser.add_argument("--cutout-size", type=int, default=100)
    parser.add_argument("--psf-cutout-size", type=int, default=49)
    parser.add_argument("--pixel-grid-shape", type=int, default=16)
    parser.add_argument("--parametric-steps", type=int, default=1)
    parser.add_argument("--pixelated-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--z-lens", type=float, default=0.5)
    parser.add_argument("--z-source-1", type=float, default=1.0)
    parser.add_argument("--z-source-2", type=float, default=2.0)
    parser.add_argument("--eta-low", type=float, default=None)
    parser.add_argument("--eta-high", type=float, default=None)
    parser.add_argument("--theta-e-low", type=float, default=0.2)
    parser.add_argument("--theta-e-high", type=float, default=2.0)
    parser.add_argument("--sis-theta-low", type=float, default=0.02)
    parser.add_argument("--sis-theta-high", type=float, default=0.6)
    parser.add_argument("--n-gauss-lens", type=int, default=3)
    parser.add_argument("--n-gauss-source", type=int, default=2)
    parser.add_argument("--source-grid-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=LOCAL_DIR / "outputs" / "teapot_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = ROOT / "Herculens" / "Data" / "Teapot"
    cutout_output_dir = args.output_dir / "cutouts"
    science_center = SkyCoord(args.science_ra, args.science_dec, unit=("hourangle", "deg"))
    psf_center = (args.psf_x, args.psf_y)
    eta_fiducial = compute_eta_fiducial(args.z_lens, args.z_source_1, args.z_source_2)
    eta_low = args.eta_low if args.eta_low is not None else 0.8 * eta_fiducial
    eta_high = args.eta_high if args.eta_high is not None else 1.2 * eta_fiducial
    eta_prior = (float(eta_low), float(eta_high))

    bands = [
        build_band_data(
            key="f475w",
            path=data_dir / "ifo601010_drc.fits",
            science_center=science_center,
            psf_center=psf_center,
            cutout_size=args.cutout_size,
            psf_cutout_size=args.psf_cutout_size,
            pixel_grid_shape=args.pixel_grid_shape,
            source_grid_scale=args.source_grid_scale,
            cutout_output_dir=cutout_output_dir,
        ),
        build_band_data(
            key="f814w",
            path=data_dir / "ifo602010_drz.fits",
            science_center=science_center,
            psf_center=psf_center,
            cutout_size=args.cutout_size,
            psf_cutout_size=args.psf_cutout_size,
            pixel_grid_shape=args.pixel_grid_shape,
            source_grid_scale=args.source_grid_scale,
            cutout_output_dir=cutout_output_dir,
        ),
    ]

    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, stage1_key, stage2_key = jax.random.split(rng_key, 3)

    model_parametric = build_parametric_model(
        bands=bands,
        eta_prior=eta_prior,
        n_gauss_lens=args.n_gauss_lens,
        n_gauss_source=args.n_gauss_source,
        theta_e_low=args.theta_e_low,
        theta_e_high=args.theta_e_high,
        sis_theta_low=args.sis_theta_low,
        sis_theta_high=args.sis_theta_high,
    )
    result_parametric, median_parametric = run_svi(
        model_parametric,
        rng_key=stage1_key,
        steps=args.parametric_steps,
    )

    fixed_lens_lights = {
        band.key: band_light_from_parametric_params(median_parametric, band.key)[0]
        for band in bands
    }

    model_pixelated = build_pixelated_model(
        bands=bands,
        eta_prior=eta_prior,
        pixel_grid_shape=args.pixel_grid_shape,
        fixed_lens_lights=fixed_lens_lights,
        theta_e_low=args.theta_e_low,
        theta_e_high=args.theta_e_high,
        sis_theta_low=args.sis_theta_low,
        sis_theta_high=args.sis_theta_high,
    )
    stage2_init_values = build_stage2_init_values(
        median_parametric,
        pixel_grid_shape=args.pixel_grid_shape,
        band_keys=[band.key for band in bands],
    )
    result_pixelated, median_pixelated = run_svi(
        model_pixelated,
        rng_key=stage2_key,
        steps=args.pixelated_steps,
        init_values=stage2_init_values,
    )

    summary = {
        "cutout_size": args.cutout_size,
        "psf_cutout_size": args.psf_cutout_size,
        "pixel_grid_shape": args.pixel_grid_shape,
        "science_center": {"ra": args.science_ra, "dec": args.science_dec},
        "psf_center": {"x": args.psf_x, "y": args.psf_y},
        "eta_fiducial": eta_fiducial,
        "eta_prior": {"low": eta_prior[0], "high": eta_prior[1]},
        "eta_parametric": float(np.asarray(median_parametric["eta"])),
        "eta_pixelated": float(np.asarray(median_pixelated["eta"])),
        "parametric_loss_final": float(result_parametric.losses[-1]),
        "pixelated_loss_final": float(result_pixelated.losses[-1]),
        "bands": {
            band.key: {
                "filter": band.filter_name,
                "rms": band.rms,
                "pixel_scale": band.pixel_scale,
                "cutout_shape": list(map(int, band.data.shape)),
                "science_cutout_path": str(band.science_cutout_path),
                "mask_cutout_path": str(band.mask_cutout_path),
                "psf_cutout_path": str(band.psf_cutout_path),
            }
            for band in bands
        },
        "main_mass": {
            key: float(np.asarray(value))
            for key, value in shared_mass_from_params(median_pixelated)[0][0].items()
        },
    }
    write_summary(args.output_dir / "smoke_summary.json", summary)

    print("Parametric loss:", float(result_parametric.losses[-1]))
    print("Parametric eta:", float(np.asarray(median_parametric["eta"])))
    print("Pixelated loss:", float(result_pixelated.losses[-1]))
    print("Pixelated eta:", float(np.asarray(median_pixelated["eta"])))
    print("Summary:", args.output_dir / "smoke_summary.json")


if __name__ == "__main__":
    main()
