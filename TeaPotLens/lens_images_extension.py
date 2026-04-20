import jax
import jax.numpy as jnp
import numpy as np
import scipy

from herculens.LensImage.Numerics.numerics import Numerics
from herculens.LensImage.lens_image import LensImage
from functools import partial


class LensImageExtension(LensImage):
    def __init__(
        self,
        grid_class,
        psf_class,
        noise_class=None,
        lens_mass_model_class=None,
        source_model_class=None,
        lens_light_model_class=None,
        point_source_model_class=None,
        source_arc_mask=None,
        source_grid_scale=1.0,
        conjugate_points=None,
        kwargs_numerics=None,
        kwargs_lens_equation_solver=None
    ):
        super().__init__(
            grid_class,
            psf_class,
            noise_class=noise_class,
            lens_mass_model_class=lens_mass_model_class,
            source_model_class=source_model_class,
            lens_light_model_class=lens_light_model_class,
            point_source_model_class=point_source_model_class,
            source_arc_mask=source_arc_mask,
            kwargs_numerics=kwargs_numerics,
            kwargs_lens_equation_solver=kwargs_lens_equation_solver
        )
        if kwargs_numerics is None:
            kwargs_numerics = {}
        self.ImageNumerics = Numerics(
            pixel_grid=self.Grid,
            psf=self.PSF,
            **kwargs_numerics
        )
        self._source_grid_scale = source_grid_scale
        self.conjugate_points = conjugate_points
        self._source_arc_mask_flat = None

        ssf = self.ImageNumerics.grid_supersampling_factor
        s_ones = np.ones([ssf, ssf])
        # promote the mask to the super sampled grid
        self.source_arc_mask_ss = np.kron(self.source_arc_mask, s_ones)
        # flatten the super sampled mask
        self._source_arc_mask_flat = self.source_arc_mask_ss.flatten()
        # calculate the (flattened) outline of the mask to use for the adaptive grid
        self._source_arc_mask_outline_flat = (
            self.source_arc_mask_ss - scipy.ndimage.binary_erosion(self.source_arc_mask_ss)
        ).flatten().astype(bool)

    def source_surface_brightness(
        self,
        kwargs_source,
        kwargs_lens=None,
        de_lensed=False,
        k=None,
        k_lens=None
    ):
        if len(self.SourceModel.profile_type_list) == 0:
            return jnp.zeros(self.Grid.num_pixel_axes)

        x_grid_img, y_grid_img = self.ImageNumerics.coordinates_evaluate
        if (self._src_adaptive_grid) or (not de_lensed):
            # ray shoot once and use those results to get the adaptive
            # source grid
            x_grid_src, y_grid_src = self.MassModel.ray_shooting(
                x_grid_img,
                y_grid_img,
                kwargs_lens,
                k=k_lens
            )
            pixels_x_coord, pixels_y_coord, _ = self.adapt_source_coordinates(
                x_grid_src,
                y_grid_src
            )
        else:
            pixels_x_coord, pixels_y_coord = None, None
        if de_lensed:
            source_light = self.SourceModel.surface_brightness(
                x_grid_img,
                y_grid_img,
                kwargs_source,
                k=k,
                pixels_x_coord=pixels_x_coord,
                pixels_y_coord=pixels_y_coord
            )
        else:
            source_light = self.SourceModel.surface_brightness(
                x_grid_src,
                y_grid_src,
                kwargs_source,
                k=k,
                pixels_x_coord=pixels_x_coord,
                pixels_y_coord=pixels_y_coord
            )
        return source_light

    def lens_surface_brightness(self, kwargs_lens_light, k=None):
        x_grid_img, y_grid_img = self.ImageNumerics.coordinates_evaluate
        lens_light = self.LensLightModel.surface_brightness(
            x_grid_img,
            y_grid_img,
            kwargs_lens_light,
            k=k
        )
        return lens_light

    def point_source_image(
        self,
        kwargs_point_source,
        kwargs_lens,
        kwargs_solver,
        k=None,
        psf_kernel=None
    ):
        """Compute PSF-convolved point sources, optionally using an explicit PSF kernel."""
        result = jnp.zeros((self.Grid.num_pixel_axes))
        if self.PointSourceModel is None:
            return result
        theta_x, theta_y, amplitude = self.PointSourceModel.get_multiple_images(
            kwargs_point_source,
            kwargs_lens=kwargs_lens,
            kwargs_solver=kwargs_solver,
            k=k,
            with_amplitude=True,
            zero_amp_duplicates=True
        )
        for i in range(len(theta_x)):
            result += self.ImageNumerics.render_point_sources(
                theta_x[i],
                theta_y[i],
                amplitude[i],
                psf_kernel=psf_kernel
            )
        return result

    @partial(jax.jit, static_argnums=(0, 5, 6, 7, 8, 9, 10, 11, 12, 13))
    def model(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_point_source=None,
        unconvolved=False,
        supersampled=False,
        source_add=True,
        lens_light_add=True,
        point_source_add=True,
        k_lens=None,
        k_source=None,
        k_lens_light=None,
        k_point_source=None,
        psf_kernel=None,
        psf_kernel_super=None
    ):
        """
        Create the 2D model image from parameter values.
        Note: due to JIT compilation, the first call to this method will be slower.

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light
            profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness
            profiles
        :param kwargs_point_source: keyword arguments corresponding to the point source model
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param supersampled: if True, returns the model on the higher resolution grid (WARNING: no convolution nor
            normalization is performed in this case!)
        :param source_add: if True, compute source, otherwise without
        :param lens_light_add: if True, compute lens light, otherwise without
        :param point_source_add: if True, add point-source images, otherwise without
        :param k_lens: list of bool or list of int to select which lens mass profiles to include
        :param k_source: list of bool or list of int to select which source profiles to include
        :param k_lens_light: list of bool or list of int to select which lens light profiles to include
        :param k_point_source: list of bool or list of int to select which point-source profiles to include
        :param psf_kernel: optional explicit PSF kernel used for both full-image convolution and point-source rendering
        :param psf_kernel_super: placeholder for future supersampled PSF support
        :return: 2d array of surface brightness pixels of the simulation
        """
        model = jnp.zeros((self.ImageNumerics.grid_class.num_grid_points,)).flatten()
        if source_add is True:
            source_model = self.source_surface_brightness(
                kwargs_source,
                kwargs_lens,
                k=k_source,
                k_lens=k_lens
            )
            if self._source_arc_mask_flat is not None:
                source_model *= self._source_arc_mask_flat
            model += source_model
        if lens_light_add is True:
            model += self.lens_surface_brightness(
                kwargs_lens_light,
                k=k_lens_light
            )
        if not supersampled:
            if psf_kernel_super is not None:
                raise NotImplementedError(
                    "Explicit supersampled PSF override is not enabled in this Numerics path."
                )
            model = self.ImageNumerics.re_size_convolve(
                model,
                unconvolved=unconvolved,
                psf_kernel=psf_kernel
            )
        if point_source_add:
            if supersampled:
                raise ValueError(
                    "point_source_add=True is not supported with supersampled=True in LensImageExtension"
                )
            model += self.point_source_image(
                kwargs_point_source,
                kwargs_lens,
                kwargs_solver=self.kwargs_lens_equation_solver,
                k=k_point_source,
                psf_kernel=psf_kernel
            )
        return model

    def trace_conjugate_points(self, kwargs_lens, k_lens=None):
        if self.conjugate_points is not None:
            x, y = self.conjugate_points.T
            conj_x, conj_y = self.MassModel.ray_shooting(x, y, kwargs_lens, k=k_lens)
            return jnp.vstack([conj_x, conj_y]).T
        else:
            return None

    def mask_extent(self, x_grid_src, y_grid_src, npix_src, grid_scale=1):
        # create grid encompassed by ray-shot coordinates
        x_left, x_right = x_grid_src.min(), x_grid_src.max()
        y_bottom, y_top = y_grid_src.min(), y_grid_src.max()
        # center of the region
        cx = 0.5 * (x_left + x_right)
        cy = 0.5 * (y_bottom + y_top)
        # get the width and height
        width = jnp.abs(x_left - x_right)
        height = jnp.abs(y_bottom - y_top)
        # choose the largest of the two to end up with a square region
        half_size = 0.5 * grid_scale * jnp.maximum(height, width)
        # recompute the new boundaries
        x_left = cx - half_size
        x_right = cx + half_size
        y_bottom = cy - half_size
        y_top = cy + half_size
        x_adapt = jnp.linspace(x_left, x_right, npix_src)
        y_adapt = jnp.linspace(y_bottom, y_top, npix_src)
        extent_adapt = [x_adapt[0], x_adapt[-1], y_adapt[0], y_adapt[-1]]
        return x_adapt, y_adapt, extent_adapt

    @partial(jax.jit, static_argnums=(0, 3, 4, 5))
    def adapt_source_coordinates(
        self,
        x_grid_src,
        y_grid_src,
        force=False,
        npix_src=100,
        source_grid_scale=1
    ):
        """Compute new source coordinates based on the outline of the ray-traced arc-mask"""
        if self._src_adaptive_grid or force:
            if not force:
                npix_src, npix_src_y = self.SourceModel.pixel_grid.num_pixel_axes
                if npix_src_y != npix_src:
                    raise ValueError("Adaptive source plane grid only works with square grids")
                grid_scale = self._source_grid_scale
            else:
                grid_scale = source_grid_scale
            if self.Grid.x_is_inverted or self.Grid.y_is_inverted:
                # TODO: fix this
                raise NotImplementedError("invert x and y not yet supported for adaptive source grid")
            # only pass in the outline of the mask to save on computation time
            return self.mask_extent(
                x_grid_src[self._source_arc_mask_outline_flat],
                y_grid_src[self._source_arc_mask_outline_flat],
                npix_src,
                grid_scale
            )
        else:
            return None, None, None

    def get_source_coordinates(
        self,
        kwargs_lens,
        force=False,
        npix_src=100,
        source_grid_scale=1.0,
        k_lens=None
    ):
        if (not self._src_adaptive_grid) and (self.SourceModel.pixel_grid is not None):
            x_grid, y_grid = self.SourceModel.pixel_grid.pixel_coordinates
            extent = self.SourceModel.pixel_grid.extent
        else:
            x_grid_img, y_grid_img = self.ImageNumerics.coordinates_evaluate
            x_grid_src, y_grid_src = self.MassModel.ray_shooting(
                x_grid_img,
                y_grid_img,
                kwargs_lens,
                k=k_lens
            )
            x_grid, y_grid, extent = self.adapt_source_coordinates(
                x_grid_src,
                y_grid_src,
                force=force,
                npix_src=npix_src,
                source_grid_scale=source_grid_scale
            )
        return x_grid, y_grid, extent


def pixelize_plane(
    lens_image,
    herc_dict,
    num_pix,
    source_grid_scale=None
):
    if source_grid_scale is None:
        source_grid_scale = lens_image._source_grid_scale
    x, y, extent = lens_image.get_source_coordinates(
        herc_dict['kwargs_lens'],
        force=True,
        npix_src=num_pix,
        source_grid_scale=source_grid_scale
    )
    xgrid, ygrid = jnp.meshgrid(x, y)
    image_grid = lens_image.SourceModel.surface_brightness(
        xgrid,
        ygrid,
        herc_dict['kwargs_source'],
        pixels_x_coord=xgrid[0],
        pixels_y_coord=ygrid[:, 0],
    ) * lens_image.Grid.pixel_area
    return image_grid, extent
