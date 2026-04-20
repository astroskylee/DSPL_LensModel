import optax
import numpyro
import jax_tqdm
import jax
import matplotlib.pyplot as plt
import numpy as np

from functools import partial


def split_scheduler(
    max_iterations,
    init_value=0.1,
    decay_rates=[0.99, 0.99],
    transition_steps=[50, 10],
    boundary=0.5
):
    boundary = int(max_iterations * boundary)

    scheduler1 = optax.exponential_decay(
        init_value=init_value,
        decay_rate=decay_rates[0],
        transition_steps=transition_steps[0]
    )

    scheduler2 = optax.exponential_decay(
        init_value=scheduler1(boundary),
        decay_rate=decay_rates[1],
        transition_steps=transition_steps[1]
    )

    return optax.join_schedules(
        [scheduler1, scheduler2],
        boundaries=[boundary]
    )


def plot_loss(losses, max_iterations, ax=None, axins=None, inset=True, percentile_limit=99.95, **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 3.5))
    ax.plot(losses, **kwargs)
    ax.set_yscale('asinh')

    if (inset) and (axins is None):
        axins = ax.inset_axes([0.3, 0.5, 0.64, 0.45])
    N_end = max_iterations // 3
    x_plot = np.linspace(max_iterations - N_end, max_iterations, N_end)
    if inset:
        trimmed = losses[max_iterations - N_end:]
        ylow, yhigh = np.percentile(trimmed.T, np.array([0.0, percentile_limit]))
        delta = 0.02 * (yhigh - ylow)
        axins.plot(x_plot, losses[max_iterations - N_end:], **kwargs)
        axins.set_ylim(ylow - delta, yhigh + delta)
        ax.indicate_inset_zoom(axins, edgecolor='k')
    return ax, axins


class SVI_vec(numpyro.infer.SVI):
    def run(
        self,
        rng_key,
        num_chains,
        num_steps,
        *args,
        stable_update=False,
        forward_mode_differentiation=False,
        init_states=None,
        init_params=None,
        **kwargs
    ):
        @jax_tqdm.scan_tqdm(num_steps)
        def body_fn(svi_state, _):
            if stable_update:
                svi_state, loss = self.stable_update(
                    svi_state,
                    *args,
                    forward_mode_differentiation=forward_mode_differentiation,
                    **kwargs,
                )
            else:
                svi_state, loss = self.update(
                    svi_state,
                    *args,
                    forward_mode_differentiation=forward_mode_differentiation,
                    **kwargs,
                )
            return svi_state, loss

        @jax.vmap
        def map_func(i, init_value):
            init_bar = jax_tqdm.PBar(id=i, carry=init_value)
            final_state, losses = jax.lax.scan(body_fn, init_bar, jax.numpy.arange(num_steps))
            return final_state.carry, losses

        @partial(jax.vmap, in_axes=(0, None, 0, None))
        def vmap_init(rng_key, args, init_params, kwargs):
            return self.init(rng_key, *args, init_params=init_params, **kwargs)

        rng_keys = jax.random.split(rng_key, num_chains)
        if init_states is None:
            svi_states = vmap_init(rng_keys, args, init_params, kwargs)
        else:
            svi_states = init_states

        svi_states, losses = map_func(jax.numpy.arange(num_chains), svi_states)
        return numpyro.infer.svi.SVIRunResult(self.get_params(svi_states), svi_states, losses)
