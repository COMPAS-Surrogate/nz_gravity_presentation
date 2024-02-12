from skopt import Optimizer
import numpy as np
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process, _evenly_sample
import plotly.graph_objects as go

from plotly.offline import plot

np.random.seed(4)
noise_level = 0.1


def f(x, noise_level=noise_level):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) \
           + np.random.randn() * noise_level


acq_func_kwargs = {"xi": 10000, "kappa": 10000, "noise_level": noise_level ** 2}
opt = Optimizer([(-2.0, 2.0)], "GP", n_initial_points=3,
                acq_optimizer="sampling",
                acq_func_kwargs=acq_func_kwargs,
                acq_optimizer_kwargs={"noise_level": noise_level ** 2, }
                )

opt.run(f, n_iter=25)
res = opt.get_result()


def f_wo_noise(x):
    return f(x, noise_level=0)


def plot_gaussian_process(res, **kwargs):
    ax = kwargs.get("ax", None)
    n_calls = kwargs.get("n_calls", -1)
    objective = kwargs.get("objective", None)
    noise_level = kwargs.get("noise_level", 0)
    n_points = kwargs.get("n_points", 1000)

    if ax is None:
        ax = plt.gca()
    n_dims = res.space.n_dims
    assert n_dims == 1, "Space dimension must be 1"
    dimension = res.space.dimensions[0]
    x, x_model = _evenly_sample(dimension, n_points)
    x = x.reshape(-1, 1)
    x_model = x_model.reshape(-1, 1)

    n_random = len(res.x_iters) - len(res.models)

    fx = np.array([objective(x_i) for x_i in x])

    model = res.models[n_calls]

    curr_x_iters = res.x_iters[:n_random + n_calls]
    curr_func_vals = res.func_vals[:n_random + n_calls]

    # Plot true function + observations
    ax.plot(x, fx, "k-", label="True (unknown)")
    ax.plot(curr_x_iters, curr_func_vals,
            "k.", markersize=15, label="Observations")

    # Plot GP(x) + contours
    y_pred, sigma = model.predict(x_model, return_std=True)
    ax.plot(x, y_pred, color="tab:orange")
    ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 1.9600 * sigma,
                            (y_pred + 1.9600 * sigma)[::-1]]),
            alpha=.2, fc="tab:orange", ec="None", label=r"Model")

    ax.set_xlim(-2, 2)
    ax.set_ylim(1.2, -1.2)
    # hide axes and box
    ax.axis("off")
    ax.legend(loc="upper left", prop={'size': 12}, numpoints=1, frameon=False, )

    return ax


def matplotlib_plots():
    for n_iter in range(13):
        # Plot true function.
        fig, ax = plt.subplots()
        ax = plot_gaussian_process(res, n_calls=n_iter,
                                   objective=f_wo_noise,
                                   noise_level=noise_level,
                                   show_legend=True, show_title=False,
                                   show_next_point=False, show_acq_func=False)
        ax.set_ylabel("")
        ax.set_xlabel("")

    plt.show()


def plotly_gaussian_process(res, n_calls, objective, noise_level, n_points=1000):
    n_dims = res.space.n_dims
    assert n_dims == 1, "Space dimension must be 1"


    dimension = res.space.dimensions[0]
    x, x_model = _evenly_sample(dimension, n_points)
    x = x.reshape(-1, 1)
    x_model = x_model.reshape(-1, 1)

    n_random = len(res.x_iters) - len(res.models)

    fx = np.array([objective(x_i) for x_i in x])

    model = res.models[n_calls]

    curr_x_iters = res.x_iters[:n_random + n_calls]
    curr_func_vals = res.func_vals[:n_random + n_calls]

    # Plot true function + observations
    true_function = dict(x=x.flatten(), y=fx, mode='lines', name="True (unknown)", line=dict(color='black'))
    observations = dict(
        x=np.array(curr_x_iters).flatten(),
        y=curr_func_vals, mode='markers', name="Observations", marker=dict(size=10, color='black'),
    )

    # Plot GP(x) + contours
    y_pred, sigma = model.predict(x_model, return_std=True)
    gp_curve = dict(x=x.flatten(), y=y_pred, mode='lines', name="Model", line=dict(color='rgba(255,165,0,1)'))
    fill_lower = y_pred - 1.96 * sigma
    fill_upper = y_pred + 1.96 * sigma
    fill_trace =dict(
        x=np.concatenate([x.flatten(), x.flatten()[::-1]]),
        y=np.concatenate([fill_lower, fill_upper[::-1]]),
        fill='toself',
        fillcolor='rgba(255,165,0,0.2)',
        line=dict(color='rgba(255,165,0,0)'),
        showlegend=False
    )

    layout = go.Layout(
        xaxis=dict(range=[-2, 2]),
        yaxis=dict(range=[-1.2, 1.2]),
        legend=dict(x=0, y=1, font=dict(size=20), bgcolor='rgba(255,255,255,0)', itemwidth=30),
        showlegend=True,
        margin = go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=0,  # top margin
        )
    )

    data = dict(true=true_function, observations=observations, gp_curve=gp_curve, fill_trace=fill_trace)
    return data, layout



def plotly_plots():
    # Initialize empty list to store frames
    frames = []

    data, layout = plotly_gaussian_process(res, n_calls=0, objective=f_wo_noise, noise_level=noise_level)

    fig = go.Figure(go.Scatter(**data['true']),  layout=layout)
    fig.add_scatter(**data['observations'])
    fig.add_scatter(**data['gp_curve'])
    fig.add_scatter(**data['fill_trace'])


    n_iters = 13
    for n_iter in range(n_iters):
        data, layout = plotly_gaussian_process(res, n_calls=n_iter, objective=f_wo_noise, noise_level=noise_level)
        frames.append(go.Frame(data=[
            go.Scatter(**data['true']),
            go.Scatter(**data['observations']),
            go.Scatter(**data['gp_curve']),
            go.Scatter(**data['fill_trace']),
        ],
       traces=[0, 1, 2, 3], # traces 0-3 are updated
       name=f"iter{n_iter}"))

    fig.update(frames=frames)

    updatemenus = [dict(
        buttons=[
            dict(
                args=[None, {"frame": {"duration": 500, "redraw": True},
                               "mode": "immediate",
                               "transition": {"duration": 500}}],
                label="Play",
                method="animate"
            ),
            dict(
                args=[[None], {"frame": {"duration": 0, "redraw": True},
                               "mode": "immedsiate",
                               "transition": {"duration": 0}}],
                label="Pause",
                method="animate"
            )
        ],
        direction="left",
        pad={"r": 10, "t": 87},
        showactive=False,
        type="buttons",
        x=0.1,
        xanchor="right",
        y=0,
        yanchor="top"
    )]

    sliders = [
        dict(
            steps=[dict(
                method='animate',
                args=[
                    [f'iter{k}'],
                    dict(mode='immediate', frame=dict(duration=400, redraw=True), transition=dict(duration=0))
                ],
                label=f'{k + 1}'
            ) for k in range(n_iters)
            ],
            active=0,
            transition=dict(duration=0),
            x=0, y=0, len=1.0,
            currentvalue=dict(font=dict(size=12), prefix='iter: ', visible=False, xanchor='center'),
        )
    ]
    fig.update_layout(updatemenus=updatemenus, sliders=sliders, template='simple_white',
                      xaxis_visible=False, yaxis_visible=False
                      )
    plot(fig, filename='../gp_animation.html', auto_play=False)





plotly_plots()