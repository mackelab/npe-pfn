import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tabpfn_sbi.utils.data_utils import query

_custom_styles = ["pyloric"]
_mpl_styles = list(plt.style.available)


PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_COLORS = {
    "nle": "#76b5c5",
    "snle": "#76b5c5",
    "nre": "#3b9778",
    "snre": "#3b9778",
    "npe": "#1f77b4",
    "snpe": "#1f77b4",
    "tsnpe": "#1f77b2",
    "tabpfn": "C3",
    "ts_tabpfn": "#f2a900",
    "filtered_tabpfn": "#f2a900",
    "npe_ensemble": "#0B3954",
    "npe_sweeper": "#8EA524",
    "filtered_tabpfn_infomax": "C3",
}


def plot_calibration_curves(
    name,
    method=None,
    estimator=None,
    embedding_net=None,
    task=None,
    num_simulations=None,
    seed=None,
    metric="tarp",
    ax=None,
    figsize=(2.0, 2),
    color_map=None,
    hue=None,
    df=None,
    alpha=0.7,
    plot_mean=True,
    plot_individual=True,
    num_individual_curves=100,
    **kwargs,
):
    assert metric in ["sbc", "tarp"]
    if df is None:
        df = query(
            name=name,
            method=method,
            estimator=estimator,
            embedding_net=embedding_net,
            task=task,
            num_simulations=num_simulations,
            seed=seed,
            metric=metric,
            reduce_fn=None,
        )

    # Group by hue if specified
    transparency = alpha
    if metric == "sbc":

        def convert_to_cdf(x):
            x = np.array(eval(x))
            hist, *_ = np.histogram(x, bins=30, density=True)
            histcs = np.cumsum(hist)
            histcs = histcs / histcs[-1]
            histcs = np.concatenate([[0], histcs])
            alpha = np.linspace(0, 1, len(histcs))
            return alpha, histcs

        value = df["value"].apply(convert_to_cdf)
        alphas = value.apply(lambda x: x[0])
        ecp = value.apply(lambda x: x[1])

    elif metric == "tarp":
        alphas = df["value"].apply(lambda x: np.array(eval(x)[0]))
        ecp = df["value"].apply(lambda x: np.array(eval(x)[1]))

    ylims = (0, 1)
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        fig = None

    if hue is not None:
        labels = df[hue]
    else:
        labels = ["none"] * len(alphas)

    if color_map is None:
        if hue is not None:
            unique_labels = df[hue].unique()
            colors = plt.cm.get_cmap("tab10", len(unique_labels))
            color_map = {label: colors(i) for i, label in enumerate(unique_labels)}
        else:
            color_map = {}

    common_alpha = np.linspace(0, 1, 100)

    # Group data by label
    label_data = {}
    for alpha, ecp, l in zip(alphas, ecp, labels):
        if l not in label_data:
            label_data[l] = {"alphas": [], "ecps": []}
        label_data[l]["alphas"].append(alpha)
        label_data[l]["ecps"].append(ecp)

    # Plot individual curves and calculate common ECP per label
    for l, data in label_data.items():
        color = color_map.get(l, "C0")
        label_ecp = np.zeros_like(common_alpha)
        count = 0

        # Plot individual curves
        j = 0
        for alpha, ecp in zip(data["alphas"], data["ecps"]):
            if plot_individual and j < num_individual_curves:
                ax.plot(alpha, ecp, label=l, color=color, alpha=transparency, lw=0.7)
                j += 1
            label_ecp += np.interp(common_alpha, np.array(alpha), np.array(ecp))
            count += 1

        # Calculate and plot mean ECP for this label
        if count > 0:
            label_ecp /= count
            # Smooth the common ECP
            label_ecp_padded = np.pad(label_ecp, pad_width=5, mode="edge")
            label_ecp = np.convolve(label_ecp_padded, np.ones(10) / 10, mode="valid")[
                :-1
            ]
            if plot_mean:
                ax.plot(common_alpha, label_ecp, color=color, label=l, lw=1.4)

    exact = np.linspace(0, 1, 100)
    ax.plot(exact, exact, "--", color="black", alpha=0.25)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("ECP")
    ax.set_ylim(*ylims)
    ax.set_xlim(0, 1)
    ax.set_yticks([0.0, 1.0])
    ax.set_xticks([0.0, 1.0])
    ax.set_yticklabels([0.0, 1.0])
    ax.set_xticklabels([0.0, 1.0])

    if hue is not None:
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys())

    return fig, ax


def plot_metric_by_num_simulations(
    name,
    method=None,
    estimator=None,
    embedding_net=None,
    task=None,
    num_simulations=None,
    seed=None,
    metric=None,
    ax=None,
    figsize=(3, 2),
    color_map=None,
    hue=None,
    df=None,
    **kwargs,
):
    if df is None:
        df = query(
            name=name,
            method=method,
            estimator=estimator,
            embedding_net=embedding_net,
            task=task,
            num_simulations=num_simulations,
            seed=seed,
            metric=metric,
        )

    ylims = get_ylim_by_metric(metric)
    df = df.sort_values("num_simulations")
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        fig = None
    if hue is not None:
        df = df.sort_values(hue, key=get_sorting_key_fn(hue))

    sns.pointplot(
        x="num_simulations",
        y="value",
        data=df,
        ax=ax,
        hue=hue,
        markers=".",
        dodge=False,
        palette=color_map,
        lw=kwargs.get("lw", 2.0),
        alpha=kwargs.get("alpha", 0.7),
        ms=kwargs.get("ms", 5.0),
    )
    ax.set_xlabel("Number of simulations")
    ax.set_ylabel(get_metric_plot_name(metric))
    ax.set_ylim(*ylims)
    ax.set_xticklabels(
        [
            float_to_power_of_ten(float(label.get_text()))
            for label in ax.get_xticklabels()
        ]
    )
    if metric == "c2st":
        ax.set_yticks([0.5, 1.0])

    return fig, ax


def get_style(style, **kwargs):
    if style in _mpl_styles:
        return [style]
    elif style in _custom_styles:
        return [PATH + os.sep + style + ".mplstyle"]
    elif style == "science":
        return ["science"]
    elif style == "science_grid":
        return ["science", {"axes.grid": True}]
    elif style is None:
        return None
    else:
        return style


class use_style:
    def __init__(self, style, kwargs={}) -> None:
        super().__init__()
        self.style = get_style(style) + [kwargs]
        self.previous_style = {}

    def __enter__(self):
        self.previous_style = mpl.rcParams.copy()
        if self.style is not None:
            plt.style.use(self.style)

    def __exit__(self, *args, **kwargs):
        mpl.rcParams.update(self.previous_style)
        plt.show()  # Ensure the plot is displayed


def get_ylim_by_metric(metric):
    """Get ylim by metric"""
    if metric is None:
        return (None, None)
    elif "c2st" in metric:
        return (0.5, 1.0)
    elif "swd" in metric:
        return (0.0, None)
    else:
        return (None, None)


def get_metric_plot_name(metric):
    """Get metric plot name"""
    if metric is None:
        return None
    elif "c2st" in metric:
        return "C2ST"
    elif "nll" in metric:
        return "NLL"
    elif "swd" in metric:
        return r"s$W_1$"
    elif "standardized_distance" in metric:
        return "dist. to obs."
    else:
        return metric


def get_task_plot_name(task):
    """Get task plot name"""
    print(task)
    match task:
        case "gaussian_mixture":
            return "Gauss. Mixture"
        case "two_moons":
            return "Two Moons"
        case "gaussian_linear":
            return "Gauss. Linear"
        case "gaussian_linear_uniform":
            return "Gaussian Linear Uniform"
        case "bernoulli_glm":
            return "Bernoulli GLM"
        case "bernoulli_glm_raw":
            return "Bernoulli GLM Raw"
        case "sir":
            return "SIR"
        case "lotka_volterra":
            return "Lotka Volterra"
        case "slcp":
            return "SLCP"
        case "weinberg":
            return "Weinberg"
        case "mg1":
            return "MG1"
        case "biomolecular_docking":
            return "Docking"
        case "nonlinear_gaussian_tree":
            return "Tree"
        case "nonlinear_marcov_chain":
            return "HMM"
        case "nonlinear_marcov_chain_long":
            return "HMM[50d]"
        case "streams":
            return "Streams"
        case _:
            return task


def get_method_plot_name(method):
    match method:
        case "nle":
            return "NLE"
        case "npe":
            return "NPE"
        case "nre":
            return "NRE"
        case "nse":
            return "NSE"
        case "tabpfn":
            return "NPE-PFN (unfiltered)"
        case "filtered_tabpfn":
            return "NPE-PFN"
        case "ts_tabpfn":
            return "TSNPE-PFN"
        case "snpe":
            return "SNPE"
        case "snre":
            return "SNRE"
        case "snle":
            return "SNLE"
        case "tsnpe":
            return "TSNPE"
        case "npe_ensemble":
            return "NPE (Ensemble)"
        case "npe_sweeper":
            return "NPE (Sweep)"
        case _:
            return method


def float_to_power_of_ten(val: float):
    if val == 0:
        return "0"
    exp = math.log10(val)
    exp = int(exp)
    base = val / (10**exp)
    if base == 1:
        return rf"$10^{exp}$"
    else:
        return rf"${base:.0f}\cdot10^{exp}$"


def get_sorting_key_fn(name):
    if name == "method":

        def key_fn(method):
            if method == "npe" or method == "snpe" or method == "tsnpe":
                return 0
            elif method == "nle" or method == "snle":
                return 1
            elif method == "nre" or method == "snre":
                return 2
            elif method == "tabpfn":
                return 3
            elif method == "filtered_tabpfn" or method == "ts_tabpfn":
                return 4
            else:
                return 5

        return np.vectorize(key_fn)
    elif name == "task":

        def key_fn(task):
            if task == "gaussian_linear":
                return 0
            elif task == "gaussian_linear_uniform":
                return 1
            elif task == "gaussian_mixture":
                return 2
            elif task == "two_moons":
                return 3
            elif task == "slcp":
                return 4
            elif task == "bernoulli_glm":
                return 5
            elif task == "bernoulli_glm_raw":
                return 6
            elif task == "sir":
                return 7
            elif task == "lotka_volterra":
                return 8
            elif task == "nonlinear_marcov_chain":
                return 9
            elif task == "nonlinear_marcov_chain_long":
                return 10
            else:
                return 6

        return np.vectorize(key_fn)
    else:
        return lambda x: x


def get_plot_name_fn(name):
    """Get plot name fn"""

    if name == "task":
        return get_task_plot_name
    elif name == "metric":
        return get_metric_plot_name
    else:
        return lambda x: x


def use_all_plot_name_fn(name):
    """Get plot name fn"""

    return get_method_plot_name(get_task_plot_name(get_metric_plot_name(name)))


def multi_plot(
    name,
    cols,
    rows,
    plot_fn,
    fig_title=None,
    y_label_by_row=True,
    y_labels=None,
    scilimit=3,
    x_labels=None,
    y_lims=None,
    fontsize_title=None,
    figsize_per_row=2,
    figsize_per_col=2.3,
    legend_bbox_to_anchor=[0.5, -0.2],
    legend_title=False,
    legend_ncol=10,
    legend_kwargs={},
    fig_legend=True,
    df=None,
    verbose=False,
    **kwargs,
):
    if df is None:
        df = query(name, **kwargs)
    else:
        df = df.copy()

    df = df.sort_values(cols, na_position="first", key=get_sorting_key_fn(cols))
    cols_vals = df[cols].dropna().unique()

    df = df.sort_values(rows, na_position="first", key=get_sorting_key_fn(rows))
    rows_vals = df[rows].dropna().unique()

    # Creating a color map if hue is specified:
    if "hue" in kwargs and "color_map" not in kwargs:
        hue_col = kwargs["hue"]
        df = df.sort_values(
            hue_col, na_position="first", key=get_sorting_key_fn(hue_col)
        )
        unique_vals = df[hue_col].unique()
        unique_vals.sort()
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_map = {}
        for i in range(len(unique_vals)):
            color_map[unique_vals[i]] = colors[min(i, len(colors) - 1)]
    else:
        if "color_map" not in kwargs:
            color_map = None
        else:
            color_map = kwargs.pop("color_map")

    n_cols = len(cols_vals)
    n_rows = len(rows_vals)

    if n_cols == 0:
        raise ValueError(f"No columns found in the dataset with label {cols}")

    if n_rows == 0:
        raise ValueError(f"No rows found in the dataset with label {rows}")

    print(figsize_per_col)

    figsize = (n_cols * figsize_per_col, n_rows * figsize_per_row)

    print(figsize)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    else:
        if n_cols == 1:
            axes = np.array([[ax] for ax in axes])

        if n_rows == 1:
            axes = np.array([axes])

    max_legend_elements = 0

    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j].ticklabel_format(axis="y", scilimits=[-scilimit, scilimit])
            if y_labels is not None:
                y_label = y_labels[i]
            else:
                if y_label_by_row:
                    name_fn = get_plot_name_fn(rows)
                    y_label = name_fn(rows_vals[i])
                else:
                    y_label = None

            if x_labels is not None:
                x_label = x_labels[i]
            else:
                x_label = None

            if y_lims is not None:
                if isinstance(y_lims, tuple):
                    y_lim = y_lims
                else:
                    if isinstance(y_lims[0], tuple):
                        y_lim = y_lims[i]
                    else:
                        if isinstance(y_lims[0, 0], tuple):
                            y_lim = y_lims[i, j]
                        else:
                            raise ValueError()
            else:
                y_lim = None

            plot_dict = {cols: cols_vals[j], rows: rows_vals[i]}
            plot_kwargs = {**kwargs, **plot_dict}

            if verbose:
                print(plot_kwargs)
            try:
                plot_fn(name, ax=axes[i, j], color_map=color_map, **plot_kwargs)
            except Exception as e:
                if verbose:
                    print(str(e))
                    # Print traceback
                    import traceback

                    traceback.print_exc()

            if y_label is not None:
                axes[i, j].set_ylabel(y_label)
                axes[i, j].yaxis.set_label_coords(-0.3, 0.5)
            else:
                fn = get_plot_name_fn(cols)
                y_label = axes[i, j].get_ylabel()
                axes[i, j].set_ylabel(fn(y_label))
                axes[i, j].yaxis.set_label_coords(-0.3, 0.5)

            if x_label is not None:
                axes[i, j].set_xlabel(x_label)
            else:
                fn = get_plot_name_fn(rows)
                x_label = axes[i, j].get_xlabel()
                axes[i, j].set_xlabel(fn(x_label))
            if i == 0:
                name_fn = get_plot_name_fn(cols)
                axes[i, j].set_title(name_fn(cols_vals[j]))

            if i < n_rows - 1:
                axes[i, j].set_xlabel(None)
                axes[i, j].set_xticklabels([])

            if j > 0:
                axes[i, j].set_ylabel(None)

            if y_lim is not None:
                axes[i, j].set_ylim(y_lim)

            if i > 0:
                axes[i, j].set_title(None)

            if axes[i, j].get_legend() is not None:
                legend = axes[i, j].get_legend()
                if len(legend.get_texts()) > max_legend_elements:
                    max_legend_elements = len(legend.get_texts())
                    legend_text = [t._text for t in legend.get_texts()]
                    if legend_title:
                        legend_title = legend.get_title()._text
                    else:
                        legend_title = ""
                    legend_handles = legend.legend_handles
                legend.remove()

    for i in range(n_rows):
        for j in range(n_cols):
            if len(axes[i, j].lines) == 0 and len(axes[i, j].collections) == 0:
                axes[i, j].text(
                    0.5,
                    0.5,
                    "No data",
                    bbox={
                        "facecolor": "white",
                        "alpha": 1,
                        "edgecolor": "none",
                        "pad": 1,
                    },
                    ha="center",
                    va="center",
                )

    if fig_legend and "legend_text" in locals() and len(legend_text) > 0:
        text = [use_all_plot_name_fn(t) for t in list(dict.fromkeys(legend_text))]
        handles = list(dict.fromkeys(legend_handles))
        fig.legend(
            labels=text,
            handles=handles,
            title=use_all_plot_name_fn(str(legend_title)),
            ncol=legend_ncol,
            loc="lower center",
            bbox_to_anchor=legend_bbox_to_anchor,
            **legend_kwargs,
        )

    # fig.tight_layout()
    if fig_title is not None:
        fig.suptitle(fig_title)
    return fig, axes
