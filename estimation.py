import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from meteor import rsmap

from logger import setup_logger

logger = setup_logger()


def weighted_std(values: np.ndarray, weights: np.ndarray) -> tuple:
    """
    Calculate the weighted standard deviation.
    """
    weighted_mean = np.average(values, weights=weights)

    # This is the weighted average of the squared deviations from the weighted mean
    variance = np.average((values - weighted_mean) ** 2, weights=weights)

    return weighted_mean, np.sqrt(variance)


def _calculate_statistics(
    diffmap_np: np.ndarray, map_dark_np: np.ndarray, mask_np: np.ndarray
) -> dict:
    """
    Extracts voxel values for masked and unmasked regions and calculates
    basic weights and divisions.
    """
    # Division (Occupancy Factor proxy)
    solvent_level = -0.
    pseudo_occupancy = -diffmap_np[mask_np] / (map_dark_np[mask_np] - solvent_level)
    mask_inv = np.logical_and(~mask_np, map_dark_np != 0)
    pseudo_occupancy_inv = -diffmap_np[mask_inv] / (map_dark_np[mask_inv] - solvent_level)

    weight = np.abs(diffmap_np[mask_np])

    sigma_level = np.sqrt(
        np.sum((diffmap_np - np.mean(diffmap_np)) ** 2) / diffmap_np.size
    )

    diffmap_sigma = (diffmap_np - np.mean(diffmap_np)) / sigma_level
    diffmap_sigma = (np.abs(diffmap_sigma)) * np.sign(diffmap_sigma)
    return {
        "diffmap_raw": diffmap_np[mask_np],
        "pseudo_occupancy": pseudo_occupancy,
        "weight": weight,
        "diffmap_sigma": diffmap_sigma[mask_np],
        "diffmap_inv": diffmap_sigma[mask_inv],
        "pseudo_occupancy_inv": pseudo_occupancy_inv,
    }


def _analyze_threshold_trends(
    diffmap_vals: np.ndarray, pseudo_occupancy: np.ndarray, weights: np.ndarray
) -> dict:
    """
    Iterates through intensity thresholds to calculate running means and standard deviations.
    """

    def reweight(weights):
        return weights

    # Note: original code negated diffmap values for the threshold loop: `diffmap_mk = -diffmap_np[mask_np]`
    diffmap_mk = -diffmap_vals

    threshold = np.linspace(np.min(diffmap_mk), np.max(diffmap_mk) * 0.9, num=50)
    means, stds = [], []
    stability = []
    weight = []
    bootstrap_std = []
    bootstrap_mean = []

    prefactor = reweight(np.max(weights)) * (pseudo_occupancy[np.argmax(weights)])

    for thresh in threshold:
        # Select voxels exceeding threshold
        thresh_mask = diffmap_mk >= thresh

        current_div = pseudo_occupancy[thresh_mask]
        current_weight = weights[thresh_mask]

        if len(current_div) == 0:
            # Handle empty slices if threshold is too high
            means.append(np.nan)
            stds.append(np.nan)
            stability.append(np.nan)
            bootstrap_std.append(np.nan)
            bootstrap_mean.append(np.nan)
            weight.append(np.nan)
            continue

        div_mean, div_std = weighted_std(current_div, current_weight)

        sum_weight = reweight(np.sum(current_weight))
        general_uncertainty = prefactor / sum_weight if sum_weight != 0 else 0

        means.append(div_mean)
        stds.append(div_std)
        stability.append(general_uncertainty)
        random_sample = np.random.choice(current_div, size=1000)
        bs_std = np.std(random_sample)
        bootstrap_std.append(bs_std)
        bs_mean = np.mean(random_sample)
        bootstrap_mean.append(bs_mean)
        weight.append(sum_weight)

    return {
        "threshold": threshold,
        "mean": np.array(means),
        "std": np.array(stds),
        "stability": np.array(stability),
        "bootstrap_std": np.array(bootstrap_std),
        "bootstrap_mean": np.array(bootstrap_mean),
        "weight": np.array(weight),
    }


def _create_plot_v2(
    stats: dict,
    trend: dict,
    plot_config: dict,
    general_config: dict,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Handles all matplotlib logic.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    else:
        fig = ax.get_figure()
    # ax.set_title(general_config["name_human"])

    # Unpack data
    means = trend["mean"]
    stds = trend["std"]
    stability = trend["stability"]
    threshs = trend["threshold"]

    # 1. Plot Trends (Mean + Error Bands)
    ax.plot(-threshs, means, label="Weighted mean", color="blue")
    min_uncertainty_idx = np.argmin(stds + stability)
    # optimal_uncertainty = [
    #     -(means[min_uncertainty_idx] - stds[min_uncertainty_idx]),
    #     means[min_uncertainty_idx] + stds[min_uncertainty_idx],
    # ]
    # ax.plot(
    #     [-threshs[min_uncertainty_idx],]*2,
    #     optimal_uncertainty,
    #     marker="|",
    #     color="blue",
    #     label=f"Prediction: {means[min_uncertainty_idx]:.3f} \nUncertainty: {stds[min_uncertainty_idx]:.3f} ({stds[min_uncertainty_idx]/means[min_uncertainty_idx]:.1%})",
    # )
    ax.scatter(
        -threshs[min_uncertainty_idx],
        means[min_uncertainty_idx],
        s=200,
        facecolor="none",
        color="brown",
        label=f"Optimal threshold: {means[min_uncertainty_idx]:.3f}",
    )

    # Fill between std
    ax.fill_between(
        -threshs,
        (means - stds),
        (means + stds),
        color="gray",
        alpha=0.5,
        label="Standard Deviation",
    )

    # Fill between stability
    ax.fill_between(
        -threshs,
        (means - stds),
        (means - stds - stability),
        color="green",
        alpha=0.5,
        label="Numerical instability",
    )
    ax.fill_between(
        -threshs,
        (means + stds),
        (means + stds + stability),
        color="green",
        alpha=0.5,
    )

    # 2. Plot Voxel Scatter
    marker = "."
    # Note: X-axis is pseudo_occupancy (Occupancy), Y-axis is -DifferenceMap
    if plot_config["show_ignored_voxels"]:
        extra_mask = np.logical_and(
            stats["diffmap_sigma"] != 0, stats["pseudo_occupancy_inv"] > 0
        )
        ax.plot(
            stats["diffmap_inv"][extra_mask],
            stats["pseudo_occupancy_inv"][extra_mask],
            marker=".",
            linestyle="",
            label="excluded voxels",
            alpha=0.3,
            color="gray",
        )
    ax.plot(
        stats["diffmap_sigma"],
        stats["pseudo_occupancy"],
        label="included voxels",
        marker=marker,
        linestyle="",
        alpha=0.5,
        color="red",
    )

    ax.set_xscale("linear")

    # 4. Formatting
    if not plot_config.get("is_composite", False):
        ax.legend(loc="upper right")
        ax.set_ylabel("Implied occupancy " + r"$ \chi^{-1} = -\Delta\rho/\rho_{0}$")
        ax.set_xlabel("Difference Map " + r"$-\Delta \rho$")

    # Determine X-limits safely ignoring NaNs
    valid_means = means[~np.isnan(means)]
    valid_stds = stds[~np.isnan(stds)]
    if len(valid_means) > 0:
        max_y = np.max(valid_means + valid_stds) * 1.1
        ax.set_ylim(0, max_y)
    if plot_config["set_ylim"]:
        ax.set_ylim(*plot_config["set_ylim"])

    ax.set_xlim(
        None,
        0,
    )
    ax.grid()

    return fig, ax  # type: ignore


def _create_plot(
    stats: dict,
    trend: dict,
    plot_config: dict,
    general_config: dict,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Handles all matplotlib logic.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    else:
        fig = None
    # ax.set_title(general_config["name_human"])

    # Unpack data
    means = trend["mean"]
    stds = trend["std"]
    stability = trend["stability"]
    threshs = trend["threshold"]

    # 1. Plot Trends (Mean + Error Bands)
    ax.plot(threshs, means, label="Weighted mean", color="blue")

    min_uncertainty_idx = np.argmin(stds + stability)
    optimal_uncertainty = [
        means[min_uncertainty_idx] - stds[min_uncertainty_idx],
        means[min_uncertainty_idx] + stds[min_uncertainty_idx],
    ]
    ax.plot(
        [
            threshs[min_uncertainty_idx],
        ]
        * 2,
        optimal_uncertainty,
        marker="|",
        color="blue",
        label=f"Prediction: {means[min_uncertainty_idx]:.3f} \nUncertainty: {stds[min_uncertainty_idx]:.3f} ({stds[min_uncertainty_idx]/means[min_uncertainty_idx]:.1%})",
    )
    ax.scatter(
        means[min_uncertainty_idx],
        threshs[min_uncertainty_idx],
        s=200,
        facecolor="none",
        color="brown",
        label=f"Optimal threshold: {means[min_uncertainty_idx]:.3f}",
    )

    # Fill between std
    ax.fill_betweenx(
        threshs,
        (means - stds),
        (means + stds),
        color="gray",
        alpha=0.5,
        label="Standard Deviation",
    )

    # Fill between stability
    ax.fill_betweenx(
        threshs,
        (means - stds),
        (means - stds - stability),
        color="green",
        alpha=0.5,
        label="Numerical instability",
    )
    ax.fill_betweenx(
        threshs,
        (means + stds),
        (means + stds + stability),
        color="green",
        alpha=0.5,
    )

    # 2. Plot Voxel Scatter
    marker = "."
    # Note: X-axis is pseudo_occupancy (Occupancy), Y-axis is -DifferenceMap
    ax.plot(
        stats["pseudo_occupancy"],
        -stats["diffmap_sigma"],
        label="included voxels",
        marker=marker,
        linestyle="",
        alpha=0.5,
    )

    if plot_config["show_ignored_voxels"]:
        ax.plot(
            stats["pseudo_occupancy_inv"],
            -stats["diffmap_inv"],
            marker=".",
            linestyle="",
            label="excluded voxels",
            alpha=0.3,
        )

    # 4. Formatting
    if not plot_config.get("is_composite", False):
        ax.legend(loc="upper right")
        ax.set_ylabel("Implied occupancy " + r"$ \chi^{-1} = -\Delta\rho/\rho_{0}$")
        ax.set_xlabel("Difference Map " + r"$-\Delta \rho$")

    # Determine X-limits safely ignoring NaNs
    valid_means = means[~np.isnan(means)]
    valid_stds = stds[~np.isnan(stds)]
    if len(valid_means) > 0:
        max_x = np.max(valid_means + valid_stds) * 1.1
        ax.set_xlim(0, max_x)

    ax.set_ylim(0, None)
    ax.grid()

    return fig, ax  # type: ignore


def plot_extrapolation_estimate(
    diffmap: rsmap.Map,
    map_dark: rsmap.Map,
    inclusion_mask: np.ndarray,
    config: dict,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    general_config = config["general"]

    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=general_config["map_sampling"])
    map_dark_np = map_dark.to_3d_numpy_map(map_sampling=general_config["map_sampling"])
    logger.warning(
        f"Mean of diffmap_np: {np.mean(diffmap_np)}, Mean of map_dark_np: {np.mean(map_dark_np)}"
    )
    stats_data = _calculate_statistics(diffmap_np, map_dark_np, inclusion_mask)
    trend_data = _analyze_threshold_trends(
        stats_data["diffmap_raw"],
        stats_data["pseudo_occupancy"],
        stats_data["weight"],
    )

    # 5. Visualization
    return _create_plot_v2(stats_data, trend_data, config["plot"], general_config, ax)


def plot_extrapolation_estimate_v2(
    diffmap: rsmap.Map,
    map_dark: rsmap.Map,
    inclusion_mask: np.ndarray,
    config: dict,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    general_config = config["general"]

    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=general_config["map_sampling"])
    map_dark_np = map_dark.to_3d_numpy_map(map_sampling=general_config["map_sampling"])
    logger.warning(
        f"Mean of diffmap_np: {np.mean(diffmap_np)}, Mean of map_dark_np: {np.mean(map_dark_np)}"
    )
    stats_data = _calculate_statistics(diffmap_np, map_dark_np, inclusion_mask)
    trend_data = _analyze_threshold_trends(
        stats_data["diffmap_masked"],
        stats_data["pseudo_occupancy"],
        stats_data["weight"],
    )
    fig, ax = _create_plot_v2(
        stats_data, trend_data, config["plot"], general_config, ax
    )
    return fig, ax


def cummean_and_errors(stats_data, leng_shown=None, number_sym_ops=1, plot_config={}):
    pseudo = stats_data["pseudo_occupancy"]
    diff2 = -stats_data["diffmap_raw"]
    diff_sigma = -stats_data["diffmap_sigma"]
    leng_shown = len(diff2) if leng_shown is None else leng_shown
    argsorted = np.argsort(diff2)[::-number_sym_ops][:leng_shown]
    pseudo_sort = np.cumsum(pseudo[argsorted]) / (np.arange(len(argsorted)) + 1)
    # 1. Your existing sorted data
    sorted_vals = pseudo[argsorted]
    weights = diff2[argsorted]
    cum_mean = np.cumsum(sorted_vals) / (np.arange(len(sorted_vals)) + 1)
    cum_weighted = np.cumsum(sorted_vals * weights) / (np.cumsum(weights) + 1e-8)
    cum_mean_sq = np.cumsum(sorted_vals**2) / (np.arange(len(sorted_vals)) + 1)

    # 4. Cumulative standard deviation
    # We use np.maximum to avoid tiny negative numbers due to floating point error
    cum_std = np.sqrt(np.maximum(cum_mean_sq - cum_mean**2, 0))
    pseudo_std = cum_std
    pseudo_ste = pseudo_std / np.sqrt(np.arange(1, len(argsorted) + 1)) * 2
    bias_term = [
        np.abs(pseudo_sort[i] - pseudo_sort[i // 2]) for i in range(len(pseudo_sort))
    ]

    solvent_density = plot_config.get("solvent_density", 0.3)
    thresh_line = diff2[argsorted] / solvent_density

    return {
        "pseudo_sort": pseudo_sort,
        "pseudo_weighted": cum_weighted,
        "pseudo_ste": pseudo_ste,
        "bias_term": bias_term,
        "diff_sorted": diff2[argsorted],
        "diff_sigma": diff_sigma[argsorted],
        "thresh_line": thresh_line,
        "pseudo_std": pseudo_std,
    }


def compact_v3(cummean_dict, plot_config={}):
    std_cutoff = plot_config.get("std_cutoff", 3.0)
    thresh_line = cummean_dict["thresh_line"]
    average_distance_mask = (
        cummean_dict["pseudo_sort"] + std_cutoff * cummean_dict["pseudo_std"]
        > thresh_line
    )
    bottom_index = 0
    if np.any(average_distance_mask):
        bottom_index = np.where(average_distance_mask)[0][0]

    min_size_middle = 10
    is_there_a_middle = bottom_index > 0 and min_size_middle < bottom_index
    if is_there_a_middle:
        middle_diff = (
            cummean_dict["diff_sigma"][bottom_index] + cummean_dict["diff_sigma"][0]
        ) / 2
        middle_index = np.where(middle_diff > cummean_dict["diff_sigma"])[0][0]
        middle_index = max(middle_index, min_size_middle)
        middle_mean = cummean_dict["pseudo_sort"][middle_index]
        middle_diff = -cummean_dict["diff_sigma"][middle_index]
        return middle_mean, cummean_dict["pseudo_std"][middle_index]
    # elif bottom_index > -1:
    #     return cummean_dict["pseudo_sort"][bottom_index], cummean_dict["pseudo_std"][bottom_index]
    else:
        return np.nan, np.nan


def create_plot_v3(
    stats, cummean_dict, extra_info={}, ax=None, plot_config={}
) -> tuple[Figure, Axes, tuple[float, float]]:
    std_cutoff = plot_config.get("std_cutoff", 3.0)
    markersize = plot_config.get("markersize", 1)
    thresh_line = cummean_dict["thresh_line"]
    average_distance_mask = (
        cummean_dict["pseudo_sort"] + std_cutoff * cummean_dict["pseudo_std"]
        > thresh_line
    )
    bottom_index = 0
    if np.any(average_distance_mask):
        bottom_index = np.where(average_distance_mask)[0][0]

    min_size_middle = 10

    marker = "."
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ########################## Data ###################################
    ax.plot(
        stats["diffmap_sigma"],
        stats["pseudo_occupancy"],
        label="Voxel Prediction",
        marker=marker,
        markersize=markersize,
        linestyle="",
        alpha=0.5,
    )
    ax.fill_between(
        -cummean_dict["diff_sigma"],
        cummean_dict["pseudo_sort"] - cummean_dict["pseudo_std"],
        cummean_dict["pseudo_sort"] + cummean_dict["pseudo_std"],
        color="grey",
        alpha=0.2,
    )
    ax.plot(
        -cummean_dict["diff_sigma"][: bottom_index + 1],
        cummean_dict["pseudo_sort"][: bottom_index + 1],
        color="blue",
        label="Cumulative Mean",
    )
    ax.plot(
        -cummean_dict["diff_sigma"][bottom_index:],
        cummean_dict["pseudo_sort"][bottom_index:],
        color="gray",
        # label="Cumulative Mean",
    )
    ########################## Thresholds ###################################
    pref = 1.1
    lowest_sigma = -cummean_dict["diff_sigma"][0] * pref
    lowest_thresh = thresh_line[0] * pref

    max_xtr = 11
    ax.fill_between(
        [lowest_sigma, 0],
        [lowest_thresh, 0],
        [max_xtr, max_xtr],
        color="red",
        alpha=0.2,
    )

    ax.plot(
        [lowest_thresh, 0],
        [lowest_sigma, 0],
        "--",
        color="gray",
        label="Reference Density Cutoff = Solvent",
    )
    if bottom_index > 0:
        ax.plot(
            [
                -cummean_dict["diff_sigma"][bottom_index],
            ]
            * 2,
            [0, thresh_line[bottom_index]],
            "r--",
            label="3x Std Dev. Cutoff",
        )
        ymax = (
            cummean_dict["pseudo_sort"] + std_cutoff * 2 * cummean_dict["pseudo_std"]
        )[bottom_index]
        ax.set_ylim(0.0, ymax)

    ############## Optimimum
    stats_kwarg = dict(
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    is_there_a_middle = bottom_index > 0 and min_size_middle < bottom_index
    if is_there_a_middle:
        print(f"Bottom index: {bottom_index}, Is there a middle? {is_there_a_middle}")
        middle_diff = (
            cummean_dict["diff_sigma"][bottom_index] + cummean_dict["diff_sigma"][0]
        ) / 2
        middle_index = np.where(middle_diff > cummean_dict["diff_sigma"])[0][0]
        middle_index = max(middle_index, min_size_middle)
        pseudo_range = cummean_dict["pseudo_sort"][: bottom_index + 1]
        middle_mean = cummean_dict["pseudo_sort"][middle_index]
        middle_diff = -cummean_dict["diff_sigma"][middle_index]
        middle_std_range = np.array([-1, 1]) * cummean_dict["pseudo_std"][middle_index]
        ax.plot(middle_diff * np.ones(2), middle_mean - middle_std_range, color="blue")
        optimal_kwargs = dict(
            s=200,
            facecolor="none",
            color="brown",
            label="Optimal",
        )
        ax.scatter(middle_diff, middle_mean, **optimal_kwargs)  # type: ignore

        prediction_tuple = middle_mean, cummean_dict["pseudo_std"][middle_index]
        plot_stats = dict(
            middle_mean=middle_mean,
            middle_std=cummean_dict["pseudo_std"][middle_index],
            middle_std_rel=cummean_dict["pseudo_std"][middle_index] / middle_mean,
            estimation_range=cummean_dict["diff_sigma"][0]
            / cummean_dict["diff_sigma"][bottom_index]
            - 1,
            variation_range=(np.max(pseudo_range) - np.min(pseudo_range)) / middle_mean,
        )
        chi = "$\\chi^{-1}$"
        text = ""
        text += f" {chi} =  {middle_mean:.3f}"
        text += f"\nSt. Dev.: {plot_stats['middle_std']:.3f} ({plot_stats['middle_std_rel']:.1%})"
        if plot_config.get("comparison_to_reference", False):
            text += f"\nReference: {plot_config['comparison_to_reference']['value']:.3f} (1/{1/plot_config['comparison_to_reference']['value']:.1f})"
        if np.min(pseudo_range) < middle_mean - plot_stats["middle_std"] and False:
            text += "\nWarning: Min. est. less than\n1 std. dev. than reported est."
        if np.max(pseudo_range) > middle_mean + plot_stats["middle_std"] and False:
            text += "\nWarning: Max. est. more than\n1 std. dev. than reported est."
        # text += f"\nMin-Max Variation: {plot_stats['variation_range']:.1%}"
        # text += f"\n Estimation Range: {plot_stats['estimation_range']:.0%}"
        # place legend like box in top right corner
        ax.text(0.95, 0.95, text, **stats_kwarg)  # type: ignore

    else:
        text = "No Prediction because \n estimation range is too small"
        ax.text(0.95, 0.95, text, **stats_kwarg)  # type: ignore
        prediction_tuple = (np.nan, np.nan)

    # 4. Formatting
    if not plot_config.get("is_composite", False):
        ax.set_ylabel("Extrapolation factor " + r"$ \chi^{-1} = -\Delta\rho/\rho_{0}$")
        ax.set_xlabel("Difference Map " + r"$-\Delta \rho$ (standard deviations)")
        # ax.legend(loc="upper left")

        text_kwargs = dict(
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.4),
        )
        ax.text(0.25, 0.10, r"$\rho_0>\rho_{solvent}$", **text_kwargs)  # type: ignore
        ax.text(0.95, 0.45, r"$\rho_0<\rho_{solvent}$", **text_kwargs)  # type: ignore

    ax.grid()
    ax.set_xlim((lowest_sigma, 0.0))

    if True:
        ax2 = ax.twiny()
        ax2.plot(
            [-cummean_dict["diff_sorted"][0], 0],
            [thresh_line[0], 0],
            "--",
            color="gray",
        )
        ax2.set_xlim((-cummean_dict["diff_sorted"][0]) * 1.1, 0.0)
        ax2.set_xlabel(r"$-\Delta \rho$ (absolute units)")

    if plot_config.get("set_ylim", False):
        ax.set_ylim(*plot_config["set_ylim"])
    return fig, ax, prediction_tuple


def plot_extrapolation_estimate_new(
    diffmap: rsmap.Map,
    map_dark: rsmap.Map,
    inclusion_mask: np.ndarray,
    config: dict,
    ax: Axes | None = None,
    compact: bool = False,
) -> tuple[Figure | None, Axes | None, tuple[float, float]]:
    general_config = config["general"]

    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=general_config["map_sampling"])
    map_dark_np = map_dark.to_3d_numpy_map(map_sampling=general_config["map_sampling"])
    logger.warning(
        f"Mean of diffmap_np: {np.mean(diffmap_np)}, Mean of map_dark_np: {np.mean(map_dark_np)}"
    )
    stats_data = _calculate_statistics(diffmap_np, map_dark_np, inclusion_mask)
    cummean_dict = cummean_and_errors(stats_data, number_sym_ops=1)
    # trend_data = _analyze_threshold_trends(
    #     stats_data["diffmap_masked"],
    #     stats_data["pseudo_occupancy"],
    #     stats_data["weight"],
    # )
    # 5. Visualization
    if compact:
        return None, None, compact_v3(cummean_dict, plot_config=config["plot"])
    return create_plot_v3(stats_data, cummean_dict, plot_config=config["plot"], ax=ax)
