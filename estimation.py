import numpy as np
import matplotlib.pyplot as plt
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
):
    """
    Extracts voxel values for masked and unmasked regions and calculates
    basic weights and divisions.
    """
    # Division (Occupancy Factor proxy)
    pseudo_occupancy = -diffmap_np / map_dark_np

    weight = np.abs(diffmap_np[mask_np])

    diffmap_sigma = (diffmap_np - np.mean(diffmap_np)) / np.std(diffmap_np)

    return {
        "diffmap_masked": diffmap_sigma[mask_np],
        "pseudo_occupancy": pseudo_occupancy[mask_np],
        "weight": weight,
        "diffmap_inv": diffmap_sigma[~mask_np],
        "pseudo_occupancy_inv": pseudo_occupancy[~mask_np],
    }


def _analyze_threshold_trends(
    diffmap_vals: np.ndarray, pseudo_occupancy: np.ndarray, weights: np.ndarray
):
    """
    Iterates through intensity thresholds to calculate running means and standard deviations.
    """

    def reweight(weights):
        return weights

    # Note: original code negated diffmap values for the threshold loop: `diffmap_mk = -diffmap_np[mask_np]`
    diffmap_mk = -diffmap_vals

    threshold = np.linspace(np.min(diffmap_mk), np.max(diffmap_mk), num=50)
    means, stds = [], []
    stability = []

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
            continue

        div_mean, div_std = weighted_std(current_div, current_weight)

        sum_weight = reweight(np.sum(current_weight))
        general_uncertainty = prefactor / sum_weight if sum_weight != 0 else 0

        means.append(div_mean)
        stds.append(div_std)
        stability.append(general_uncertainty)

    return {
        "threshold": threshold,
        "mean": np.array(means),
        "std": np.array(stds),
        "stability": np.array(stability),
    }


def _create_plot(stats, trend, plot_config, general_config):
    """
    Handles all matplotlib logic.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), tight_layout=True)
    fig.suptitle(general_config["name_human"])

    # Unpack data
    means = trend["mean"]
    stds = trend["std"]
    stability = trend["stability"]
    threshs = trend["threshold"]

    # 1. Plot Trends (Mean + Error Bands)
    ax.plot(means, threshs, label="Weighted mean", color="blue")

    # Fill between std
    ax.fill_betweenx(
        threshs,
        (means - stds),
        (means + stds),
        color="gray",
        alpha=0.5,
        label="Weighted mean ± std",
    )

    # Fill between stability
    ax.fill_betweenx(
        threshs,
        (means - stds),
        (means - stds - stability),
        color="green",
        alpha=0.5,
        label="Weighted mean ± std ± general stability",
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
        -stats["diffmap_masked"],
        label="voxels to include",
        marker=marker,
        linestyle="",
        alpha=0.5,
    )

    if plot_config["show_ignored_voxels"]:
        ax.plot(
            stats["pseudo_occupancy_inv"],
            -stats["diffmap_inv"],
            marker=marker,
            linestyle="",
            label="voxels to ignore",
            alpha=0.3,
        )

    # 4. Formatting
    ax.legend(loc="upper right")
    ax.set_yscale("linear")
    ax.set_xlabel("Implied occupancy factor")
    ax.set_ylabel("Difference Map voxel values")

    # Determine X-limits safely ignoring NaNs
    valid_means = means[~np.isnan(means)]
    valid_stds = stds[~np.isnan(stds)]
    if len(valid_means) > 0:
        max_x = np.max(valid_means + valid_stds) * 1.1
        ax.set_xlim(0, max_x)

    ax.set_ylim(0, None)
    ax.grid()

    return fig, ax


def plot_extrapolation_estimate(
    diffmap: rsmap.Map,
    map_dark: rsmap.Map,
    inclusion_mask: np.ndarray,
    config: dict,
):
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

    # 5. Visualization
    return _create_plot(stats_data, trend_data, config["plot"], general_config)
