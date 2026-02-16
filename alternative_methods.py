import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import gemmi
import numpy as np
import pickle
import warnings
from pathlib import Path
import reciprocalspaceship as rs
from meteor import rsmap
from meteor.sfcalc import gemmi_structure_to_calculated_map
from configuration import load_homepath
from configuration import get_file_config
from logger import setup_logger

logger = setup_logger()


################################### Load Filenames #############################
def find_folder(folder_cond):
    weight = folder_cond["weight"]
    snr = folder_cond["snr"]

    if folder_cond["noise_type"] == "gaussian":
        noise_dark = "gaussian"
        noise_folder = "gaussian_"
    elif folder_cond["noise_type"] == "pseudo_poisson":
        noise_dark = "snr"
        noise_folder = ""
    elif folder_cond["noise_type"] == "gaussian_flat":
        noise_dark = "gaussian_flat"
        noise_folder = "flat_"
    else:
        raise ValueError

    if "q" == weight:
        diffmap_columns = dict(amplitude_column="QFOFOWT", phase_column="PHIQFOFOWT")
    elif "" == weight:
        diffmap_columns = dict(amplitude_column="FOFOWT", phase_column="PHIFOFOWT")
    else:
        raise ValueError

    dark_name = f"trans_{noise_dark}_{snr}_dmin_16.mtz"
    folder_name = f"{weight}{noise_folder}snr_{folder_cond["snr"]}/"
    fofo_name = f"xx_m{weight}FoFo.mtz"
    x8_analysis_snip = f"alpha_occupancy_determination_{weight}Fextr"

    return dark_name, fofo_name, folder_name, diffmap_columns, x8_analysis_snip


########################## Negative Sum Explosion ##############################


def get_fits2(neg_sum, alpha_invs, n_largest, return_all=False):
    a_sorted = np.argsort(alpha_invs)
    m_lowest = a_sorted <= n_largest
    m_biggest = a_sorted >= len(a_sorted) - n_largest - 2
    res_lowest = stats.linregress(alpha_invs[m_lowest], neg_sum[m_lowest])
    res_biggest = stats.linregress(alpha_invs[m_biggest], neg_sum[m_biggest])
    np.linspace(np.min(alpha_invs), np.max(alpha_invs), 5)
    fit_lowest = res_lowest.intercept + res_lowest.slope * alpha_invs
    fit_biggest = res_biggest.intercept + res_biggest.slope * alpha_invs

    # intersection = (res_2.tercept-res_1.intercept) / (res_1.slope-res_2.slope)
    intersection = (res_biggest.intercept - res_lowest.intercept) / (
        res_lowest.slope - res_biggest.slope
    )
    intersection_y = res_biggest.intercept + res_biggest.slope * intersection
    highest_low = np.max(alpha_invs[m_lowest])
    lowest_high = np.min(alpha_invs[m_biggest])
    if intersection > lowest_high or intersection < highest_low:
        logger.warning(
            f"Intersection at {intersection:.2f} should be between {highest_low:.2f} and {lowest_high:.2f}"
        )
    else:
        logger.debug(
            f"Intersection at {intersection:.2f} is between {highest_low:.2f} and {lowest_high:.2f}"
        )
    hlf, llf = 1, 1
    if (
        intersection < highest_low * hlf or intersection > lowest_high * llf
    ) and not return_all:
        logger.warning("    Intersection declared invalid")
        intersection = np.nan
    if intersection < highest_low * hlf:
        logger.warning(
            f"Intersection declared invalid, because it is greater than {highest_low * hlf:.2f}"
        )
    if intersection > lowest_high * llf:
        logger.warning(
            f"Intersection declared invalid, because it is less than {lowest_high * llf:.2f}"
        )
    if intersection < 0:
        logger.error("Negative intersection found, this should not happen")

    if np.max(np.abs(fit_lowest - fit_biggest)) < 0.1:
        logger.warning(f"Fits are (close to) parallel: {intersection:.1f} )")
        intersection = np.nan

    return fit_lowest, fit_biggest, intersection, intersection_y


def _calculate_negative_density_trends(diffmap_np, map_dark_np, mask_np, xtr_range):
    """
    Performs the 'extrapolation factor' analysis.
    Reason: It generates a series of synthetic maps by scaling the difference map
    and adding it to the dark map, then counting negative density. This is
    computationally distinct from basic masking.
    """

    # Create synthetic maps: map_new = map_dark + factor * diffmap
    # Using broadcasting to create a 4D array (N_factors, X, Y, Z)
    xtr_maps = (
        diffmap_np[None, ...] * xtr_range[:, None, None, None] + map_dark_np[None, ...]
    )

    neg_dens = np.zeros(len(xtr_range))
    for i in range(len(xtr_range)):
        # Apply mask to the synthetic map
        xtr_temp = xtr_maps[i][mask_np]
        # Sum only the negative values
        neg_dens[i] = np.sum(xtr_temp[xtr_temp < 0], axis=0)

    return {
        "xtr_range": xtr_range,
        "neg_dens": neg_dens,
    }


def plot_negative_density_trends(neg_density_plot, neg_density_fit, ax=None):
    xtr_range_plot = neg_density_plot["xtr_range"]
    neg_dens_plot = neg_density_plot["neg_dens"]
    xtr_range_fit = neg_density_fit["xtr_range"]
    neg_dens_fit = neg_density_fit["neg_dens"]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    ax.plot(xtr_range_plot, -neg_dens_plot, "x")

    fit_lowest2, fit_biggest2, intersect, intersect_y = get_fits2(
        -neg_dens_fit, xtr_range_fit, 3, return_all=True
    )
    ax.plot(xtr_range_fit, fit_lowest2, "--", color="gray")
    ax.plot(
        xtr_range_fit,
        fit_biggest2,
        "--",
        color="gray",
        label=f"Fits (Intersect at {intersect:.2f} i.e. {1/intersect:.2f})",
    )
    print()
    ax.scatter(
        intersect,
        intersect_y,
        s=200,
        facecolor="none",
        color="brown",
        label=f"Intersection at {intersect:.2f} (i.e. {1/intersect:.2f})",
    )
    ax.legend(loc="upper right")
    ax.set_xlabel("Extrapolation factor")
    ax.set_ylabel("Negative density sum")
    xmax = np.max(xtr_range_plot) * 1.1
    ymax = np.max(-neg_dens_plot) * 1.1
    ax.set_xlim(0, xmax)
    ax.set_ylim(-ymax / 50, ymax)
    return fig, ax


def nse_analysis(diffmap, map_dark, inclusion_mask, ax):
    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=3)
    map_dark_np = map_dark.to_3d_numpy_map(map_sampling=3)

    xtr_range_show = np.arange(1, 15)
    xtr_range_fit = np.concatenate(
        (np.linspace(0, 0.8, 8), xtr_range_show, np.linspace(50, 80, 8))
    )
    neg_density_fit = _calculate_negative_density_trends(
        diffmap_np, map_dark_np, inclusion_mask, xtr_range_fit
    )
    neg_density_plot = _calculate_negative_density_trends(
        diffmap_np, map_dark_np, inclusion_mask, xtr_range_show
    )

    return plot_negative_density_trends(neg_density_plot, neg_density_fit, ax)


################################ PANDDA ########################################
def _calculate_pandda(diffmap_np, map_dark_np, mask_np):
    """
    Performs the 'extrapolation factor' analysis.
    Reason: It generates a series of synthetic maps by scaling the difference map
    and adding it to the dark map, then counting negative density. This is
    computationally distinct from basic masking.
    """

    # Define range for extrapolation
    #
    xtr_range = np.linspace(0.0001, 1, 50) ** 2

    # Create synthetic maps: map_new = map_dark + factor * diffmap
    # Using broadcasting to create a 4D array (N_factors, X, Y, Z)
    xtr_maps = (
        diffmap_np[None, ...] * 1 / xtr_range[:, None, None, None]
        + map_dark_np[None, ...]
    )

    mean_global = np.empty(len(xtr_range))
    mean_local = np.empty(len(xtr_range))
    for ii in range(len(xtr_range)):
        # Apply mask to the synthetic map
        # Sum only the negative values
        # neg_dens[i] = np.sum(xtr_temp[xtr_temp < 0], axis=0)
        mean_global[ii] = stats.pearsonr(xtr_maps[ii].flatten(), map_dark_np.flatten())[
            0
        ]
        mean_local[ii] = stats.pearsonr(
            xtr_maps[ii][mask_np].flatten(), map_dark_np[mask_np].flatten()
        )[0]

    return {
        "xtr_range": xtr_range,
        "mean_global": mean_global,
        "mean_local": mean_local,
    }


def plot_pandda_results(pandda_dict, alpha=None, axs=None):
    pseudo_occupancy = pandda_dict["xtr_range"]
    mean_local = pandda_dict["mean_local"]
    mean_global = pandda_dict["mean_global"]

    if axs is None:
        fig, axs = plt.subplots(2, figsize=(8, 4), sharex=True)
    else:
        fig = None

    ax = axs[0]
    if alpha is not None:
        ax.axvline(alpha, c="k", linestyle="-.", label="alpha_true")
    mean_diff = mean_global - mean_local
    ax.plot(pseudo_occupancy, +mean_diff, label="global-local")
    pk_val_idx = np.argmax(mean_diff)
    pk_val_narrow = pseudo_occupancy[pk_val_idx]
    # pseudo_occ = np.argwhere(pseudo_occupancy == pk_val_narrow)[0]
    ax.scatter(
        pseudo_occupancy[pk_val_idx],
        mean_diff[pk_val_idx],
        s=200,
        facecolor="none",
        color="brown",
        label=f"Peak at {pk_val_narrow:.2f}",
    )
    # ax.axvline(pseudo_occupancy[pk_val_idx], color="green", label=f"Peak at {pk_val_narrow:.2f}")
    ax.legend()
    ax.set_title("PanDDA method")
    ax = axs[1]
    if alpha is not None:
        ax.axvline(alpha, c="k", linestyle="-.", label="alpha_true")
    ax.plot(pseudo_occupancy, mean_local, label="local")
    ax.plot(pseudo_occupancy, mean_global, label="global")
    ax.set_ylim(-1, 1)
    ax.legend()
    return fig, axs


################################## Xtrapol8 #####################################
def load_xtrapol8_data(xtrapolate_pickle):
    warnings.filterwarnings("ignore", category=UserWarning)
    with open(xtrapolate_pickle, "rb") as file:
        # 'latin1' handles Python 2 strings and NumPy arrays correctly
        data = pickle.load(file, encoding="latin1")
    return data


def replot_xtrapol8(data, axes=None):

    (
        alphas,
        occupancies,
        pos,
        neg,
        sum,
        reference,
        pearsonCC,
        alpha,
        occ,
        alpha_CC,
        occ_CC,
        alpha_found,
    ) = data

    pos_features = np.asarray(pos) / (reference[0] if reference[0] else 1)
    neg_features = np.asarray(neg) / (reference[1] if reference[1] else 1)
    all_features = np.asarray(sum) / (reference[2] if reference[2] else 1)

    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    else:
        fig = None
    ax = axes
    ax.plot(occupancies, pos_features, "o", color="green", label="Positive features")
    ax.plot(
        occupancies,
        neg_features,
        "s",
        markersize=5,
        color="red",
        label="Negative features",
    )
    ax.plot(
        occupancies,
        all_features,
        "^",
        color="k",
        label=f"All features: Peak at {occ:.2f}",
    )
    mask = np.isclose(occupancies, occ)
    ax.scatter(
        occupancies[mask],
        all_features[mask],
        s=200,
        facecolor="none",
        color="brown",
    )
    mask = np.isclose(occupancies, occ_CC)
    ax.scatter(
        occupancies[mask],
        np.array(pearsonCC)[mask],
        s=200,
        facecolor="none",
        color="brown",
    )
    ax.set_xlabel("Triggered state occupancy")
    ax.set_ylabel("Normalized difference map signal")

    ax.plot(
        occupancies,
        pearsonCC,
        "X",
        color="blue",
        label=f"PearsonCC: Peak at {occ_CC:.2f}",
    )
    ax.set_xlabel("Triggered state occupancy")
    delta_occupancy = 0.05 * np.max(occupancies)
    ax.set_xlim(
        np.min(occupancies) - delta_occupancy, np.max(occupancies) + delta_occupancy
    )
    ax.legend()
    # ax = axes.twinx()
    return fig, axes
