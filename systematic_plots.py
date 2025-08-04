import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import os, sys
current_path = os.getcwd()
path_parts = current_path.split(os.sep)
idx = path_parts.index("occupancy-estimation")
homepath = os.sep.join(path_parts[:idx + 1]) +"/"
path = homepath + "../meteor/"
# sys.path.append(path)

from meteor import rsmap
from meteor.utils import cut_resolution
from meteor.scale import compute_scale_factors

from compare_conds import get_intersect_and_angle
from photolyase import get_hists, plot_hists, find_wasserstein_dip
from plotting3d import slice_3d, add_fit
from meteorize import *
from meteorize import many_negsum

import os
from scipy import stats

from skimage.feature import match_template
import multiprocessing as mp
import logging

################################################################################
##############################  Helper Functions  ###################################
################################################################################


def make_plot_name(filename_dict, plot_name):
    plot_path = filename_dict["diffmap_path"] + filename_dict["filestart"]
    plot_path += "_" + plot_name
    for ending in [".png", ".pdf"]:
        yield plot_path + ending


def title_and_saving(filename_dict, plot_title, plot_name, fig, ax=None):
    if filename_dict is not None:
        plot_title += f"\n{filename_dict['tname']}"
        plot_title += f"\n{filename_dict['diffmap_tname']}"
    if ax is None:
        fig.suptitle(plot_title)
    else:
        ax.set_title(plot_title)
    if filename_dict is not None and filename_dict.get("save_fig", False):
        for filename_final in make_plot_name(filename_dict, plot_name):
            plt.savefig(filename_final, bbox_inches="tight")
            # only print the last 80 characters of the filename
            logger.info(f"Figure saved as {filename_final}")
            logger.info(
                f"Figure saved with Diffmap title {filename_dict['diffmap_tname']}"
            )

    if filename_dict.get("display", False):
        plt.show()
    else:
        plt.close(fig)


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # datefmt = "%Y-%m-%d %H:%M:%S"
    datefmt = "%H:%M:%S"
    format = "%(levelname)s %(asctime)s - %(message)s (%(filename)s:%(lineno)d)"
    format = (
        "%(asctime)s: %(name)s: %(levelname)s: %(message)s (%(filename)s:%(lineno)d)"
    )
    format = (
        "%(asctime)s: %(name)s: %(levelname)s: %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        # format = "%(asctime)s: %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
        # datefmt = "%H:%M:%S"
        log_fmt = self.FORMATS.get(record.levelno)
        # Indent line breaks in the message to align with end of levelname and time
        levelname_len = len(
            record.levelname
        )  # + len(record.asctime) + 3  # levelname + space + time + ' - '
        # asctime will be formatted as time only (HH:MM:SS)
        # record.asctime = self.formatTime(record, "%H:%M:%S")
        indent = " " * (levelname_len + 12 + 12 + 2)  # levelname + space + time + ' - '
        if record.msg and isinstance(record.msg, str):
            record.msg = record.msg.replace("\n", "\n" + indent)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(log_level=logging.DEBUG):
    """
    Set up the logger with a custom formatter.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(CustomFormatter())
    root_logger.handlers = []  # Remove any default handlers
    root_logger.addHandler(ch)
    logging.getLogger("generate_objects").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    return logger


################################################################################
##############################  Many Negsum  ###################################
################################################################################


def calculate_single_nse(
    diffmap: rsmap.Map,
    map_xtrs: list[rsmap.Map],
    extrapolation_factors: np.ndarray,
    map_sampling: float,
    thresholds: np.ndarray,
):
    negsums = np.empty((len(thresholds), len(extrapolation_factors)))
    for ii, thresh in enumerate(thresholds):
        # print("running ", thresh)
        neg_blobs = find_largest_blobs2(
            diffmap,
            map_sampling,
            threshold=thresh,
            find_pos=False,
        )
        neg_blobs_masks = neg_blobs > 0
        negsums[ii] = negsum_meteor(
            map_xtrs, map_sampling=map_sampling, mask=neg_blobs_masks
        )
    return negsums


def plot_one_nse(extrapolation_factors, negsum, ax=None):
    def get_initial_mask(alpha_invs, n_largest):
        a_sorted = np.argsort(alpha_invs)
        m_lowest = a_sorted <= n_largest
        m_biggest = a_sorted >= len(a_sorted) - n_largest
        return m_lowest, m_biggest

    def get_fits2(neg_sum, alpha_invs, m_lowest, m_biggest):
        res_lowest = stats.linregress(alpha_invs[m_lowest], neg_sum[m_lowest])
        res_biggest = stats.linregress(alpha_invs[m_biggest], neg_sum[m_biggest])
        np.linspace(np.min(alpha_invs), np.max(alpha_invs), 5)
        fit_lowest = res_lowest.intercept + res_lowest.slope * alpha_invs
        fit_biggest = res_biggest.intercept + res_biggest.slope * alpha_invs

        # intersection = (res_2.tercept-res_1.intercept) / (res_1.slope-res_2.slope)
        intersection = (res_biggest.intercept - res_lowest.intercept) / (
            res_lowest.slope - res_biggest.slope
        )
        return fit_lowest, fit_biggest, intersection

    # Create a ScalarMappable for the colorbar
    comp_low = 0.1
    ax_was_none = ax is None
    if ax_was_none:
        fig, ax = plt.subplots(1, figsize=(8, 4))

    number_reruns = 4
    norm = mcolors.Normalize(vmin=0, vmax=number_reruns)
    cmap = cm.viridis
    negsum = negsum / np.min(negsum)
    ax.plot(extrapolation_factors, negsum, "x")
    m_biggest, m_lowest = get_initial_mask(extrapolation_factors, 3)
    for ii in range(number_reruns):
        logger.info(f"Lowest: {np.sum(m_lowest)}, Biggest: {np.sum(m_biggest)}")
        color = cmap(norm(ii))  # Map threshold to color
        fit_lowest2, fit_biggest2, intersect = get_fits2(
            negsum, extrapolation_factors, m_lowest, m_biggest
        )
        logger.info(f"Intersection: {intersect:.2f}")
        diff_low = -(fit_lowest2 - negsum)  # is still negative
        diff_high = -(fit_biggest2 - negsum)  # is still negative
        smaller_diff = np.min([diff_low, diff_high], axis=0)
        peak_diff = smaller_diff[(np.argmax(smaller_diff))]
        m_lowest = diff_low < peak_diff * comp_low
        m_biggest = diff_high < peak_diff * comp_low
        if ax_was_none:
            ax.plot(extrapolation_factors, fit_lowest2, "-", color=color)
            ax.plot(extrapolation_factors, fit_biggest2, "-", color=color)
    ax.set_ylabel("Normalized Negative Sum")
    ax.set_xlabel("Extrapolation Factor")


def plot_single_nse_overview(
    extrapolation_factors,
    negsums,
    thresholds,
    filename_dict=None,
    plot_name=None,
):

    def get_fits2(neg_sum, alpha_invs, n_largest):
        a_sorted = np.argsort(alpha_invs)
        m_lowest = a_sorted <= n_largest
        m_biggest = a_sorted >= len(a_sorted) - n_largest
        res_lowest = stats.linregress(alpha_invs[m_lowest], neg_sum[m_lowest])
        res_biggest = stats.linregress(alpha_invs[m_biggest], neg_sum[m_biggest])
        np.linspace(np.min(alpha_invs), np.max(alpha_invs), 5)
        fit_lowest = res_lowest.intercept + res_lowest.slope * alpha_invs
        fit_biggest = res_biggest.intercept + res_biggest.slope * alpha_invs

        # intersection = (res_2.tercept-res_1.intercept) / (res_1.slope-res_2.slope)
        intersection = (res_biggest.intercept - res_lowest.intercept) / (
            res_lowest.slope - res_biggest.slope
        )

        return fit_lowest, fit_biggest, intersection

    # Normalize the threshold values to [0, 1] for the colormap
    norm = mcolors.Normalize(vmin=min(thresholds), vmax=max(thresholds))
    cmap = cm.viridis

    # Create a ScalarMappable for the colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.close("all")
    fig, axs = plt.subplots(
        3,
        figsize=(8, 10),
    )
    intersects = []
    for ii, (thresh, negsum) in enumerate(zip(thresholds, negsums)):
        ax = axs[0]
        negsum = negsum / np.min(negsum)
        # ax = axs.flat[ii]
        color = cmap(norm(thresh))  # Map threshold to color
        ax.plot(extrapolation_factors, negsum, "x", color=color)
        # plt.plot(extrapolation_factors, negsum)
        fit_lowest2, fit_biggest2, intersect = get_fits2(
            negsum, extrapolation_factors, 3
        )
        diff_low = -(fit_lowest2 - negsum)  # is still negative
        diff_high = -(fit_biggest2 - negsum)  # is still negative
        smaller_diff = np.min([diff_low, diff_high], axis=0)
        ax.plot(
            extrapolation_factors,
            negsum,
            "x",
            color=color,
        )
        ax = axs[1]
        ax.plot(extrapolation_factors, fit_lowest2, "-", color=color)
        ax.plot(
            extrapolation_factors,
            fit_biggest2,
            "-",
            color=color,
        )
        intersects.append(intersect)
    ax = axs[2]
    ax.set_ylabel("Intersection Point")
    ax.set_xlabel("Mask Threshold")
    ax.scatter(thresholds, intersects, c=thresholds, cmap=cmap, norm=norm)
    ax = axs[0]
    ax.set_ylabel("Normalized Negative Sum")
    ax.set_xlabel("Extrapolation Factor")
    ax = axs[1]
    ax.set_ylabel("Normalized Negative Sum")
    ax.set_xlabel("Optimal Extrapolation Factor")
    # ax2.plot(extrapolation_factors,smaller_diff/negsum[np.argmax(smaller_diff)])

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Mask Threshold (Share of Max)")
    title = "Single Negative Sums for various Thresholds"
    title_and_saving(filename_dict, title, plot_name, fig, ax=axs[0])
    return intersects


################################################################################
##############################  Many Negsum  ###################################
################################################################################


def calculate_many_negsum_best_guesses(
    diffmap: rsmap.Map,
    map_xtrs: list[rsmap.Map],
    extrapolation_factors: np.ndarray,
    thresholds_many: np.ndarray,
    map_sampling: float,
) -> dict:

    intersection_tuples = []
    mask_tuples = []
    for thresh in thresholds_many:
        # print("running ", thresh)
        neg_blobs = find_largest_blobs2(
            diffmap,
            map_sampling,
            threshold=thresh,
            find_pos=False,
            maximum_quantity=1500,
        )
        neg_blobs_masks = [(neg_blobs == idx) for idx in np.unique(neg_blobs) if idx]
        logger.info(f"Number of Masks: {(len(neg_blobs_masks))}")
        intersection_tuple, mask_tuple = many_negsum(
            map_xtrs,
            extrapolation_factors,
            map_sampling=map_sampling,
            masks=neg_blobs_masks,
            detailed=True,
            diffmap=diffmap,
        )
        intersection_tuples.append(intersection_tuple)
        mask_tuples.append(mask_tuple)

    intersection_tuples = np.array(intersection_tuples).T
    mask_tuples = np.array(mask_tuples).T
    best_guesses_dict = {
        "intersection_average": intersection_tuples[0],
        "intersection_std": intersection_tuples[1],
        "intersection_average_inv": intersection_tuples[2],
        "intersection_std_inv": intersection_tuples[3],
        "thresholds": thresholds_many,
        "mask_counts": mask_tuples[0],
        "mask_weight": mask_tuples[1],
    }
    return best_guesses_dict


def calculate_many_negsum_all_lines(
    diffmap,
    map_xtrs,
    extrapolation_factors,
    map_sampling,
    thresholds=[0.35, 0.5, 0.7],
):
    processing_dicts = []
    for jj, thresh in enumerate(thresholds):
        neg_blobs = find_largest_blobs2(
            diffmap,
            map_sampling,
            threshold=thresh,
            find_pos=False,
            maximum_quantity=1500,
        )
        neg_blobs_masks = [(neg_blobs == idx) for idx in np.unique(neg_blobs) if idx]
        _, negsums, intersection_points, weight = many_negsum(
            map_xtrs,
            extrapolation_factors,
            map_sampling=map_sampling,
            masks=neg_blobs_masks,
            return_neg_sum=True,
            detailed=True,
            diffmap=diffmap,
        )

        proccesing_dict = {
            "threshold": thresh,
            "extrapolation_factors": extrapolation_factors,
            "negsums": negsums,
            "intersection_points": intersection_points,
            "mask_weight": weight,
        }

        processing_dicts.append(proccesing_dict)
    return processing_dicts


def plot_many_negsum_best_guesses(
    best_guesses_dict, filename_dict=None, plot_name=None
):

    thresholds_many = best_guesses_dict["thresholds"]
    intersection_averages = best_guesses_dict["intersection_average"]
    intersection_stds = best_guesses_dict["intersection_std"]
    intersection_averages_inv = best_guesses_dict["intersection_average_inv"]
    intersection_stds_inv = best_guesses_dict["intersection_std_inv"]
    weights = best_guesses_dict.get("mask_weight", None)
    counts = best_guesses_dict.get("mask_counts", None)

    fig, axs = plt.subplots(3, sharex=True, tight_layout=True)
    ax = axs[0]
    ax.set_title("Many Negative Sums")
    ax.errorbar(
        thresholds_many,
        intersection_averages,
        intersection_stds,
        linestyle="",
        marker=".",
        capsize=2,
    )
    # ax.axhline(21, linewidth=0.5, color="k", linestyle="--")
    ax.set_ylabel("Extrapolation Factor")
    ax.set_ylim(0, None)
    ax = axs[1]
    ax.errorbar(
        thresholds_many,
        intersection_averages_inv,
        intersection_stds_inv,
        linestyle="",
        marker=".",
        capsize=2,
    )
    ax.set_ylabel('"Occupancy"')
    ax.set_xlabel("Mask Threshold (Percentage of Maximum)")
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax = axs[2]
    ax.plot(thresholds_many, counts, "o-k", label="Mask Count")
    ax2 = ax.twinx()
    ln = ax2.plot(thresholds_many, weights, "o-b", label="Mask Weight")
    # add ln to legend of axs[2]
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.set_ylabel("Mask Count")
    ax2.set_ylabel("Mask Weight")
    ax.set_ylim(0, None)
    ax2.set_ylim(0, None)

    title = "Best Guess of Many Negative Sums"

    title_and_saving(filename_dict, title, plot_name, fig, axs[0])


def plot_many_negsum_all_lines(
    proccesing_dict,
    colors=None,
    axs=None,
    jj=0,
    filename_dict=None,
):

    axs_was_none = axs is None
    if axs_was_none:
        fig, axs = plt.subplots(2, figsize=(10, 6), sharex=True)
    extrapolation_factors = proccesing_dict["extrapolation_factors"]
    negsums = proccesing_dict["negsums"]
    intersection_points = proccesing_dict["intersection_points"]
    mask_weights = proccesing_dict["mask_weight"]
    thresh = proccesing_dict["threshold"]
    if colors is None:
        colors = ["grey"]
        kwargs = {
            "marker": "x",
            "linestyle": "",
            # "color": "grey",
            "alpha": 0.5,
        }
    else:
        kwargs = {
            "color": colors[jj],
            "linewidth": 0.2 + jj * 0.3,
        }
    for ii, negsum in enumerate(negsums.T):
        ax = axs[0]
        negsum = negsum / np.min(negsum)  # normalize
        ax.plot(extrapolation_factors, negsum, **kwargs)
    ax = axs[1]
    bins = np.linspace(0, 100, 50)
    ax.hist(
        intersection_points,
        bins=bins,
        weights=mask_weights,
        density=True,
        alpha=0.5,
        color=colors[jj],
        label=f"Threshold {thresh:.2f}",
    )
    axs[1].legend()
    axs[0].set_ylabel("Negative Sums")
    axs[1].set_xlabel("Extrapolation Factor")
    axs[1].set_ylabel("Weighted Intersections")

    if axs_was_none:
        title = f"Many Negative Sums for Threshold {thresh:.2f}"
        plot_name = f"many_negsum_{thresh:.2f}"
        title_and_saving(
            filename_dict, plot_title=title, plot_name=plot_name, fig=fig, ax=axs[0]
        )


def plot_many_negsum_all_lines_all_thresh(
    processing_dicts, filename_dict=None, plot_name=None
):
    """Plot the results of many_negsum."""
    fig, axs = plt.subplots(2, figsize=(10, 6), sharex=True)
    colors = "red", "orange", "yellow", "green", "blue", "purple"
    for jj, processing_dict in enumerate(processing_dicts):
        plot_many_negsum_all_lines(processing_dict, axs=axs, colors=colors, jj=jj)
    title = "Many Negative Sums for various Thresholds"
    title_and_saving(
        filename_dict, plot_title=title, plot_name=plot_name, fig=fig, ax=axs[0]
    )


################################################################################
##############################  Histogram Best Guess  ##########################
################################################################################


def calculate_histogram_best_guess(
    diffmap: rsmap.Map,
    map_xtrs: list[rsmap.Map],
    map_dark: rsmap.Map,
    map_sampling: float,
    thresholds: np.ndarray,
):
    dens_xtrs = [
        map_xtr.to_3d_numpy_map(map_sampling=map_sampling) for map_xtr in map_xtrs
    ]
    logger.info(f"{len(dens_xtrs)=}, {len(map_xtrs)=}")
    rho_dark = map_dark.to_3d_numpy_map(map_sampling=map_sampling)
    rmin = np.min(rho_dark)
    rmax = np.max(rho_dark)
    offset = (rmax - rmin) * 0.5
    bins = np.linspace(rmin - offset, rmax + offset, 100)
    len_dens = len(dens_xtrs)
    logger.info(f"{len_dens=}, {len(bins)=}")

    wassersteins = []
    for ii, thresh in enumerate(thresholds):
        # print("running ", thresh)
        pos_blobs, neg_blobs = find_largest_blobs2(
            diffmap, map_sampling, threshold=thresh, maximum_quantity=1500
        )
        mask = np.logical_or(neg_blobs > 0, pos_blobs > 0)
        if mask.any():
            _, _, wdists, bin_centers = get_hists(dens_xtrs, rho_dark, mask, bins)
        else:
            wdists = np.zeros(len(dens_xtrs)) * np.nan

        wassersteins.append(wdists)
    return wassersteins


def plot_wasserstein_dists(
    extrapolation_factors, thresholds, wassersteins, filename_dict=None, plot_name=None
):
    # Normalize the threshold values to [0, 1] for the colormap
    norm = mcolors.Normalize(vmin=min(thresholds), vmax=max(thresholds))
    cmap = cm.viridis

    # Create a ScalarMappable for the colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    # fig, ax = plt.subplots()

    fig, axs = plt.subplots(2, figsize=(8, 8))
    wmins = []
    for ii, (thresh, wdists) in enumerate(zip(thresholds, wassersteins)):
        ax = axs[0]
        try:
            wmin = find_wasserstein_dip(extrapolation_factors, wdists)
        except ValueError as e:
            logger.exception(
                f"Wasserstein dip mismatch: {len(extrapolation_factors)=}, {len(wdists)=}"
            )
        wmins.append(wmin)
        # ax = axs.flat[ii]
        color = cmap(norm(thresh))  # Map threshold to color
        wdists = wdists / np.max(wdists)
        ax.plot(extrapolation_factors, wdists, "-", color=color)
        ax = axs[1]
        if wmin:
            ax.plot(thresh, wmin, ".", color=color)
    ax = axs[0]
    ax.set_ylabel("Wasserstein Distance")
    ax.set_xlabel("Extrapolation Factor")
    ax = axs[1]
    ax.set_ylabel("Best Extrapolation Factor")
    ax.set_xlabel("Mask Threshold (Percentage of Maximum)")
    title = "Wasserstein Distances for various Thresholds"
    # plot_name = "histograms_best_guess"
    title_and_saving(filename_dict, title, plot_name, fig, ax=axs[0])
    return wmins


################################################################################
################################################################################
################################################################################


def structure2realspace(struc, hs_limit, noise_level):
    map_vals = meteor.sfcalc.gemmi_structure_to_calculated_map(
        struc, high_resolution_limit=hs_limit
    )
    struc_vals = rs.DataSet(map_vals)
    noise = np.random.normal(loc=0.0, scale=noise_level * np.abs(struc_vals["F"]))
    struc_vals["F"] = struc_vals["F"] + noise
    struc_vals["sf"] = struc_vals.to_structurefactor("F", "PHI")
    struc_vals_grid = struc_vals.to_reciprocal_grid("sf")
    struc_vals_real = np.fft.fftn(struc_vals_grid).real / struc_vals_grid.size
    return struc_vals_real


################################################################################
################################################################################
################################################################################

from scipy.signal import correlate


def compute_correlation_coefficient(image1, image2):
    matched = correlate(image1, image2, mode="valid")
    norm1 = correlate(image1, image1, mode="valid")
    norm2 = correlate(image2, image2, mode="valid")
    matched = (matched / np.sqrt(norm1 * norm2)).flatten()[0]
    return matched


def crosscorrelation_groundtruth(
    pdbloc_light, map_xtrs, *, map_sampling, hs_limit, mask=None
):
    struc_light = gemmi.read_structure(pdbloc_light)
    density_light_ground = structure2realspace(struc_light, hs_limit, 0)

    density_light_ground
    density_xtrs = [
        map_xtr.to_3d_numpy_map(map_sampling=map_sampling) for map_xtr in map_xtrs
    ]
    if mask is not None:
        density_xtrs = [density_xtr[mask] for density_xtr in density_xtrs]
        density_light_ground = density_light_ground[mask]
    cross_correls = []
    for ii, density_xtr in enumerate(density_xtrs):
        matched = compute_correlation_coefficient(density_light_ground, density_xtr)
        cross_correls.append(matched)
    return np.array(cross_correls)


def plot_true_correlations(
    true_correlations, extrapolation_factors, filename_dict, plot_name
):
    fig, ax = plt.subplots(figsize=(8, 6))
    for key, cross_correls in true_correlations.items():
        ax.plot(
            extrapolation_factors,
            cross_correls,
            label=key,
            marker="o",
            linestyle="--",
        )
    ax.set_xlabel("Extrapolation Factor")
    ax.set_ylabel("Cross-Correlation Coefficient")
    ax.legend()
    title = f"Cross-Correlation Coefficients"
    title_and_saving(
        filename_dict, plot_title=title, plot_name=plot_name, fig=fig, ax=ax
    )
    plt.show()


def plot_overall_comparison(comparison_device, filename_dict):
    fig, ax = plt.subplots(figsize=(8, 6))
    for key, value_dict in comparison_device.items():
        if key == "true":
            continue
        # if "uncertainty" in value_dict:
        if value_dict.get("uncertainty") is not None:
            # Plot uncertainty
            ax.errorbar(
                value_dict["thresholds"],
                value_dict["best_guess"],
                yerr=value_dict["uncertainty"],
                fmt="o",
                capsize=2,
                label=f"{key}",
            )
        else:
            ax.plot(
                value_dict["thresholds"],
                value_dict["best_guess"],
                "x",
                label=key,
            )
    if "true" in comparison_device:
        true_values = comparison_device["true"]

        # get colors from set1
        colors = cm.get_cmap("Set1", len(true_values))
        for ii, (key, thresh) in enumerate(true_values.items()):
            ax.axhline(
                thresh,
                linestyle="--",
                label=f"Best CC for {key}",
                color=colors(ii),
            )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Best Extrapolation Factor")
    ax.legend()
    title = f"Comparison of different Methods"
    plot_name = "comparison_of_methods"
    title_and_saving(
        filename_dict, plot_title=title, plot_name=plot_name, fig=fig, ax=ax
    )


################################################################################
##############################  Iteration Function #############################
################################################################################


# def load_photolyase_paths():
def load_homepath():
    # This function returns the path in which t
    current_path = os.getcwd()
    path_parts = current_path.split(os.sep)
    idx = path_parts.index("occupancy-estimation")
    homepath = os.sep.join(path_parts[: idx + 1]) + "/"
    return homepath


def load_photolyase_paths() -> list[dict]:
    logger.info("Loading Photolyase paths")
    homepath = load_homepath()
    folderloc = homepath + "../photolyase_share_reviewers/"
    dataloc_dark = folderloc + "1_superdark/superdark_deposit.mtz"
    pdbloc_dark = folderloc + "1_superdark/superdark_deposit.pdb"

    mask_loc_sphere = (
        homepath + "../photolyase_masks/photolyase_TT_chainA_spherical_mask.ccp4"
    )
    mask_loc_handcraft = (
        homepath + "../photolyase_masks/photolyase_chainA_spherical_masks.ccp4"
    )
    # logger.info(f"Loading dark data from {dataloc_dark}")
    # loads all folders in the folderloc
    info_containers = []
    for f in os.listdir(folderloc):
        if not os.path.isdir(os.path.join(folderloc, f)):
            continue
        if f[:2] == "1_":
            continue
        if not f[0].isdigit():
            continue
        changing_bit = f + "/" + f.split("_")[-1]
        dataloc_light = folderloc + changing_bit + "_deposit.mtz"
        pdbloc_light = folderloc + changing_bit + "_deposit.pdb"

        timepoint = changing_bit.split("/")[-1]
        fname = "photolyase_" + timepoint
        tname = "Photolyase " + timepoint
        info_container = {
            "dataloc_dark": dataloc_dark,
            "pdbloc_dark": pdbloc_dark,
            "dataloc_light": dataloc_light,
            "pdbloc_light": pdbloc_light,
            "mask_loc_sphere": mask_loc_sphere,
            "mask_hand": mask_loc_handcraft,
            "fname": fname,
            "tname": tname,
            "fshort": "PL" + timepoint,
            "hs_limit": 2.4,
            "map_sampling": 3,
            "datatype": "photolyase",
        }
        info_containers.append(info_container)
    for info_container in info_containers:
        logger.info(f"Loaded {info_container['tname']} with {info_container['fname']}")
    return info_containers


def load_mpro_paths() -> list[dict]:
    logger.info("Loading MPro paths")
    homepath = load_homepath()
    folderloc = homepath + "../data/meteor_data/"
    dataloc_dark = folderloc + "k.mtz"
    dataloc_light = folderloc + "on.mtz"
    folderloc = homepath + "../meteor/test/data/"
    dataloc_dark = folderloc + "scaled-test-data.mtz"
    dataloc_light = folderloc + "scaled-test-data.mtz"
    pdbloc_light = folderloc + "8a6g.pdb"
    pdbloc_dark = folderloc + "8a6g-chromophore-removed.pdb"
    fname = "mpro"
    tname = "MPRO"
    info_container = {
        "dataloc_dark": dataloc_dark,
        "pdbloc_dark": pdbloc_dark,
        "dataloc_light": dataloc_light,
        "pdbloc": pdbloc_light,
        "hs_limit": 2.4,
        "map_sampling": 3,
        "fname": fname,
        "tname": tname,
        "fshort": fname,
        "datatype": "mpro",
    }
    return [info_container]


def load_inputs():
    folders = load_photolyase_paths()
    folders2 = load_mpro_paths()
    folders = folders + folders2
    return folders


from photolyase import make_diffmap_config


def loading_diffmaps(
    map_light: rsmap.Map,
    map_dark: rsmap.Map,
    map_sampling: float,
    path: str,
    force_compute=False,
):
    """
    Loads the difference maps for the dark and light datasets.
    """
    diffmap_config = make_diffmap_config(map_sampling)
    map_types = [
        "direct_realspace",
        "vanilla_diffmap",
        "kweighted",
        "tv",
    ]
    mtz_name = f"{path}diffmaps.mtz"
    diffmaps = {}
    if not os.path.exists(mtz_name) or force_compute:
        logger.info(f"Calculating from maps")
        for key in map_types:
            config = diffmap_config[key]
            loader = config["loader"]
            kwargs = config.get("kwargs", {})
            diffmaps[key] = loader(map_light, map_dark, **kwargs)

        diffmaps_mtz = rs.DataSet(diffmaps["tv"])
        diffmaps_mtz = diffmaps_mtz.rename(
            columns=lambda x: f"{x}_tv" if x != "index" else x
        )
        for key in diffmaps.keys():
            if key != "tv":
                for col in diffmaps[key].columns:
                    if col != "index":
                        diffmaps_mtz[f"{col}_{key}"] = diffmaps[key][col]
        rs.DataSet(diffmaps_mtz).write_mtz(mtz_name)

    else:
        logger.info(f"Reading from {mtz_name}")
        diffmaps_mtz = rs.read_mtz(mtz_name)
        for key in map_types:
            diffmap = rsmap.Map(
                diffmaps_mtz,
                amplitude_column=f"F_{key}",
                phase_column=f"PHI_{key}",
                uncertainty_column=f"SigF_{key}",
            )
            diffmaps[key] = diffmap
    return diffmaps, diffmap_config


def add_phi_from_pdb(ds_dark, pdbloc, hs_limit, new_phi_name="PHI"):
    # 2. Calculate the map from the structure using gemmi (with slightly lower resolution)
    struc_dark = gemmi.read_structure(pdbloc)
    map_dark_comp = meteor.sfcalc.gemmi_structure_to_calculated_map(
        struc_dark, high_resolution_limit=hs_limit - 0.7
    )

    # 3. Convert the calculated map to a reciprocalspaceship DataSet and expand to P1 and anomalous
    ds_dark_comp = rs.DataSet(map_dark_comp)
    ds_dark_comp = ds_dark_comp.expand_to_p1()
    ds_dark_comp = ds_dark_comp.expand_anomalous()

    # 4. Filter calculated map indices to match those in the experimental map
    not_in_dark = ds_dark_comp.index.difference(ds_dark.index)
    ds_dark_comp_filtered = ds_dark_comp[~ds_dark_comp.index.isin(not_in_dark)]

    ds_dark.sort_index(inplace=True)
    ds_dark_comp_filtered.sort_index(inplace=True)

    # 5. Inspect the DataFrames for further analysis or debugging

    ds_dark[new_phi_name] = ds_dark_comp_filtered["PHI"]
    return ds_dark


def get_mpro_maps(ds_dark, ds_light, info_container):
    dark_columns = {
        "amplitude_column": "F_off",
        "uncertainty_column": "SIGF_off",
        "phase_column": "PHI",
    }
    light_columns = {
        "amplitude_column": "F_on",
        "uncertainty_column": "SIGF_on",
        "phase_column": "PHI",
    }
    ds_dark = add_phi_from_pdb(
        ds_dark, info_container["pdbloc_dark"], info_container["hs_limit"]
    )
    ds_light["PHI"] = ds_dark["PHI"]
    map_dark = rsmap.Map(ds_dark, **dark_columns)
    map_light = rsmap.Map(ds_light, **light_columns)
    return map_dark, map_light


def calculate_scaled_maps(ds_dark, ds_light, info_container):
    if info_container["datatype"] == "photolyase":
        map_dark, map_light = get_scaled_maps(ds_dark, ds_light)
    if info_container["datatype"] == "mpro":
        map_dark, map_light = get_mpro_maps(ds_dark, ds_light, info_container)
    return map_dark, map_light


def calculate_objects(info_container, evaluation_path, choose_diffmaps=None):
    dataloc_dark = info_container["dataloc_dark"]
    dataloc_light = info_container["dataloc_light"]

    hs_limit = info_container["hs_limit"]
    map_sampling = info_container["map_sampling"]

    ds_light = rs.read_mtz(dataloc_light)
    ds_dark = rs.read_mtz(dataloc_dark)

    ds_dark = cut_resolution(ds_dark, high_resolution_limit=hs_limit)
    ds_light = cut_resolution(ds_light, high_resolution_limit=hs_limit)

    map_dark, map_light = calculate_scaled_maps(ds_dark, ds_light, info_container)

    evaluation_path_basis = evaluation_path + info_container["fname"] + "/"

    if not os.path.exists(evaluation_path_basis):
        os.makedirs(evaluation_path_basis)

    diffmaps, diffmap_config = loading_diffmaps(
        map_dark=map_dark,
        map_light=map_light,
        path=evaluation_path_basis,
        map_sampling=map_sampling,
    )
    info_container["diffmap_config"] = diffmap_config
    return (diffmaps, map_dark, info_container, evaluation_path_basis)


def file_is_old(filename, days=0, hours=0, minutes=0):
    """
    Check if the file was modified within the last specified time period.
    """
    if days == 0 and hours == 0 and minutes == 0:
        raise ValueError("At least one time unit must be greater than zero.")
    if not os.path.exists(filename):
        return False
    file_mod_time = os.path.getmtime(filename)
    current_time = time.time()
    delta_seconds = days * 86400 + hours * 3600 + minutes * 60
    return (current_time - file_mod_time) > delta_seconds


def redo_plot(filename_dict, plot_name):
    if filename_dict["display"]:
        return True
    plot_loc = make_plot_name(filename_dict, plot_name)
    file_exists = os.path.exists(next(plot_loc))

    if filename_dict.get("rerun_old_only", False):
        rerun_required = filename_dict["rerun"] and file_is_old(
            next(plot_loc), minutes=1
        )
    else:
        rerun_required = filename_dict["rerun"]

    do_run_analysis = (not file_exists) or rerun_required
    if do_run_analysis:
        logger.info(f"Running {plot_name}")
    else:
        warn_str = f"Not running {plot_name} because a file exists: {file_exists} "
        warn_str += f"\n and rerun is not required: {rerun_required}"
        logger.warning(warn_str)
    return do_run_analysis


def run_plots(
    diffmap: rsmap.Map,
    map_dark: rsmap.Map,
    info_container: dict,
    diffmap_key: str,
    filename_dict: dict = {},
):
    """
    Generates and saves a series of systematic plots for occupancy estimation analysis.

    This function performs multiple analyses and visualizations using the provided difference map,
    dark map, and associated metadata. It computes extrapolated maps, negative sum statistics,
    histogram-based metrics, and cross-correlation with ground truth data, then plots the results.

    Parameters
    ----------
    diffmap : np.ndarray
        The difference map to be analyzed.
    map_dark : np.ndarray
        The dark map used for extrapolation and comparison.
    info_container : dict
        Dictionary containing metadata and configuration for the analysis, including keys such as
        'map_sampling', 'hs_limit', 'masks', 'pdbloc_light', 'tname', 'diffmap_config', and 'fname'.
    diffmap_key : str
        Identifier or filename for the difference map.

    Returns
    -------
    None
        This function generates and saves plots but does not return any value.

    Notes
    -----
    The function expects several plotting and calculation utilities to be available in the scope,
    such as `make_k_space_xtr`, `calculate_many_negsum_best_guesses`, `plot_many_negsum_best_guesses`,
    `calculate_many_negsum_all_lines`, `plot_many_negsum_all_lines_all_thresh`, `plot_many_negsum_all_lines`,
    `calculate_single_nse`, `plot_single_nse_overview`, `calculate_histogram_best_guess`,
    `plot_wasserstein_dists`, `crosscorrelation_groundtruth`, and `plot_true_correlations`.
    """

    map_sampling = info_container.get("map_sampling", 3)
    hs_limit = info_container.get("hs_limit", 2.4)
    masks = info_container.get("masks", None)
    pdbloc_light = info_container.get("pdbloc_light", None)

    filename_dict = (
        {
            "tname": info_container["tname"],
            "diffmap_fname": diffmap_key,
            "fname": info_container["fname"],
            "fshort": info_container["fshort"],
        }
        | filename_dict
        | {
            "diffmap_tname": info_container["diffmap_config"][diffmap_key]["title"],
        }
    )

    extrapolation_factors = np.arange(1, 100, 2.5)
    map_xtrs = make_k_space_xtr(map_dark, diffmap, extrapolation_factors)
    threshold_default_minimum = 0.25
    threshold_default_maximum = 0.95
    if info_container.get("minimum_threshold", None) is None:
        logger.info(
            f"Minimum Threshold not set, using default {threshold_default_minimum}"
        )
    if info_container.get("maximum_threshold", None) is None:
        logger.info(
            f"Maximum Threshold not set, using default {threshold_default_maximum}"
        )

    minimum_threshold = info_container.get(
        "minimum_threshold", threshold_default_minimum
    )
    maximum_threshold = info_container.get(
        "maximum_threshold", threshold_default_maximum
    )

    thresholds = np.arange(minimum_threshold, maximum_threshold, 0.05)
    thresholds_few = thresholds[2::4]
    comparison_device = {}

    # Many NegSum
    # if

    plot_name = "many_negsum_best_guess"
    if redo_plot(filename_dict, plot_name):
        logger.info("Running Many NegSum - Best Guess")
        mns_best_guesses_dict = calculate_many_negsum_best_guesses(
            diffmap, map_xtrs, extrapolation_factors, thresholds, map_sampling
        )
        plot_many_negsum_best_guesses(
            mns_best_guesses_dict, filename_dict=filename_dict, plot_name=plot_name
        )
        comparison_device["many_negsum"] = {
            "thresholds": thresholds,
            "best_guess": mns_best_guesses_dict["intersection_average"],
            "uncertainty": mns_best_guesses_dict["intersection_std"],
        }

    plot_name = "many_negsum_many_thresh"
    if redo_plot(filename_dict, plot_name):
        logger.info("Running Many NegSum - Best Guess")
        processing_dicts = calculate_many_negsum_all_lines(
            diffmap,
            map_xtrs,
            extrapolation_factors,
            map_sampling=map_sampling,
            thresholds=thresholds_few,
        )

        plot_many_negsum_all_lines_all_thresh(
            processing_dicts, filename_dict, plot_name=plot_name
        )
        plot_many_negsum_all_lines(
            processing_dicts[0],
            filename_dict=filename_dict,
        )
        plot_many_negsum_all_lines(
            processing_dicts[1],
            filename_dict=filename_dict,
        )

    # Single NegSum
    plot_name = "single_negsum_overview"
    if redo_plot(filename_dict, plot_name):
        logger.info("Running Single NegSum")
        negsums = calculate_single_nse(
            diffmap, map_xtrs, extrapolation_factors, map_sampling, thresholds
        )
        intersects = plot_single_nse_overview(
            extrapolation_factors,
            negsums,
            thresholds,
            filename_dict=filename_dict,
            plot_name=plot_name,
        )
        comparison_device["negsum"] = {
            "thresholds": thresholds,
            "best_guess": intersects,
        }

    # histogram
    logger.info("Running Histogram Best Guess")
    plot_name = "histograms_best_guess"
    if redo_plot(filename_dict, plot_name):
        wassersteins = calculate_histogram_best_guess(
            diffmap, map_xtrs, map_dark, map_sampling, thresholds
        )
        wmins = plot_wasserstein_dists(
            extrapolation_factors, thresholds, wassersteins, filename_dict, plot_name
        )
        comparison_device["histogram"] = {
            "thresholds": thresholds,
            "best_guess": wmins,
        }

    plot_name = "cross_correlation_coefficients"
    # logger.info(f"pdbloc_light: {pdbloc_light}")
    # logger.info(f"redo_plot: {redo_plot(filename_dict, plot_name)}")
    logger.warning(f"masks: {masks is None}")

    if pdbloc_light is not None and redo_plot(filename_dict, plot_name):
        cross_correls_all = crosscorrelation_groundtruth(
            pdbloc_light, map_xtrs, map_sampling=map_sampling, hs_limit=hs_limit
        )
        true_correlations = {"all": cross_correls_all}
        if masks is not None:
            cross_correls_ball = crosscorrelation_groundtruth(
                pdbloc_light,
                map_xtrs,
                map_sampling=map_sampling,
                hs_limit=hs_limit,
                mask=masks["ball"],
            )
            true_correlations["ball"] = cross_correls_ball

            cross_correls_hand = crosscorrelation_groundtruth(
                pdbloc_light,
                map_xtrs,
                map_sampling=map_sampling,
                hs_limit=hs_limit,
                mask=masks["handpicked"],
            )
            true_correlations["handpicked"] = cross_correls_hand

            comparison_device["true"] = {
                "all": extrapolation_factors[np.argmax(cross_correls_all)],
                "ball": extrapolation_factors[np.argmax(cross_correls_ball)],
                "handpicked": extrapolation_factors[np.argmax(cross_correls_hand)],
            }
        else:
            comparison_device["true"] = {
                "all": extrapolation_factors[np.argmax(cross_correls_all)],
            }

        plot_true_correlations(
            true_correlations, extrapolation_factors, filename_dict, plot_name
        )
    if comparison_device != {}:
        plot_overall_comparison(comparison_device, filename_dict)
        pass
    else:
        logger.warning(f"Noting to compare - because no tests were run this iteration")


from photolyase import load_mask_config, load_masks


def get_pl_masks(diffmap, info_container):
    map_sampling = info_container["map_sampling"]
    mask_types = ["ball", "handpicked"]
    if info_container["datatype"] == "photolyase":
        mask_loc_sphere = info_container["mask_loc_sphere"]
        mask_loc_handcraft = info_container["mask_hand"]
        mask_configs = load_mask_config(
            diffmap, map_sampling, mask_loc_sphere, mask_loc_handcraft
        )
        masks = load_masks(diffmap, map_sampling, mask_configs, mask_types=mask_types)
        info_container["masks"] = masks

def rescaling_diffmaps(diffmaps:list[rsmap.Map], rescale_key:str):
    for key, diffmap in diffmaps.items():
        if key != rescale_key:
            # print(diffmap.head(10))
            # print(diffmaps[rescale_key].head(10))
            scaling_factors = compute_scale_factors(
                reference_values=diffmaps[rescale_key].amplitudes,
                values_to_scale=diffmap.amplitudes,
            )
            finite_amps = np.isfinite(diffmaps[key].amplitudes)
            diffmaps[key].loc[finite_amps,diffmap.amplitude_column_name] *= scaling_factors
            logger.info(f"Rescaled {key} to match {rescale_key}")


def main():
    # pdbloc_light, map_xtrs = load_photolyase_paths()
    filename_dict = {
        "save_fig": True,
        "display": False,
        "rerun": True,
        "rerun_old_only": False,
    }
    extra_info = {"minimum_threshold": 0.25}
    diffmap_ids = ["vanilla_diffmap", "kweighted", "tv"]
    rescale_key = "vanilla_diffmap"

    info_containers = load_inputs()
    evaluation_path = load_homepath() + "../evaluation/"
    if not os.path.exists(evaluation_path):
        os.makedirs(evaluation_path)

    for info_container in info_containers:
        logger.info(f"\n\nrunning {info_container["tname"]}\n\n")

        diffmaps, map_dark, info_container, evaluation_path_basis = calculate_objects(
            info_container, evaluation_path
        )
        info_container = info_container | extra_info

        # reduce diffmaps to kweigthed and tv:
        diffmaps = {diffmap_id: diffmaps[diffmap_id] for diffmap_id in diffmap_ids}
        logger.warning(f"Rescaling Diffmaps to match key:{rescale_key}")
        rescaling_diffmaps(diffmaps, rescale_key)


        for diffmap_id, diffmap in diffmaps.items():
            logger.info(f"\nrunning {info_container["tname"]}: {diffmap_id}\n")
            diffmap_path = evaluation_path_basis + f"{diffmap_id}/"
            filestart = f"{info_container['fshort']}_{diffmap_id}_"
            os.makedirs(diffmap_path, exist_ok=True)
            logger.info(f"diffmap_path: {diffmap_path}")
            filename_dict = {
                "diffmap_path": diffmap_path,
                "filestart": filestart,
            } | filename_dict
            get_pl_masks(diffmap, info_container)

            run_plots(diffmap, map_dark, info_container, diffmap_id, filename_dict)


if __name__ == "__main__":

    logger = setup_logger()
    main()
