import time
import numpy as np
import gemmi
import reciprocalspaceship as rs

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from scipy import stats
from scipy.signal import correlate

# import multiprocessing as mp
import logging


##################### relevant if in meteor-dev mode ###########################
import os
import sys

current_path = os.getcwd()
path_parts = current_path.split(os.sep)
idx = path_parts.index("occupancy-estimation")
homepath = os.sep.join(path_parts[: idx + 1]) + "/"
path = homepath + "../meteor/"
sys.path.append(path)
################################ end if ########################################

import meteor
from meteor import rsmap
from meteor.utils import cut_resolution
from meteor.scale import compute_scale_factors

from photolyase import get_hists, find_wasserstein_dip
from plotting3d import slice_3d, add_fit

from photolyase import load_mask_config, load_masks, make_diffmap_config
from meteorize import (
    find_most_positive_blobs_np,
    find_most_positive_blobs_rmsd,
    find_most_positive_blobs_fixed_basis,
    make_k_space_xtr,
    find_largest_blobs2,
    negsum_meteor,
    many_negsum,
    get_photolyase_maps,
    adding_maps,
)


################################################################################
############################  Helper Functions  ################################
################################################################################


def make_plot_name(filename_dict, plot_name):
    scaled = filename_dict.get("rescaling_diffmaps", False)
    fscale = "rescaled" if scaled else "not_rescaled"
    plot_path = filename_dict["diffmap_path"] + filename_dict["filestart"]
    plot_path += fscale + "_" + plot_name
    for ending in [".png"]:
        yield plot_path + ending


def title_and_saving(filename_dict, plot_title, plot_name, fig, ax=None):
    if filename_dict is not None:
        plot_title += f"\n{filename_dict['tname']}"
        plot_title += f"\n{filename_dict['diffmap_tname']}"
        plot_title += f"\n{filename_dict['diffmap_tname']}"
        scaled = filename_dict.get("rescaling_diffmaps", False)
        plot_title += " rescaled" if scaled else " not rescaled"
    if ax is None:
        fig.suptitle(plot_title)
    else:
        ax.set_title(plot_title)
    if filename_dict is not None and filename_dict.get("save_fig", False):
        for filename_final in make_plot_name(filename_dict, plot_name):
            plt.savefig(filename_final, bbox_inches="tight")
            logger.info(f"Figure saved as {filename_final}")

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
##############################  Processing  ###################################
################################################################################

############################ Single Negative Sums ##############################


def calculate_single_nse(
    diffmap: rsmap.Map,
    map_xtrs: list[rsmap.Map],
    extrapolation_factors: np.ndarray,
    map_sampling: float,
    thresholds: np.ndarray,
    blob_selection_func: callable = None,
):
    negsums = np.empty((len(thresholds), len(extrapolation_factors)))

    def fast_blobs(diffmap, thresh):
        diffmap_np = diffmap.to_3d_numpy_map(map_sampling=map_sampling)
        min_val = np.min(diffmap_np)
        neg_blob_mask = diffmap_np <= thresh * min_val
        return neg_blob_mask

    logger.warning(f"Calculating negsum for for single NSE 'manually'")
    logger.warning(
        f"This may cause slight differences to many nse version but is a lot faster'"
    )
    for ii, thresh in enumerate(thresholds):

        if False:
            neg_blob_mask = fast_blobs(diffmap, thresh)
        else:
            neg_blobs = find_largest_blobs2(
                diffmap,
                map_sampling,
                threshold=thresh,
                find_pos=False,
                blob_selection_func=blob_selection_func,
            )
            neg_blob_mask = neg_blobs > 0

        negsums[ii] = negsum_meteor(
            map_xtrs, map_sampling=map_sampling, mask=neg_blob_mask
        )
    return negsums


##############################  Many Negsum  ###################################


def calculate_many_negsum_best_guesses(
    diffmap: rsmap.Map,
    map_xtrs: list[rsmap.Map],
    extrapolation_factors: np.ndarray,
    thresholds_many: np.ndarray,
    map_sampling: float,
    blob_selection_func: callable = None,
) -> dict:

    best_guesses_dict = {
        "intersection_average": [],
        "intersection_std": [],
        "intersection_average_inv": [],
        "intersection_std_inv": [],
        "mask_counts": [],
        "mask_weight": [],
    }
    intersection_point_list = []
    intersection_weight_list = []
    for thresh in thresholds_many:
        start_time = time.time()
        neg_blobs = find_largest_blobs2(
            diffmap,
            map_sampling,
            threshold=thresh,
            find_pos=False,
            maximum_quantity=1500,
            blob_selection_func=blob_selection_func,
        )
        t1 = time.time()
        neg_blobs_masks = [(neg_blobs == idx) for idx in np.unique(neg_blobs) if idx]
        t2 = time.time()
        logger.debug("")
        logger.debug(f"seconds for finding blobs: {t1-start_time:.2f} ")
        logger.debug(f"seconds for making masks: {t2-t1:.2f} ")
        share_second = (t2 - t1) / (t2 - start_time)
        share_first = (t1 - start_time) / (t2 - start_time)
        logger.debug(f"Share First: {share_first:.2f} ")
        logger.debug(f"Share Second: {share_second:.2f} ")
        logger.debug("")
        logger.info(f"Number of Masks: {(len(neg_blobs_masks))}")
        try:
            neg_sum, intersection_points, weight = many_negsum(
                map_xtrs,
                extrapolation_factors,
                map_sampling=map_sampling,
                masks=neg_blobs_masks,
                diffmap=diffmap,
                # return_data=True,
            )

            output_dict = process_many_negsum(
                intersection_points, weight, masks=neg_blobs_masks
            )

        except NotEnoughMasksError:
            output_dict = {}

        for key in best_guesses_dict.keys():
            val = output_dict.get(key, np.nan)
            best_guesses_dict[key].append(val)

        intersection_point_list.append(intersection_points)
        intersection_weight_list.append(weight)

    best_guesses_dict["thresholds"] = thresholds_many
    best_guesses_dict["intersection_points"] = intersection_point_list
    best_guesses_dict["intersection_weight"] = intersection_weight_list
    return best_guesses_dict


from meteorize import NotEnoughMasksError, calculate_and_process_many_negsum
from meteorize import many_negsum, process_many_negsum


def calculate_many_negsum_all_lines(
    diffmap,
    map_xtrs,
    extrapolation_factors,
    map_sampling,
    thresholds=[0.35, 0.5, 0.7],
    blob_selection_func=None,
):
    processing_dicts = []
    for jj, thresh in enumerate(thresholds):
        neg_blobs = find_largest_blobs2(
            diffmap,
            map_sampling,
            threshold=thresh,
            find_pos=False,
            maximum_quantity=1500,
            blob_selection_func=blob_selection_func,
        )
        neg_blobs_masks = [(neg_blobs == idx) for idx in np.unique(neg_blobs) if idx]
        # fig, axs = plt.subplots(2,2)
        # idcs = np.linspace(0, len(map_xtrs) - 1, 4, dtype=int)
        # for ii, ax in zip(idcs, axs.flat):
        #     ax.imshow(neg_blobs[ii])
        # plt.show()
        try:
            logger.info(f"Calculating many negsum for threshold {thresh:.2f}")
            negsums, intersection_points, weight = calculate_and_process_many_negsum(
                map_xtrs,
                extrapolation_factors,
                map_sampling=map_sampling,
                masks=neg_blobs_masks,
                diffmap=diffmap,
                return_data=True,
            )
        except NotEnoughMasksError:
            arrlen = len(map_xtrs)
            masklen = len(neg_blobs_masks)
            logger.error(f"Error: No masks provided, received {masklen} masks.")
            negsums, intersection_points, weight = (
                np.empty((arrlen, masklen)) * np.nan,
                np.empty(masklen) * np.nan,
                np.empty(masklen) * np.nan,
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


################################################################################
#################################  Plotting  ###################################
################################################################################


############################ Single Negative Sums ##############################
def plot_one_nse(extrapolation_factors, negsum, ax=None):
    def get_initial_mask(alpha_invs, n_largest):
        a_sorted = np.argsort(alpha_invs)
        m_lowest = a_sorted <= n_largest
        m_biggest = a_sorted >= len(a_sorted) - n_largest
        return m_lowest, m_biggest

    def get_fits4(neg_sum, alpha_invs, m_lowest, m_biggest):
        res_lowest = stats.linregress(alpha_invs[m_lowest], neg_sum[m_lowest])
        res_biggest = stats.linregress(alpha_invs[m_biggest], neg_sum[m_biggest])
        np.linspace(np.min(alpha_invs), np.max(alpha_invs), 5)
        fit_lowest = res_lowest.intercept + res_lowest.slope * alpha_invs
        fit_biggest = res_biggest.intercept + res_biggest.slope * alpha_invs

        # intersection = (res_2.tercept-res_1.intercept) / (res_1.slope-res_2.slope)
        intersection = (res_biggest.intercept - res_lowest.intercept) / (
            res_lowest.slope - res_biggest.slope
        )

        if np.max(alpha_invs[m_lowest]) > np.max(alpha_invs[m_biggest]):
            alow = np.max(alpha_invs[m_biggest])
            ahigh = np.min(alpha_invs[m_lowest])
        else:
            alow = np.max(alpha_invs[m_lowest])
            ahigh = np.min(alpha_invs[m_biggest])
        if intersection < 0:
            logger.warning(f"Negative intersection at {intersection:.2f}")
        elif intersection < alow or intersection > ahigh:
            logger.warning(
                f"Intersection at {intersection:.2f} should be between {alow:.2f} and {ahigh:.2f}"
            )
            intersection = np.nan
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
        fit_lowest2, fit_biggest2, intersect = get_fits4(
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


from meteorize import get_fits2


def plot_single_nse_overview(
    extrapolation_factors,
    negsums,
    thresholds,
    filename_dict=None,
    plot_name=None,
):

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
            negsum, extrapolation_factors, 3, return_all=True
        )
        diff_low = -(fit_lowest2 - negsum)  # is still negative
        diff_high = -(fit_biggest2 - negsum)  # is still negative
        smaller_diff = np.min([diff_low, diff_high], axis=0)
        ax.plot( extrapolation_factors, negsum, "x", color=color)
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
    title += f"\n Median Guess: {np.median(intersects):.2f}"
    title_and_saving(filename_dict, title, plot_name, fig, ax=axs[0])
    return intersects


##############################  Many Negsum  ###################################


def plot_many_negsum_best_violin(best_guesses_dict, filename_dict=None, plot_name=None):

    thresholds_many = best_guesses_dict["thresholds_rmsd"]
    intersections = best_guesses_dict["intersection_points"]
    weights = best_guesses_dict.get("mask_weight", None)
    weights = best_guesses_dict.get("mask_weight", None)
    counts = best_guesses_dict.get("mask_counts", None)

    fig, axs = plt.subplots(3, sharex=True, tight_layout=True, figsize=(8, 10))
    ax = axs[0]
    ax.set_title("Many Negative Sums")
    # Calculate width proportional to the spacing between positions
    positions = np.array(thresholds_many)
    # Calculate minimum distance between adjacent positions
    min_spacing = np.min(np.diff(np.sort(positions)))
    # Set width as a fraction of the minimum spacing
    width_factor = 0.5  # You can adjust this factor for more/less overlap
    violin_width = min_spacing * width_factor

    logging.info(f"Violin Width: {violin_width:.4f}")
    ax.violinplot(
        [ints[np.isfinite(ints)] for ints in intersections],
        positions=positions,
        widths=violin_width,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    #     showmeans=False,
    #     showmedians=True,
    #     showextrema=True
    # )
    # ax.axhline(21, linewidth=0.5, color="k", linestyle="--")
    ax.set_ylabel("Extrapolation Factor")
    ax.set_ylim(0, None)
    ax.grid(True)
    ax = axs[1]
    ax.violinplot(
        [(1 / ints)[np.isfinite(1 / ints)] for ints in intersections],
        positions=thresholds_many,
        widths=violin_width,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    ax.set_ylabel('"Occupancy"')
    # ax.set_xlabel("Mask Threshold (Percentage of Maximum)")
    ax.set_xlabel("Mask Threshold (Peak Sigma in Blob)")
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax.grid(True)
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

    title = "Best Guess of Many Negative Sums(violin)"

    # nan_warning = mask_weights[np.isnan(intersection_points)].sum() / mask_weights.sum() > 0.2
    # if nan_warning:
    #     title += " \n(Warning: Many NaN Intersections)"
    title_and_saving(filename_dict, title, plot_name, fig, axs[0])


def plot_many_negsum_best_guesses(
    best_guesses_dict, filename_dict=None, plot_name=None
):

    thresholds_many = best_guesses_dict["thresholds_rmsd"]
    intersection_averages = best_guesses_dict["intersection_average"]
    intersection_stds = best_guesses_dict["intersection_std"]
    intersection_averages_inv = best_guesses_dict["intersection_average_inv"]
    intersection_stds_inv = best_guesses_dict["intersection_std_inv"]
    weights = best_guesses_dict.get("mask_weight", None)
    counts = best_guesses_dict.get("mask_counts", None)

    fig, axs = plt.subplots(3, sharex=True, tight_layout=True, figsize=(8, 11))
    ax = axs[0]
    ax.set_title("Many Negative Sums")
    ax.grid(True)
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
    ax.grid(True)
    ax.errorbar(
        thresholds_many,
        intersection_averages_inv,
        intersection_stds_inv,
        linestyle="",
        marker=".",
        capsize=2,
    )
    ax.set_ylabel('"Occupancy"')
    ax.set_xlabel("Mask Threshold (Peak Sigma in Blob)")
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
    title += f"\n Median Guess: {np.median(intersection_averages):.2f}"

    # nan_warning = mask_weights[np.isnan(intersection_points)].sum() / mask_weights.sum() > 0.2
    # if nan_warning:
    #     title += " \n(Warning: Many NaN Intersections)"
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
    clr = [
        "grey",
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "purple",
        "black",
        "pink",
        "brown",
        "cyan",
        "magenta",
        "lime",
        "teal",
        "navy",
        "maroon",
        "olive",
        "silver",
        "gold",
        "coral",
        "salmon",
        "plum",
        "orchid",
        "turquoise",
        "lavender",
    ]
    if colors is None:
        colors = ["grey"]
        kwargs = {
            "marker": "x",
            "linestyle": "",
            # "color": "grey",
            "alpha": 0.5,
        }
        kwargs2 = {
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
        if colors == ["grey"]:
            fit_lowest, fit_biggest, intersection = get_fits2(
                negsum, extrapolation_factors, n_largest=3
            )
            ax.plot(
                extrapolation_factors, fit_lowest, color=clr[ii % len(clr)], **kwargs2
            )
            ax.plot(
                extrapolation_factors, fit_biggest, color=clr[ii % len(clr)], **kwargs2
            )

        # ax = axs.flat[ii]

    # if share of intersection_points == nan > 20 % of weight, raise warning
    nan_warning = (
        mask_weights[np.isnan(intersection_points)].sum() / mask_weights.sum() > 0.2
    )
    if nan_warning:
        raise Warning(
            f"More than 20% of the weights are not valid for threshold {thresh:.2f}. "
        )

    # ax.set_ylim(0,0.5)
    ax = axs[1]
    bins = np.linspace(5, 15, 50)
    bins = np.linspace(0, np.max(extrapolation_factors), 140)
    # logging.info(f"Unique intersection points: {np.unique(intersection_points)}")
    ax.hist(
        intersection_points,
        bins=bins,
        weights=mask_weights,
        density=True,
        alpha=0.5,
        color=colors[jj],
        label=f"Threshold {thresh:.2f}",
    )
    # from sklearn.neighbors import KernelDensity
    # from scipy.signal import argrelextrema
    # # Fit a Kernel Density Estimation to the intersection points
    # print("intersection_points", intersection_points.shape)
    # s = np.linspace(0, np.max(intersection_points), 280)

    # for bw in [0.5, 1, 2, 3, 4]:
    #     kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(intersection_points[np.isfinite(intersection_points)].reshape(-1, 1))
    #     e = kde.score_samples(s.reshape(-1,1))
    #     print(np.max(e))
    #     ax.plot(s, e)
    #     mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    #     print(f"Bandwidth {bw}: Minima at {mi}, Maxima at {ma}")
    #     ax.plot(s[:mi[0]+1], e[:mi[0]+1], 'r',
    #     s[mi[0]:mi[1]+1], e[mi[0]:mi[1]+1], 'g',
    #     s[mi[1]:], e[mi[1]:], 'b',
    #     s[ma], e[ma], 'go',
    #     s[mi], e[mi], 'ro')

    # ax.set_xlim(2,10)

    axs[1].legend()
    axs[0].set_ylim(-0.1, 1)
    # axs[1].set_ylim(-5, 5)
    axs[0].set_ylabel("Negative Sums")
    axs[1].set_xlabel("Extrapolation Factor")
    axs[1].set_ylabel("Weighted Intersections")

    if axs_was_none:
        title = f"Many Negative Sums for Threshold {thresh:.2f}"
        if nan_warning:
            title += " \n(Warning: Many NaN Intersections)"
        plot_name = f"many_negsum_{thresh:.2f}"
        title_and_saving(
            filename_dict, plot_title=title, plot_name=plot_name, fig=fig, ax=axs[0]
        )

    # try:
    #     df_modes, df_final_modes = analyze_modes(intersection_points, 4)
    #     plot_modes(intersection_points,df_modes, df_final_modes)
    # except Exception as e:
    #     logger.warning(f"Error analyzing/plotting modes: {e}")


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


##############################  Histogram Best Guess  ##########################


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

from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde
import pandas as pd


def create_histogram_data(n=10000, mu=None, sigma=None, p=None, bins=1000):
    if mu is None:
        mu = [25, 60, 130, 190]
    if sigma is None:
        sigma = [8, 13, 15, 19]
    if p is None:
        p = [0.18, 0.2, 0.24, 0.28]

    def rmix(n, mu, sigma, p):
        k = len(mu)
        components = np.random.choice(k, size=n, p=np.array(p) / np.sum(p))
        samples = np.array([np.random.normal(mu[i], sigma[i]) for i in components])
        return samples

    x = rmix(n, mu, sigma, p)
    hist, bin_edges = np.histogram(x, bins=bins, density=True)
    return x, hist, bin_edges


def analyze_modes(x, n_modes=7):

    def find_modes(kde_x, kde_y):
        y_prev = np.roll(kde_y, -1)
        y_next = np.roll(kde_y, 1)
        peaks = (kde_y > y_prev) & (kde_y > y_next)
        # Exclude endpoints which are artifacts of np.roll
        peaks[0] = peaks[-1] = False
        return kde_x[peaks]

    kde_default = gaussian_kde(x)
    bw_default = kde_default.factor
    bw_values = bw_default * 10 ** np.linspace(1, -1, 24)

    mode_records = []
    m = np.mean(x)
    id_list = np.array([1])
    id_counter = 1

    for h in bw_values:
        kde = gaussian_kde(x, bw_method=h)
        kde_x = np.linspace(np.min(x), np.max(x), 1000)
        kde_y = kde(kde_x)
        m_new = np.sort(find_modes(kde_x, kde_y))
        if len(m_new) == 0:
            continue
        delta = np.inf if len(m_new) == 1 else np.min(np.diff(m_new)) / 2
        d = np.abs(m_new[:, None] - m)
        g = np.full(len(m_new), np.nan)
        if m.size:
            i = np.argmin(d, axis=0)
            g[i] = np.where(d[i, np.arange(len(i))] < delta, id_list, np.nan)
        unmatched = np.isnan(g)
        n_new = np.sum(unmatched)
        g[unmatched] = np.arange(id_counter + 1, id_counter + 1 + n_new)
        id_counter += n_new
        id_list = g.astype(int)
        m = m_new
        mode_records.extend(
            {"bw": h, "Mode": mode_val, "id": mode_id}
            for mode_val, mode_id in zip(m_new, id_list)
        )
    df_modes = pd.DataFrame(mode_records)
    df_modes["id"] = df_modes["id"].astype("category")
    # Plot x and density for every tenth bandwidth
    plt.figure()
    plt.hist(x, bins=100, density=True, alpha=0.3, label="Histogram")
    for idx, h in enumerate(bw_values[::10]):
        kde = gaussian_kde(x, bw_method=h)
        kde_x = np.linspace(np.min(x), np.max(x), 1000)
        kde_y = kde(kde_x)
        plt.plot(kde_x, kde_y, label=f"KDE (bw={h:.3f})")
    plt.title(f"Density Estimate at Bandwidth {h:.3f}")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    def minslope(x_vals, y_vals):
        def f(z):
            return np.interp(z, x_vals, y_vals)

        e = (np.max(x_vals) - np.min(x_vals)) * 1e-4

        def df2(z):
            return ((f(z + e) - f(z - e)) / (2 * e)) ** 2

        result = minimize_scalar(
            df2, bounds=(np.min(x_vals), np.max(x_vals)), method="bounded"
        )
        return {"bw": result.x, "slope": result.fun, "Mode": f(result.x)}

    bw_max = df_modes[df_modes["id"] == n_modes]["bw"].max()
    final_modes = []
    for i in range(1, n_modes + 1):
        Y = df_modes[(df_modes["id"] == i) & (df_modes["bw"] <= bw_max)]
        if len(Y) >= 2:
            res = minslope(Y["bw"].values, Y["Mode"].values)
            final_modes.append(res)
    df_final_modes = pd.DataFrame(final_modes)
    return df_modes, df_final_modes


# Usage example:
def plot_modes(x, df_modes, df_final_modes):
    # Plot 1: Mode Trace
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode_id, group in df_modes.groupby("id"):
        ax.plot(group["Mode"], group["bw"], label=f"Mode {mode_id}", linewidth=1.2)
    ax.scatter(
        df_final_modes["Mode"],
        df_final_modes["bw"],
        color="black",
        s=60,
        alpha=0.5,
        zorder=5,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Bandwidth (log scale)")
    ax.set_xlabel("Mode")
    ax.set_title("Mode Trace")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Histogram With Modes
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(x, bins=500, density=True, color="#2E75B2", edgecolor="none")
    for mode in df_final_modes["Mode"]:
        ax.axvline(mode, color="#D18A4e", linewidth=2)
    ax.set_title("Histogram With Modes")
    plt.tight_layout()
    plt.show()

    # x, _ ,_  = create_histogram_data()
    # analyze_modes(x, n_modes=4)


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
    for key, value_device in comparison_device.items():
        if key == "cross_correlation_coefficients":
            continue
        # if "uncertainty" in value_device:
        if value_device.get("uncertainty") is not None:
            # Plot uncertainty
            ax.errorbar(
                value_device["thresholds"],
                value_device["best_guess"],
                yerr=value_device["uncertainty"],
                fmt="o",
                capsize=2,
                label=f"{key}",
            )
        else:
            ax.plot(
                value_device["thresholds"],
                value_device["best_guess"],
                "x",
                label=key,
            )
    if "cross_correlation_coefficients" in comparison_device:
        true_values = comparison_device["cross_correlation_coefficients"]

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

###############################  Dataset Specifics ##############################


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
        "datatype": fname,
    }
    return [info_container]


def load_maxiv_paths() -> list[dict]:
    dataloc = homepath + "../data/MAXIV_ECH_new/"
    dataloc_dark = dataloc + "initial/ech-full_dark_dimple.mtz"
    dataloc_light = dataloc + "initial/ech-light_dimple.mtz"
    info_container = {
        "dataloc_dark": dataloc_dark,
        "pdbloc_dark": None,
        "dataloc_light": dataloc_light,
        "tname": "MAX IV OCP data",
        "fname": "maxiv",
        "fshort": "M4",
        "datatype": "maxiv",
        "hs_limit": 1.8,
        "map_sampling": 3,
    }
    return [info_container]


def load_ocp_paths() -> list[dict]:
    dataloc = homepath + "../data/MAXIV_ECH_new/"
    dataloc_dark = dataloc + "updated/ech-full_dark_dimple.mtz"
    dataloc_light = dataloc + "updated/ech-laser_dimple.mtz"
    info_container = {
        "dataloc_dark": dataloc_dark,
        "pdbloc_dark": None,
        "dataloc_light": dataloc_light,
        "tname": "MAX IV OCP data 2",
        "fname": "OCP",
        "fshort": "OCP",
        "datatype": "OCP",
        "hs_limit": 1.7,
        "map_sampling": 3,
    }
    return [info_container]


def load_doeke_paths() -> list[dict]:
    logger.info("Loading Doeke paths")
    homepath = load_homepath()
    folderloc = homepath + "../data/Precognition_hkls/"
    dataloc_dark = folderloc + "processing_me/combined_200_OFF.mtz"
    dataloc_light = (
        folderloc
        + "small_P1_v2/e35cdef_OFF200ns_HD_2sig_varEll.hkl_small_P1_no_sys_abs.mtz"
    )
    pdbloc_dark = None
    pdbloc_light = folderloc + "PDBs/5e11.pdb"
    tname = "Doeke"
    fname = "doeke"
    # logger.info(f"Loading dark data from {dataloc_dark}")
    info_container = {
        "dataloc_dark": dataloc_dark,
        "pdbloc_dark": pdbloc_dark,
        "dataloc_light": dataloc_light,
        "pdbloc_light": pdbloc_light,
        "hs_limit": 1.8,
        "map_sampling": 3,
        "fname": fname,
        "tname": tname,
        "fshort": fname,
        "datatype": fname,
    }
    return [info_container]


def load_inputs():
    folders1 = load_photolyase_paths()
    folders2 = load_mpro_paths()
    folders3 = load_doeke_paths()
    folders4 = load_ocp_paths()
    folders5 = load_maxiv_paths()
    folders = folders4 #+folders5  + folders1 # + folders1# + folders2 + folders3
    return folders


###############################  Map Calculation ###############################

from meteor.scale import _compute_anisotropic_scale_factors, ScaleParameters
import scipy.optimize as opt


def compute_scale_factors2(
    *,
    reference_values: rs.DataSeries,
    values_to_scale: rs.DataSeries,
    reference_uncertainties: rs.DataSeries | None = None,
    to_scale_uncertainties: rs.DataSeries | None = None,
    only_global_constant: bool = False,
) -> rs.DataSeries:
    """
    Compute anisotropic scale factors to modify `values_to_scale` to be on the same scale as
    `reference_values`.

    Following SCALEIT, the scaling model is an anisotropic model, applying a transformation of the
    form:

        C * exp{ -(h**2 B11 + k**2 B22 + l**2 B33 +
                    2hk B12 + 2hl  B13 +  2kl B23) }

    The parameters Bxy are fit using least squares, optionally with uncertainty weighting.

    Parameters
    ----------
    reference_values : rs.DataSeries
        The reference dataset against which scaling is performed, indexed by Miller indices.
    values_to_scale : rs.DataSeries
        The dataset to be scaled, also Miller indexed.
    reference_uncertainties : rs.DataSeries, optional
        Uncertainty values associated with `reference_values`. If provided, they are used in
        weighting the scaling process. Must have the same index as `reference_values`.
    to_scale_uncertainties : rs.DataSeries, optional
        Uncertainty values associated with `values_to_scale`. If provided, they are used in
        weighting the scaling process. Must have the same index as `values_to_scale`.

    Returns
    -------
    rs.DataSeries
        The computed anisotropic scale factors for each Miller index in `values_to_scale`.

    See Also
    --------
    scale_datasets : higher-level interface that operates on entire DataSets, typically more
    convienent.

    Citations:
    ----------
    [1] SCALEIT https://www.ccp4.ac.uk/html/scaleit.html
    """
    reference_values.dropna(axis="index", how="any", inplace=True)
    # keep all indices to ensure we scale all values
    index_locations = np.isfinite(values_to_scale)
    all_finite_indices = values_to_scale[index_locations].index.copy()

    values_to_scale.dropna(axis="index", how="any", inplace=True)

    common_miller_indices: pd.Index = reference_values.index.intersection(
        values_to_scale.index
    )
    common_reference_values: np.ndarray = reference_values.loc[
        common_miller_indices
    ].to_numpy()
    common_values_to_scale: np.ndarray = values_to_scale.loc[
        common_miller_indices
    ].to_numpy()

    half_root_two = np.array(
        np.sqrt(2) / 2.0
    )  # weights are one if no uncertainties provided
    ref_variance: np.ndarray = (
        np.square(reference_uncertainties.loc[common_miller_indices].to_numpy())
        if reference_uncertainties is not None
        else half_root_two
    )
    to_scale_variance: np.ndarray = (
        to_scale_uncertainties.loc[common_miller_indices].to_numpy()
        if to_scale_uncertainties is not None
        else half_root_two
    )
    inverse_variance = 1.0 / (ref_variance + to_scale_variance)

    def compute_residuals(scaling_parameters: ScaleParameters) -> np.ndarray:
        scale_factors = _compute_anisotropic_scale_factors(
            common_miller_indices,
            scaling_parameters,
        )

        difference_after_scaling = (
            scale_factors * common_values_to_scale - common_reference_values
        )
        residuals = inverse_variance * difference_after_scaling

        if not isinstance(residuals, np.ndarray):
            msg = "scipy optimizers' behavior is unstable unless `np.ndarray`s are used"
            raise TypeError(msg)

        return residuals

    def compute_constant(scale_factor: float):
        difference_after_scaling = (
            scale_factor * common_values_to_scale - common_reference_values
        )
        residuals = inverse_variance * difference_after_scaling

        if not isinstance(residuals, np.ndarray):
            msg = "scipy optimizers' behavior is unstable unless `np.ndarray`s are used"
            raise TypeError(msg)

        return residuals

    if only_global_constant:
        initial_scaling_parameter = 1.0
        optimization_result = opt.least_squares(
            compute_constant, initial_scaling_parameter
        )
        optimized_parameter = optimization_result.x[0]

        return optimized_parameter

    initial_scaling_parameters: ScaleParameters = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    optimization_result = opt.least_squares(
        compute_residuals, initial_scaling_parameters
    )
    optimized_parameters: ScaleParameters = optimization_result.x

    # now be sure to compute the scale factors for all miller indices in `values_to_scale`
    optimized_scale_factors = _compute_anisotropic_scale_factors(
        all_finite_indices,
        optimized_parameters,
    )
    if len(optimized_scale_factors) != len(all_finite_indices):
        msg1 = "length mismatch: `optimized_scale_factors`"
        msg2 = f"({len(optimized_scale_factors)}) vs `values_to_scale` ({len(all_finite_indices)})"
        raise RuntimeError(msg1, msg2)

    return optimized_scale_factors


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


from meteor.scale import scale_maps


def get_maxiv_maps(ds_dark, ds_light):
    dark_columns = {
        "amplitude_column": "F",
        "uncertainty_column": "SIGF",
        "phase_column": "PHIC",
    }
    light_columns = {
        "amplitude_column": "F",
        "uncertainty_column": "SIGF",
        "phase_column": "PHI_dark",
    }
    ds_light[light_columns["phase_column"]] = ds_dark[dark_columns["phase_column"]]
    map_dark = rsmap.Map(ds_dark, **dark_columns)
    map_light_unscaled = rsmap.Map(ds_light, **light_columns)
    from meteor.scale import scale_maps

    map_light = scale_maps(
        reference_map=map_dark,
        map_to_scale=map_light_unscaled,
    )
    return map_dark, map_light


def get_doeke_maps(ds_dark, ds_light):
    dark_columns = dict(
        amplitude_column="dark2",
        uncertainty_column="SIGF_dark2",
        phase_column="PHI_maps",
    )

    dark_columns = dict(
        amplitude_column="F",
        uncertainty_column="SigF",
        phase_column="PHI_maps",
    )
    light_columns = dict(
        amplitude_column="F",
        uncertainty_column="SigF",
        phase_column=dark_columns["phase_column"],
    )
    ds_dark2 = {
        "I": ds_dark["dark"] ** 2,
        "SIGI": 2 * ds_dark["dark"] * ds_dark["SIGF_dark"],
        "PHI_maps": ds_dark["PHI_maps"],
    }
    ds_dark2 = rs.DataSet(ds_dark2, spacegroup=ds_dark.spacegroup, cell=ds_dark.cell)
    ds_dark2 = rs.algorithms.merge(ds_dark2)
    print(ds_dark2.columns)
    # add phase from dark to light
    ds_dark2["F"] = np.sqrt(ds_dark2["IMEAN"])
    ds_dark2["SigF"] = ds_dark2["SIGIMEAN"] / (2 * ds_dark2["F"])
    ds_dark2["PHI_maps"] = ds_dark["PHI_maps"]
    # ds_light = rs.algorithms.merge(ds_light)
    ds_light[light_columns["phase_column"]] = ds_dark2[dark_columns["phase_column"]]
    unscaled_dark = rsmap.Map(ds_dark2, **dark_columns)
    unscaled_light = rsmap.Map(ds_light, **light_columns)
    scaled_light = scale_maps(reference_map=unscaled_dark, map_to_scale=unscaled_light)
    map_dark = unscaled_dark
    map_light = scaled_light
    logger.info(f"Merged maps: {map_dark.merged}, {map_light.merged}")

    return map_dark, map_light


def get_doeke_maps2(ds_dark, ds_light):
    dark_columns = dict(
        amplitude_column="dark2",
        uncertainty_column="SIGF_dark2",
        phase_column="PHI_maps",
    )

    dark_columns = dict(
        amplitude_column="F",
        uncertainty_column="SigF",
        phase_column="PHI_maps",
    )
    light_columns = dict(
        amplitude_column="F",
        uncertainty_column="SigF",
        phase_column=dark_columns["phase_column"],
    )
    ds_dark2 = {
        "I": ds_dark["dark"] ** 2,
        "SIGI": 2 * ds_dark["dark"] * ds_dark["SIGF_dark"],
        "PHI_maps": ds_dark["PHI_maps"],
    }
    ds_dark2 = rs.DataSet(ds_dark2, spacegroup=ds_dark.spacegroup, cell=ds_dark.cell)
    ds_dark2 = rs.algorithms.merge(ds_dark2)
    print(ds_dark2.columns)
    # add phase from dark to light
    ds_dark2["F"] = np.sqrt(ds_dark2["IMEAN"])
    ds_dark2["SigF"] = ds_dark2["SIGIMEAN"] / (2 * ds_dark2["F"])
    ds_dark2["PHI_maps"] = ds_dark["PHI_maps"]
    # ds_light = rs.algorithms.merge(ds_light)
    ds_light[light_columns["phase_column"]] = ds_dark2[dark_columns["phase_column"]]
    unscaled_dark = rsmap.Map(ds_dark2, **dark_columns)
    unscaled_light = rsmap.Map(ds_light, **light_columns)
    scaled_light = scale_maps(reference_map=unscaled_dark, map_to_scale=unscaled_light)
    map_dark = unscaled_dark
    map_light = scaled_light
    logger.info(f"Merged maps: {map_dark.merged}, {map_light.merged}")

    return map_dark, map_light


def calculate_scaled_maps(ds_dark, ds_light, info_container):
    if info_container["datatype"] == "photolyase":
        map_dark, map_light = get_photolyase_maps(ds_dark, ds_light)
    elif info_container["datatype"] == "mpro":
        map_dark, map_light = get_mpro_maps(ds_dark, ds_light, info_container)
    elif info_container["datatype"] == "doeke":
        map_dark, map_light = get_doeke_maps(ds_dark, ds_light)
    elif info_container["datatype"] in ["maxiv", "OCP"]:
        map_dark, map_light = get_maxiv_maps(ds_dark, ds_light)
    else:
        raise ValueError(
            f"Unknown datatype {info_container['datatype']}. Please implement map calculation."
        )
    return map_dark, map_light


#############################  Not Dataset Specifics ############################


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
        # "direct_realspace",
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

    # if filename_dict.get("rerun_old_only", False):
    #     rerun_required = filename_dict["rerun"] and file_is_old(
    #         next(plot_loc), minutes=1
    #     )
    # else:
    rerun_required = filename_dict["rerun"]

    do_run_analysis = (not file_exists) or rerun_required
    if do_run_analysis:
        logger.info(f"Running {plot_name}")
    else:
        warn_str = f"Not running {plot_name} because a file exists: {file_exists} "
        warn_str += f"\n and rerun is not required: {rerun_required}"
        logger.warning(warn_str)
    return do_run_analysis


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


def run_many_negsum_best_guess(value_device, filename_dict, plot_name):
    logger.info("Running Many NegSum - Best Guess")
    mns_best_guesses_dict = calculate_many_negsum_best_guesses(
        value_device["diffmap"],
        value_device["map_xtrs"],
        value_device["extrapolation_factors"],
        value_device["thresholds"],
        value_device["map_sampling"],
        blob_selection_func=value_device["blob_selection_func"],
    )
    mns_best_guesses_dict["thresholds_rmsd"] = value_device["thresholds_rmsd"]

    plot_many_negsum_best_violin(
        mns_best_guesses_dict,
        filename_dict=filename_dict,
        plot_name=plot_name + "_violin",
    )
    plot_many_negsum_best_guesses(
        mns_best_guesses_dict, filename_dict=filename_dict, plot_name=plot_name
    )
    best_guess = np.median(mns_best_guesses_dict["intersection_average"])
    map_xtr = adding_maps(
        value_device["diffmap"], value_device["map_dark"], factor1=best_guess
    )

    map_xtr.write_mtz(
        filename_dict["diffmap_path"]
        + filename_dict["filestart"]
        + f"_best_guess_{best_guess:.2f}.mtz"
    )

    return {
        "thresholds": mns_best_guesses_dict["thresholds"],
        "thresholds_rmsd": mns_best_guesses_dict["thresholds_rmsd"],
        "best_guess": mns_best_guesses_dict["intersection_average"],
        "uncertainty": mns_best_guesses_dict["intersection_std"],
    }


def run_many_negsum_many_thresh(value_device, filename_dict, plot_name):
    logger.info("Running Many NegSum - Best Guess")
    processing_dicts = calculate_many_negsum_all_lines(
        value_device["diffmap"],
        value_device["map_xtrs"],
        value_device["extrapolation_factors"],
        map_sampling=value_device["map_sampling"],
        thresholds=value_device["thresholds_few"],
        blob_selection_func=value_device["blob_selection_func"],
    )
    plot_many_negsum_all_lines_all_thresh(
        processing_dicts, filename_dict, plot_name=plot_name
    )
    for pd in processing_dicts:
        plot_many_negsum_all_lines(
            pd,
            filename_dict=filename_dict,
        )
    return None


def run_single_negsum_overview(value_device, filename_dict, plot_name):
    logger.info("Running Single NegSum")
    thresholds = value_device["thresholds"]
    negsums = calculate_single_nse(
        value_device["diffmap"],
        value_device["map_xtrs"],
        value_device["extrapolation_factors"],
        map_sampling=value_device["map_sampling"],
        thresholds=thresholds,
        blob_selection_func=value_device["blob_selection_func"],
    )
    # plot_single_nse_overview is a function that plots the results of the single negsum
    # and returns the intersection points

    intersects = plot_single_nse_overview(
        value_device["extrapolation_factors"],
        negsums,
        thresholds,
        filename_dict=filename_dict,
        plot_name=plot_name,
    )
    return {
        "thresholds": thresholds,
        "best_guess": intersects,
    }


def run_histograms_best_guess(value_device, filename_dict, plot_name):

    logger.info("Running Histogram Best Guess")
    thresholds = value_device["thresholds"]
    wassersteins = calculate_histogram_best_guess(
        value_device["diffmap"],
        value_device["map_xtrs"],
        value_device["map_dark"],
        map_sampling=value_device["map_sampling"],
        thresholds=thresholds,
    )
    wmins = plot_wasserstein_dists(
        value_device["extrapolation_factors"],
        thresholds,
        wassersteins,
        filename_dict,
        plot_name,
    )
    return {
        "thresholds": thresholds,
        "best_guess": wmins,
    }


def run_cross_correlation_coefficients(value_device, filename_dict, plot_name):
    logger.warning(f"masks: {value_device['masks'] is None}")
    if value_device["pdbloc_light"] is None:
        return None
    extrapolation_factors = value_device["extrapolation_factors"]
    masks = value_device.get("masks", None)

    cross_correls_all = crosscorrelation_groundtruth(
        value_device["pdbloc_light"],
        value_device["map_xtrs"],
        map_sampling=value_device["map_sampling"],
        hs_limit=value_device["hs_limit"],
    )
    true_correlations = {"all": cross_correls_all}
    comparison_true = {}
    if masks is not None:
        cross_correls_ball = crosscorrelation_groundtruth(
            value_device["pdbloc_light"],
            value_device["map_xtrs"],
            map_sampling=value_device["map_sampling"],
            hs_limit=value_device["hs_limit"],
            mask=masks["ball"],
        )
        true_correlations["ball"] = cross_correls_ball

        cross_correls_hand = crosscorrelation_groundtruth(
            value_device["pdbloc_light"],
            value_device["map_xtrs"],
            map_sampling=value_device["map_sampling"],
            hs_limit=value_device["hs_limit"],
            mask=masks["handpicked"],
        )
        true_correlations["handpicked"] = cross_correls_hand

        comparison_true = {
            "all": extrapolation_factors[np.argmax(cross_correls_all)],
            "ball": extrapolation_factors[np.argmax(cross_correls_ball)],
            "handpicked": extrapolation_factors[np.argmax(cross_correls_hand)],
        }
    else:
        comparison_true = {
            "all": extrapolation_factors[np.argmax(cross_correls_all)],
        }
    plot_true_correlations(
        true_correlations, extrapolation_factors, filename_dict, plot_name
    )
    return comparison_true


def make_thresholds(info_container):
    thresh_default_minimum = 0.25
    thresh_default_maximum = 0.95
    thresh_default_step = 0.05
    minimum_threshold = info_container.get("minimum_threshold", thresh_default_minimum)
    maximum_threshold = info_container.get("maximum_threshold", thresh_default_maximum)
    step_threshold = info_container.get("step_threshold", thresh_default_step)
    info_msg = f"Setting thresholds from {minimum_threshold} to {maximum_threshold} with step {step_threshold}"
    logger.info(info_msg)
    return np.arange(minimum_threshold, maximum_threshold, step_threshold)


def make_extrapolation_factors(info_container):
    xtr_default_min = 1
    xtr_default_mid = 10
    xtr_default_max = 100
    minimum_extrapolation_factor = info_container.get("min_xtr_factor", xtr_default_min)
    middle_extrapolation_factor = info_container.get("mid_xtr_factor", xtr_default_mid)
    maximum_extrapolation_factor = info_container.get("max_xtr_factor", xtr_default_max)
    # minmax = maximum_extrapolation_factor * 0.8
    x1 = np.linspace(minimum_extrapolation_factor, middle_extrapolation_factor, 8)
    x3 = np.linspace(middle_extrapolation_factor, maximum_extrapolation_factor, 12)
    extrapolation_factors = np.concatenate((x1, x3))
    info_msg = f"Created extrapolation factors ranging from {minimum_extrapolation_factor} to {maximum_extrapolation_factor} with mid at {middle_extrapolation_factor}"
    logger.info(info_msg)
    return extrapolation_factors


def convert_thresholds_to_rmsd(
    diffmap: rsmap.Map, thresholds: np.ndarray, map_sampling: float = 3
):
    diffmap_np = diffmap.to_3d_numpy_map(
        map_sampling=map_sampling,
    )
    diffmap_negmax = diffmap_np.min()
    diffmap_mean = diffmap_np.mean()
    diffmap_std = diffmap_np.std()
    max_sig = -(diffmap_negmax - diffmap_mean) / diffmap_std
    return thresholds * max_sig


def run_plots(
    diffmap: rsmap.Map,
    map_dark: rsmap.Map,
    info_container: dict,
    diffmap_key: str,
    filename_dict: dict = {},
    function_selection: list[str] = None,
    blob_selection_func: callable = None,
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

    -----
    """

    map_sampling = info_container.get("map_sampling", 3)
    hs_limit = info_container.get("hs_limit", 2.4)
    masks = get_pl_masks(diffmap, info_container)
    pdbloc_light = info_container.get("pdbloc_light", None)

    extrapolation_factors = make_extrapolation_factors(info_container)
    thresholds = make_thresholds(info_container)
    thresholds_few = thresholds[::4]

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

    map_xtrs = make_k_space_xtr(map_dark, diffmap, extrapolation_factors)

    comparison_device = {}
    default_bsf = find_most_positive_blobs_fixed_basis
    bsf = blob_selection_func if blob_selection_func is not None else default_bsf
    threshold_rmsd = convert_thresholds_to_rmsd(
        diffmap, thresholds, map_sampling=map_sampling
    )
    value_device = {
        "extrapolation_factors": extrapolation_factors,
        "map_sampling": map_sampling,
        "thresholds": thresholds,
        "thresholds_rmsd": threshold_rmsd,
        "thresholds_few": thresholds_few,
        "diffmap": diffmap,
        "map_xtrs": map_xtrs,
        "map_dark": map_dark,
        "hs_limit": hs_limit,
        "masks": masks,
        "pdbloc_light": pdbloc_light,
        "blob_selection_func": bsf,
    }

    # Many NegSum Best Guess
    functions_dict = {
        "many_negsum_best_guess": run_many_negsum_best_guess,
        "many_negsum_many_thresh": run_many_negsum_many_thresh,
        "single_negsum_overview": run_single_negsum_overview,
        "histograms_best_guess": run_histograms_best_guess,
        "cross_correlation_coefficients": run_cross_correlation_coefficients,
        "show_comparison": lambda *args, **kwargs: None,
    }

    if function_selection is not None:
        logger.info(f"Function selection provided: {function_selection}")
        # assert all entries in function_selection are keys in functions_dict

        assert all(
            key in functions_dict for key in function_selection
        ), f"Function selection {function_selection} contains keys not in functions_dict: {functions_dict.keys()}"
        # If function_selection is provided, filter the functions_dict
        functions_dict = {
            key: func
            for key, func in functions_dict.items()
            if key in function_selection
        }
    else:
        logger.info("No function selection provided, running all functions")

    # if info_container.get("save_extrapolated", False):
    #     # Save the extrapolated data
    #     save_extrapolated_data(value_device, filename_dict)

    for plot_name, func in functions_dict.items():
        if redo_plot(filename_dict, plot_name):
            ouput_dict = func(value_device, filename_dict, plot_name=plot_name)
            if ouput_dict is not None:
                comparison_device[plot_name] = ouput_dict

    if comparison_device != {} and filename_dict.get("show_comparison", True):
        plot_overall_comparison(comparison_device, filename_dict)
    elif comparison_device != {}:
        logger.info(
            f"Not plotting comparison - because show_comparison is set to False in {filename_dict}"
        )
    elif filename_dict.get("show_comparison", False):
        logger.warning(f"Noting to compare - because no tests were run this iteration")
    else:
        logger.info(f"Nothing to compare and not even the desire to do so")

    del value_device["diffmap"]
    del value_device["map_xtrs"]
    del value_device["map_dark"]
    del value_device["masks"]
    return value_device


def save_extrapolated_data(value_device, filename_dict):
    # Save the extrapolated maps to MTZ files
    # The extrapolated maps are stored in value_device["map_xtrs"]
    scaled = filename_dict.get("rescaling_diffmaps", False)
    fscale = "rescaled" if scaled else "_not_rescaled"
    plot_path = filename_dict["diffmap_path"] + filename_dict["filestart"]
    plot_path += fscale + "_" + "xtr"
    for extrapolation_factor, map_xtr in zip(
        value_device["extrapolation_factors"], value_device["map_xtrs"]
    ):
        # Save each extrapolated map
        logger.warning(
            f"Saving extrapolated map {extrapolation_factor:.2f} to MTZ files"
        )
        plot_path_xtr = f"{plot_path}_{extrapolation_factor:.2f}.mtz"
        map_xtr.write_mtz(plot_path_xtr)


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
        return masks


def realspace_maximum_scaling(
    reference_map: rsmap.Map,
    map_to_scale: rsmap.Map,
    map_sampling: float = 3.0,
):
    ref_map = reference_map.to_3d_numpy_map(map_sampling=map_sampling)
    scale_map = map_to_scale.to_3d_numpy_map(map_sampling=map_sampling)
    shares = [0.01, 0.005, 0.001, 0.0005]
    for share in shares:
        k = int(share * ref_map.size)
        # Get k largest values
        largest_indices = np.argpartition(ref_map.flatten(), -k)[-k:]
        largest_values_ref = ref_map.flatten()[largest_indices]
        largest_indices = np.argpartition(scale_map.flatten(), -k)[-k:]
        largest_values_scale = scale_map.flatten()[largest_indices]

        # Get k smallest values
        smallest_indices = np.argpartition(ref_map.flatten(), k)[:k]
        smallest_values_ref = ref_map.flatten()[smallest_indices]
        smallest_indices = np.argpartition(scale_map.flatten(), k)[:k]
        smallest_values_scale = scale_map.flatten()[smallest_indices]
        # Compute scaling factors using percentiles
        pos_scale = np.mean(largest_values_ref) / np.mean(largest_values_scale)
        neg_scale = np.min(smallest_values_ref) / np.min(smallest_values_scale)
        logger.info(
            f"Computed scaling factors: pos_scale={pos_scale}, neg_scale={neg_scale}"
        )
    return pos_scale, neg_scale


def rescaling_diffmaps(diffmaps: list[rsmap.Map], rescale_key: str):
    for key, diffmap in diffmaps.items():
        if key != rescale_key:
            # scaling_factor = compute_scale_factors2(
            #     reference_values=diffmaps[rescale_key].amplitudes,
            #     values_to_scale=diffmap.amplitudes,
            #     only_global_constant=True,
            # )

            pos_scale, scaling_factor = realspace_maximum_scaling(
                reference_map=diffmaps[rescale_key],
                map_to_scale=diffmap,
            )
            logger.info(f"Computed scaling factor for {key}: {scaling_factor}")
            finite_amps = np.isfinite(diffmaps[key].amplitudes)
            diffmaps[key].loc[
                finite_amps, diffmap.amplitude_column_name
            ] *= scaling_factor
            logger.info(f"Rescaled {key} to match {rescale_key}")
    return diffmaps


def main():
    # pdbloc_light, map_xtrs = load_photolyase_paths()
    filename_dict = {
        "save_fig": True,
        "display": False,
        "rerun": True,
        "rerun_old_only": False,
        "rescaling_diffmaps": True,
    }
    if filename_dict.get("rescaling_diffmaps", False):
        xtr_settings = {
            "mid_xtr_factor": 3.5,
            "max_xtr_factor": 60,
        }
    else:
        xtr_settings = {
            "mid_xtr_factor": 6,
            "max_xtr_factor": 180,
        }
    if False:
        extra_info = {
            "minimum_threshold": 4,
            "maximum_threshold": 14,
        } | xtr_settings
    extra_info = {
        "minimum_threshold": 0.25,
        "save_extrapolated": True,
    } | xtr_settings

    diffmap_ids = ["kweighted"]
    diffmap_ids = ["kweighted", "tv"]
    diffmap_ids = ["tv"]
    rescale_key = "vanilla_diffmap"
    function_selection = [
        # "single_negsum_overview",
        "many_negsum_best_guess",
        "many_negsum_many_thresh",
        # "show_comparison",
    ]
    blob_selection_func = find_most_positive_blobs_fixed_basis

    info_containers = load_inputs()
    evaluation_path = load_homepath() + "../evaluation/"
    if not os.path.exists(evaluation_path):
        os.makedirs(evaluation_path)
    outputs = []
    for info_container in info_containers:
        # logger.info(f"\n\nrunning {info_container['tname']}\n\n")

        diffmaps, map_dark, info_container, evaluation_path_basis = calculate_objects(
            info_container, evaluation_path
        )
        info_container = info_container | extra_info

        # reduce diffmaps to kweigthed and tv:
        # diffmaps = {diffmap_id: diffmaps[diffmap_id] for diffmap_id in diffmap_ids}
        logger.warning(f"Rescaling Diffmaps to match key:{rescale_key}")
        if filename_dict.get("rescaling_diffmaps", False):

            diffmaps = rescaling_diffmaps(diffmaps, rescale_key)

        for diffmap_id in diffmap_ids:
            diffmap = diffmaps[diffmap_id]
            logger.info(f"\nrunning {info_container['tname']}: {diffmap_id}\n")
            diffmap_path = evaluation_path_basis + f"{diffmap_id}/"
            filestart = f"{info_container['fshort']}_{diffmap_id}_"
            os.makedirs(diffmap_path, exist_ok=True)
            logger.info(f"diffmap_path: {diffmap_path}")
            filename_dict2 = {
                "diffmap_path": diffmap_path,
                "filestart": filestart,
            } | filename_dict

            output = run_plots(
                diffmap,
                map_dark,
                info_container,
                diffmap_id,
                filename_dict2,
                function_selection=function_selection,
                blob_selection_func=blob_selection_func,
            )
            outputs.append((info_container, output))


logger = setup_logger()

if __name__ == "__main__":

    main()
###
#TODOS
# - Rewrite Blob function to a) calculate once and b) be inherently about sigmas
# - Figure out why noise in lower levels of thresholds moves results