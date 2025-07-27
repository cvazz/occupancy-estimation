import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from meteor import rsmap
from meteor.utils import cut_resolution

from meteor.diffmaps import (
    compute_difference_map,
    max_negentropy_kweighted_difference_map,
)
from meteor.tv import tv_denoise_difference_map
from meteor.validate import map_negentropy

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
logger = logging.getLogger(__name__)

################################################################################
##############################  Helper Functions  ###################################
################################################################################


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
        pos_blobs, neg_blobs = find_largest_blobs(
            diffmap, map_sampling, threshold=thresh
        )
        neg_blobs_masks = neg_blobs > 0
        negsums[ii] = negsum_meteor(
            map_xtrs, map_sampling=map_sampling, mask=neg_blobs_masks
        )
    return negsums


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
    for thresh in thresholds_many:
        # print("running ", thresh)
        pos_blobs, neg_blobs = find_largest_blobs(
            diffmap, map_sampling, threshold=thresh
        )
        neg_blobs_masks = [(neg_blobs == idx) for idx in np.unique(neg_blobs) if idx]
        print(len(neg_blobs_masks))
        intersection_tuple = many_negsum(
            map_xtrs,
            extrapolation_factors,
            map_sampling=map_sampling,
            masks=neg_blobs_masks,
            detailed=True,
        )
        intersection_tuples.append(intersection_tuple)

    intersection_tuples = np.array(intersection_tuples).T
    best_guesses_dict = {
        "intersection_average": intersection_tuples[0],
        "intersection_std": intersection_tuples[1],
        "intersection_average_inv": intersection_tuples[2],
        "intersection_std_inv": intersection_tuples[3],
        "thresholds": thresholds_many,
    }
    return best_guesses_dict


def calculate_many_negsum_all_lines(
    diffmap,
    map_xtrs,
    extrapolation_factors,
    map_sampling,
):
    thresholds = [0.35, 0.5]
    processing_dicts = []
    for jj, thresh in enumerate(thresholds):
        _, neg_blobs = find_largest_blobs(diffmap, map_sampling, threshold=thresh)
        neg_blobs_masks = [(neg_blobs == idx) for idx in np.unique(neg_blobs) if idx]
        _, negsums, intersection_points = many_negsum(
            map_xtrs,
            extrapolation_factors,
            map_sampling=map_sampling,
            masks=neg_blobs_masks,
            return_neg_sum=True,
            detailed=True,
        )

        proccesing_dict = {
            "threshold": thresh,
            "extrapolation_factors": extrapolation_factors,
            "negsums": negsums,
            "intersection_points": intersection_points,
            "neg_blobs_masks": neg_blobs_masks,
        }

        processing_dicts.append(proccesing_dict)
    return processing_dicts


def plot_many_negsum_best_guesses(best_guesses_dict, filename_dict=None, plot_name=None):

    thresholds_many = best_guesses_dict["thresholds"]
    intersection_averages = best_guesses_dict["intersection_average"]
    intersection_stds = best_guesses_dict["intersection_std"]
    intersection_averages_inv = best_guesses_dict["intersection_average_inv"]
    intersection_stds_inv = best_guesses_dict["intersection_std_inv"]

    fig, axs = plt.subplots(2, sharex=True)
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
    ax.axhline(21, linewidth=0.5, color="k", linestyle="--")
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
    title = "Best Guess of Many Negative Sums"

    title_and_saving(filename_dict, title, plot_name, fig, axs[0])


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
            logger.info(f"Figure saved as {filename_final[-80:]}")

    if filename_dict.get("display", False):
        plt.show()
    else:
        plt.close(fig)


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
    neg_blobs_masks = proccesing_dict["neg_blobs_masks"]
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
        print(colors)
        kwargs = {
            "color": colors[jj],
            "linewidth": 0.2 + jj * 0.3,
        }
    for ii, negsum in enumerate(negsums.T):
        ax = axs[0]
        negsum = negsum / np.min(negsum)  # normalize
        ax.plot(extrapolation_factors, negsum, **kwargs)
    ax = axs[1]
    weights = np.sum(neg_blobs_masks, axis=(1, 2, 3))
    bins = np.linspace(0, 100, 50)
    ax.hist(
        intersection_points,
        bins=bins,
        weights=weights,
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


def plot_many_negsum_all_lines_all_thresh(processing_dicts, filename_dict=None, plot_name=None):
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
    rho_dark = map_dark.to_3d_numpy_map(map_sampling=map_sampling)
    rmin = np.min(rho_dark)
    rmax = np.max(rho_dark)
    offset = (rmax - rmin) * 0.5
    bins = np.linspace(rmin - offset, rmax + offset, 100)

    wassersteins = []
    for ii, thresh in enumerate(thresholds):
        # print("running ", thresh)
        pos_blobs, neg_blobs = find_largest_blobs(
            diffmap, map_sampling, threshold=thresh
        )
        mask = np.logical_or(neg_blobs > 0, pos_blobs > 0)
        _, _, wdists, bin_centers = get_hists(dens_xtrs, rho_dark, mask, bins)
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
        wmin = find_wasserstein_dip(extrapolation_factors, wdists)
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


def plot_true_correlations(true_correlations, extrapolation_factors, filename_dict, plot_name):
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
        for key, thresh in true_values.items():
            ax.axhline(
                thresh,
                linestyle="--",
                label=f"Best CC for {key}",
                # color="black",
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
    logger.info(f"Loading dark data from {dataloc_dark}")
    # loads all folders in the folderloc
    info_containers = []
    for f in os.listdir(folderloc):
        if not os.path.isdir(os.path.join(folderloc, f)):
            continue
        if f[:1] == "1_":
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
            "pdbloc": pdbloc_light,
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

    return info_containers[::-1]


def load_mpro_paths() -> list[dict]:
    homepath = load_homepath()
    folderloc = homepath + "../data/meteor_data/"
    dataloc_dark = folderloc + "k.mtz"
    dataloc_light = folderloc + "on.mtz"
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


def get_mpro_maps(ds_dark, ds_light):
    dark_columns = {
        "amplitude_column": "F_off",
        "uncertainty_column": "SIGF_off",
        "phase_column": "PHI_off",
    }
    dark_columns = {
        "amplitude_column": "F_k",
        "uncertainty_column": "SIGF_k",
        "phase_column": "PHI_k",
    }
    light_columns = {
        "amplitude_column": "F_on",
        "uncertainty_column": "SIGF_on",
        "phase_column": "PHI",
    }
    ds_light["PHI"] = ds_dark["PHI_k"]
    map_dark = rsmap.Map(ds_dark, **dark_columns)
    map_light = rsmap.Map(ds_light, **light_columns)
    return map_dark, map_light


def calculate_scaled_maps(ds_dark, ds_light, info_container):
    if info_container["datatype"] == "photolyase":
        map_dark, map_light = get_scaled_maps(ds_dark, ds_light)
    if info_container["datatype"] == "mpro":
        map_dark, map_light = get_mpro_maps(ds_dark, ds_light)
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


def redo_plot(filename_dict, plot_name):
    plot_loc = make_plot_name(filename_dict, plot_name)
    do_run_analysis = not os.path.exists(next(plot_loc)) and filename_dict.get("rerun", False)
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
    filename_dict = {
        "tname": info_container["tname"],
        "diffmap_tname": info_container["diffmap_config"][diffmap_key]["title"],
        "diffmap_fname": diffmap_key,
        "fname": info_container["fname"],
        "fshort": info_container["fshort"],
    } | filename_dict

    extrapolation_factors = np.arange(1, 100, 2.5)
    map_xtrs = make_k_space_xtr(map_dark, diffmap, extrapolation_factors)

    thresholds_many = np.arange(0.25, 0.75, 0.05)
    thresholds = np.arange(0.25, 0.8, 0.05)
    comparison_device = {}

    # Many NegSum
    logger.info("Running Many NegSum")
    # if 

    plot_name = "many_negsum_best_guess"
    if redo_plot(filename_dict, plot_name):
        mns_best_guesses_dict = calculate_many_negsum_best_guesses(
            diffmap, map_xtrs, extrapolation_factors, thresholds_many, map_sampling

        )
        plot_many_negsum_best_guesses(mns_best_guesses_dict, filename_dict=filename_dict)
        comparison_device["many_negsum"] = {
            "thresholds": thresholds_many,
            "best_guess": mns_best_guesses_dict["intersection_average"],
            "uncertainty": mns_best_guesses_dict["intersection_std"],
        }

    plot_name = "many_negsum_many_thresh"
    if redo_plot(filename_dict, plot_name):
        processing_dicts = calculate_many_negsum_all_lines(
            diffmap,
            map_xtrs,
            extrapolation_factors,
            map_sampling=map_sampling,
        )

        plot_many_negsum_all_lines_all_thresh(processing_dicts, filename_dict, plot_name=plot_name)
        plot_many_negsum_all_lines(
            processing_dicts[0],
            filename_dict=filename_dict,
        )
        plot_many_negsum_all_lines(
            processing_dicts[1],
            filename_dict=filename_dict,
        )

    # Single NegSum
    logger.info("Running Single NegSum")
    plot_name = "single_negsum_overview"
    if redo_plot(filename_dict, plot_name):
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
        plot_true_correlations(true_correlations, extrapolation_factors, filename_dict)
    if comparison_device != {}:
        plot_overall_comparison(comparison_device, filename_dict)
        pass
    else:
        print("Noting to compare")


def main():
    # pdbloc_light, map_xtrs = load_photolyase_paths()
    info_containers = load_inputs()
    evaluation_path = load_homepath() + "../evaluation/"
    if not os.path.exists(evaluation_path):
        os.makedirs(evaluation_path)

    for info_container in info_containers:

        diffmaps, map_dark, info_container, evaluation_path_basis = calculate_objects(
            info_container, evaluation_path
        )

        # reduce diffmaps to kweigthed and tv:
        diffmaps = {
            "kweighted": diffmaps["kweighted"],
            "tv": diffmaps["tv"],
        }
        for diffmap_id, diffmap in diffmaps.items():
            diffmap_path = evaluation_path_basis + f"{diffmap_id}/"
            filestart = f"{info_container['fshort']}_{diffmap_id}_"
            os.makedirs(diffmap_path, exist_ok=True)
            logger.info(f"diffmap_path: {diffmap_path}")
            filename_dict = {
                "save_fig": True,
                "display": False,
                "diffmap_path": diffmap_path,
                "filestart": filestart,
            }
            run_plots(diffmap, map_dark, info_container, diffmap_id, filename_dict)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
