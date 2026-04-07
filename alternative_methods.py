import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import copy 
import gemmi
import pickle
import warnings
import shutil
import reciprocalspaceship as rs
from multiprocessing import Pool
from functools import partial
import pandas as pd

from matplotlib.ticker import MaxNLocator

from meteor import rsmap
from meteor.sfcalc import gemmi_structure_to_calculated_map
from meteor.utils import cut_resolution

from configuration import load_homepath, minimal_masking_config
from configuration import get_file_config_diff_only
from logger import setup_logger
from masking import make_inclusion_mask
from estimation import plot_extrapolation_estimate_new

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
    if np.isclose(res_lowest.slope, res_biggest.slope):
        intersection = np.nan
    else:
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


def plot_negative_density_trends(
    neg_density_plot, neg_density_fit, figargs={}, ax=None
):
    xtr_range_plot = neg_density_plot["xtr_range"]
    neg_dens_plot = neg_density_plot["neg_dens"]
    xtr_range_fit = neg_density_fit["xtr_range"]
    neg_dens_fit = neg_density_fit["neg_dens"]
    fit_subtext = figargs.get("title", "")
    fit_text = f"Fit: {fit_subtext}" if fit_subtext else "Linear Fit"
    color = figargs.get("color", "blue")
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.plot(xtr_range_plot, -neg_dens_plot, "x", color=color)

    fit_lowest2, fit_biggest2, intersect, intersect_y = get_fits2(
        -neg_dens_fit, xtr_range_fit, 3, return_all=True
    )
    fit_text += f" {intersect:.2f} (i.e. {1/intersect:.2f})"
    ax.plot(xtr_range_fit, fit_lowest2, "--", color=color)
    ax.plot(
        xtr_range_fit,
        fit_biggest2,
        "--",
        color=color,
        label=fit_text,
    )
    ax.scatter(
        intersect,
        intersect_y,
        s=200,
        facecolor="none",
        color="brown",
        label="Intersection",
    )
    if figargs.get("legend", False):
        ax.legend(loc="upper left")
    else:
        ax.legend(loc="center left")
    ax.set_xlabel("Extrapolation factor")
    ax.set_ylabel("Negative density sum")
    xmax = np.max(xtr_range_plot) * 1.1
    ymax = np.max(-neg_dens_plot) * 1.1
    ymax = max(ymax, intersect_y * 2)
    xmax = max(xmax, intersect * 1.2)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-ymax / 50, ymax)
    return fig, ax


def nse_analysis(diffmap, map_dark, inclusion_mask, figargs={}, ax=None):
    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=3)
    map_dark_np = map_dark.to_3d_numpy_map(map_sampling=3)
    xtr_show_max = figargs.get("xtr_show_max", 15)
    xtr_range_show = np.arange(1, xtr_show_max + 1)
    xtr_range_fit = np.concatenate(
        (np.linspace(1.4, 0.8, 8), xtr_range_show, np.linspace(50, 80, 8))
    )
    neg_density_fit = _calculate_negative_density_trends(
        diffmap_np, map_dark_np, inclusion_mask, xtr_range_fit
    )
    neg_density_plot = _calculate_negative_density_trends(
        diffmap_np, map_dark_np, inclusion_mask, xtr_range_show
    )
    if ax is None:
        fig, ax = plt.subplots()

    return plot_negative_density_trends(
        neg_density_plot, neg_density_fit, figargs, ax=ax
    )


def compact_nse(diffmap, map_dark, inclusion_mask, figargs={}, ax=None):
    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=3)
    map_dark_np = map_dark.to_3d_numpy_map(map_sampling=3)

    xtr_range_fit = np.concatenate((np.linspace(1.4, 0.8, 8), np.linspace(50, 80, 8)))
    neg_density_fit = _calculate_negative_density_trends(
        diffmap_np, map_dark_np, inclusion_mask, xtr_range_fit
    )
    fit_lowest2, fit_biggest2, intersect, intersect_y = get_fits2(
        -neg_density_fit["neg_dens"], neg_density_fit["xtr_range"], 3, return_all=True
    )
    return 1 / intersect


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


def compact_pandda_results(pandda_dict, alpha=None):
    pseudo_occupancy = pandda_dict["xtr_range"]
    mean_local = pandda_dict["mean_local"]
    mean_global = pandda_dict["mean_global"]

    def find_max(mean_global, mean_local):
        mean_diff = mean_global - mean_local
        pk_val_idx = np.argmax(mean_diff)
        return pseudo_occupancy[pk_val_idx]

    max_trad = find_max(mean_global, mean_local)
    mean_local_improved = mean_local * (1 + np.sign(mean_local)) / 2
    max_improved = find_max(mean_global, mean_local_improved)
    return max_trad, max_improved


def plot_pandda_results(pandda_dict, alpha=None, axs=None, improved=False):
    pseudo_occupancy = pandda_dict["xtr_range"]
    mean_local = pandda_dict["mean_local"]
    mean_global = pandda_dict["mean_global"]

    if axs is None:
        fig, axs = plt.subplots(2, figsize=(8, 4), sharex=True)
    else:
        fig = None

    ax = axs[0]
    scatter_kwargs = dict(
        s=200,
        facecolor="none",
        color="brown",
    )
    if alpha is not None:
        ax.axvline(alpha, c="k", linestyle="-.", label="alpha_true")
    mean_diff = mean_global - mean_local
    ax.plot(pseudo_occupancy, +mean_diff, label="global-local")

    pk_val_idx = np.argmax(mean_diff)
    pk_val_narrow = pseudo_occupancy[pk_val_idx]
    # pseudo_occ = np.argwhere(pseudo_occupancy == pk_val_narrow)[0]
    scatter_kwargs["label"] = f"Peak at {pk_val_narrow:.2f}"
    ax.scatter(pseudo_occupancy[pk_val_idx], mean_diff[pk_val_idx], **scatter_kwargs)

    # ax.axvline(pseudo_occupancy[pk_val_idx], color="green", label=f"Peak at {pk_val_narrow:.2f}")
    ax.set_title("PanDDA method")
    ax = axs[1]
    if alpha is not None:
        ax.axvline(alpha, c="k", linestyle="-.", label="alpha_true")
    ax.plot(pseudo_occupancy, mean_local, label="local")
    ax.plot(pseudo_occupancy, mean_global, label="global")
    ax.set_ylim(-1, 1)

    if improved:
        mean_local_alt = mean_local * (1 + np.sign(mean_local)) / 2
        mean_diff = mean_global - mean_local_alt
        axs[0].plot(pseudo_occupancy, +mean_diff, label="global - non-neg. local")

        pk_val_idx = np.argmax(mean_diff)
        pk_val_narrow = pseudo_occupancy[pk_val_idx]
        scatter_kwargs["label"] = f"Peak at {pk_val_narrow:.2f}"
        axs[0].scatter(
            pseudo_occupancy[pk_val_idx], mean_diff[pk_val_idx], **scatter_kwargs
        )
        axs[1].plot(pseudo_occupancy, mean_local_alt, label="local \n(nonnegative)")
    for ax in axs:
        ax.legend()
    return fig, axs


################################## Xtrapol8 #####################################
def load_xtrapol8_data(xtrapolate_pickle):
    warnings.filterwarnings("ignore", category=UserWarning)
    with open(xtrapolate_pickle, "rb") as file:
        # 'latin1' handles Python 2 strings and NumPy arrays correctly
        data = pickle.load(file, encoding="latin1")
    return data


def compact_x8(data):
    (_, _, _, _, _, _, _, _, occ, _, occ_CC, _) = data
    return occ, occ_CC


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
    pos_feature_kwarg = dict(marker="o", color="green", label="Positive features")
    neg_feature_kwarg = dict(
        marker="s", color="red", label="Negative features", markersize=5
    )
    all_feature_kwarg = dict(
        marker="^", color="k", label=f"All features: Peak at {occ:.2f}"
    )

    ax.plot(occupancies, pos_features, **pos_feature_kwarg)
    ax.plot(occupancies, neg_features, **neg_feature_kwarg)
    ax.plot(occupancies, all_features, **all_feature_kwarg)
    # "^",
    # color="k",
    # label=f"All features: Peak at {occ:.2f}",
    mask = np.isclose(occupancies, occ)
    correct_occ_kwarg = dict(s=200, facecolor="none", color="brown")

    ax.scatter(occupancies[mask], all_features[mask], **correct_occ_kwarg)
    mask = np.isclose(occupancies, occ_CC)
    ax.scatter(occupancies[mask], np.array(pearsonCC)[mask], **correct_occ_kwarg)
    ax.set_xlabel("Triggered state occupancy")
    ax.set_ylabel("Normalized difference map signal")

    cc_kwarg = dict(marker="X", color="blue", label=f"PearsonCC: Peak at {occ_CC:.2f}")
    ax.plot(occupancies, pearsonCC, **cc_kwarg)
    ax.set_xlabel("Triggered state occupancy")
    delta_occupancy = 0.05 * np.max(occupancies)
    ax.set_xlim(
        np.min(occupancies) - delta_occupancy, np.max(occupancies) + delta_occupancy
    )
    ax.legend()
    # ax = axes.twinx()
    return fig, axes




################################################################################

def cleanup_data(folder_path):
    actual_folder = list(folder_path.keys())[0]
    print("Cleaning up data for folder: ", actual_folder)
    difference_mtz = load_homepath() + "occupancy-estimation/" + actual_folder + "run/x8x8x8_mFoFo.mtz"
    filepaths = [difference_mtz]
    out_folder = load_homepath() + "occupancy-estimation/" + actual_folder
    for ending in ['pickle', 'png', 'pdf']:
        file_path = load_homepath() + "occupancy-estimation/" + actual_folder + "run/alpha_occupancy_determination_Fextr." + ending
        filepaths.append(file_path)
    for file_path in filepaths:
        shutil.copy(file_path, out_folder)
    run_folder = load_homepath() + "occupancy-estimation/" + actual_folder + "run/"
    shutil.rmtree(run_folder)

    # Path(run_folder).unlink()
    


def prepare_data(folder_path, reference_pdb, dmin):
    actual_folder = list(folder_path.keys())[0]
    options = folder_path[actual_folder]
    reference_mtz = load_homepath() + "occupancy-estimation/" + actual_folder + "map_dark.mtz"
    difference_mtz = load_homepath() + "occupancy-estimation/" + actual_folder + "x8x8x8_mFoFo.mtz"

    name_machine = "sim_rsEFGP2"

    col_dict = {
        "amplitude_column": "F",
        "phase_column": "PHIC",
        "uncertainty_column": "SIGF",
    }
    config = get_file_config_diff_only(
        reference_mtz,
        difference_mtz,
        reference_pdb,
        col_dict,
        col_dict,
        name_machine = name_machine,
        high_resolution_limit=dmin,
    )
    struc_dark = gemmi.read_pdb(reference_pdb)
    map_dark_comp = gemmi_structure_to_calculated_map(
        struc_dark, high_resolution_limit=dmin,
    )
    # config["diffmap_path"] = diffmap_path
    config["masking"]["exclude_large_occupancy_outliers"]= False
    config["masking"]["dark_size_threshold"]= -1

    ds_dark = rs.read_mtz(reference_mtz)
    ds_dark["PHIC"] = map_dark_comp.phases

    map_dark = rsmap.Map(ds_dark, amplitude_column="F", phase_column="PHIC") # type: ignore
    comp_F000 = map_dark_comp.loc[(0,0,0), "F"]
    map_dark.loc[(0,0,0), ["F", "PHIC"]] = [comp_F000, 0]
    map_dark = cut_resolution(map_dark, high_resolution_limit=dmin)
    map_dark.sort_index(inplace=True)

    ds_diff = rs.read_mtz(difference_mtz)
    diffmap_columns = dict(amplitude_column="FOFOWT", phase_column="PHIFOFOWT")
    diffmap = rsmap.Map(ds_diff, **diffmap_columns) # type: ignore


    xtrapolate_pickle = load_homepath() + "occupancy-estimation/" + actual_folder + "alpha_occupancy_determination_Fextr.pickle"
    data = load_xtrapol8_data(xtrapolate_pickle)
    output = {
        "config": config,
        "map_dark": map_dark,
        "diffmap": diffmap,
        "xtrapolate_data": data,
        "options": options,
    }
    return output

def make_figure_vary_inside(input_data):
    config = input_data["config"]
    map_dark = input_data["map_dark"]
    diffmap = input_data["diffmap"]
    x8_data = input_data["xtrapolate_data"]
    options = input_data["options"]
    map_dark_np = map_dark.to_3d_numpy_map(map_sampling=3)
    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=3)
    try:
        inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)
    except ValueError:
        inclusion_mask = None
    config_nse = copy.deepcopy(config)
    config_nse["masking"] = minimal_masking_config()
    inclusion_mask_nse = make_inclusion_mask(diffmap, map_dark, config_nse)

    try:
        pandda_dict = _calculate_pandda(diffmap_np, map_dark_np, inclusion_mask)
    except ValueError:
        pandda_dict = _calculate_pandda(diffmap_np, map_dark_np, inclusion_mask_nse)

    pandda_trad, pandda_imp = compact_pandda_results(
        pandda_dict,
    )
    occ, occ_CC = compact_x8(x8_data)
    try:
        simple_nse = compact_nse(diffmap, map_dark, inclusion_mask_nse)
    except (ValueError, RuntimeError):
        simple_nse = np.nan
    # if inclusion_mask is not None:
    try:
        advanced_nse = compact_nse(
            diffmap,
            map_dark,
            inclusion_mask,
        )
    except (ValueError, RuntimeError):
        advanced_nse = np.nan

    # --------------------------- Cell (1,1): Plot 4 ---------------------------
    _, _, (vacuum_mean, vacuum_std) = plot_extrapolation_estimate_new(
        diffmap, map_dark, inclusion_mask_nse, config, compact=True
    )
    return {
        "type": options["noise_type"],
        "snr": options["snr_factor"],
        "true_occupancy": options["occupancy_level"],
        "attempt_no": options["attempt_no"],
        "panda_trad": pandda_trad,
        "pandda_imp": pandda_imp,
        "occ": occ,
        "occ_CC": occ_CC,
        "simple_nse": simple_nse,
        "advanced_nse": advanced_nse,
        "vacuum_mean": vacuum_mean,
        "vacuum_std": vacuum_std,
    }

def change_legend_color(ax, figargs):
    color = figargs['color']
    ax.tick_params(axis='y', labelcolor=color)
    ax.spines.left.set_position(("axes", -0))
    ax.yaxis.set_label_position('left')
    ax.yaxis.set_ticks_position('left')
    ylabel = ax.get_ylabel()
    ax.set_ylabel("")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    return ylabel
def reduce_legends(axs):
    handles = []        
    labels = []
    for ax_in in axs:
        handle, label = ax_in.get_legend_handles_labels() 
        handles.extend(handle)
        labels.extend(label)
        if ax_in.legend_:
            ax_in.legend_.remove()
    sorted_pairs = dict(sorted(zip(labels, handles), key=lambda x: x[0]))
    # hl = dict(zip(labels, handles))

    axs[0].legend(sorted_pairs.values(), sorted_pairs.keys(), loc="upper left")

def make_comb_nse_figure(nse_data, ax):
    map_dark = nse_data["map_dark"]
    diffmap = nse_data["diffmap"]
    xtr_show_max = nse_data["xtr_show_max"]
    masking_options = nse_data["masks"]
    map_dark_no_zero = copy.deepcopy(map_dark)
    map_dark_no_zero.loc[(0,0,0), "F"] = 0
    axs = []
    lab = ""
    for ii, (mask_settings, mask) in enumerate(masking_options):
        if ii:
            ax2 = ax.twinx()
        else:
            ax2 = ax
        figargs=dict(**mask_settings, legend=True, xtr_show_max=xtr_show_max)
        map_dark_in = map_dark_no_zero if mask_settings["title"] == "Naive" else map_dark
        nse_analysis(diffmap, map_dark_in, mask, figargs=figargs, ax=ax2)
        lab = change_legend_color(ax2, figargs)
        axs.append(ax2)
    reduce_legends(axs)
    # figargs=dict(title="Basic Mask", color="orange", xtr_show_max=xtr_show_max)
    # nse_analysis(diffmap, map_dark, inclusion_mask_nse, figargs=figargs, ax=ax2_twin)
    ax.set_ylabel(lab, labelpad=20)
    ax.set_title("Negative Sum Explosion", )

def make_single_figure(input_data):
    config = input_data["config"]
    map_dark = input_data["map_dark"]
    diffmap = input_data["diffmap"]
    x8_data = input_data["xtrapolate_data"]
    map_dark_np = map_dark.to_3d_numpy_map(map_sampling=3)
    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=3)
    try: 
        inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)
    except ValueError:
        inclusion_mask = None
    config_nse = copy.deepcopy(config)
    config_nse["masking"] = minimal_masking_config()
    inclusion_mask_nse = make_inclusion_mask(diffmap, map_dark, config_nse)

    # 1. Create the main figure
    fig = plt.figure(figsize=(12, 12))
    title = "Simulated rsEGFP2\n"
    occu_level = input_data['options']['occupancy_level']
    title += f"light state with {occu_level:.0%} occupancy \nand expeccted $\\chi^{{-1}}$ of {occu_level/2:.2f}"
    # title += f"with {folder_cond['weight']}mFoFo maps"
    # title += f"\n with {folder_cond['noise_type']} noise"
    # title += f"\nwith a signal to noise ratio: {folder_cond['snr']}"
    fig.suptitle(title, fontsize=12)

    # 2. Define the 2x2 outer grid
    outer_grid = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.3)

    # ------------------------ Cell (0,0): The "Double Plot" -----------------------
    # We split this specific cell into 2 rows and 1 column
    inner_grid = outer_grid[0, 0].subgridspec(2, 1, hspace=0.4)
    ax0_top = fig.add_subplot(inner_grid[0, 0])
    ax0_bottom = fig.add_subplot(inner_grid[1, 0])
    ax_pandda= [ax0_top, ax0_bottom]
    
    pandda_dict = _calculate_pandda(diffmap_np, map_dark_np, inclusion_mask)
    _ = plot_pandda_results(pandda_dict, axs = ax_pandda, improved=True)
    pandda_dict_zero = _calculate_pandda(diffmap_np, map_dark_np-map_dark_np.mean(), inclusion_mask)
    _ = plot_pandda_results(pandda_dict_zero, axs = ax_pandda, improved=True)

    # --------------------------- Cell (0,1): Plot 2 --------------------------- 
    ax1 = fig.add_subplot(outer_grid[1, 0])
    replot_xtrapol8(x8_data, axes=ax1)
    ax1.set_title("Xtrapol8")

    # --------------------------- Cell (1,0): Plot 3 ---------------------------
    ax2 = fig.add_subplot(outer_grid[0, 1])
    inclusion_mask_none = np.ones_like(map_dark.to_3d_numpy_map(map_sampling=3), dtype=bool)
    nse_data = dict(
        map_dark = map_dark,
        diffmap = diffmap,
        xtr_show_max = max(15,2.1/input_data['options']['occupancy_level']),
        masks = [
            (dict(title="Vacuum Matching \ninspired Mask", color="blue"), inclusion_mask),
            (dict(title="Basic Mask", color="orange"), inclusion_mask_nse),
            (dict(title="Naive", color="green"), inclusion_mask_none),
        ],
    )
    make_comb_nse_figure(nse_data, ax2)
    # xtr_show_max = 25
    # figargs=dict(title="Basic Mask", color="orange", xtr_show_max=xtr_show_max)
    # ax2_twin = ax2.twinx()
    # nse_analysis(diffmap, map_dark, inclusion_mask_nse, figargs=figargs, ax=ax2_twin)
    # figargs=dict(title="Vacuum Matching \ninspired Mask", color="blue", legend=True,  xtr_show_max=xtr_show_max)
    # if inclusion_mask is not None:
    #     nse_analysis(diffmap, map_dark, inclusion_mask, figargs=figargs, ax=ax2, )
    # map_dark_no_zero = copy.deepcopy(map_dark)
    # map_dark_no_zero.loc[(0,0,0), "F"] = 0
    # inclusion_mask_none = np.ones_like(map_dark.to_3d_numpy_map(map_sampling=3), dtype=bool)
    # figargs=dict(title="Naive", color="green", legend=True, xtr_show_max=xtr_show_max)
    # ax2_twin2 = ax2_twin.twinx()
    # ax2_twin2.tick_params(axis='y', labelcolor='green')
    # ax2_twin.tick_params(axis='y', labelcolor='orange')
    # nse_analysis(diffmap, map_dark_no_zero, inclusion_mask_none, figargs=figargs, ax=ax2_twin2)
    # reduce_legends([ax2, ax2_twin, ax2_twin2])

    # ax2.set_title("Negative Sum Explosion")

    # --------------------------- Cell (1,1): Plot 4 ---------------------------
    ax3 = fig.add_subplot(outer_grid[1, 1])
    plot_extrapolation_estimate_new(diffmap, map_dark, inclusion_mask_nse, config, ax3)
    ax3.set_title("Vacuum Matching")
    # ax3.set_xlim(0,1)
    # ax3.set_ylim(0,0.81)


    # loc = load_figurepath()
    # fileloc = loc + name
    # print('saving to ', fileloc)
    # fig.savefig(fileloc)

    plt.show()

def make_comb_nse_figure_wrapper(input_data):
    config = input_data["config"]
    map_dark = input_data["map_dark"]
    diffmap = input_data["diffmap"]
    try: 
        inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)
    except ValueError:
        inclusion_mask = None
    config_nse = copy.deepcopy(config)
    config_nse["masking"] = minimal_masking_config()
    inclusion_mask_nse = make_inclusion_mask(diffmap, map_dark, config_nse)
    fig,ax2 = plt.subplots()
    inclusion_mask_none = np.ones_like(map_dark.to_3d_numpy_map(map_sampling=3), dtype=bool)
    nse_data = dict(
        map_dark = map_dark,
        diffmap = diffmap,
        xtr_show_max = 25,
        masks = [
            (dict(title="Vacuum Matching \ninspired Mask", color="blue"), inclusion_mask),
            (dict(title="Basic Mask", color="orange"), inclusion_mask_nse),
            (dict(title="Naive", color="green"), inclusion_mask_none),
        ],
    )
    make_comb_nse_figure(nse_data, ax2)
    fig.tight_layout()
    plt.show()

def process_single_folder(folder_path, reference_pdb, dmin):
    """
    Worker function to process a single folder.
    Returns a tuple of (output_dict, error_folder_path)
    """
    logger_alt = setup_logger()
    logger_alt.setLevel(40) 
    logger.setLevel(40)
    key = list(folder_path.keys())[0]
    try:
        print(f"Processing folder: {key}")
        data = prepare_data(folder_path, reference_pdb, dmin)
        output = make_figure_vary_inside(data)
        return output, None
    
    except (KeyError, ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Failed for {key}: {e}")
        return None, folder_path


def run_parallel_processing(folder_paths, reference_pdb, dmin, num_processes=4, pool_it =True):
    """
    Main orchestrator to handle the multiprocessing pool.
    """
    # 1. Get paths
    
    # 2. Prepare the worker function with fixed arguments
    worker = partial(process_single_folder, reference_pdb=reference_pdb, dmin=dmin)
    
    # 3. Execute Pool
    if pool_it:
        with Pool(processes=num_processes) as pool:
            results = pool.map(worker, folder_paths)
    else:
        results = [worker(folder_path) for folder_path in folder_paths]
    
    # 4. Aggregate results
    output_dict_list = [r[0] for r in results if r[0] is not None]
    failed_list = [r[1] for r in results if r[1] is not None]
    
    successes = len(output_dict_list)
    faileds = len(failed_list)
    
    print(f"Successfully processed {successes} and failed on {faileds} folders")
    
    # 5. Save and Return
    if output_dict_list:
        ds = pd.DataFrame(output_dict_list)
        return ds
    
    return pd.DataFrame()