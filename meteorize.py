import numpy as np
import reciprocalspaceship as rs

from scipy.ndimage import label, generate_binary_structure
from scipy.stats import pearsonr

from meteor import rsmap
from meteor.tv import tv_denoise_difference_map
from meteor.diffmaps import max_negentropy_kweighted_difference_map
from meteor.scale import scale_maps

from compare_conds import get_intersect_and_angle
from generate_objects import run_scaleit

import logging

logger = logging.getLogger(__name__)

################################################################################
###################  Scaling  ##################################################
################################################################################


def scale_structure_factors(ds_dark, ds_light, dark_columns, light_columns):

    dark_f = dark_columns["amplitude_column"]
    dark_sig = dark_columns["uncertainty_column"]
    light_f = light_columns["amplitude_column"]
    light_sig = light_columns["uncertainty_column"]

    out_columns = {
        "dark_f": "F_dark",
        "dark_sig": "SIGF_dark",
        "light_f": "F_light",
        "light_sig": "SIGF_light",
    }

    ds_scaleit = rs.DataSet(cell=ds_dark.cell, spacegroup=ds_dark.spacegroup)
    logger.info(f"Dark columns: {ds_dark.columns}")
    logger.info(f"Dark amplitude column: {dark_f}")
    try:
        ds_scaleit[out_columns["dark_f"]] = ds_dark[dark_f]
        ds_scaleit[out_columns["dark_sig"]] = ds_dark[dark_sig]
        ds_scaleit[out_columns["light_f"]] = ds_light[light_f]
        ds_scaleit[out_columns["light_sig"]] = ds_light[light_sig]
    except KeyError:

        logger.error(ds_dark.columns)
        logger.error(ds_light.columns)
        logger.error(dark_columns)
        logger.error(light_columns)
        raise KeyError

    logger.info(f"Scaled dataset columns: {ds_scaleit.columns}")

    ds_scaleit = run_scaleit(ds_scaleit, None, False, columns=out_columns)

    light_columns2 = {}
    dark_columns2 = {}
    dark_columns2["amplitude_column"] = out_columns["light_f"]
    dark_columns2["uncertainty_column"] = out_columns["light_sig"]
    light_columns2["amplitude_column"] = out_columns["dark_f"]
    light_columns2["uncertainty_column"] = out_columns["dark_sig"]

    return ds_scaleit, light_columns2, dark_columns2


def get_photolyase_maps(ds_dark, ds_light):
    make_dict = lambda **x: x
    dark_columns = make_dict(
        amplitude_column="F-obs-filtered",
        uncertainty_column="SIGF-obs-filtered",
        phase_column="PHIF-model",
    )

    light_columns = make_dict(
        amplitude_column="F", uncertainty_column="SIGF", phase_column="PHIF-model"
    )
    ds_light[light_columns["phase_column"]] = ds_dark[dark_columns["phase_column"]]
    unscaled_dark = rsmap.Map(ds_dark, **dark_columns)
    unscaled_light = rsmap.Map(ds_light, **light_columns)
    scaled_light = scale_maps(reference_map=unscaled_dark, map_to_scale=unscaled_light)
    map_dark = unscaled_dark
    map_light = scaled_light

    return map_dark, map_light


################################################################################
###################  Occupancy Estimation  #####################################
################################################################################


def pandda(
    map_dark: rsmap.Map,
    map_xtrs: list[rsmap.Map],
    mask_region_of_change: np.ndarray,
    map_sampling: float,
):
    """
    Compute local and global Pearson correlation coefficients between a reference map and a list of experimental maps.

    Parameters
    ----------
    map_dark : rsmap.Map
        The reference (dark) map.
    map_xtrs : list of rsmap.Map
        List of experimental maps to compare against the reference map.
    mask_region_of_change : np.ndarray
        Boolean mask indicating the region of change within the map.
    map_sampling : float
        The sampling rate for converting maps to 3D numpy arrays.

    Returns
    -------
    mean_local : np.ndarray
        Array of local Pearson correlation coefficients (within the region of change) for each experimental map.
    mean_global : np.ndarray
        Array of global Pearson correlation coefficients (over the entire map) for each experimental map.
    """

    rho_dark = map_dark.to_3d_numpy_map(map_sampling=map_sampling)
    mean_global = np.empty(len(map_xtrs))
    mean_local = np.empty(len(map_xtrs))
    for ii, map_xtr in enumerate(map_xtrs):
        rho_xtr = map_xtr.to_3d_numpy_map(map_sampling=map_sampling)
        mean_global[ii] = pearsonr(
            rho_xtr[~mask_region_of_change].flatten(),
            rho_dark[~mask_region_of_change].flatten(),
        )[0]
        mean_global[ii] = pearsonr(rho_xtr.flatten(), rho_dark.flatten())[0]
        mean_local[ii] = pearsonr(
            rho_xtr[mask_region_of_change].flatten(),
            rho_dark[mask_region_of_change].flatten(),
        )[0]
    return mean_local, mean_global


def negsum_meteor(
    rho_xtrs: list[rsmap.Map],
    *,
    map_sampling: float,
    mask: None | np.ndarray = None,
):
    rho_shape = rho_xtrs[0].to_3d_numpy_map(map_sampling=map_sampling).shape
    mask = np.ones(rho_shape, bool) if mask is None else mask
    arrlen = len(rho_xtrs)
    neg_sum = np.empty((arrlen))
    for ii, dens in enumerate(rho_xtrs):
        dens = dens.to_3d_numpy_map(map_sampling=map_sampling)[mask]
        neg_sum[ii] = np.sum(dens[dens < 0])
    return neg_sum


class NotEnoughMasksError(Exception):
    pass
from scipy.stats import stats
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
    if (intersection < highest_low * hlf or intersection > lowest_high * llf) and not return_all:
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

    return fit_lowest, fit_biggest, intersection

def many_negsum(
    rho_xtrs: list[rsmap.Map],
    extrapolation_factors: list[float],
    *,
    map_sampling: float,
    masks: np.ndarray = None,
    diffmap: rsmap.Map = None,
    return_data: bool = False,
):
    n_largest = 3
    arrlen = len(rho_xtrs)

    if len(masks) < 1:
        logger.error(f"Error: No masks provided, masks has {len(masks)}  .")
        raise NotEnoughMasksError(
            f"Error: No masks provided, masks has {len(masks)}  ."
        )

    weight = np.empty(len(masks))
    if diffmap is not None:
        diffmap_density = diffmap.to_3d_numpy_map(map_sampling=map_sampling)
        for ii, mask in enumerate(masks):
            weight[ii] = np.sum(np.abs(diffmap_density[mask]))
    else:
        logger.warning(f"No Diffmap provided for Weighting Calculation")
        for ii, mask in enumerate(masks):
            weight[ii] = np.sum(mask) ** 2
    # find the 1000 masks with the heaviest weight
    max_masks = 1000
    if len(weight) > max_masks:
        logger.warning(
            f"Too many masks ({len(weight)}), selecting {max_masks} with the largest diffmap contribution."
        )
        sorted_indices = np.argsort(weight)[-max_masks:]
        logger.debug(f"Selected masks: {sorted_indices[sorted_indices>max_masks]}")

        masks = np.array(masks)[sorted_indices]
        weight = weight[sorted_indices]

    neg_sum = np.empty((arrlen, len(masks)))
    for ii, density in enumerate(rho_xtrs):
        density = density.to_3d_numpy_map(map_sampling=map_sampling)
        for jj, mask in enumerate(masks):
            neg_sum[ii, jj] = np.sum(density[mask][density[mask] < 0])

    intersection_points = np.empty(len(masks))
    from matplotlib import pyplot as plt
    colors = plt.cm.viridis(np.linspace(0, 1, len(masks)))
    for jj, mask in enumerate(masks):
        fit1, fit2, intersect = get_fits2(
            neg_sum[:, jj]/np.min(neg_sum[:, jj]), extrapolation_factors, n_largest
        )
        intersection_points[jj] = intersect

        if np.max(np.abs(fit1 - fit2)) < 0.1:
            logger.warning(f"Fits are parallel (would have been {intersect:.2f}, {np.min(fit1):.1f}, {np.max(fit1):.1f}, {np.min(fit2):.1f},  {np.max(fit2):.1f} )") 
            intersection_points[jj] = np.nan
            # plt.show()
    return neg_sum, intersection_points, weight


def process_many_negsum(intersection_points, weight, masks=None):

    # calculate standard deviation with weights, masking NaNs
    finite_intersections = np.isfinite(intersection_points)
    masked_intersect = intersection_points[finite_intersections]
    masked_weight = weight[finite_intersections]
    intersection_average = np.average(masked_intersect, weights=masked_weight)
    intersection_std = np.sqrt(np.cov(masked_intersect, aweights=masked_weight))
    intersection_average_inv = np.average(1 / masked_intersect, weights=masked_weight)
    intersection_std_inv = np.sqrt(np.cov(1 / masked_intersect, aweights=masked_weight))

    logstart = "Intersection Average"
    log_msg = f"{logstart}: {intersection_average:.2f} ± {intersection_std:.2f}"
    log_msg += f"Inverse: {intersection_average_inv:.2f} ± {intersection_std_inv:.2f}"
    logger.info(log_msg)
    if masks is not None:
        if len(masks) == len(intersection_points):
            masks = np.asarray(masks)[finite_intersections]
    mask_sum = np.sum(masks) if masks is not None else None
    weight_sum = np.sum(masked_weight)
    logger.info(f"Total mask sum: {mask_sum}, Total weight sum: {weight_sum}")
    output_dict = {
        "intersection_average": intersection_average,
        "intersection_std": intersection_std,
        "intersection_average_inv": intersection_average_inv,
        "intersection_std_inv": intersection_std_inv,
        "mask_counts": mask_sum,
        "mask_weight": weight_sum,
    }
    share_nan = 1-np.sum(finite_intersections)/len(finite_intersections) 
    if share_nan > 0.2:
        logger.error(f"Many NaN values in intersection points {share_nan:.2f} , consider adjusting your analysis.")
    share_nan_weight = 1 - np.sum(masked_weight) / np.sum(weight)
    if share_nan_weight:
        logger.warning(f"Share of NaN values in intersection points (weighted) {share_nan_weight:.2f} out of 1")

    return output_dict


def calculate_and_process_many_negsum(
    rho_xtrs: list[rsmap.Map],
    extrapolation_factors: list[float],
    *,
    map_sampling: float,
    masks: np.ndarray = None,
    diffmap: rsmap.Map = None,
    return_data: bool = False,
):

    neg_sum, intersection_points, weight = many_negsum(
        rho_xtrs,
        extrapolation_factors=extrapolation_factors,
        map_sampling=map_sampling,
        masks=masks,
        diffmap=diffmap,
        return_data=return_data,
    )
    if return_data:
        return neg_sum, intersection_points, weight

    return process_many_negsum(intersection_points, weight, masks=masks)


################################################################################
##########################  Mask Calculations ##################################
################################################################################


def fetch_map2numpy_args(rsmap: rsmap.Map):
    return (lambda **kwargs: kwargs)(
        spacegroup=rsmap.spacegroup,
        cell=rsmap.cell,
        high_resolution_limit=rsmap.resolution_limits[1],
    )


def calc_direct_difference(map_light, map_dark, map_sampling):
    direct_diff = map_light.to_3d_numpy_map(
        map_sampling=map_sampling
    ) - map_dark.to_3d_numpy_map(map_sampling=map_sampling)

    return rsmap.Map.from_3d_numpy_map(direct_diff, **fetch_map2numpy_args(map_dark))


def fetch_without_meta(map_light, map_dark, diffmap_maker):
    k_weighted_diffmap, kparameter_metadata = diffmap_maker(map_light, map_dark)
    return k_weighted_diffmap


def fetch_tv_denoised(map_light, map_dark):
    k_weighted_diffmap, kparameter_metadata = max_negentropy_kweighted_difference_map(
        map_light, map_dark
    )
    tv_denoised_map, metadata = tv_denoise_difference_map(
        k_weighted_diffmap, full_output=True
    )
    return tv_denoised_map


def find_largest_blobs(
    diffmap: rsmap.Map,
    map_sampling: float,
    *,
    threshold: float | None = None,
    thresh_pos: float | None = None,
    thresh_neg: float | None = None,
    minimum_size: int = 3,
):
    """
    Finds all positive and negative blobs in the difference map, respecting periodic boundary conditions.

    Identifies all contiguous regions (blobs) above (positive) or below (negative) a threshold,
    where the threshold is a fraction of the global maximum (for positive) or minimum (for negative) value.
    Returns two lists of masks (one for positive blobs, one for negative blobs), each ordered by blob size (largest first).
    Each mask is a boolean array of the same shape as the map, True where the blob is present.
    """
    # Convert map to numpy array
    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=map_sampling)

    # Find max and min values
    max_val = np.max(diffmap_np)
    min_val = np.min(diffmap_np)

    # add assertions allowing only for thresh_pos and thresh_neg
    # or threshold, not both
    if threshold is not None:
        thresh_pos = threshold
        thresh_neg = threshold

    # Thresholds for positive and negative blobs
    pos_thresh = max_val * thresh_pos
    neg_thresh = min_val * thresh_neg

    # Create masks for positive and negative blobs
    pos_mask = diffmap_np >= pos_thresh
    neg_mask = diffmap_np <= neg_thresh

    # Use 3D connectivity for labeling
    structure = generate_binary_structure(3, 3)

    # Label positive and negative blobs
    pos_labeled, pos_num = label(pos_mask, structure=structure)
    neg_labeled, neg_num = label(neg_mask, structure=structure)

    # Get sizes and sort order for positive blobs
    pos_blob_sizes = np.bincount(pos_labeled.ravel())
    pos_blob_sizes[0] = 0  # background
    pos_blob_sizes[pos_blob_sizes < minimum_size] = 0  # filter out small blobs
    pos_order = np.argsort(pos_blob_sizes)[::-1]  # largest first, skip 0

    # Get sizes and sort order for negative blobs
    neg_blob_sizes = np.bincount(neg_labeled.ravel())
    neg_blob_sizes[0] = 0  # background
    neg_blob_sizes[neg_blob_sizes < minimum_size] = 0  # filter out small blobs
    neg_order = np.argsort(neg_blob_sizes)[::-1]  # largest first, skip 0

    # Create masks for all positive blobs, ordered by size
    pos_blob_masks = np.zeros_like(pos_labeled, dtype=int)

    for new_idx, old_idx in enumerate(pos_order):
        if old_idx != 0 and pos_blob_sizes[old_idx] > 0:
            pos_blob_masks[pos_labeled == old_idx] = new_idx

    # Create masks for all negative blobs, ordered by size
    neg_blob_masks = np.zeros_like(neg_labeled, dtype=int)

    for new_idx, old_idx in enumerate(neg_order):
        if old_idx != 0 and neg_blob_sizes[old_idx] > 0:
            neg_blob_masks[neg_labeled == old_idx] = new_idx

    return pos_blob_masks, neg_blob_masks


def find_most_positive_blobs_np(
    diffmap_np: np.ndarray,
    *,
    threshold: float,
    minimum_size: int,
    maximum_quantity: int,
):
    """
    Finds the largest positive blobs in the difference map.

    Returns a list of boolean masks, each mask corresponding to a blob, ordered by size (largest first).
    """
    max_val = np.max(diffmap_np)
    pos_thresh = max_val * threshold
    pos_mask = diffmap_np >= pos_thresh
    structure = generate_binary_structure(3, 3)
    pos_labeled, pos_num = label(pos_mask, structure=structure)
    pos_blob_sizes = np.bincount(pos_labeled.ravel())
    pos_blob_sizes[0] = 0  # background
    pos_blob_sizes[pos_blob_sizes < minimum_size] = 0
    pos_order = np.argsort(pos_blob_sizes)[::-1]
    maximum_quantity = min(maximum_quantity, len(pos_order))
    pos_order = pos_order[:maximum_quantity]
    pos_blob_masks = []
    for new_idx, old_idx in enumerate(pos_order):
        if old_idx != 0 and pos_blob_sizes[old_idx] > 0:
            logger.warning(f"Blob {new_idx}, inserting {old_idx}")
            pos_blob_masks[pos_labeled == old_idx] = new_idx
    return pos_blob_masks


def find_most_positive_blobs_np(
    diffmap_np: np.ndarray,
    *,
    threshold: float,
    minimum_size: int,
    maximum_quantity: int,
):
    max_val = np.max(diffmap_np)

    # add assertions allowing only for thresh_pos and thresh_neg
    # or threshold, not both
    if threshold is not None:
        thresh_pos = threshold

    # Thresholds for positive and negative blobs
    pos_thresh = max_val * thresh_pos

    # Create masks for positive and negative blobs
    pos_mask = diffmap_np >= pos_thresh

    # Use 3D connectivity for labeling
    structure = generate_binary_structure(3, 3)

    # Label positive and negative blobs
    pos_labeled, pos_num = label(pos_mask, structure=structure)

    # Get sizes and sort order for positive blobs
    pos_blob_sizes = np.bincount(pos_labeled.ravel())
    pos_blob_sizes[0] = 0  # background
    pos_blob_sizes[pos_blob_sizes < minimum_size] = 0  # filter out small blobs
    pos_order = np.argsort(pos_blob_sizes)[::-1]  # largest first, skip 0

    maximum_quantity = min(maximum_quantity, len(pos_order))
    pos_order = pos_order[:maximum_quantity]

    # Create masks for all positive blobs, ordered by size
    pos_blob_masks = np.zeros_like(pos_labeled, dtype=int)

    for new_idx, old_idx in enumerate(pos_order):
        if old_idx != 0 and pos_blob_sizes[old_idx] > 0:
            pos_blob_masks[pos_labeled == old_idx] = new_idx

    return pos_blob_masks

def find_most_positive_blobs_rmsd(
    diffmap_np: np.ndarray,
    *,
    threshold: float,
    minimum_size: int,
    maximum_quantity: int,
):
    threshold2 = threshold*np.max(diffmap_np)
    thresh_pos = np.percentile(diffmap_np, 99.9) 
    if threshold2<thresh_pos:
        threshold2 = thresh_pos 
        logger.info(f"Using threshold: {thresh_pos:.3f} ({threshold2:.3f})")
    diffmean = np.mean(diffmap_np)
    sigma = diffmap_np.std()
    peak_threshold = threshold*sigma+diffmean
    
    # threshold2 = thresh_pos*1.2
    # Thresholds for positive and negative blobs
    # pos_thresh = max_val * thresh_pos
    pos_thresh = np.percentile(diffmap_np, 99.9) 
    logger.info(f"Using pos_thresh: {pos_thresh:.3f}, in sigmas: {(pos_thresh-diffmean)/sigma:.3f}")

    # Create masks for positive and negative blobs
    pos_mask = diffmap_np >= pos_thresh

    # Use 3D connectivity for labeling
    structure = generate_binary_structure(3, 3)

    # Label positive and negative blobs
    pos_labeled, pos_num = label(pos_mask, structure=structure)
    for label_id in range(1, pos_num + 1):
        blob_peak = np.max(diffmap_np[pos_labeled == label_id])
        if blob_peak < peak_threshold:
            pos_labeled[pos_labeled == label_id] = 0  # remove blob below threshold

    # Get sizes and sort order for positive blobs
    pos_blob_sizes = np.bincount(pos_labeled.ravel())
    pos_blob_sizes[0] = 0  # background

    pos_blob_sizes[pos_blob_sizes < minimum_size] = 0  # filter out small blobs
    pos_order = np.argsort(pos_blob_sizes)[::-1]  # largest first, skip 0

    maximum_quantity = min(maximum_quantity, len(pos_order))
    pos_order = pos_order[:maximum_quantity]

    # Create masks for all positive blobs, ordered by size
    pos_blob_masks = np.zeros_like(pos_labeled, dtype=int)

    for new_idx, old_idx in enumerate(pos_order):
        if old_idx != 0 and pos_blob_sizes[old_idx] > 0:
            pos_blob_masks[pos_labeled == old_idx] = new_idx

    return pos_blob_masks

def calculate_all_pos_blobs(
    diffmap_np: np.ndarray,
    sigma: float
):
    
    threshold = diffmap_np.std()*sigma-diffmap_np.mean()

    # Create masks for positive and negative blobs
    pos_mask = diffmap_np >= threshold

    # Use 3D connectivity for labeling
    structure = generate_binary_structure(3, 3)

    # Label positive and negative blobs
    pos_labeled, pos_num = label(pos_mask, structure=structure)
    logger.warning(f"Used threshold for posmask: {threshold/np.max(diffmap_np):.3f}, found {pos_num} blobs")
    return pos_labeled


def filter_pos_blobs_rmsd(
    diffmap_np: np.ndarray,
    pos_labeled: np.ndarray,
    *,
    threshold: float,
    maximum_quantity: int,
    minimum_size: int = 10,
):
    threshold2 = threshold*diffmap_np.std()-np.mean(diffmap_np)
    for label_id in range(1, np.max(pos_labeled) + 1):
        blob_peak = np.max(diffmap_np[pos_labeled == label_id])
        if blob_peak < threshold2:
            pos_labeled[pos_labeled == label_id] = 0  # remove blob below threshold

    # Get sizes and sort order for positive blobs
    pos_blob_sizes = np.bincount(pos_labeled.ravel())
    pos_blob_sizes[0] = 0  # background
    # from matplotlib import pyplot as plt
    # bins = np.arange(0.1, np.max(pos_blob_sizes)+1, 1)
    # plt.figure()
    # plt.hist(pos_blob_sizes, bins=bins)
    # plt.show()

    pos_blob_sizes[pos_blob_sizes < minimum_size] = 0  # filter out small blobs
    pos_order = np.argsort(pos_blob_sizes)[::-1]  # largest first, skip 0


    maximum_quantity = min(maximum_quantity, len(pos_order))
    pos_order = pos_order[:maximum_quantity]

    # Create masks for all positive blobs, ordered by size
    pos_blobs = np.zeros_like(pos_labeled, dtype=int)

    pos_blob_mask_list = []
    for new_idx, old_idx in enumerate(pos_order):
        if old_idx != 0 and pos_blob_sizes[old_idx] > 0:
            pos_blobs[pos_labeled == old_idx] = new_idx
            pos_blob_mask_list.append(pos_labeled == old_idx)
    return pos_blobs, pos_blob_mask_list

def filter_pos_blobs(
    diffmap_np: np.ndarray,
    pos_labeled: np.ndarray,
    *,
    threshold: float,
    maximum_quantity: int,
    minimum_size: int = 10,
):
    threshold2 = threshold*np.max(diffmap_np)
    for label_id in range(1, np.max(pos_labeled) + 1):
        blob_peak = np.max(diffmap_np[pos_labeled == label_id])
        if blob_peak < threshold2:
            pos_labeled[pos_labeled == label_id] = 0  # remove blob below threshold

    # Get sizes and sort order for positive blobs
    pos_blob_sizes = np.bincount(pos_labeled.ravel())
    pos_blob_sizes[0] = 0  # background

    pos_blob_sizes[pos_blob_sizes < minimum_size] = 0  # filter out small blobs
    pos_order = np.argsort(pos_blob_sizes)[::-1]  # largest first, skip 0

    maximum_quantity = min(maximum_quantity, len(pos_order))
    pos_order = pos_order[:maximum_quantity]

    # Create masks for all positive blobs, ordered by size
    pos_blobs = np.zeros_like(pos_labeled, dtype=int)

    pos_blob_mask_list = []
    for new_idx, old_idx in enumerate(pos_order):
        if old_idx != 0 and pos_blob_sizes[old_idx] > 0:
            pos_blobs[pos_labeled == old_idx] = new_idx
            pos_blob_mask_list.append(pos_labeled == old_idx)
    return pos_blobs, pos_blob_mask_list

def find_most_positive_blobs_fixed_basis(
    diffmap_np: np.ndarray,
    *,
    threshold: float,
    minimum_size: int,
    maximum_quantity: int,
):
    max_val = np.max(diffmap_np)
    # add assertions allowing only for thresh_pos and thresh_neg
    # or threshold, not both
    if threshold is not None:
        thresh_pos = threshold
        threshold2 = threshold*np.max(diffmap_np)
        thresh_pos = np.percentile(diffmap_np, 99.) 
        if threshold2<thresh_pos:
            threshold2 = thresh_pos 
            logger.info(f"Using threshold: {thresh_pos:.3f} ({threshold2:.3f})")
    # threshold2 = thresh_pos*1.2
    # Thresholds for positive and negative blobs
    # pos_thresh = max_val * thresh_pos
    pos_thresh = np.percentile(diffmap_np, 99.8) 

    logger.error(f"Using threshold for posmask: {pos_thresh/np.max(diffmap_np):.3f}")
    # Create masks for positive and negative blobs
    pos_mask = diffmap_np >= pos_thresh

    # Use 3D connectivity for labeling
    structure = generate_binary_structure(3, 3)

    # Label positive and negative blobs
    pos_labeled, pos_num = label(pos_mask, structure=structure)
    for label_id in range(1, pos_num + 1):
        blob_peak = np.max(diffmap_np[pos_labeled == label_id])
        if blob_peak < threshold2:
            pos_labeled[pos_labeled == label_id] = 0  # remove blob below threshold

    # Get sizes and sort order for positive blobs
    pos_blob_sizes = np.bincount(pos_labeled.ravel())
    pos_blob_sizes[0] = 0  # background

    pos_blob_sizes[pos_blob_sizes < minimum_size] = 0  # filter out small blobs
    pos_order = np.argsort(pos_blob_sizes)[::-1]  # largest first, skip 0

    maximum_quantity = min(maximum_quantity, len(pos_order))
    pos_order = pos_order[:maximum_quantity]

    # Create masks for all positive blobs, ordered by size
    pos_blob_masks = np.zeros_like(pos_labeled, dtype=int)

    for new_idx, old_idx in enumerate(pos_order):
        if old_idx != 0 and pos_blob_sizes[old_idx] > 0:
            pos_blob_masks[pos_labeled == old_idx] = new_idx

    return pos_blob_masks


def find_largest_blobs2(
    diffmap: rsmap.Map,
    map_sampling: float,
    *,
    threshold: float = 0.5,
    minimum_size: int = 10,
    maximum_quantity: int = np.inf,
    find_pos: bool = True,
    find_neg: bool = True,
    blob_selection_func: callable = None,
):
    blob_selection_func = (
        find_most_positive_blobs_np
        if blob_selection_func is None
        else blob_selection_func
    )
    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=map_sampling)
    if find_pos:
        pos_blob_mask = blob_selection_func(
            diffmap_np,
            threshold=threshold,
            minimum_size=minimum_size,
            maximum_quantity=maximum_quantity,
        )
    if find_neg:
        neg_blob_mask = blob_selection_func(
            -diffmap_np,
            threshold=threshold,
            minimum_size=minimum_size,
            maximum_quantity=maximum_quantity,
        )
    if find_pos and find_neg:
        return pos_blob_mask, neg_blob_mask
    elif find_pos:
        return pos_blob_mask
    elif find_neg:
        return neg_blob_mask
    else:
        raise ValueError("At least one of find_pos or find_neg must be True.")


################################################################################
################################################################################
################################################################################
import pandas as pd
def realspace_maximum_scaling(
    reference_map: rsmap.Map,
    map_to_scale: rsmap.Map,
    map_sampling: float = 3.0,
):
    ref_map = reference_map.to_3d_numpy_map(map_sampling=map_sampling)
    scale_map = map_to_scale.to_3d_numpy_map(map_sampling=map_sampling)
    shares = [0.01, 0.005, 0.001, 0.0005]
    pos_scales, neg_scales = [], []
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
        pos_scales.append(pos_scale)
        neg_scales.append(neg_scale)
    if np.std(neg_scales) / np.mean(neg_scales) > 0.1:
        logger.warning(
            f"High variation in negative scaling factors: {neg_scales}, using last."
        )
    return pos_scale, neg_scale


def rescaling_diffmap(diffmap_to_scale: rsmap.Map, diffmap_reference: rsmap.Map):
    pos_scale, neg_scale = realspace_maximum_scaling(
        reference_map=diffmap_reference,
        map_to_scale=diffmap_to_scale,
    )
    scaling_factor =  neg_scale * 2 # scaling to vanilla map
    finite_amps = np.isfinite(diffmap_to_scale.amplitudes)
    diffmap_to_scale.loc[
        finite_amps, diffmap_to_scale.amplitude_column_name
    ] *= scaling_factor
    if diffmap_to_scale.has_uncertainties:
        diffmap_to_scale.loc[
            finite_amps, diffmap_to_scale.uncertainties_column_name
        ] *= scaling_factor
    logger.info(f"Rescaled map by factor {scaling_factor}")
    return diffmap_to_scale

def adding_maps(map1: rsmap.Map, map2: rsmap.Map, *, factor1=1, factor2=1):
    common_indices = map1.index.intersection(map2.index)
    structure_factors1 = map1.loc[common_indices].to_structurefactor()
    structure_factors2 = map2.loc[common_indices].to_structurefactor()
    added_structure_factors = (
        factor1 * structure_factors1 + factor2 * structure_factors2
    )

    sum_of_map  = rsmap.Map.from_structurefactor(
        added_structure_factors,
        index=common_indices,
        cell=map1.cell,
        spacegroup=map1.spacegroup,
    )

    if  map1.has_uncertainties and map2.has_uncertainties:
        sigmaF = np.sqrt(
          (factor1 * map1.uncertainties[common_indices]) ** 2
        + (factor2 * map2.uncertainties[common_indices]) ** 2
        )
        sum_of_map.set_uncertainties(pd.Series(sigmaF, index=common_indices))
    return sum_of_map

def saving_xtrapolated_map(
        diffmap:rsmap.Map, 
        *,
        map_dark:rsmap.Map, 
        xtr_factor:float,
        reference_diffmap:rsmap.Map = None,
        seed_dataset:rs.DataSet = None,
        save_diffmap:bool = False,
        filename : str = "xtr_map"
        ):
    diffmap =rescaling_diffmap(diffmap.copy(), reference_diffmap) 
    xtr_map = adding_maps(map_dark, diffmap, factor2=xtr_factor)
    if seed_dataset is not None:
        implant = rs.DataSet(seed_dataset.copy())
        implant[["F", "SIGF", "PHIC"]] = xtr_map[["F", "SIGF", "PHI"]]
        necessary_cols = ["H", "K", "L", "F", "SIGF", "PHIC", "FreeR_flag"]
        implant = implant.drop(columns=[col for col in implant.columns if col not in necessary_cols])
    mask = np.logical_or((~implant["F"].isna() & implant["SIGF"].isna()), (implant["F"].isna() & ~implant["SIGF"].isna()))
    non_matching_indices = np.sum(np.array(mask))
    if non_matching_indices > 0:
        print(non_matching_indices)
        logger.warning(f"Number of rows with not shared NaNs in F and SIGF: {non_matching_indices}")
    implant.loc[mask, ["F", "PHIC"]] = np.nan
    implant.write_mtz(f"{filename}_{xtr_factor}.mtz")
    if save_diffmap:
        diffmap.write_mtz(f"diffmap_kweighted.mtz")
        return xtr_map, diffmap
    return xtr_map

def make_k_space_xtr(
    map_dark,
    diffmap,
    extrapolation_factors,
):

    map_xtrs = []
    for xtr_factor in extrapolation_factors:
        map_xtr = adding_maps(diffmap, map_dark, factor1=xtr_factor)
        map_xtrs.append(map_xtr)
    return map_xtrs


def make_real_space_xtr(
    map_dark,
    diffmap,
    extrapolation_factors,
    map_sampling,
):
    rho_xtrs = []
    for xtr_factor in extrapolation_factors:

        xtr = diffmap.to_3d_numpy_map(
            map_sampling=map_sampling
        ) * xtr_factor + map_dark.to_3d_numpy_map(map_sampling=map_sampling)
        rho_xtr = rsmap.Map.from_3d_numpy_map(
            xtr,
            spacegroup=map_dark.spacegroup,
            cell=map_dark.cell,
            high_resolution_limit=map_dark.resolution_limits[1],
        )
        rho_xtrs.append(rho_xtr)
    return rho_xtrs