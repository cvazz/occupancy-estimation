import numpy as np
import reciprocalspaceship as rs




from meteor import rsmap
from meteor.tv import tv_denoise_difference_map
from meteor.diffmaps import max_negentropy_kweighted_difference_map,
from meteor.scale import scale_maps

from scipy.ndimage import label, generate_binary_structure

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

def get_scaled_maps(ds_dark, ds_light):
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
    scaled_light = scale_maps(
        reference_map=unscaled_dark, map_to_scale=unscaled_light)
    map_dark = unscaled_dark
    map_light = scaled_light

    return map_dark, map_light


################################################################################
###################  Occupancy Estimation  #####################################
################################################################################

from scipy.stats import pearsonr


def pandda(
    map_dark: rsmap.Map,
    map_xtrs: list[rsmap.Map],
    mask_pks: np.ndarray,
    map_sampling: float,
):
    rho_dark = map_dark.to_3d_numpy_map(map_sampling=map_sampling)
    mean_global = np.empty(len(map_xtrs))
    mean_local = np.empty(len(map_xtrs))
    for ii, map_xtr in enumerate(map_xtrs):
        rho_xtr = map_xtr.to_3d_numpy_map(map_sampling=map_sampling)
        mean_global[ii] = pearsonr(
            rho_xtr[~mask_pks].flatten(), rho_dark[~mask_pks].flatten()
        )[0]
        mean_global[ii] = pearsonr(rho_xtr.flatten(), rho_dark.flatten())[0]
        mean_local[ii] = pearsonr(
            rho_xtr[mask_pks].flatten(), rho_dark[mask_pks].flatten()
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



from compare_conds import get_intersect_and_angle


def many_negsum(
    rho_xtrs: list[rsmap.Map],
    extrapolation_factors: list[float],
    *,
    map_sampling: float,
    masks: np.ndarray = None,
    detailed: bool = False,
    return_neg_sum: bool = False,
    diffmap: rsmap.Map = None,
):
    n_largest = 4
    arrlen = len(rho_xtrs)

    if len(masks) < 1:
        logger.error(f"Error: No masks provided")
        value_tuple = (np.nan, np.nan, np.nan, np.nan)
        if detailed and return_neg_sum:
            return (
                value_tuple,
                np.empty((arrlen, len(masks))) * np.nan,
                np.empty(len(masks)) * np.nan,
                np.empty(len(masks)) * np.nan,
            )
        elif detailed:
            return value_tuple, (np.nan, np.nan)
        if return_neg_sum:
            return np.empty((arrlen, len(masks))) * np.nan
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
        # logger.info(masks.shape)
        # masks.shape

        masks = np.array(masks)[sorted_indices]
        weight = weight[sorted_indices]


    neg_sum = np.empty((arrlen, len(masks)))
    for ii, density in enumerate(rho_xtrs):
        density = density.to_3d_numpy_map(map_sampling=map_sampling)
        for jj, mask in enumerate(masks):
            neg_sum[ii, jj] = np.sum(density[mask][density[mask] < 0])

    intersection_points = np.empty(len(masks))
    for jj, mask in enumerate(masks):
        intersect, angle = get_intersect_and_angle(
            extrapolation_factors, neg_sum[:, jj], n_largest
        )
        intersection_points[jj] = intersect
    # calculate standard deviation with weights, masking NaNs
    finite_intersections = np.isfinite(intersection_points)
    masked_intersection = intersection_points[finite_intersections]
    masked_weight = weight[finite_intersections]
    intersection_average = np.ma.average(masked_intersection, weights=masked_weight)
    intersection_std = np.sqrt(np.cov(masked_intersection, aweights=masked_weight))
    logger.info(
        f"Intersection Average 2: {intersection_average:.2f} ± {intersection_std:.2f}"
    )
    intersection_average_inv = np.ma.average(
        2 / masked_intersection, weights=masked_weight
    )
    intersection_std_inv = np.sqrt(
        np.cov(2 / masked_intersection, aweights=masked_weight)
    )
    logger.info(
        f"Intersection Average Inverse: {intersection_average_inv:.2f} ± {intersection_std_inv:.2f}"
    )
    value_tuple = (
        intersection_average,
        intersection_std,
        intersection_average_inv,
        intersection_std_inv,
    )
    mask_tuple = np.sum(np.asarray(masks)[finite_intersections]), np.sum(masked_weight)
    if detailed and return_neg_sum:
        return value_tuple, neg_sum, intersection_points, weight
    elif detailed:
        return value_tuple, mask_tuple
    if return_neg_sum:
        return neg_sum

    return (
        intersection_average,
        intersection_std,
    )


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


def find_largest_blobs2(
    diffmap: rsmap.Map,
    map_sampling: float,
    *,
    threshold: float = 0.5,
    minimum_size: int = 3,
    maximum_quantity: int = np.inf,
    find_pos: bool = True,
    find_neg: bool = True,
):
    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=map_sampling)
    if find_pos:
        pos_blob_mask = find_most_positive_blobs_np(
            diffmap_np,
            threshold=threshold,
            minimum_size=minimum_size,
            maximum_quantity=maximum_quantity,
        )
    if find_neg:
        neg_blob_mask = find_most_positive_blobs_np(
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


def adding_maps(map1: rsmap.Map, map2: rsmap.Map, *, factor1=1, factor2=1):
    common_indices = map1.index.intersection(map2.index)
    structure_factors1 = map1.loc[common_indices].to_structurefactor()
    structure_factors2 = map2.loc[common_indices].to_structurefactor()
    added_structure_factors = (
        factor1 * structure_factors1 + factor2 * structure_factors2
    )
    return rsmap.Map.from_structurefactor(
        added_structure_factors,
        index=common_indices,
        cell=map1.cell,
        spacegroup=map1.spacegroup,
    )


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
