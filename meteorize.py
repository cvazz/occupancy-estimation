import numpy as np
import pandas as pd
import gemmi
import matplotlib.pyplot as plt
import meteor
import reciprocalspaceship as rs


from generate_objects import run_scaleit

from meteor.diffmaps import (
    compute_difference_map,
    max_negentropy_kweighted_difference_map,
)
from meteor.tv import tv_denoise_difference_map

from meteor.utils import cut_resolution


from meteor import rsmap
from scipy.ndimage import label, generate_binary_structure

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
    print(ds_dark.columns)
    print(dark_f)
    ds_scaleit[out_columns["dark_f"]] = ds_dark[dark_f]
    ds_scaleit[out_columns["dark_sig"]] = ds_dark[dark_sig]
    ds_scaleit[out_columns["light_f"]] = ds_light[light_f]
    ds_scaleit[out_columns["light_sig"]] = ds_light[light_sig]
    print("cols here", ds_scaleit.columns)

    ds_scaleit = run_scaleit(ds_scaleit, None, False, columns=out_columns)

    light_columns2 = {}
    dark_columns2 = {}
    dark_columns2["amplitude_column"] = out_columns["light_f"]
    dark_columns2["uncertainty_column"] = out_columns["light_sig"]
    light_columns2["amplitude_column"] = out_columns["dark_f"]
    light_columns2["uncertainty_column"] = out_columns["dark_sig"]

    print("cols", ds_scaleit.columns)

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
    ds_comb, dark_columns_out, light_columns_out = scale_structure_factors(
        ds_dark, ds_light, dark_columns, light_columns
    )

    ds_comb[dark_columns["phase_column"]] = ds_dark[dark_columns["phase_column"]]
    dark_columns_out["phase_column"] = dark_columns["phase_column"]
    light_columns_out["phase_column"] = light_columns["phase_column"]

    map_dark = rsmap.Map(ds_comb, **dark_columns_out)
    map_light = rsmap.Map(ds_comb, **light_columns_out)
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
):
    n_largest = 4
    arrlen = len(rho_xtrs)
    if len(masks) > 1_000:
        print("Warning: Many masks")
    weight = np.empty(len(masks))
    for ii, mask in enumerate(masks):
        weight[ii] = np.sum(mask) ** 2

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
    ma = np.ma.MaskedArray(intersection_points, mask=np.isnan(intersection_points))
    intersection_average = np.ma.average(ma, weights=weight)
    intersection_std = np.sqrt(np.cov(ma, aweights=weight))
    print(f"Intersection Average: {intersection_average:.2f} ± {intersection_std:.2f}")

    intersection_average_inv = np.ma.average(2 / ma, weights=weight)
    intersection_std_inv = np.sqrt(np.cov(2 / ma, aweights=weight))
    print(
        f"Intersection Average Inverse: {intersection_average_inv:.2f} ± {intersection_std_inv:.2f}"
    )
    if detailed:
        return (
            intersection_average,
            intersection_std,
            intersection_average_inv,
            intersection_std_inv,
        )
    if False:
        return intersection_average, intersection_std, neg_sum, intersection_points

    return intersection_average, intersection_std


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
    threshold: float = 0.3,
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

    # Thresholds for positive and negative blobs
    pos_thresh = max_val * threshold
    neg_thresh = min_val * threshold

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
