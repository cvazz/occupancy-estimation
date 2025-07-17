import pandas as pd
import gemmi
import matplotlib.pyplot as plt
import meteor
import reciprocalspaceship as rs
from occupancy import *
from plotting3d import *

from generate_objects import run_scaleit


def k2real(f):
    f_prime = f.copy()
    f_prime[np.isnan(f)] = 0
    return np.real(np.fft.fftn(f_prime))


def get_exp_structure_factors(ds_dark, ds_light, return_ds=False):

    dark_f = "F-obs-filtered"
    dark_phi = "PHIF-model"
    light_f = "F"
    light_sig = "SIGF"
    dark_sig = "SIGF-obs-filtered"
    phi_corr = "phi_corr"

    columns = {
        "light_f": light_f,
        "dark_f": dark_f,
        "light_sig": light_sig,
        "dark_sig": dark_sig,
    }

    ds_comb = ds_dark
    # ds_comb[light_f] = np.sqrt(np.abs(ds_light[light_f]))
    # ds_comb[light_sig] = np.sqrt(np.abs(ds_light[light_sig]))
    ds_comb[light_f] = ds_light[light_f]
    ds_comb[light_sig] = ds_light[light_sig]

    ds_scaleit = rs.DataSet(cell=ds_comb.cell, spacegroup=ds_comb.spacegroup)
    for column in columns.values():
        ds_scaleit[column] = ds_comb[column]

    ds_scaleit = run_scaleit(ds_scaleit, None, False, columns=columns)

    for column in columns.values():
        ds_comb[column] = ds_scaleit[column]
    if return_ds:
        return ds_comb

    ds_comb[phi_corr] = (ds_light[light_f] < 0) * np.pi

    ds_comb["sf_dark"] = ds_comb.to_structurefactor(dark_f, dark_phi)
    ds_comb["F_delta"] = ds_comb[light_f] - ds_comb[dark_f]
    ds_comb["sf_delta"] = ds_comb.to_structurefactor("F_delta", dark_phi)
    ds_comb["sf_light"] = ds_comb.to_structurefactor(light_f, dark_phi)

    f_dark = ds_comb.to_reciprocal_grid("sf_dark")
    # f_delta = ds_comb.to_reciprocal_grid( "sf_delta")
    f_light = ds_comb.to_reciprocal_grid("sf_light")

    f_dark[np.isnan(f_dark)] = 0
    f_light[np.isnan(f_light)] = 0

    return f_dark, f_light


from generate_objects import get_pdb_pairs
from generate_objects import struc2realspace


def find_directory_in_path(target_dir: str) -> str | None:
    """
    Parses the path of the current file and returns the full path to a specific directory
    if it exists in the path.

    Args:
        target_dir (str): The name of the directory to look for in the current file's path.

    Returns:
        str | None: Full path to the directory if found, otherwise None.
    """
    # Get the absolute path of the current file
    current_path = os.path.abspath(__file__)
    # Split the path into components
    path_parts = current_path.split(os.sep)

    # Attempt to find the index of the target directory
    try:
        idx = path_parts.index(target_dir)
        # Rebuild the path up to and including the target directory
        matched_path = os.sep.join(path_parts[: idx + 1])
        return matched_path
    except ValueError:
        # target_dir not found in path
        return None


def get_parent_dir():
    current_path = os.path.abspath(__file__)
    path_parts = current_path.split(os.sep)
    parent = os.sep.join(path_parts[:-1])
    return parent

    # parent=get_parent_dir()
    # pdb_pair = get_pdb_pairs(photolyase_hash)
    # fname_dark = f"{parent}/synthetic_data/{pdb_pair[0]}.pdb"
    # fname_mix = f"{parent}/synthetic_data/{pdb_pair[1]}.pdb"


def get_model_densities(fname_dark, fname_mix, hs_limit):
    struc_dark = gemmi.read_structure(fname_dark)
    struc_light = gemmi.read_structure(fname_mix)
    density_darkT = struc2realspace(struc_dark, hs_limit, 0)
    density_lightT = struc2realspace(struc_light, hs_limit, 0)
    return density_darkT, density_lightT


################################################################################
############################## DoG Filter ######################################
################################################################################


def undo_3d_tiling(mask, target_shape, lob_off):
    mask_comb = np.zeros((mask.shape[0], mask.shape[1], target_shape[2]))
    mask_comb[..., lob_off[2] :] += mask[..., : target_shape[2] - lob_off[2]]
    mask_comb[..., : -lob_off[2]] += mask[..., target_shape[2] - lob_off[2] :]

    mask_comb2 = np.zeros((mask.shape[0], target_shape[1], target_shape[2]))
    mask_comb2[:, lob_off[1] :, :] += mask_comb[:, : target_shape[1] - lob_off[1], :]
    mask_comb2[:, : -lob_off[1], :] += mask_comb[..., target_shape[1] - lob_off[1] :, :]

    mask_comb3 = np.zeros((target_shape))
    mask_comb3[lob_off[0] :, ...] += mask_comb2[: target_shape[0] - lob_off[0], ...]
    mask_comb3[: -lob_off[0], ...] += mask_comb2[target_shape[0] - lob_off[0] :, ...]

    return mask_comb3


def calculate_dog_filter_mask_v2(
    delta_of_rho, sigma, thresh_pos, thresh_neg, nth_val, target_shape, detailed=False
):
    delta_rho_tiled = np.tile(delta_of_rho, (2, 2, 2))
    dog_filter = ndimage.gaussian_laplace(
        delta_rho_tiled,
        sigma=sigma,
    )

    print("dogged")

    lob_off = target_shape // 3

    slc = tuple(slice(lob, shp * 2 - lob) for lob, shp in zip(lob_off, target_shape))
    mask_pos = pick_largest(dog_filter[slc], thresh_fac=thresh_pos, nth_val=nth_val)
    mask_neg = pick_largest(-dog_filter[slc], thresh_fac=thresh_neg, nth_val=nth_val)

    mask_pos_comb = undo_3d_tiling(mask_pos, target_shape, lob_off)
    mask_pos_comb = np.array(mask_pos_comb > 0, float)
    mask_neg_comb = undo_3d_tiling(mask_neg, target_shape, lob_off)
    mask_neg_comb = np.array(mask_neg_comb > 0, float)
    if detailed:
        return mask_pos_comb, mask_neg_comb, dog_filter
    mask_comb = (mask_pos_comb + mask_neg_comb) > 0

    return mask_comb


def periodic_label_3d1(label_image):
    for dim in range(label_image.ndim):
        for idx_1 in range(label_image.shape[dim]):
            for idx_2 in range(label_image.shape[(dim + 1) % 3]):
                first = tuple(np.roll((idx_1, idx_2, 0), dim))
                last = tuple(np.roll((idx_1, idx_2, -1), dim))
                if label_image[first] > 0 and label_image[last] > 0:
                    label_image[label_image == label_image[last]] = label_image[first]
    return label_image


def pick_largest(
    result,
    thresh_fac,
    nth_val=70,
):
    idcs_flat = np.argpartition(result.flatten(), -nth_val)[-nth_val:]
    idcs = np.unravel_index(idcs_flat, result.shape)
    thresh = np.min(result[idcs]) * thresh_fac
    mask_dog = (result) > thresh
    labeled_dog, total_num = ndimage.label(mask_dog)
    labeled_dog = periodic_label_3d1(labeled_dog)
    # labeled_dog =label_periodic_3d_consistent(result, structure)

    # np.unique(labeled_dog, return_counts=True)
    idcs_labels = {}
    for idx in np.array(idcs).T:
        label = labeled_dog[tuple(idx)]
        if label not in idcs_labels.keys():
            size = np.sum(labeled_dog == label)
            idcs_labels[label] = size
    print(thresh, idcs_labels)
    new_mask = np.zeros_like(labeled_dog)
    for key in idcs_labels.keys():
        new_mask += labeled_dog == key
    return new_mask


from scipy.ndimage import zoom


def ccp4_to_mask(mask_loc, target_shape):
    mask_map = gemmi.read_ccp4_map(mask_loc)
    mask_map = np.array(mask_map.grid)
    new_shape = np.array(target_shape) / np.array(mask_map.shape)

    ball_of_truth_rescaled = zoom(mask_map, new_shape)
    mask = ball_of_truth_rescaled > 0.2
    # mask = mask[::-1,::-1,::-1] # ccp4 convention is opposite mtz convention?
    return mask


from plotting3d import val_distributions3

def find_wasserstein_dip(xvalues,yvalues):
    yvalues = np.asarray(yvalues)
    xvalues = np.asarray(xvalues)

    if yvalues.shape != xvalues.shape:
        raise ValueError("yvalues and xvalues must have the same shape.")

    # Step 1: Find the global minimum
    min_idx = np.argmin(yvalues)
    if min_idx != 0:
        return xvalues[min_idx]

    # Step 2: Compute gradient
    grad = np.gradient(yvalues)

    # Step 3: Find where gradient first turns negative
    neg_grad_indices = np.where(grad < 0)[0]
    if neg_grad_indices.size == 0:
        return None

    start_idx = neg_grad_indices[0]

    # Step 4: Find minimum from that point forward
    y_sub = yvalues[start_idx:]
    rel_min_idx = np.argmin(y_sub)
    true_min_idx = start_idx + rel_min_idx

    return xvalues[true_min_idx]



def get_hists(dens_xtrs, rho_dark, mask, bins):
    dhists = []
    lhists = []
    wdists = []
    for dens_xtr in dens_xtrs:
        wasser_dist, bin_centers, hist_dark, hist_light = val_distributions3(
            rho_dark,
            dens_xtr,
            mask.astype(bool),
            bins,
        )
        wdists.append(wasser_dist)
        lhists.append(hist_light)
        dhists.append(hist_dark)

    return dhists, lhists, wdists, bin_centers


def plot_hists(dhists, lhists, wdists, bin_centers, alphas):
    fig, axs = plt.subplots(4, 5, figsize=(12, 10), tight_layout=True)
    bin_width = bin_centers[1] - bin_centers[0]
    for ii, (ax, alpha) in enumerate(zip(axs.flat, alphas)):
        # ax = axs.flat[0]
        ax.bar(bin_centers, dhists[ii], alpha=0.5, label="Dark", width=bin_width)
        ax.bar(bin_centers, lhists[ii], alpha=0.5, label="Extrapol.", width=bin_width)
        ax.set_title(f"Alpha: {alpha:.2f} \n Wasserstein: {wdists[ii]:.4f}")
    ax.legend()
    return fig, axs
