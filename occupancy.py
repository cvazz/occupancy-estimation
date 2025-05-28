import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import ndimage
from scipy import stats


from generate_objects import (
    make_objs,
    make_working_vars,
    generate_obj,
    generate_obj_cistrans,
    get_fig_folders,
)

################################################################################
######################### Data preparation #####################################
################################################################################


def make_f_xtr(
    alphas, f_dark, f_light, f_angle, version, noise_level=0, poisson_noise=-1
):
    """
    Make extrapolated structure factor amplitudes.

    Parameters
    ----------
    alphas : ndarray
        1D array of floating point extrapolation factors
    f_dark : ndarray
        Complex array of structure factors for the dark state
    f_light : ndarray
        Complex array of structure factors for the light state
    f_angle : ndarray
        Array of phase angles (in radians)
    version : int
        Integer specifying the calculation method (only version 1 is used)
    noise_level : float, optional
        Float describing the noise level to add, by default 0
    poisson_noise : float, optional
        Parameter for Poisson noise; disabled if < 0, by default -1

    Returns
    -------
    ndarray
        Array of extrapolated structure factors
    """

    f_dark_abs = np.abs(f_dark)
    f_light_abs = np.abs(f_light)
    noise = np.random.normal(size=f_dark_abs.shape) * np.mean(f_dark_abs) * noise_level
    delta_f = (f_light_abs - f_dark_abs) + noise
    delta_f_with_phase = delta_f * np.exp(1j * f_angle)
    many_none = (None,) * f_dark.ndim
    f_xtr_abs = np.abs(
        2 / alphas[(slice(None), *many_none)] * (delta_f)[None, ...]
        + f_dark_abs[None, ...]
    )
    if poisson_noise > 0:
        pixel_count = poisson_noise / f_xtr_abs.sum()
        f_xtr_abs = np.random.poisson(f_xtr_abs * pixel_count) / pixel_count

    match version:
        case 1:
            f_xtr = (
                2 / alphas[(slice(None), *many_none)] * (delta_f_with_phase[None, ...])
                + f_dark[None, ...]
            )
        case 2:
            f_xtr = f_xtr_abs * np.exp(1j * f_angle)[None, ...]
        case 3:
            f_xtr = 2 / alphas * f_light
        case 4:
            pass

    return f_xtr


def make_f_xtr_phased(alphas, f_dark, delta_f, noise_level=0):
    f_dark_abs = np.abs(f_dark)
    noise = np.random.normal(size=f_dark_abs.shape) * np.mean(f_dark_abs) * noise_level
    delta_f_with_phase = delta_f + noise
    many_none = (None,) * f_dark.ndim
    f_xtr = (
        1 / alphas[(slice(None), *many_none)] * (delta_f_with_phase[None, ...])
        + f_dark[None, ...]
    )
    return f_xtr


################################################################################
############################## Processing ######################################
################################################################################


def comp_cc(f_xtra, f_dark, f_light):
    delta_meas = f_light - f_dark
    delta_xtra = f_xtra - f_dark
    out = pearsonr(delta_meas.flatten(), delta_xtra.flatten())
    return out[0]


def x8_inspired(
    f_dark,
    f_light,
    f_xtrs,
):
    corrs = np.empty(len(f_xtrs))
    for ii, f_xtr in enumerate(f_xtrs):
        corrs[ii] = comp_cc(f_xtr, f_dark, f_light)
    return corrs


################################ PanDDA ########################################


def pandda(
    f_dark,
    f_xtrs,
    mask_pks,
):
    rho_dark = np.fft.ifftn(f_dark).real
    mean_global = np.empty(len(f_xtrs))
    mean_local = np.empty(len(f_xtrs))
    for ii, f_xtr in enumerate(f_xtrs):
        rho_xtr = np.fft.ifftn(f_xtr).real
        mean_global[ii] = pearsonr(
            rho_xtr[~mask_pks].flatten(), rho_dark[~mask_pks].flatten()
        )[0]
        mean_global[ii] = pearsonr(rho_xtr.flatten(), rho_dark.flatten())[0]
        mean_local[ii] = pearsonr(
            rho_xtr[mask_pks].flatten(), rho_dark[mask_pks].flatten()
        )[0]
    return mean_local, mean_global


######################## Negative Sum Explosion ################################


def get_fits(neg_sum, alpha_invs, n_largest):
    a_sorted = np.argsort(alpha_invs)
    m_lowest = a_sorted <= n_largest
    m_biggest = a_sorted >= len(a_sorted) - n_largest
    res_lowest = stats.linregress(alpha_invs[m_lowest], -neg_sum[m_lowest])
    res_biggest = stats.linregress(alpha_invs[m_biggest], -neg_sum[m_biggest])
    alpha_line = np.linspace(np.min(alpha_invs), np.max(alpha_invs), 5)
    fit_lowest = res_lowest.intercept + res_lowest.slope * alpha_line
    fit_biggest = res_biggest.intercept + res_biggest.slope * alpha_line
    return alpha_line, fit_lowest, fit_biggest


def marius(f_xtrs, mask=None):
    mask = np.ones(f_xtrs.shape[1:], bool) if mask is None else mask
    arrlen = len(f_xtrs)
    neg_sum = np.empty((arrlen))
    densities = np.empty(f_xtrs.shape)
    for ii, f_xtr in enumerate(f_xtrs):
        dens = np.fft.ifftn(f_xtr).real
        densities[ii] = dens
        dens = dens[mask]
        neg_sum[ii] = np.sum(dens[dens < 0])
        # print(ii, neg_sum[ii])
    return densities, neg_sum


def marius_masked(f_xtrs, mask_pks):
    arrlen = len(f_xtrs)
    neg_sum = np.empty((arrlen))
    densities = np.empty(f_xtrs.shape)
    for ii, f_xtr in enumerate(f_xtrs):
        dens = np.fft.ifftn(f_xtr).real
        densities[ii] = dens
        dens = dens[mask_pks]
        neg_sum[ii] = np.sum(dens[dens < 0])
    return densities, neg_sum


############################### Xtrapol8 #######################################


def x8_density_map_f1(f_xtrs, mask_pks, fofo, obj0):
    arrlen = len(f_xtrs)
    peak_sum = np.empty((arrlen))
    real_CC = np.empty((arrlen))
    for ii, f_xtr in enumerate(f_xtrs):
        dens = np.fft.ifftn(f_xtr).real
        real_CC[ii] = pearsonr(dens.flatten(), obj0.flatten())[0]
        peak_sum[ii] = np.abs(dens[mask_pks]).sum() / np.abs(dens).sum()
        # peak_sum[ii] = np.abs(dens[mask_pks]).sum()/np.abs(fofo[mask_pks]).sum()
    return peak_sum, real_CC


def x8_density_map_fdiff(f_xtrs, mask_pks, obj0, fofo):
    arrlen = len(f_xtrs)
    peak_sum = np.empty((arrlen))
    real_CC = np.empty((arrlen))
    for ii, f_xtr in enumerate(f_xtrs):
        dens = np.fft.ifftn(f_xtr).real
        real_CC[ii] = pearsonr((dens - obj0).flatten(), fofo.flatten())[0]
        peak_sum[ii] = np.abs((dens - obj0)[mask_pks]).sum() / np.abs(dens - obj0).sum()

    return peak_sum, real_CC


def x8_density_map_fdiff_norm(f_xtrs, mask_pks, obj0, fofo):
    arrlen = len(f_xtrs)
    peak_sum = np.empty((arrlen))
    real_CC = np.empty((arrlen))
    obj0 = obj0 / np.max(obj0)
    for ii, f_xtr in enumerate(f_xtrs):
        dens = np.fft.ifftn(f_xtr).real
        dens = dens / np.max(dens)
        real_CC[ii] = pearsonr((dens - obj0).flatten(), fofo.flatten())[0]
        peak_sum[ii] = np.abs((dens - obj0)[mask_pks]).sum() / np.abs(dens - obj0).sum()

    return peak_sum, real_CC


def x8_density_map_fdiff_noisyf0(f_xtrs, mask_pks, obj0, fofo):
    arrlen = len(f_xtrs)
    peak_sum = np.empty((arrlen))
    real_CC = np.empty((arrlen))
    f0 = np.fft.fftn(obj0)
    noise = np.random.normal(size=obj0.shape) * np.mean(np.abs(f0))
    obj_mod = np.fft.ifftn(f0 + noise).real

    for ii, f_xtr in enumerate(f_xtrs):
        dens = np.fft.ifftn(f_xtr).real
        real_CC[ii] = pearsonr((dens - obj_mod).flatten(), fofo.flatten())[0]
        peak_sum[ii] = (
            np.abs((dens - obj_mod)[mask_pks]).sum() / np.abs(dens - obj0).sum()
        )

    return peak_sum, real_CC


def x8_density_map_fdiff_alpha(f_xtrs, mask_pks, obj0, fofo, alpha_xtrs):
    arrlen = len(f_xtrs)
    peak_sum = np.empty((arrlen))
    real_CC = np.empty((arrlen))
    for ii, f_xtr in enumerate(f_xtrs):
        dens = np.fft.ifftn(f_xtr).real
        dens = dens / alpha_xtrs[ii] / 10
        real_CC[ii] = pearsonr((dens - obj0).flatten(), fofo.flatten())[0]
        diff = dens - obj0
        peak_sum[ii] = np.abs(diff[mask_pks]).sum() / np.abs(diff).sum()

    return peak_sum, real_CC


############################### my way #########################################
def root_finding(f_xtrs, alpha_xtrs):
    """
    finds all densities that
    """
    dens_xtrs = np.array([np.fft.ifftn(f_xtr).real for f_xtr in f_xtrs])
    changing_at = np.ones(dens_xtrs.shape[1:]) * -1
    adiff = np.diff(alpha_xtrs)  # determine whether counting up or down
    assert (np.sign(adiff) == np.sign(adiff)[0]).all()
    assert np.sign(adiff)[0] != 0
    counting_order = slice(None, None, int(np.sign(adiff[0])))
    all_negs = np.argwhere(dens_xtrs < 0)
    for all_neg in all_negs[counting_order]:
        best, index = all_neg[0], all_neg[1:]
        changing_at[tuple(index)] = alpha_xtrs[best]
    return changing_at


def root_finding2(f_xtrs, alpha_xtrs):
    """
    finds all densities that
    """
    dens_xtrs = np.array([np.fft.ifftn(f_xtr).real for f_xtr in f_xtrs])
    changing_at = np.ones(dens_xtrs.shape[1:]) * np.nan
    adiff = np.diff(alpha_xtrs)  # determine whether counting up or down
    assert (np.sign(adiff) == np.sign(adiff)[0]).all()
    assert np.sign(adiff)[0] != 0
    counting_order = slice(None, None, int(np.sign(adiff[0])))
    only_with_change = np.sign(dens_xtrs[0]) != np.sign(dens_xtrs[-1])
    # only_with_change = np.stack([only_with_change for _ in range(len(dens_xtrs))],axis=0)
    changing_at = np.ones(np.sum(only_with_change)) * np.nan
    # all_negs = np.argwhere([only_with_change]<0)
    for ii, dens_xtr in enumerate(dens_xtrs[counting_order]):
        # best, index = all_neg[0], all_neg[1:]
        indices = np.argwhere(dens_xtr[only_with_change] < 0)
        for index in indices:
            changing_at[tuple(index)] = alpha_xtrs[ii]
    return changing_at


def root_finding_blobs(f_xtrs, alpha_xtrs, mask_pks_neg):
    dens_xtrs = np.array([np.fft.ifftn(f_xtr).real for f_xtr in f_xtrs])
    blobs, blob_number = ndimage.label(mask_pks_neg)
    adiff = np.diff(alpha_xtrs)  # determine whether counting up or down
    assert (np.sign(adiff) == np.sign(adiff)[0]).all()
    assert np.sign(adiff)[0] != 0
    counting_order = slice(None, None, int(np.sign(adiff[0])))
    integrated_peaks = np.empty((len(dens_xtrs), blob_number))
    for dens_id, dens_xtr in enumerate(dens_xtrs):
        for blob_id in range(blob_number):
            integrated_peaks[dens_id, blob_id] = np.sum(dens_xtr[blobs == blob_id + 1])
    # print(integrated_peaks)
    all_negs = np.argwhere(integrated_peaks < 0)
    changing_at = np.ones(blob_number) * -1
    for all_neg in all_negs[counting_order]:
        best, index = all_neg[0], all_neg[1:]
        changing_at[tuple(index)] = alpha_xtrs[best]
    return changing_at


def root_finding_blobs2(f_xtrs, alpha_xtrs, mask_pks_neg):
    dens_xtrs = np.array([np.fft.ifftn(f_xtr).real for f_xtr in f_xtrs])
    blobs, blob_number = ndimage.label(mask_pks_neg)
    adiff = np.diff(alpha_xtrs)  # determine whether counting up or down
    assert (np.sign(adiff) == np.sign(adiff)[0]).all()
    assert np.sign(adiff)[0] != 0
    counting_order = slice(None, None, int(np.sign(adiff[0])))
    integrated_peaks = np.empty((len(dens_xtrs), blob_number))
    for dens_id, dens_xtr in enumerate(dens_xtrs):
        for blob_id in range(blob_number):
            integrated_peaks[dens_id, blob_id] = np.sum(dens_xtr[blobs == blob_id + 1])
    # print(integrated_peaks)
    # all_negs = np.argwhere(integrated_peaks<0)
    # changing_at = np.ones(blob_number)*-1
    # for all_neg in all_negs[counting_order]:
    #     best, index = all_neg[0], all_neg[1:]
    #     changing_at[tuple(index)] = alpha_xtrs[best]
    only_with_change = np.sign(integrated_peaks[0]) != np.sign(integrated_peaks[-1])
    changing_at = np.ones(np.sum(only_with_change)) * np.nan
    for ii, int_pks in enumerate(integrated_peaks[counting_order]):
        indices = np.argwhere(int_pks[only_with_change] < 0)
        for index in indices:
            changing_at[tuple(index)] = alpha_xtrs[counting_order][ii]
    return changing_at
